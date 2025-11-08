import time

import nasdaqdatalink
from multiprocessing import Pool
import pandas as pd
import os
from typing import List, NamedTuple, Dict
from gain_events_finder import GainStrategyIF, FirstGainAbovePercent
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sql_connector import DropRun
from utils import LOCAL_HISTORIC_DATA_DIR

DB_URL = "mysql+pymysql://root:brsz8bvb@localhost/stocks_project"
engine = create_engine(DB_URL)
Session = sessionmaker(bind=engine)

API_KEY = "yZMdFB7ybWMNkpv_NA3v"
nasdaqdatalink.ApiConfig.api_key = API_KEY

class TickerInterval(NamedTuple):
    ticker: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp

class Snp500TickersExtractor:

    def __init__(self, snp_tickers: pd.DataFrame):
        self._df = snp_tickers
        self._df['start_date'] = pd.to_datetime(self._df['start_date'])
        self._df['end_date'] = pd.to_datetime(self._df['end_date'])

    def generate_overlap_mask(self, requested_start_date: pd.Timestamp) -> pd.Series:

        # Condition 1: If interval_end_date is missing (open-ended interval)
        cond_open_ended = self._df['end_date'].isna()

        # Condition 2: If requested_date is strictly *between* interval dates
        cond_between = (requested_start_date < self._df['end_date']) & \
                       (requested_start_date > self._df['start_date'])

        # Condition 3: If requested_date is *before* the interval's start
        cond_before_start = requested_start_date < self._df['start_date']

        filter_mask = cond_open_ended | cond_between | cond_before_start

        return filter_mask

    def find_relevant_tickers(self, requested_start_date: pd.Timestamp) -> pd.DataFrame:
        overlap_mask = self.generate_overlap_mask(requested_start_date)
        filtered_df = self._df[overlap_mask]
        filtered_df['start_date'] = filtered_df['start_date'].apply(lambda x: max(x, requested_start_date))
        return filtered_df


class DropsEventsFinder:

    def __init__(self,
                 directory:str,
                 start_date: pd.Timestamp,
                 drop_percent: float,
                 MA: int):

        self._sql_run_id = self._create_new_run_in_sql(MA, drop_percent, start_date)
        self._start_date = start_date
        self._drop_percent = drop_percent
        self._MA = MA

        self._logs_dir = f"{os.path.dirname(os.path.abspath(__file__))}/{directory}/logs"
        os.makedirs(self._logs_dir, exist_ok=True)

    @staticmethod
    def _create_new_run_in_sql(MA, drop_percent, start_date) -> int:
        session = Session()
        new_run = DropRun(
            MA_interval=MA,
            drop_percent=drop_percent,
            since_year=start_date
        )
        session.add(new_run)
        session.commit()
        run_id = new_run.id
        session.close()
        return run_id

    def _enrich_df_pre_process(self, full_df: pd.DataFrame):
        full_df[f"MA"] = full_df["close"].rolling(window=self._MA).mean()
        full_df["PctChange"] = full_df[f"MA"].pct_change(periods=self._MA)

    def _find_drops(self, full_df: pd.DataFrame, start_Date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        cond_percentage_drop = full_df["PctChange"] < (-1 * self._drop_percent / 100)
        cond_start_date = full_df.index > start_Date
        if pd.isna(end_date) or end_date is None:
            cond_end_date = pd.Series(True, index=full_df.index)
        else:
            cond_end_date = full_df.index < end_date

        matches = full_df[cond_percentage_drop & cond_start_date & cond_end_date]
        return matches

    def _find_drop_and_record_price(self, relevant_ticker: TickerInterval) -> List[Dict]:
        try:
            # data = nasdaqdatalink.get_table('SHARADAR/SEP', date={'gte': self._start_date}, ticker=relevant_ticker.ticker).iloc[::-1]
            data = pd.read_csv(f'{LOCAL_HISTORIC_DATA_DIR}/{relevant_ticker.ticker}.csv')
            data.index = pd.to_datetime(data.date)
            data = data[data.index > self._start_date]
            if data.empty:
                raise Exception(f"empty dataframe for ticker {relevant_ticker}")
            self._enrich_df_pre_process(data)
            matches = self._find_drops(data, relevant_ticker.start_date, relevant_ticker.end_date)
            return matches[['ticker','date','close','MA','high','low','PctChange']]


            #TODO: this line should be done outside of this scope, with matches as input
            return self._gain_strategy.record_gain_strategy(matches, data, relevant_ticker.ticker)

        except Exception as e:
            log_file_path = f"{self._logs_dir}/{relevant_ticker.ticker}.csv"
            with open(log_file_path, 'w') as f:
                f.write(str(e))

    def _upload_to_sql_database(self, results: List[List[Dict]]):
        combined_df = pd.concat([df for df in results if (df is not None and not df.empty)], ignore_index=True)

        #put results in SQL tables
        combined_df["drop_run_id"] = self._sql_run_id
        combined_df.to_sql("drop_events", con=engine, if_exists="append", index=False, method="multi")


    def run_workers_pool(self) -> int:
        snp_tickers_df = pd.read_csv("sp500_ticker_start_end.csv", usecols=["ticker", "start_date","end_date"])
        ticker_extractor = Snp500TickersExtractor(snp_tickers_df)
        relevant_tickers_df = ticker_extractor.find_relevant_tickers(self._start_date)
        relevant_tickers = [TickerInterval(*x) for _, x in relevant_tickers_df.iterrows()]

        with Pool(processes=8) as pool:
            results: List[List[Dict]] = pool.map(self._find_drop_and_record_price, relevant_tickers)
        self._upload_to_sql_database(results)

        # # DEBUGGING CODE
        # results = []
        # relevant_tickers = [TickerInterval(ticker="NEM", start_date=pd.Timestamp(2015,1,1), end_date=None),
        #                     TickerInterval(ticker="NVDA", start_date=pd.Timestamp(2015,1,1), end_date=None)]
        # for y in relevant_tickers:
        #     results.append(self._find_drop_and_record_price(y))
        # self.aggregate_results_in_sql(results)

        # self._gain_strategy.aggregate_results(results, self._output_file)


if __name__ == "__main__":
    # MA = 1
    # drop_percent = 20
    tick = time.time()
    since_year = 2015
    for MA in [1,5,10,20,30]:
        for drop_percent in [10,15,20,25]:
            directory = f"nasdaqdatalink_artifacts/drops_finder/since_{since_year}/MA{MA}_drop_{drop_percent}%"
            DropsEventsFinder(directory=directory,
                              start_date=pd.Timestamp(since_year, 1, 1),
                              drop_percent=drop_percent,
                              MA=MA).run_workers_pool()
    tock = time.time()
    print(tock-tick)
