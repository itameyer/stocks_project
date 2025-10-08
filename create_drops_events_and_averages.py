import multiprocessing
import yfinance as yf
from multiprocessing import Pool
import pandas as pd
import os
from typing import List, NamedTuple

from gain_strategy import GainStrategyIF, LookupPricesAtRandomMonths, FirstGainAbovePercent


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


class DropsFinder:

    INTERVAL = "1d"
    PERIOD_MONTH_LOOKUP_AFTER_DROP = 48

    def __init__(self,
                 directory:str,
                 lock: multiprocessing.Lock,
                 start_date: pd.Timestamp,
                 percentage_drop_threshold: float,
                 moving_days_average_window: int,
                 gain_strategy: GainStrategyIF):

        self._start_date = start_date
        self._percentage_drop_threshold = percentage_drop_threshold
        self._moving_days_average_window = moving_days_average_window
        self._gain_strategy = gain_strategy

        self._output_file = f"{os.path.dirname(os.path.abspath(__file__))}/{directory}/prices.csv"
        if os.path.exists(self._output_file):
            os.remove(self._output_file)

        self._output_file_lock = lock

        self._logs_dir = f"{os.path.dirname(os.path.abspath(__file__))}/{directory}/logs"
        os.makedirs(self._logs_dir, exist_ok=True)


    def _enrich_df(self, full_df: pd.DataFrame):
        full_df[f"MA{self._moving_days_average_window}"] = full_df["Close"].rolling(window=self._moving_days_average_window).mean()
        full_df["PctChange"] = full_df[f"MA{self._moving_days_average_window}"].pct_change(periods=self._moving_days_average_window)

    def _find_drops(self, full_df: pd.DataFrame, start_Date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        cond_percentage_drop = full_df["PctChange"] < self._percentage_drop_threshold
        cond_start_date = full_df.index > start_Date
        if pd.isna(end_date) or end_date is None:
            cond_end_date = pd.Series(True, index=full_df.index)
        else:
            cond_end_date = full_df.index < end_date

        matches = full_df[cond_percentage_drop & cond_start_date & cond_end_date]
        return matches

    def _find_drop_and_record_price(self, relevant_ticker: TickerInterval):
        try:
            data = yf.download(relevant_ticker.ticker, start=self._start_date, interval=self.INTERVAL)
            data.index = pd.to_datetime(data.index)

            self._enrich_df(data)
            matches = self._find_drops(data, relevant_ticker.start_date, relevant_ticker.end_date)
            self._gain_strategy.record_gain_strategy(matches, data, relevant_ticker.ticker, self._output_file, self._output_file_lock)

        except Exception as e:
            log_file_path = f"{self._logs_dir}/{relevant_ticker.ticker}.csv"
            with open(log_file_path, 'w') as f:
                f.write(str(e))

    def run_workers_pool(self):
        snp_tickers_df = pd.read_csv("sp500_ticker_start_end.csv")
        ticker_extractor = Snp500TickersExtractor(snp_tickers_df)
        relevant_tickers_df = ticker_extractor.find_relevant_tickers(self._start_date)
        relevant_tickers = [TickerInterval(*x) for _, x in relevant_tickers_df.iterrows()]

        # with Pool(processes=8) as pool:
        #     results = [pool.apply_async(self._find_drop_and_record_price, (y,) ) for y in relevant_tickers]
        #
        #     for r in results:
        #         r.wait()

        for y in relevant_tickers:
            self._find_drop_and_record_price(y)


if __name__ == "__main__":
    MA = 5
    drop_percent = 10
    since_year = 2015
    rise_percent = 0.1
    directory = f"time_to_rise/since_{since_year}/time_to_rise_{rise_percent*100}%_MA{MA}_drop_{drop_percent}%"
    gain_strategy = FirstGainAbovePercent(rise_percent)
    with multiprocessing.Manager() as manager:
        DropsFinder(directory=directory,
                    lock=manager.Lock(),
                    start_date=pd.Timestamp(since_year, 1, 1),
                    percentage_drop_threshold=(-1* drop_percent / 100),
                    moving_days_average_window=MA,
                    gain_strategy=gain_strategy).run_workers_pool()
