from abc import ABC
import multiprocessing
import pandas as pd
from pandas.tseries.offsets import DateOffset
import numpy as np
from typing import List, Dict
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from sql_connector import UpRun, DropEvent, DropRun
from utils import LOCAL_HISTORIC_DATA_DIR

DB_URL = "mysql+pymysql://root:brsz8bvb@localhost/stocks_project"
engine = create_engine(DB_URL)
Session = sessionmaker(bind=engine)

class GainStrategyIF(ABC):
    def record_gain_strategy(self, drop_matches: pd.DataFrame, ticker: str):
        pass

    def aggregate_results(self,results: List[List[Dict]], output_file_path: str):
        pass

class LookupPricesAtRandomMonths(GainStrategyIF):
    def __init__(self, period_month_lookup_after_drop: int):
        self._period_month_lookup_after_drop = period_month_lookup_after_drop

    def record_gain_strategy(self,
                           drop_matches: pd.DataFrame,
                           entire_ticker_data: pd.DataFrame,
                           ticker: str):
        for idx, row in drop_matches.iterrows():
            drop_date = idx
            record = {
                "ticker": ticker,
                "DropDate": drop_date.date(),
                "Close": round(row["close"].iloc[0], 2),
                "PctChange": f"{round(100 * row['PctChange'].iloc[0], 2)}%"
            }

            offsets = {f"{i}M": DateOffset(months=i) for i in range(1, self._period_month_lookup_after_drop + 1)}

            for label, offset in offsets.items():
                target_date = drop_date + offset
                # get the next available trading day
                future_dates = entire_ticker_data.index[entire_ticker_data.index >= target_date]
                if len(future_dates) > 0:
                    future_date = future_dates[0]
                    # percentage_diff = round(100*(data.loc[future_date, 'Close'][0]-row['Close'][0])/row['Close'][0],1)
                    record[f"Close+{label}"] = entire_ticker_data.loc[future_date, 'close'].iloc[0]
                else:
                    record[f"Close+{label}"] = None

            # Convert to DataFrame for table representation
            df_table = pd.DataFrame([record])

            # # Save table to CSV or log file
            # with output_file_lock:
            #     output_file_exists = os.path.exists(output_file_path)
            #     df_table.to_csv(output_file_path, mode='a', index=False, header=not output_file_exists)

    def aggregate_results(self,results: List[List[Dict]], output_file_path: str):
        pass

class FirstGainAbovePercent(GainStrategyIF):

    AGGREGATION_FILE_NAME = "aggregation.csv"

    def __init__(self, rise_percent: int, ultimate_sell_period: int, drop_run_id:int):
        self._sql_run_id = self._create_new_run_in_sql(drop_run_id, rise_percent, ultimate_sell_period)

        self._drop_run_id = drop_run_id
        self._rise_percent = rise_percent
        self._ultimate_sell_period = ultimate_sell_period
        self._lower_expectations_period = 90 #days

    @staticmethod
    def _create_new_run_in_sql(drop_run_id: int, rise_percent: int, ultimate_sell_period: int) -> int:
        session = Session()
        new_run = UpRun(
            drop_run_id=drop_run_id,
            rise_percent=rise_percent,
            ultimate_sell_period=ultimate_sell_period
        )
        session.add(new_run)
        session.commit()
        run_id = new_run.id
        session.close()
        return run_id

    def _get_optimal_rise(self, closes: np.array, highs: np.array, lows: np.array, dates: np.array, idx_of_drop_date: int):
        offset = idx_of_drop_date + 1
        drop_price = closes[idx_of_drop_date]
        previos_lower_exp_interval = 0

        for lower_exp_power_factor, lower_expectations_interval in enumerate(range(self._lower_expectations_period, self._ultimate_sell_period+30, 30)):
            if offset > len(closes)-1:
                break

            target = drop_price * (1 + (self._rise_percent / 100) * (0.5**lower_exp_power_factor))
            mask = highs[offset: offset + lower_expectations_interval - previos_lower_exp_interval] > target
            if mask.any():
                # return the first index of the rise
                idx = offset + np.argmax(mask)
                return dates[idx], max(target, lows[idx])  # target might be lower than the acctual value since the low might be higher than target

            offset += lower_expectations_interval - previos_lower_exp_interval
            previos_lower_exp_interval = lower_expectations_interval

        # if we're here we didnt find a suitable sell point.
        # return the min(end of stock record, ultimate sell point)
        idx = min(offset, len(closes)-1)
        return dates[idx], closes[idx]

    def _get_entire_ticker_data(self, ticker:str):
        data = pd.read_csv(f'{LOCAL_HISTORIC_DATA_DIR}/{ticker}.csv')
        return data

    def record_gain_strategy(self, drop_matches: pd.DataFrame, ticker: str) -> List[Dict]:
        entire_ticker_data = self._get_entire_ticker_data(ticker)
        entire_ticker_data.date = pd.to_datetime(entire_ticker_data.date)
        entire_ticker_data.index = entire_ticker_data.date

        all_records = []
        for idx, row in drop_matches.iterrows():
            record = {
                "ticker": ticker,
                "DropDate": row["date"].date(),
                "Close": round(row["close"], 2),
                "PctChange": f"{round(100 * row['PctChange'], 2)}%"
            }
            closes = entire_ticker_data["close"].to_numpy()
            highs = entire_ticker_data["high"].to_numpy()
            lows = entire_ticker_data["low"].to_numpy()
            dates = entire_ticker_data["date"].to_numpy()
            i = entire_ticker_data.index.get_loc(row["date"])

            first_up_date, first_up_price = self._get_optimal_rise(closes, highs, lows, dates, i)

            record[f"date_of_up"] = first_up_date
            record[f"price_at_up"] = first_up_price
            record[f'days_delta_to_be_up'] = (first_up_date - row["date"]).days

            all_records.append(record)

        return all_records

    def _upload_post_process_monthly_distribution(self, df: pd.DataFrame):
        record = {"total_events": len(df)}
        df["DropDate"] = pd.to_datetime(df["DropDate"])

        for i in range(1,13):
            mask_below = df[f"days_delta_to_be_up"] <= 30*i
            mask_above = df[f"days_delta_to_be_up"] > 30*(i-1)
            mask = mask_above & mask_below
            record[f"_{i}_month"] = len(df[mask])

        record[f"above_year"] = len(df[df[f"days_delta_to_be_up"] > 30*12])
        record[f"never_sold"] = 2*record['total_events'] - sum(record.values())
        record["drops_after_2020"] = len(df[(df["DropDate"] > pd.Timestamp(2020,1,1)) & (df["DropDate"] < pd.Timestamp(2021,1,1))])

        with Session() as session:
            session.query(UpRun).filter(UpRun.id == self._sql_run_id).update(record)
            session.commit()

    def _get_drop_matches(self):
        query = select(DropEvent).where(DropEvent.drop_run_id == self._drop_run_id)
        df = pd.read_sql(query, con=engine)
        return df

    def _upload_to_sql_database(self, results_df: pd.DataFrame):
        results_df["up_run_id"] = self._sql_run_id
        results_df.to_sql("up_events", con=engine, if_exists="append", index=False, method="multi")

    def run_workers_pool(self):
        drop_matches = self._get_drop_matches()
        grouped_dfs = {ticker: sub_df for ticker, sub_df in drop_matches.groupby("ticker")}
        with multiprocessing.Pool(processes=8) as Pool:
            results = Pool.starmap(self.record_gain_strategy, [(sub_df, ticker) for ticker, sub_df in grouped_dfs.items()])

        # # DEBUG POINT
        # results = [self.record_gain_strategy(list(grouped_dfs.values())[0],list(grouped_dfs.keys())[0])]

        combined_results: List[Dict] = []
        for r in results:
            combined_results.extend(r)
        combined_df = pd.DataFrame(combined_results)

        self._upload_to_sql_database(combined_df)
        self._upload_post_process_monthly_distribution(combined_df)


if __name__ == "__main__":

    with Session() as session:
        stmt = select(DropRun.id)
        drop_run_ids = session.execute(stmt).scalars().all()

    for drop_run_id in drop_run_ids:
        for rise_percent in [25,30]:
            FirstGainAbovePercent(rise_percent=rise_percent,ultimate_sell_period=180,drop_run_id=drop_run_id).run_workers_pool()