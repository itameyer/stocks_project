from abc import ABC
import os
import pandas as pd
from pandas.tseries.offsets import DateOffset
import numpy as np
from typing import List, Dict
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sql_connector import Run

DB_URL = "mysql+pymysql://root:brsz8bvb@localhost/stocks_project"
engine = create_engine(DB_URL)
Session = sessionmaker(bind=engine)

class GainStrategyIF(ABC):
    def record_gain_strategy(self, drop_matches: pd.DataFrame, entire_ticker_data: pd.DataFrame, ticker: str, output_file_path: str):
        pass

    def aggregate_results(self,results: List[List[Dict]], output_file_path: str):
        pass

class LookupPricesAtRandomMonths(GainStrategyIF):
    def __init__(self, period_month_lookup_after_drop: int):
        self._period_month_lookup_after_drop = period_month_lookup_after_drop

    def record_gain_strategy(self,
                           drop_matches: pd.DataFrame,
                           entire_ticker_data: pd.DataFrame,
                           ticker: str,
                           output_file_path: str):
        for idx, row in drop_matches.iterrows():
            drop_date = idx
            record = {
                "ticker": ticker,
                "DropDate": drop_date.date(),
                "Close": round(row["closeadj"].iloc[0], 2),
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
                    record[f"Close+{label}"] = entire_ticker_data.loc[future_date, 'closeadj'].iloc[0]
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

    def __init__(self, rise_percent: float, ultimate_sell_period: int, sql_run_id):
        self._rise_percent = rise_percent
        self._ultimate_sell_period = ultimate_sell_period
        self._sql_run_id = sql_run_id
        self._lower_expectations_period = 90 #days

    def _get_optimal_rise(self, closes: np.array, idx_of_drop_date: int):
        offset = idx_of_drop_date + 1
        drop_price = closes[idx_of_drop_date]
        previos_lower_exp_interval = 0

        for lower_exp_power_factor, lower_expectations_interval in enumerate(range(self._lower_expectations_period, self._ultimate_sell_period+30, 30)):
            if offset > len(closes)-1:
                break

            target = drop_price * (1 + (self._rise_percent / 100) * (0.5**lower_exp_power_factor))
            mask = closes[offset: offset + lower_expectations_interval - previos_lower_exp_interval] > target
            if mask.any():
                # return the first index of the rise
                return offset + np.argmax(mask)

            offset += lower_expectations_interval - previos_lower_exp_interval
            previos_lower_exp_interval = lower_expectations_interval

        # if we're here we didnt find a suitable sell point.
        # return the min(end of stock record, ultimate sell point)
        return min(offset, len(closes)-1)


    def record_gain_strategy(self,
                             drop_matches: pd.DataFrame,
                             entire_ticker_data: pd.DataFrame,
                             ticker: str,
                             output_file_path: str) -> List[Dict]:

        all_records = []
        for idx, row in drop_matches.iterrows():
            drop_date = idx
            record = {
                "ticker": ticker,
                "DropDate": drop_date.date(),
                "Close": round(row["closeadj"], 2),
                "PctChange": f"{round(100 * row['PctChange'], 2)}%"
            }
            closes = entire_ticker_data["closeadj"].to_numpy()
            dates = entire_ticker_data.index.to_numpy()
            i = entire_ticker_data.index.get_loc(idx)

            j = self._get_optimal_rise(closes, i)
            first_up_date = dates[j] if j else pd.NaT
            first_up_price = round(closes[j].item(),2) if j else pd.NaT

            record[f"date_of_up"] = first_up_date
            record[f"price_at_up"] = first_up_price
            record[f'days_delta_to_be_up'] = (first_up_date - idx).days

            all_records.append(record)

        return all_records

    def aggregate_results(self, results: List[List[Dict]], output_file_path: str) -> pd.DataFrame:
        all_records = []
        for result in results:
            if result:
                for item in result:
                    all_records.append(item)

        df = pd.DataFrame(all_records)
        output_file_exists = os.path.exists(output_file_path)
        df.to_csv(output_file_path, mode='a', index=False, header=not output_file_exists)

        #put results in SQL tables
        df["run_id"] = self._sql_run_id
        df.to_sql("events", con=engine, if_exists="append", index=False, method="multi")

        self._post_process_monthly_distribution(df, output_file_path)

    def _post_process_monthly_distribution(self, df:pd.DataFrame, output_file_path: str):
        record = {"total_events": len(df)}
        df["DropDate"] = pd.to_datetime(df["DropDate"])

        for i in range(1,13):
            mask_below = df[f"days_delta_to_be_up"] <= 30*i
            mask_above = df[f"days_delta_to_be_up"] > 30*(i-1)
            mask = mask_above & mask_below
            record[f"month #{i}"] = len(df[mask])

        record[f"above year"] = len(df[df[f"days_delta_to_be_up"] > 30*12])
        record[f"never_sold"] = 2*record['total_events'] - sum(record.values())
        record["events_in_2020(COVID)"] = len(df[(df["DropDate"] > pd.Timestamp(2020,1,1)) & (df["DropDate"] < pd.Timestamp(2021,1,1))])

        df_table = pd.DataFrame([record])
        aggregation_file_path = f"{os.path.dirname(output_file_path)}/{self.AGGREGATION_FILE_NAME}"
        if os.path.exists(aggregation_file_path):
            os.remove(aggregation_file_path)
        df_table.to_csv(aggregation_file_path, mode='a', index=False, header=True)



if __name__ == "__main__":
    FirstGainAbovePercent(5).aggregate_results(
        "yfinance_artifacts/time_to_rise/since_2015/time_to_rise_5%_MA5_drop_10%/prices.csv")