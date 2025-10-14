import multiprocessing
from abc import ABC
import os
import pandas as pd
from pandas.tseries.offsets import DateOffset
import numpy as np

class GainStrategyIF(ABC):
    def record_gain_strategy(self, drop_matches: pd.DataFrame, entire_ticker_data: pd.DataFrame, ticker: str, output_file_path: str, output_file_lock: multiprocessing.Lock):
        pass

    def aggregate_results(self, output_file_path: str):
        pass

class LookupPricesAtRandomMonths(GainStrategyIF):
    def __init__(self, period_month_lookup_after_drop: int):
        self._period_month_lookup_after_drop = period_month_lookup_after_drop

    def record_gain_strategy(self,
                           drop_matches: pd.DataFrame,
                           entire_ticker_data: pd.DataFrame,
                           ticker: str,
                           output_file_path: str,
                           output_file_lock: multiprocessing.Lock):
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

            # Save table to CSV or log file
            with output_file_lock:
                output_file_exists = os.path.exists(output_file_path)
                df_table.to_csv(output_file_path, mode='a', index=False, header=not output_file_exists)

    def aggregate_results(self, output_file_path: str):
        pass

class FirstGainAbovePercent(GainStrategyIF):
    def __init__(self, percentage_to_pass: float):
        self._percentage_to_pass = percentage_to_pass

    def record_gain_strategy(self, drop_matches: pd.DataFrame,
                             entire_ticker_data: pd.DataFrame,
                             ticker: str,
                             output_file_path: str,
                             output_file_lock: multiprocessing.Lock):

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

            target = row["closeadj"] * (1 + self._percentage_to_pass / 100)

            # Find the first index *after i* where Close > target
            mask = closes[i + 1:] > target
            j = i + 1 + np.argmax(mask)
            first_up_date = dates[j]
            first_up_price = round(closes[j].item(),2)

            record[f"up_{self._percentage_to_pass}%_from_drop_date"] = first_up_date
            record[f"price_at_up"] = first_up_price
            record[f'Days_to_be_up_{self._percentage_to_pass}%_from_drop'] = (first_up_date - idx).days

            df_table = pd.DataFrame([record])
            with output_file_lock:
                output_file_exists = os.path.exists(output_file_path)
                df_table.to_csv(output_file_path, mode='a', index=False, header=not output_file_exists)

    def aggregate_results(self, output_file_path: str):
        days_until_up = pd.read_csv(output_file_path)
        record = {"total_events": len(days_until_up)}

        for i in range(1,13):
            mask_below = days_until_up[f"Days_to_be_up_{self._percentage_to_pass}%_from_drop"] <= 30*i
            mask_above = days_until_up[f"Days_to_be_up_{self._percentage_to_pass}%_from_drop"] > 30*(i-1)
            mask = mask_above & mask_below
            record[f"month #{i}"] = len(days_until_up[mask])

        record[f"above year"] = len(days_until_up[days_until_up[f"Days_to_be_up_{self._percentage_to_pass}%_from_drop"] > 30*12])


        df_table = pd.DataFrame([record])
        aggregation_file_path = f"{os.path.dirname(output_file_path)}/aggregation.csv"
        if os.path.exists(aggregation_file_path):
            os.remove(aggregation_file_path)
        df_table.to_csv(aggregation_file_path, mode='a', index=False, header=True)

if __name__ == "__main__":
    FirstGainAbovePercent(5).aggregate_results(
        "yfinance_artifacts/time_to_rise/since_2015/time_to_rise_5%_MA5_drop_10%/prices.csv")