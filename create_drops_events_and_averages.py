import multiprocessing
import yfinance as yf
from multiprocessing import Pool
import pandas as pd
import os
from pandas.tseries.offsets import DateOffset
from typing import List, NamedTuple

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

    PERCENTAGE_DROP_THRESHOLD = -0.1
    START_DATE = pd.Timestamp(2020, 1, 1)
    INTERVAL = "1d"
    PERIOD_MONTH_LOOKUP_AFTER_DROP = 48

    def __init__(self, directory:str, lock: multiprocessing.Lock):
        self._output_file = f"{os.path.dirname(os.path.abspath(__file__))}/{directory}/prices.csv"
        if os.path.exists(self._output_file):
            os.remove(self._output_file)

        self._output_file_lock = lock

        self._logs_dir = f"{os.path.dirname(os.path.abspath(__file__))}/{directory}/logs"
        os.makedirs(self._logs_dir, exist_ok=True)

    def _find_drops(self, full_df: pd.DataFrame, start_Date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        full_df["PctChange"] = full_df["Close"].pct_change()
        cond_percentage_drop = full_df["PctChange"] < self.PERCENTAGE_DROP_THRESHOLD
        cond_start_date = full_df.index > start_Date
        if pd.isna(end_date) or end_date is None:
            cond_end_date = pd.Series(True, index=full_df.index)
        else:
            cond_end_date = full_df.index < end_date

        matches = full_df[cond_percentage_drop & cond_start_date & cond_end_date]
        return matches

    def _iterate_rows_and_append_future_prices(self, drops_df: pd.DataFrame, full_df: pd.DataFrame, ticker: str):
        for idx, row in drops_df.iterrows():
            drop_date = idx
            record = {
                "ticker": ticker,
                "DropDate": drop_date.date(),
                "Close": round(row["Close"].iloc[0], 2),
                "PctChange": f"{round(100 * row['PctChange'].iloc[0], 2)}%"
            }

            offsets = {f"{i}M": DateOffset(months=i) for i in range(1, self.PERIOD_MONTH_LOOKUP_AFTER_DROP + 1)}

            for label, offset in offsets.items():
                target_date = drop_date + offset
                # get the next available trading day
                future_dates = full_df.index[full_df.index >= target_date]
                if len(future_dates) > 0:
                    future_date = future_dates[0]
                    # percentage_diff = round(100*(data.loc[future_date, 'Close'][0]-row['Close'][0])/row['Close'][0],1)
                    record[f"Close+{label}"] = full_df.loc[future_date, 'Close'].iloc[0]
                else:
                    record[f"Close+{label}"] = None

            # Convert to DataFrame for table representation
            df_table = pd.DataFrame([record])

            # Save table to CSV or log file
            with self._output_file_lock:
                output_file_exists = os.path.exists(self._output_file)
                df_table.to_csv(self._output_file, mode='a', index=False, header=not output_file_exists)

    def _find_drop_and_record_price(self, relevant_ticker: TickerInterval):
        try:
            data = yf.download(relevant_ticker.ticker, start=self.START_DATE, interval=self.INTERVAL)
            data.index = pd.to_datetime(data.index)

            matches = self._find_drops(data, relevant_ticker.start_date, relevant_ticker.end_date)
            self._iterate_rows_and_append_future_prices(matches, data, relevant_ticker.ticker)

        except Exception as e:
            log_file_path = f"{self._logs_dir}/{relevant_ticker.ticker}.csv"
            with open(log_file_path, 'w') as f:
                f.write(str(e))

    def run_workers_pool(self):
        snp_tickers_df = pd.read_csv("sp500_ticker_start_end.csv")
        ticker_extractor = Snp500TickersExtractor(snp_tickers_df)
        relevant_tickers_df = ticker_extractor.find_relevant_tickers(self.START_DATE)
        relevant_tickers = [TickerInterval(*x) for _, x in relevant_tickers_df.iterrows()]

        with Pool(processes=8) as pool:
            results = [pool.apply_async(self._find_drop_and_record_price, (y,) ) for y in relevant_tickers]

            for r in results:
                r.wait()


if __name__ == "__main__":
    directory = "alternating_snp500"
    with multiprocessing.Manager() as manager:
        lock = manager.Lock()
        d = DropsFinder(directory,lock).run_workers_pool()
