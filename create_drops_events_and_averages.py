import multiprocessing
import yfinance as yf
from multiprocessing import Pool
import pandas as pd
import os
from pandas.tseries.offsets import DateOffset


class DropsFinder:

    PERCENTAGE_DROP_THRESHOLD = -0.1
    PERIOD = "10y"
    INTERVAL = "1d"
    PERIOD_MONTH_LOOKUP_AFTER_DROP = 48

    def __init__(self, directory:str, lock: multiprocessing.Lock):
        self._tickers = self._extract_tickers()
        self._logs_dir = f"{os.path.dirname(os.path.abspath(__file__))}/{directory}/logs"
        self._output_file = f"{os.path.dirname(os.path.abspath(__file__))}/{directory}/prices.csv"
        if os.path.exists(self._output_file):
            os.remove(self._output_file)

        self._output_file_lock = lock
        os.makedirs(self._logs_dir, exist_ok=True)

    def _extract_tickers(self):
        CSV_FILE_PATH = f"{os.path.dirname(os.path.abspath(__file__))}/{directory}/{directory}.csv"
        csv = pd.read_csv(CSV_FILE_PATH)
        return list(set(csv["Ticker"]))

    def _find_drop_and_record_price(self, ticker):
        try:
            data = yf.download(ticker, period=self.PERIOD, interval=self.INTERVAL)
            data.index = pd.to_datetime(data.index)
            data["PctChange"] = data["Close"].pct_change()

            # Find drops
            matches = data[data["PctChange"] < self.PERCENTAGE_DROP_THRESHOLD]

            if matches.empty:
                return
            else:
                for idx, row in matches.iterrows():
                    first_date = idx
                    first_row = row
                    record = {
                        "ticker": ticker,
                        "DropDate": first_date.date(),
                        "Close": round(first_row["Close"].iloc[0],2),
                        "PctChange": f"{round(100 * first_row['PctChange'].iloc[0], 2)}%"
                    }

                    offsets = {f"{i}M": DateOffset(months=i) for i in range(1, self.PERIOD_MONTH_LOOKUP_AFTER_DROP + 1)}

                    for label, offset in offsets.items():
                        target_date = first_date + offset
                        # get the next available trading day
                        future_dates = data.index[data.index >= target_date]
                        if len(future_dates) > 0:
                            future_date = future_dates[0]
                            # percentage_diff = round(100*(data.loc[future_date, 'Close'][0]-first_row['Close'][0])/first_row['Close'][0],1)
                            record[f"Close+{label}"] = data.loc[future_date, 'Close'].iloc[0]
                        else:
                            record[f"Close+{label}"] = None

                    # Convert to DataFrame for table representation
                    df_table = pd.DataFrame([record])

                    # Save table to CSV or log file
                    with self._output_file_lock:
                        output_file_exists = os.path.exists(self._output_file)
                        df_table.to_csv(self._output_file, mode='a', index=False, header=not output_file_exists)

        except Exception as e:
            log_file_path = f"{self._logs_dir}/{ticker}.csv"
            with open(log_file_path, 'w') as f:
                f.write(str(e))

    def run_workers_pool(self):
        with Pool(processes=8) as pool:
            results = [pool.apply_async(self._find_drop_and_record_price, (ticker,)) for ticker in self._tickers]

            for r in results:
                r.wait()

if __name__ == "__main__":
    directory = "100_companies"
    with multiprocessing.Manager() as manager:
        lock = manager.Lock()
        d = DropsFinder(directory,lock).run_workers_pool()


