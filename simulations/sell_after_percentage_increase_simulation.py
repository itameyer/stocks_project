
import pandas as pd
from enum import Enum, auto
from typing import NamedTuple
from datetime import datetime
import numpy
from sql_connector import UpRun, DropEvent, DropRun, UpEvent
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
DB_URL = "mysql+pymysql://root:brsz8bvb@localhost/stocks_project"
engine = create_engine(DB_URL)
Session = sessionmaker(bind=engine)

class CommandType(Enum):
    BUY = auto()
    SELL = auto()

class SellPoint(NamedTuple):
    date: datetime
    price: float

class Event:
    def __init__(self, date: datetime, command_type: CommandType, num_stocks: int, price: float, event_id: int, ticker: str):
        self.date = date
        self.command_type = command_type
        self.num_stocks = num_stocks
        self.price = price
        self.event_id = event_id
        self.ticker = ticker

    def __lt__(self, other):
        return self.date < other.date

class SellEvent(Event):
    def __init__(self, *args, **kwargs):
        self.date_bought = kwargs.pop('date_bought', None)
        self.price_on_buy = kwargs.pop('price_on_buy', None)
        super().__init__(*args, **kwargs)

class BuyEvent(Event):
    def __init__(self, *args, **kwargs):
        self.date_to_sell = kwargs.pop('date_to_sell', None)
        self.price_on_sell = kwargs.pop('price_on_sell', None)
        super().__init__(*args, **kwargs)


class SellAfterPercentageIncreaseSimulation:

    INITIAL_SUM = 150_000

    def __init__(self, prices_df: pd.DataFrame, sum_per_buy_order: int, max_concurrent_buys_per_stock: int):
        self._curr_sum = self.INITIAL_SUM
        self._prices_df = prices_df
        self._sum_per_buy_order = sum_per_buy_order
        self._max_concurrent_buys_per_stock = max_concurrent_buys_per_stock

    def _create_list_of_buy_and_sell_dates(self):
        event_list = []
        for idx, row in self._prices_df.iterrows():
            ticker = row["ticker"]
            buy_date = pd.to_datetime(row["DropDate"])
            buy_price = round(row["Close"],2)
            sell_date = pd.to_datetime(row[f"date_of_up"])
            sell_price = row["price_at_up"]

            num_of_stocks_to_buy_and_sell = numpy.ceil(self._sum_per_buy_order / buy_price)
            if buy_date < pd.Timestamp(2025,6,1):
                event_list.append(BuyEvent(buy_date,CommandType.BUY,num_of_stocks_to_buy_and_sell, buy_price, idx, ticker, date_to_sell=sell_date, price_on_sell=sell_price))
                event_list.append(SellEvent(sell_date,CommandType.SELL,num_of_stocks_to_buy_and_sell, sell_price, idx, ticker, date_bought=buy_date, price_on_buy=buy_price))

        return event_list

    def calc_profitability(self):
        events_list = self._create_list_of_buy_and_sell_dates()
        unsold_events = [x for x in events_list if x.date is pd.NaT]
        events_list = [x for x in events_list if x.date is not pd.NaT]  # in case there was no point in time where the price rose above 5%
        sorted_event_list = sorted(events_list, key=lambda x: x.date)
        sorted_event_list = [x for x in sorted_event_list]

        bought_events = set()
        ticker_last_sold = {}
        tickers_in_portfolio = {}
        min_total_cash = self.INITIAL_SUM
        date_for_cash_exhaustion = None
        count_buy_events = 0
        for event in sorted_event_list:
            event_sum = event.num_stocks * event.price

            if event.command_type == CommandType.BUY and self._curr_sum > event_sum and (event_sum < min(self._sum_per_buy_order * 5, self.INITIAL_SUM/10))\
                    and tickers_in_portfolio.get(event.ticker,0) < self._max_concurrent_buys_per_stock \
                    and ticker_last_sold.get(event.ticker, pd.Timestamp(2000,1,1)) + pd.Timedelta(days=30) < event.date:
                tickers_in_portfolio[event.ticker] = tickers_in_portfolio.get(event.ticker,0) + 1
                # ticker_last_sold[event.ticker] = event.date
                bought_events.add(event.event_id)
                self._curr_sum -= event_sum
                min_total_cash = min(self._curr_sum, min_total_cash)
                count_buy_events += 1
                if not date_for_cash_exhaustion and self._curr_sum < self._sum_per_buy_order:
                    date_for_cash_exhaustion = event.date

            elif event.command_type == CommandType.SELL and event.event_id in bought_events:
                self._curr_sum += event_sum
                ticker_last_sold[event.ticker] = event.date
                bought_events.remove(event.event_id)
                tickers_in_portfolio[event.ticker] -= 1

        print(f"total cash: {self._curr_sum}")
        print(f"min total cash: {min_total_cash}")
        print(f"cash exhaustion date: {date_for_cash_exhaustion}")
        print(f"number of buy events: {count_buy_events}")
        print(f"unique tickers bought: {len(tickers_in_portfolio)}")
        return self._curr_sum

if __name__ == "__main__":
    up_run_id = 118
    sum_per_buy_order = 10000
    max_concurrent_buys_per_stock = 3

    query = select(UpEvent).where(UpEvent.up_run_id == up_run_id)
    prices_df = pd.read_sql(query, con=engine)
    s = SellAfterPercentageIncreaseSimulation(prices_df, sum_per_buy_order, max_concurrent_buys_per_stock).calc_profitability()

    # https://github.com/fja05680/sp500