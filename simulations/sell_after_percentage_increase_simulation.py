import numpy as np
import time
import pandas as pd
from enum import Enum, auto
from typing import NamedTuple
from datetime import datetime
import numpy


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

    IBKR_COMMISION = 0.005 # % of every trade
    INITIAL_SUM = 150_000

    def __init__(self, prices_df: pd.DataFrame, rise_percent: int, sum_per_buy_order: int, max_concurrent_buys_per_stock: int):
        self._curr_sum = self.INITIAL_SUM
        self._prices_df = prices_df
        self._rise_percent = rise_percent
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
            if buy_date < pd.Timestamp(2025,1,1):
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
        tickers_in_portfolio = {}
        min_total_cash = self.INITIAL_SUM
        date_for_cash_exhaustion = None
        count_buy_events = 0
        for event in sorted_event_list:
            event_sum = event.num_stocks * event.price

            if event.command_type == CommandType.BUY and self._curr_sum - event_sum > 0 and (event_sum < min(self._sum_per_buy_order * 5, self.INITIAL_SUM/10))\
                    and tickers_in_portfolio.get(event.ticker,0) < self._max_concurrent_buys_per_stock:
                tickers_in_portfolio[event.ticker] = tickers_in_portfolio.get(event.ticker,0) + 1
                bought_events.add(event.event_id)
                self._curr_sum -= event_sum * (1 + self.IBKR_COMMISION)
                min_total_cash = min(self._curr_sum, min_total_cash)
                count_buy_events += 1
                if not date_for_cash_exhaustion and self._curr_sum < self._sum_per_buy_order:
                    date_for_cash_exhaustion = event.date

            elif event.command_type == CommandType.SELL and event.event_id in bought_events:
                self._curr_sum += event_sum * (1- self.IBKR_COMMISION)
                bought_events.remove(event.event_id)
                tickers_in_portfolio[event.ticker] -= 1

        print(f"total cash: {self._curr_sum}")
        print(f"min total cash: {min_total_cash}")
        print(f"cash exhaustion date: {date_for_cash_exhaustion}")
        print(f"number of buy events: {count_buy_events}")
        return self._curr_sum

if __name__ == "__main__":
    MA = 30
    drop_percent = 20
    since_year = 2015
    rise_percent = 12
    ultimate_sell_period = 180
    sum_per_buy_order = 2000
    max_concurrent_buys_per_stock = 3
    prices_df = pd.read_csv(f"../nasdaqdatalink_artifacts/time_to_rise/since_{since_year}/time_to_rise_{rise_percent}%_MA{MA}_drop_{drop_percent}%_ultimate_sell_period_{ultimate_sell_period}/prices.csv")
    s = SellAfterPercentageIncreaseSimulation(prices_df, rise_percent, sum_per_buy_order, max_concurrent_buys_per_stock).calc_profitability()

    # https://github.com/fja05680/sp500