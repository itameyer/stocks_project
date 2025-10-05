import heapq
import pandas as pd
from enum import Enum, auto
from typing import NamedTuple, Dict, List, Union
from pandas.tseries.offsets import DateOffset
from datetime import datetime, date
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
        super().__init__(*args, **kwargs)

class BuyEvent(Event):
    def __init__(self, *args, **kwargs):
        self.date_to_sell = kwargs.pop('date_to_sell', None)
        self.price_on_sell = kwargs.pop('price_on_sell', None)
        super().__init__(*args, **kwargs)


class Simulation:

    INITIAL_SUM = 100_000
    HOLD_PERIOD_MONTHS = 1
    SUM_PER_BUY_ORDER = 1

    def __init__(self, prices_csv_path: str):
        self._curr_sum = self.INITIAL_SUM
        self._prices_csv_path = prices_csv_path

    def _create_list_of_buy_and_sell_dates(self):
        df = pd.read_csv(self._prices_csv_path)
        event_list = []
        for idx, row in df.iterrows():
            ticker = row["ticker"]
            buy_date = pd.to_datetime(row["DropDate"])
            buy_price = round(row["Close"],2)
            num_of_stocks_to_buy_and_sell = numpy.ceil(self.SUM_PER_BUY_ORDER / buy_price)
            sell_date = buy_date + DateOffset(months=self.HOLD_PERIOD_MONTHS)
            sell_price = row[f"Close+{self.HOLD_PERIOD_MONTHS}M"]

            if sell_date.date() < date.today() and buy_price>0 and sell_price>0:
                event_list.append(BuyEvent(buy_date,CommandType.BUY,num_of_stocks_to_buy_and_sell, buy_price, idx, ticker, date_to_sell=sell_date, price_on_sell=sell_price))
                event_list.append(SellEvent(sell_date,CommandType.SELL,num_of_stocks_to_buy_and_sell, sell_price, idx, ticker))

        return event_list

    def _sell_stocks_or_postpone_sale(self, event: SellEvent,
                                      events_min_heap: List[Union[BuyEvent,SellEvent]],
                                      soonest_selling_point: Dict[str,SellPoint]) -> float:

        if event.date >= soonest_selling_point[event.ticker].date:
            return event.num_stocks * event.price
        else:
            new_sell_event = SellEvent(soonest_selling_point[event.ticker].date, CommandType.SELL,
                                       event.num_stocks, soonest_selling_point[event.ticker].price,
                                       event.event_id, event.ticker)
            heapq.heappush(events_min_heap, new_sell_event)
            return 0    # nothing was sold

    def calc_profitability(self):
        #heapify the events list with date as key
        events_min_heap = self._create_list_of_buy_and_sell_dates()
        heapq.heapify(events_min_heap)

        bought_events = set()
        soonest_selling_point = {}
        min_total_cash = self.INITIAL_SUM
        while events_min_heap:
            event = heapq.heappop(events_min_heap)
            event_sum = event.num_stocks * event.price

            if event.command_type == CommandType.BUY:
                # always append soonest buying point, even if not bought
                soonest_selling_point[event.ticker] = SellPoint(event.date_to_sell, event.price_on_sell)
                if self._curr_sum - event_sum > 0:
                    bought_events.add(event.event_id)
                    self._curr_sum -= event_sum
                    min_total_cash = min(self._curr_sum, min_total_cash)

            elif event.command_type == CommandType.SELL and event.event_id in bought_events:
                self._curr_sum += self._sell_stocks_or_postpone_sale(event, events_min_heap, soonest_selling_point)

        print(f"total cash: {self._curr_sum}")
        print(f"min total cash: {min_total_cash}")

if __name__ == "__main__":
    s = Simulation("alternating_snp500_2020_at_20%_drop_MA20/prices.csv").calc_profitability()

    # https://github.com/fja05680/sp500