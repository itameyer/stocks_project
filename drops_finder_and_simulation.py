import pandas as pd

from create_drops_events_and_averages import DropsAndUpsFinder
from gain_strategy import FirstGainAbovePercent
from simulations.sell_after_percentage_increase_simulation import SellAfterPercentageIncreaseSimulation
from sqlalchemy import create_engine, update
from sqlalchemy.orm import sessionmaker
from sql_connector import Run, Event

DB_URL = "mysql+pymysql://root:brsz8bvb@localhost/stocks_project"
engine = create_engine(DB_URL)
Session = sessionmaker(bind=engine)

class StrategyRunner:
    def __init__(self,
                 directory:str,
                 start_date: pd.Timestamp,
                 drop_percent: float,
                 MA: int,
                 rise_percent: int,
                 ultimate_sell_period: int):
        self._run_id = self._create_new_run_in_sql(MA, drop_percent, start_date, rise_percent, ultimate_sell_period)
        self._gain_strategy = FirstGainAbovePercent(rise_percent, ultimate_sell_period, self._run_id)
        self._drops_finder = DropsAndUpsFinder(directory=directory,
                                               start_date=pd.Timestamp(since_year, 1, 1),
                                               drop_percent=drop_percent,
                                               MA=MA,
                                               gain_strategy=self._gain_strategy)

    @staticmethod
    def _create_new_run_in_sql(MA, drop_percent, start_date, rise_percent, ultimate_sell_period) -> int:
        session = Session()
        new_run = Run(
            MA_interval=MA,
            drop_percent=drop_percent,
            since_year=start_date,
            rise_percent=rise_percent,
            ultimate_sell_period=ultimate_sell_period,
            average_gain=0,
            std_dev_gain=0
        )
        session.add(new_run)
        session.commit()
        run_id = new_run.id
        session.close()
        return run_id

    def _add_simulation_output_to_sql_run(self, avg: float, std_dev: float):
        session = Session()
        stmt = (
            update(Run)
            .where(Run.id == self._run_id)
            .values(average_gain=avg, std_dev_gain=std_dev)
        )

        session.execute(stmt)
        session.commit()
        session.close()

    def _get_all_drop_rise_events(self) -> pd.DataFrame:
        query = f"SELECT * FROM {Event.__table__.name} WHERE run_id = {self._run_id};"
        df = pd.read_sql(query, con=engine)
        return df

    def _run_simulation(self) -> tuple[float, float]:
        prices_df = self._get_all_drop_rise_events()

        df = pd.DataFrame()
        for sum_per_buy_order in [1000, 2000, 3000, 5000, 10_000]:
            for max_concurrent_buys_per_stock in [1, 2, 3]:
                s = SellAfterPercentageIncreaseSimulation(prices_df,
                                                          rise_percent,
                                                          sum_per_buy_order,
                                                          max_concurrent_buys_per_stock).calc_profitability()
                df.loc[f"sum_per_buy_order:{sum_per_buy_order}", max_concurrent_buys_per_stock] = s

        std_dev = df.values.std()
        avg = df.values.mean()

        avg_row = pd.DataFrame({1: [avg]})
        avg_row.index = ['Overall Average']
        df = pd.concat([df, avg_row])
        std_dev_row = pd.DataFrame({1: [std_dev]})
        std_dev_row.index = ['std dev']
        df = pd.concat([df, std_dev_row])

        aggregation_file = f"{directory}/simulation_output.csv"
        df.to_csv(aggregation_file)
        return avg, std_dev

    def run_strategy(self):
        self._drops_finder.run_workers_pool()
        avg, std_dev = self._run_simulation()
        self._add_simulation_output_to_sql_run(avg, std_dev)


if __name__ == "__main__":
    MA = 1
    drop_percent = 5
    since_year = 2021
    rise_percent = 15
    ultimate_sell_period = 180
    # for MA in [1,5,10,15,20,30]:
    # for drop_percent in [10,15,20,25,30]:
    #     for rise_percent in [20,25,30]:
            # for ultimate_sell_period in [120,150,180]:
    directory = f"nasdaqdatalink_artifacts/time_to_rise/since_{since_year}/time_to_rise_{rise_percent}%_MA{MA}_drop_{drop_percent}%_ultimate_sell_period_{ultimate_sell_period}"

    s = StrategyRunner(directory=directory,
                       start_date=pd.Timestamp(since_year, 1, 1),
                       drop_percent=drop_percent,
                       MA=MA,
                       rise_percent=rise_percent,
                       ultimate_sell_period=ultimate_sell_period)

    s.run_strategy()