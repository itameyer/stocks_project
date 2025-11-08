import numpy as np
import pandas as pd

from drop_events_finder import DropsEventsFinder
from gain_events_finder import FirstGainAbovePercent
from simulations.sell_after_percentage_increase_simulation import SellAfterPercentageIncreaseSimulation
from sqlalchemy import create_engine, update, select
from sqlalchemy.orm import sessionmaker
from sql_connector import SimulationOutput, UpEvent

DB_URL = "mysql+pymysql://root:brsz8bvb@localhost/stocks_project"
engine = create_engine(DB_URL)
Session = sessionmaker(bind=engine)

class StrategyRunner:
    def __init__(self, up_run_id: int):
        self._up_run_id =  up_run_id
        self._run_id = self._create_new_run_in_sql(up_run_id)

    @staticmethod
    def _create_new_run_in_sql(up_run_id) -> int:
        session = Session()
        new_run = SimulationOutput(
            up_run_id=up_run_id
        )
        session.add(new_run)
        session.commit()
        run_id = new_run.id
        session.close()
        return run_id

    def _get_up_run_data(self) -> pd.DataFrame:
        query = select(UpEvent).where(UpEvent.up_run_id == self._up_run_id)
        df = pd.read_sql(query, con=engine)
        return df

    def run_simulation(self):
        prices_df = self._get_up_run_data()

        simulation_outputs = {}
        for sum_per_buy_order in [1000,2000,3000,5000,10000]:
            for max_concurrent_buys_per_stock in [1,2,3]:
                s = SellAfterPercentageIncreaseSimulation(prices_df,
                                                          sum_per_buy_order,
                                                          max_concurrent_buys_per_stock).calc_profitability()
                simulation_outputs[f"_{max_concurrent_buys_per_stock}_buys_{int(sum_per_buy_order/1000)}k"] = s

        simulation_outputs['average_gain'] = np.average(list(simulation_outputs.values()))
        simulation_outputs['std_dev_gain'] = np.std(list(simulation_outputs.values()))

        with Session() as session:
            session.query(SimulationOutput).filter(SimulationOutput.id == self._run_id).update(simulation_outputs)
            session.commit()


if __name__ == "__main__":
    for i in range(121,161):
        s = StrategyRunner(i)
        s.run_simulation()