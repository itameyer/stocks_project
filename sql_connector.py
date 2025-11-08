from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, ForeignKey, BigInteger, Index
)
from sqlalchemy.orm import declarative_base, relationship

DB_URL = "mysql+pymysql://root:brsz8bvb@localhost/stocks_project"
engine = create_engine(DB_URL, echo=True)

Base = declarative_base()

class DropRun(Base):
    __tablename__ = "drop_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)

    MA_interval = Column(Integer)
    drop_percent = Column(Integer)
    since_year = Column(DateTime)

    # Relationship (1-to-many)
    drop_events = relationship("DropEvent", back_populates="drop_run", cascade="all, delete-orphan")
    up_runs = relationship("UpRun", back_populates="drop_run", cascade="all, delete-orphan")


class DropEvent(Base):
    __tablename__ = "drop_events"

    event_id = Column(BigInteger, primary_key=True, autoincrement=True)
    drop_run_id = Column(Integer, ForeignKey("drop_runs.id", ondelete="CASCADE"), index=True)

    ticker = Column(String(12))
    date = Column(DateTime)
    close = Column(Float)
    MA = Column(Float)
    high = Column(Float)
    low = Column(Float)
    PctChange = Column(Float)

    drop_run = relationship("DropRun", back_populates="drop_events")


    __table_args__ = (
        Index("idx_drop_events_runid", "drop_run_id"),  # ⚙️ Helps with joins & filtering by run
    )

class UpRun(Base):
    __tablename__ = "up_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    drop_run_id = Column(Integer, ForeignKey(f"{DropRun.__tablename__}.id", ondelete="CASCADE"), index=True)

    rise_percent = Column(Integer)
    ultimate_sell_period = Column(Integer)
    total_events = Column(Integer)
    _1_month = Column(Integer)
    _2_month = Column(Integer)
    _3_month = Column(Integer)
    _4_month = Column(Integer)
    _5_month = Column(Integer)
    _6_month = Column(Integer)
    _7_month = Column(Integer)
    _8_month = Column(Integer)
    _9_month = Column(Integer)
    _10_month = Column(Integer)
    _11_month = Column(Integer)
    _12_month = Column(Integer)
    above_year = Column(Integer)
    never_sold = Column(Integer)
    drops_after_2020 = Column(Integer)


    up_events = relationship("UpEvent", back_populates="up_run", cascade="all, delete-orphan")
    simulation_outputs = relationship("SimulationOutput", back_populates="up_run", cascade="all, delete-orphan")
    drop_run = relationship("DropRun", back_populates="up_runs")


class UpEvent(Base):
    __tablename__ = "up_events"

    event_id = Column(BigInteger, primary_key=True, autoincrement=True)
    up_run_id = Column(Integer, ForeignKey(f"{UpRun.__tablename__}.id", ondelete="CASCADE"), index=True)

    ticker = Column(String(12))
    DropDate = Column(DateTime)
    Close = Column(Float)
    PctChange = Column(String(10))
    date_of_up = Column(DateTime)
    price_at_up = Column(Float)
    days_delta_to_be_up = Column(Integer)

    # Relationship (many-to-one)
    up_run = relationship(UpRun.__name__, back_populates="up_events")

    __table_args__ = (
        Index("idx_events_runid", "up_run_id"),  # ⚙️ Helps with joins & filtering by run
    )

class SimulationOutput(Base):
    __tablename__ = "simulation_outputs"

    id = Column(Integer, primary_key=True, autoincrement=True)

    up_run_id = Column(Integer, ForeignKey(f"{UpRun.__tablename__}.id", ondelete="CASCADE"), nullable=False)

    average_gain = Column(Float)
    std_dev_gain = Column(Float)
    _1_buys_1k = Column(Float)
    _2_buys_1k = Column(Float)
    _3_buys_1k = Column(Float)
    _1_buys_2k = Column(Float)
    _2_buys_2k = Column(Float)
    _3_buys_2k = Column(Float)
    _1_buys_3k = Column(Float)
    _2_buys_3k = Column(Float)
    _3_buys_3k = Column(Float)
    _1_buys_5k = Column(Float)
    _2_buys_5k = Column(Float)
    _3_buys_5k = Column(Float)
    _1_buys_10k = Column(Float)
    _2_buys_10k = Column(Float)
    _3_buys_10k = Column(Float)

    # Relationships (optional, for ORM navigation)
    up_run = relationship("UpRun", back_populates="simulation_outputs")

    __table_args__ = (
        Index("idx_simulation_outputs_up_run_id", "up_run_id"),
    )

def create_tables():
    Base.metadata.create_all(engine)
    print("✅ Tables created successfully.")


if __name__ == "__main__":
    create_tables()
