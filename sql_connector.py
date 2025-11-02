from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, ForeignKey, BigInteger, Index
)
from sqlalchemy.orm import declarative_base, relationship

DB_URL = "mysql+pymysql://root:brsz8bvb@localhost/stocks_project"
engine = create_engine(DB_URL, echo=True)

Base = declarative_base()


class Run(Base):
    __tablename__ = "runs"

    id = Column(Integer, primary_key=True, autoincrement=True)

    MA_interval = Column(Integer)
    drop_percent = Column(Integer)
    since_year = Column(DateTime)
    rise_percent = Column(Integer)
    ultimate_sell_period = Column(Integer)
    average_gain = Column(Float)
    std_dev_gain = Column(Float)
    

    # Relationship (1-to-many)
    events = relationship("Event", back_populates="run", cascade="all, delete-orphan")


class Event(Base):
    __tablename__ = "events"

    event_id = Column(BigInteger, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("runs.id", ondelete="CASCADE"), index=True)

    ticker = Column(String(12))
    DropDate = Column(DateTime)
    Close = Column(Float)
    PctChange = Column(String(10))
    date_of_up = Column(DateTime)
    price_at_up = Column(Float)
    days_delta_to_be_up = Column(Integer)

    # Relationship (many-to-one)
    run = relationship("Run", back_populates="events")

    __table_args__ = (
        Index("idx_events_runid", "run_id"),  # ⚙️ Helps with joins & filtering by run
    )


def create_tables():
    Base.metadata.create_all(engine)
    print("✅ Tables created successfully.")


if __name__ == "__main__":
    create_tables()
