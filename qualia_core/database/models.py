"""
Database models for the trading system
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
Base = declarative_base()

class Trade(Base):
    """Model for storing trade history"""
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)  # 'buy' or 'sell'
    price = Column(Float, nullable=False)
    amount = Column(Float, nullable=False)
    total_value = Column(Float, nullable=False)
    quantum_metrics = Column(String)  # JSON string of quantum state

class Portfolio(Base):
    """Model for storing portfolio balances"""
    __tablename__ = 'portfolio'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    currency = Column(String, nullable=False)
    balance = Column(Float, nullable=False)
    last_price = Column(Float)

class MarketData(Base):
    """Model for storing market data"""
    __tablename__ = 'market_data'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    symbol = Column(String, nullable=False)
    price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

def init_db(
    poolclass=None,
    pool_size=None,
    max_overflow=None,
    pool_timeout=None,
    pool_recycle=None
):
    """Initialize database connection and create tables"""
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")

    # Add SSL requirements for PostgreSQL
    connect_args = {
        "sslmode": "require",
        "connect_timeout": 10,
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5
    }

    engine_args = {
        "connect_args": connect_args,
        "pool_pre_ping": True  # Enable connection health checks
    }

    # Add pooling parameters if provided
    if poolclass:
        engine_args["poolclass"] = poolclass
    if pool_size:
        engine_args["pool_size"] = pool_size
    if max_overflow:
        engine_args["max_overflow"] = max_overflow
    if pool_timeout:
        engine_args["pool_timeout"] = pool_timeout
    if pool_recycle:
        engine_args["pool_recycle"] = pool_recycle

    try:
        engine = create_engine(database_url, **engine_args)
        Base.metadata.create_all(engine)
        logger.info("Database initialized successfully")
        return engine
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise