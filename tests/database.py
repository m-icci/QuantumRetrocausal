"""Database module for quantum trading analysis"""
import os
import time
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Integer, Float, DateTime, JSON, String, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from datetime import datetime
from sqlalchemy.schema import CreateTable
from sqlalchemy import inspect
from sqlalchemy.exc import OperationalError, SQLAlchemyError

# Initialize SQLAlchemy
Base = declarative_base()

def get_db_engine():
    """Get SQLAlchemy engine instance"""
    if 'DATABASE_URL' not in os.environ:
        raise ValueError("DATABASE_URL environment variable not set")

    url = os.environ['DATABASE_URL']

    try:
        # Create engine with appropriate connection parameters
        engine = create_engine(
            url,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800
        )

        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        return engine
    except Exception as e:
        raise ConnectionError(f"Failed to connect to database: {str(e)}")

# Create global engine with retry logic
engine = None
for attempt in range(3):  # Try 3 times
    try:
        engine = get_db_engine()
        break
    except Exception as e:
        if attempt == 2:  # Last attempt
            raise
        time.sleep(2)  # Wait before retrying

session_factory = sessionmaker(bind=engine)
Session = scoped_session(session_factory)

class MarketData(Base):
    """Store market price data"""
    __tablename__ = 'market_data'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    symbol = Column(String, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)

class ConsciousnessMetrics(Base):
    """Store consciousness analysis results"""
    __tablename__ = 'consciousness_metrics'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    symbol = Column(String, nullable=False)
    coherence = Column(Float)
    entanglement = Column(Float)
    resonance = Column(Float)
    integration = Column(Float)
    field_strength = Column(Float)

class MorphicPatterns(Base):
    """Store detected morphic patterns"""
    __tablename__ = 'morphic_patterns'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    symbol = Column(String, nullable=False)
    pattern_data = Column(JSON)
    strength = Column(Float)
    scale = Column(Float)
    position = Column(Float)

def init_db(max_retries=3, retry_interval=5):
    """Initialize database tables if they don't exist"""
    for attempt in range(max_retries):
        try:
            inspector = inspect(engine)

            # Create each table only if it doesn't exist
            for table in Base.metadata.sorted_tables:
                if not inspector.has_table(table.name):
                    table.create(engine)
            return True
        except OperationalError as e:
            if attempt == max_retries - 1:
                raise Exception(f"Failed to initialize database after {max_retries} attempts: {str(e)}")
            print(f"Database initialization attempt {attempt + 1} failed, retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)

def get_session():
    """Get a database session with retry logic"""
    session = Session()
    try:
        # Test the session with explicit text type
        session.execute(text("SELECT 1::integer"))
        return session
    except SQLAlchemyError:
        Session.remove()
        raise

def store_market_data(df: pd.DataFrame, symbol: str):
    """Store market data in database"""
    session = get_session()
    try:
        for _, row in df.iterrows():
            market_data = MarketData(
                timestamp=row['timestamp'],
                symbol=symbol,
                open=float(row.get('open', 0)),
                high=float(row.get('high', 0)),
                low=float(row.get('low', 0)),
                close=float(row['close']),
                volume=float(row.get('volume', 0))
            )
            session.add(market_data)
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        raise Exception(f"Failed to store market data: {str(e)}")
    finally:
        session.close()

def store_consciousness_metrics(metrics: dict, symbol: str):
    """Store consciousness analysis results"""
    session = get_session()
    try:
        consciousness = ConsciousnessMetrics(
            timestamp=datetime.now(),
            symbol=symbol,
            coherence=float(metrics['coherence']),
            entanglement=float(metrics['entanglement']),
            resonance=float(metrics['resonance']),
            integration=float(metrics['integration']),
            field_strength=float(metrics['field_strength'])
        )
        session.add(consciousness)
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        raise Exception(f"Failed to store consciousness metrics: {str(e)}")
    finally:
        session.close()

def store_morphic_patterns(patterns: list, symbol: str):
    """Store detected morphic patterns"""
    session = get_session()
    try:
        for pattern in patterns:
            morphic_pattern = MorphicPatterns(
                timestamp=datetime.now(),
                symbol=symbol,
                pattern_data=pattern,
                strength=float(pattern.get('strength', 0)),
                scale=float(pattern.get('scale', 0)),
                position=float(pattern.get('position', 0))
            )
            session.add(morphic_pattern)
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        raise Exception(f"Failed to store morphic patterns: {str(e)}")
    finally:
        session.close()