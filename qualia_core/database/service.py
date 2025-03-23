"""
Database service for handling database operations
"""
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from .models import Trade, Portfolio, MarketData, init_db
import json
from datetime import datetime
from typing import Dict, List, Optional
import logging
import time

logger = logging.getLogger(__name__)

class DatabaseService:
    def __init__(self):
        """Initialize database service"""
        self.engine = init_db(
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800  # Recycle connections after 30 minutes
        )
        self.Session = sessionmaker(bind=self.engine)

    def execute_with_retry(self, operation, *args, **kwargs):
        """Execute database operation with retry logic"""
        max_retries = 3
        retry_count = 0
        base_delay = 1.0

        while retry_count < max_retries:
            try:
                with self.Session() as session:
                    try:
                        result = operation(session, *args, **kwargs)
                        session.commit()
                        return result
                    except Exception as e:
                        session.rollback()
                        raise e
            except (OperationalError, SQLAlchemyError) as e:
                retry_count += 1
                if retry_count == max_retries:
                    logger.error(f"Database error after {max_retries} retries: {str(e)}")
                    raise
                delay = base_delay * (2 ** (retry_count - 1))  # Exponential backoff
                logger.warning(f"Database error, retrying in {delay}s: {str(e)}")
                time.sleep(delay)
            except Exception as e:
                logger.error(f"Database error: {str(e)}")
                raise

    def add_trade(self, trade_data: Dict) -> None:
        """Add a new trade to the database"""
        def _add_trade(session, data):
            trade = Trade(
                symbol=data['symbol'],
                side=data['side'],
                price=float(data['price']),
                amount=float(data['amount']),
                total_value=float(data['total_value']),
                quantum_metrics=json.dumps(data.get('quantum_metrics', {}))
            )
            session.add(trade)
            return trade

        self.execute_with_retry(_add_trade, trade_data)

    def update_portfolio(self, currency: str, balance: float, last_price: Optional[float] = None) -> None:
        """Update portfolio balance"""
        def _update_portfolio(session, curr, bal, price):
            portfolio = Portfolio(
                currency=curr,
                balance=bal,
                last_price=price
            )
            session.add(portfolio)
            return portfolio

        self.execute_with_retry(_update_portfolio, currency, balance, last_price)

    def add_market_data(self, symbol: str, price: float, volume: float) -> None:
        """Add market data point"""
        def _add_market_data(session, sym, p, v):
            market_data = MarketData(
                symbol=sym,
                price=p,
                volume=v
            )
            session.add(market_data)
            return market_data

        self.execute_with_retry(_add_market_data, symbol, price, volume)

    def get_latest_trades(self, limit: int = 100) -> List[Dict]:
        """Get latest trades"""
        def _get_trades(session, lim):
            trades = session.query(Trade)\
                .order_by(Trade.timestamp.desc())\
                .limit(lim)\
                .all()
            return [{
                'timestamp': trade.timestamp,
                'symbol': trade.symbol,
                'side': trade.side,
                'price': trade.price,
                'amount': trade.amount,
                'total_value': trade.total_value,
                'quantum_metrics': json.loads(trade.quantum_metrics)
            } for trade in trades]

        return self.execute_with_retry(_get_trades, limit)

    def get_portfolio_snapshot(self) -> Dict[str, float]:
        """Get latest portfolio balances"""
        def _get_snapshot(session):
            latest_balances = {}
            for balance in session.query(Portfolio)\
                .order_by(Portfolio.timestamp.desc())\
                .all():
                if balance.currency not in latest_balances:
                    latest_balances[balance.currency] = balance.balance
            return latest_balances

        return self.execute_with_retry(_get_snapshot)