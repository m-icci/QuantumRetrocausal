"""
Market data provider module for QUALIA Trading System
"""
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional

@dataclass
class MarketState:
    """Market state dataclass with quantum-aware metrics"""
    timestamp: float
    ohlcv: np.ndarray  # Shape: (N, 5) for OHLCV data
    quantum_features: np.ndarray  # Quantum state features
    market_phase: str  # Current market phase (accumulation, distribution, etc)
    entropy: float  # Market entropy measure
    complexity: float  # Market complexity measure

class MarketDataProvider:
    """Provides market data with quantum-enhanced features"""
    
    def __init__(self):
        self.quantum_dimension = 64  # Default quantum dimension
        self.cache = {}  # Simple cache for market data

    def get_market_data(
        self, 
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100
    ) -> MarketState:
        """
        Get market data with quantum features
        
        Args:
            symbol: Trading pair symbol (e.g. 'BTC/USDT')
            timeframe: Candle timeframe
            limit: Number of candles to fetch
        """
        # For testing/mocking, return random data
        ohlcv = np.random.rand(limit, 5)
        quantum_features = np.random.rand(self.quantum_dimension)

        return MarketState(
            timestamp=float(datetime.now().timestamp()),
            ohlcv=ohlcv,
            quantum_features=quantum_features,
            market_phase='accumulation',
            entropy=0.5,
            complexity=1.0
        )

    def get_current_price(self, symbol: str) -> float:
        """Get current price for symbol"""
        data = self.get_market_data(symbol, limit=1)
        return float(data.ohlcv[-1][4])  # Last close price

    def validate_market_state(self, symbol: str) -> bool:
        """Validate market data availability and quality"""
        try:
            data = self.get_market_data(symbol)
            return (
                data is not None and 
                len(data.ohlcv) > 0 and
                not np.isnan(data.ohlcv).any()
            )
        except Exception:
            return False
