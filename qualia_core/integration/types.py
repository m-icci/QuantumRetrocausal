"""
Core type definitions for quantum integration components
"""
from typing import Dict, Any
from datetime import datetime

class ConsciousnessState:
    """Consciousness state class"""

    def __init__(self, awareness: float = 0.0, coherence: float = 0.0,
                integration: float = 0.0, complexity: float = 0.0,
                timestamp: datetime | None = None):
        self.awareness = awareness
        self.coherence = coherence
        self.integration = integration
        self.complexity = complexity
        self.timestamp = timestamp or datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'awareness': self.awareness,
            'coherence': self.coherence,
            'integration': self.integration,
            'complexity': self.complexity,
            'timestamp': self.timestamp.isoformat()
        }

class MarketData:
    """Market data class"""

    def __init__(self, symbol: str, price: float, volume: float,
                timestamp: datetime, high: float, low: float,
                open: float, close: float):
        self.symbol = symbol
        self.price = price
        self.volume = volume
        self.timestamp = timestamp
        self.high = high
        self.low = low
        self.open = open
        self.close = close

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'symbol': self.symbol,
            'price': self.price,
            'volume': self.volume,
            'timestamp': self.timestamp.isoformat(),
            'high': self.high,
            'low': self.low,
            'open': self.open,
            'close': self.close
        }

# Export all types
__all__ = ['ConsciousnessState', 'MarketData']