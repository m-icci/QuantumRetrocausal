"""
Core quantum type definitions and system behaviors.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np

@dataclass
class MarketData:
    """Market data representation for quantum trading system"""
    symbol: str
    price: float
    volume: float
    timestamp: datetime
    high: float
    low: float
    open: float
    close: float

    def to_array(self) -> np.ndarray:
        """Convert market data to numpy array for quantum processing"""
        return np.array([
            self.price,
            self.volume,
            self.high,
            self.low,
            self.open,
            self.close
        ])

@dataclass
class ConsciousnessMetrics:
    """Quantum consciousness system metrics"""
    awareness: float      # Overall system awareness level (0-1)
    coherence: float     # Quantum state coherence (0-1)
    entanglement: float  # System entanglement measure (0-1)
    complexity: float    # Computational complexity (0-1)
    integration: float   # Information integration measure (0-1)

from .quantum_metrics import MetricsConfig, QuantumMetrics
from .quantum_state import QuantumState
from .system_behavior import SystemBehavior

__all__ = [
    'MarketData',
    'ConsciousnessMetrics',
    'SystemBehavior',
    'MetricsConfig',
    'QuantumMetrics',
    'QuantumState'
]