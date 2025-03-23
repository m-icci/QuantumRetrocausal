"""
QUALIA package initialization.
Following layered architecture from theoretical foundation.
"""
from typing import Dict, Any, Optional, List

# Constants from theoretical foundation
PHI = 1.618033988749895  # Golden ratio

# Import core base types
from .base_types import QuantumState, QuantumPattern, QuantumMetric, QuantumOperator

# Import core trading components
from .trading import AdvancedHolographicMarketMemory

__all__ = [
    'PHI',
    'QuantumState',
    'QuantumPattern',
    'QuantumMetric',
    'QuantumOperator',
    'AdvancedHolographicMarketMemory'
]