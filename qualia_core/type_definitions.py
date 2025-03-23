"""
Custom type definitions for quantum core modules.
Serves as base definitions and compatibility layer during qtypes transition.
"""

from typing import Dict, List, Union, TypeVar, Generic
import numpy as np
import numpy.typing as npt

# Re-export from qtypes for compatibility
try:
    from .qtypes.quantum_state import StateVector, QuantumStateVector
    from .qtypes.base_types import StateIndex, StateProbability
    from .qtypes.measurement_types import MeasurementResult, MeasurementBasis
    from .qtypes.trading_types import MarketData, TradeParameters
except ImportError:
    # Direct type definitions as fallback
    StateVector = npt.NDArray[np.complex128]
    QuantumStateVector = StateVector
    StateIndex = int  # Using direct type instead of NewType for simple integer indices
    StateProbability = float  # Using direct type for probabilities

    # Measurement types
    MeasurementResult = Dict[str, Union[float, str, List[float]]]
    MeasurementBasis = npt.NDArray[np.complex128]

    # Trading types
    MarketData = Dict[str, Union[float, str, List[float]]]
    TradeParameters = Dict[str, Union[float, str, bool]]

# Consciousness types
T = TypeVar('T')
class ConsciousnessState(Generic[T]):
    """Generic container for consciousness state data"""
    def __init__(self, data: T):
        self.data = data

FieldConfiguration = Dict[str, Union[float, str, List[float]]]

# Export all types
__all__ = [
    'QuantumStateVector',
    'StateIndex',
    'StateProbability',
    'MeasurementResult',
    'MeasurementBasis',
    'MarketData',
    'TradeParameters',
    'ConsciousnessState',
    'FieldConfiguration'
]