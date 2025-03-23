"""
Compatibility layer for quantum types system migration.
Provides backward compatibility during the transition from types to qtypes.
"""

import logging
from typing import Dict, List, Union, TypeVar, Generic
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from .qtypes.quantum_metrics import BaseQuantumMetric, MetricsConfig
    from .qtypes.quantum_state import QuantumState, QuantumPattern
    from .qtypes.quantum_types import ConsciousnessObservation, QualiaState, SystemBehavior
    logger.info("Successfully imported from qtypes")
except ImportError:
    logger.warning("Falling back to type_definitions")
    from .type_definitions import *

# Re-export all types for backward compatibility
__all__ = [
    'BaseQuantumMetric',
    'MetricsConfig',
    'QuantumState',
    'QuantumPattern',
    'ConsciousnessObservation',
    'QualiaState',
    'SystemBehavior'
]
