"""
Quantum Types - Tipos Quânticos

Este módulo define os tipos quânticos básicos do sistema.
"""

from .quantum_base import (
    QuantumState,
    QuantumPattern,
    QuantumMetric,
    QuantumOperator
)

from .quantum_types import (
    QualiaState,
    SystemBehavior,
    ConsciousnessObservation,
    SuperPosition,
    CosmicFactor
)

from .pattern_types import PatternType, PatternDescription
from .base_types import BaseQuantumMetric
from .qualia_types import QualiaState as QualiaMainState, SystemBehavior as SystemMainBehavior
from .qualia_state import QualiaState as QualiaStateBase
from .system_behavior import SystemBehavior as SystemBehaviorBase
from .validation_types import ValidationResult

__all__ = [
    'QuantumState',
    'QuantumPattern',
    'QuantumMetric',
    'QuantumOperator',
    'QualiaState',
    'SystemBehavior',
    'ConsciousnessObservation',
    'SuperPosition',
    'CosmicFactor',
    'PatternType',
    'PatternDescription',
    'BaseQuantumMetric',
    'QualiaMainState',
    'SystemMainBehavior',
    'QualiaStateBase',
    'SystemBehaviorBase',
    'ValidationResult'
]