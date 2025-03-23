"""
Consciousness Module - Módulo de Consciência

Este módulo implementa a consciência quântica do sistema.
"""

from qualia_core.qtypes import (
    QuantumState,
    QuantumPattern,
    QuantumMetric,
    QuantumOperator
)

from .consciousness_operator import QuantumConsciousnessOperator
from .consciousness_field import QuantumConsciousnessField
from .consciousness_network import QuantumConsciousnessNetwork

__all__ = [
    'QuantumConsciousnessOperator',
    'QuantumConsciousnessField',
    'QuantumConsciousnessNetwork',
    'QuantumState',
    'QuantumPattern',
    'QuantumMetric',
    'QuantumOperator'
]