"""
QUALIA Quantum Utilities
-----------------------

Utilitários para operações quânticas, incluindo cálculos de coerência,
entropia e outras métricas quânticas.
"""

from .merge_utils import (
    calculate_quantum_coherence,
    calculate_phase_coherence,
    calculate_entropy,
    merge_potential
)

__all__ = [
    'calculate_quantum_coherence',
    'calculate_phase_coherence',
    'calculate_entropy',
    'merge_potential'
]
