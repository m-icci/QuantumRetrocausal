"""
QUALIA Quantum State Management
-----------------------------

Este módulo fornece classes e funções para gerenciamento de estados quânticos,
incluindo criação, manipulação e análise de estados.
"""

from .quantum_state import QuantumState
from .state_manager import StateManager

__all__ = [
    'QuantumState',
    'StateManager'
]
