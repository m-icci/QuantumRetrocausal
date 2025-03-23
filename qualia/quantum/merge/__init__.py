"""
QUALIA Quantum Merge System
--------------------------

Este módulo implementa um sistema de merge quântico que utiliza princípios
de computação quântica para realizar merges inteligentes de código e dados.

Componentes principais:
- QuantumMergeSimulator: Realiza operações de merge usando superposição quântica
- QuantumMergeMonitor: Monitora métricas e estado do sistema durante merges
"""

from .merge_simulator import (
    QuantumMergeSimulator,
    QuantumMergeMonitor,
    QuantumMergeLogger
)

__all__ = [
    'QuantumMergeSimulator',
    'QuantumMergeMonitor',
    'QuantumMergeLogger'
]
