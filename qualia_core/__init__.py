"""
QUALIA - Sistema Quântico-Computacional Auto-Evolutivo

Este pacote implementa o sistema QUALIA, um sistema quântico-computacional
auto-evolutivo focado em mineração de Monero, integrando conceitos de:
- Computação quântica
- Memória holográfica
- Comunicação retrocausal
- Evolução adaptativa
"""

from .config import QuantumConfig
from .qualia import Qualia
from .quantum.quantum_computer import QuantumComputer, QuantumState, QuantumMetrics
from .processing.quantum_parallel import QuantumParallelProcessor
from .storage.holographic_memory import HolographicMemory
from .network.retrocausal_network import RetrocausalNetwork

__version__ = "0.1.0"
__author__ = "QUALIA Team"
__email__ = "qualia@example.com"

__all__ = [
    'QuantumConfig',
    'Qualia',
    'QuantumComputer',
    'QuantumState',
    'QuantumMetrics',
    'QuantumParallelProcessor',
    'HolographicMemory',
    'RetrocausalNetwork'
]
