"""
Quantum Merge Package
Provides quantum-enhanced merge operations with git integration
"""
from .unified_quantum_merge import UnifiedQuantumMerge
from .merge_simulator import QuantumMergeSimulator
from .cli import main as merge_cli

__all__ = [
    'UnifiedQuantumMerge',
    'QuantumMergeSimulator',
    'merge_cli'
]