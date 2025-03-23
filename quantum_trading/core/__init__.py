"""
Core do Sistema
=============

Módulo principal que implementa a lógica de trading
e execução de operações em tempo real.
"""

from .mining import (
    MiningConfig,
    MiningStrategy,
    QuantumMiningStrategy,
    MinerIntegration,
    PoolManager,
    QuantumMiner
)

__all__ = [
    'MiningConfig',
    'MiningStrategy',
    'QuantumMiningStrategy',
    'MinerIntegration',
    'PoolManager',
    'QuantumMiner'
] 