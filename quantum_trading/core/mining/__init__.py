"""
Módulo de mineração quântica para Monero.

Este módulo implementa estratégias de mineração quântica para Monero,
utilizando princípios de computação quântica e teoria da informação.
"""

from .mining_config import MiningConfig
from .mining_strategy import MiningStrategy, QuantumMiningStrategy
from .miner_integration import MinerIntegration
from .pool_manager import PoolManager
from .quantum_miner import QuantumMiner

__all__ = [
    'MiningConfig',
    'MiningStrategy',
    'QuantumMiningStrategy',
    'MinerIntegration',
    'PoolManager',
    'QuantumMiner'
] 