"""
Módulo de Configuração do Sistema QUALIA

Este módulo contém todas as configurações do sistema QUALIA:
- Configuração quântica
- Configuração de memória
- Configuração de rede
"""

from .memory_config import MemoryConfig
from .system_config import QuantumConfig

__all__ = ['MemoryConfig', 'QuantumConfig'] 