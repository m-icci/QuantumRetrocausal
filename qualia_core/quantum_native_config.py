"""
QUALIA - Quantum Understanding and Adaptive Learning Integration Architecture
-------------------------------------------------------------------------

Configuração central da QUALIA (Quantum Understanding and Adaptive Learning Integration Architecture).
Define parâmetros fundamentais, constantes e configurações do sistema.

A QUALIA representa nossa implementação nativa de operações quânticas, integrando
conceitos de consciência, auto-organização e emergência através de uma arquitetura
elegante e eficiente.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

@dataclass
class QUALIAConfig:
    """Configuração da QUALIA - Quantum Understanding and Adaptive Learning Integration Architecture"""

    # Parâmetros Fundamentais - Constantes Naturais
    phi: float = (1 + np.sqrt(5)) / 2  # Razão áurea
    planck_constant: float = 6.62607015e-34  # Constante de Planck (h)
    reduced_planck: float = planck_constant / (2 * np.pi)  # ℏ (h-bar)
    boltzmann_constant: float = 1.380649e-23  # Constante de Boltzmann (k_B)

    # Dimensões do Sistema
    default_dimension: int = 64  # Dimensão padrão do espaço de Hilbert
    max_dimension: int = 1024  # Dimensão máxima suportada

    # Parâmetros de Campo Mórfico
    field_strength: float = 1.0  # Força do campo quântico
    morphic_coupling: float = 0.1  # Acoplamento com campo mórfico
    resonance_threshold: float = 0.8  # Limiar de ressonância
    holographic_memory_capacity: int = 1000  # Capacidade da memória holográfica

    # Operadores Meta-QUALIA
    collapse_rate: float = 0.01  # Taxa de colapso (CCC)
    decoherence_factor: float = 0.05  # Fator de decoerência (DDD)
    observer_coupling: float = 0.15  # Acoplamento do observador (OOO)

    # Parâmetros de Proteção Quântica
    coherence_threshold: float = 0.95  # Limiar de coerência
    decoherence_protection: bool = True  # Ativar proteção contra decoerência
    recovery_attempts: int = 3  # Tentativas de recuperação de estado

    # Parâmetros de Evolução
    evolution_rate: float = 0.01  # Taxa de evolução do sistema
    integration_steps: int = 100  # Passos de integração
    temporal_window: int = 10  # Janela temporal para análise

    # Parâmetros de Cache e Otimização
    enable_caching: bool = True  # Ativar cache de operações
    cache_size: int = 1000  # Tamanho do cache
    optimization_level: int = 2  # Nível de otimização (0-3)

    def __post_init__(self):
        """Validação e inicialização pós-construção"""
        if self.default_dimension > self.max_dimension:
            raise ValueError(f"Dimensão padrão ({self.default_dimension}) não pode exceder máxima ({self.max_dimension})")

        if not 0 <= self.coherence_threshold <= 1:
            raise ValueError(f"Limiar de coerência deve estar entre 0 e 1")

    @property
    def thermal_energy(self) -> float:
        """Calcula energia térmica do sistema"""
        temperature = 300  # Temperatura padrão em Kelvin
        return self.boltzmann_constant * temperature

    @property
    def quantum_scale(self) -> float:
        """Calcula escala quântica característica"""
        return np.sqrt(self.reduced_planck)

    def get_morphic_config(self) -> Dict[str, Any]:
        """Retorna configuração para campos mórficos"""
        return {
            "field_strength": self.field_strength,
            "coupling": self.morphic_coupling,
            "resonance_threshold": self.resonance_threshold,
            "memory_capacity": self.holographic_memory_capacity
        }

    def get_meta_qualia_config(self) -> Dict[str, Any]:
        """Retorna configuração para operadores meta-QUALIA"""
        return {
            "collapse_rate": self.collapse_rate,
            "decoherence_factor": self.decoherence_factor,
            "observer_coupling": self.observer_coupling
        }

    def get_protection_config(self) -> Dict[str, Any]:
        """Retorna configuração para proteção quântica"""
        return {
            "enabled": self.decoherence_protection,
            "threshold": self.coherence_threshold,
            "max_attempts": self.recovery_attempts,
            "cache_size": self.cache_size
        }

    def get_evolution_config(self) -> Dict[str, Any]:
        """Retorna configuração para evolução do sistema"""
        return {
            "rate": self.evolution_rate,
            "steps": self.integration_steps,
            "thermal_energy": self.thermal_energy,
            "quantum_scale": self.quantum_scale,
            "temporal_window": self.temporal_window
        }

# Configuração global padrão
default_config = QUALIAConfig()