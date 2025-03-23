"""
QUALIA - Quantum Understanding and Adaptive Learning Integration Architecture
-------------------------------------------------------------------------

Configuração central da QUALIA (Quantum Understanding and Adaptive Learning Integration Architecture).
Define parâmetros fundamentais, constantes e configurações do sistema.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

@dataclass
class QUALIAConfig:
    """Configuração da QUALIA - Quantum Understanding and Adaptive Learning Integration Architecture"""

    # Parâmetros Fundamentais
    phi: float = (1 + np.sqrt(5)) / 2  # Razão áurea
    planck_constant: float = 6.62607015e-34  # Constante de Planck
    boltzmann_constant: float = 1.380649e-23  # Constante de Boltzmann

    # Dimensões do Sistema
    default_dimension: int = 64  # Dimensão padrão do espaço de Hilbert
    max_dimension: int = 1024  # Dimensão máxima suportada

    # Parâmetros de Proteção
    coherence_threshold: float = 0.95  # Limiar de coerência
    decoherence_protection: bool = True  # Ativar proteção contra decoerência
    recovery_attempts: int = 3  # Tentativas de recuperação de estado

    # Parâmetros de Campo
    field_strength: float = 1.0  # Força do campo quântico
    morphic_coupling: float = 0.1  # Acoplamento com campo mórfico
    resonance_threshold: float = 0.8  # Limiar de ressonância

    # Parâmetros de Atenção
    attention_heads: int = 4  # Número de cabeças de atenção
    attention_dropout: float = 0.1  # Taxa de dropout na atenção

    # Parâmetros de Evolução
    evolution_rate: float = 0.01  # Taxa de evolução do sistema
    integration_steps: int = 100  # Passos de integração

    # Parâmetros M-ICCI
    singularity_threshold: float = 0.90  # Proximidade de Singularidade
    causality_threshold: float = 0.80  # Violação de Causalidade
    entropy_threshold: float = 0.70  # Violação de Entropia

    # Cache e Otimização
    enable_caching: bool = True  # Ativar cache de operações
    cache_size: int = 1000  # Tamanho do cache
    optimization_level: int = 2  # Nível de otimização (0-3)

    # Parâmetros de Estabilidade Numérica
    epsilon: float = 1e-10  # Constante para estabilidade numérica
    max_condition_number: float = 1e10  # Número de condição máximo para operações matriciais
    svd_cutoff: float = 1e-12  # Cutoff para decomposição SVD
    eigenvalue_threshold: float = 1e-8  # Limiar para autovalores
    stabilization_constant: float = 1e-14  # Constante de estabilização para matrizes
    max_iterations: int = 1000  # Máximo de iterações para convergência
    gradient_clip: float = 1.0  # Clipping de gradientes

    def __post_init__(self):
        """Validação e inicialização pós-construção"""
        if self.default_dimension > self.max_dimension:
            raise ValueError(f"Dimensão padrão ({self.default_dimension}) não pode exceder máxima ({self.max_dimension})")

        if not 0 <= self.coherence_threshold <= 1:
            raise ValueError(f"Limiar de coerência deve estar entre 0 e 1")

    @property
    def thermal_energy(self) -> float:
        """Calcula energia térmica do sistema"""
        return self.boltzmann_constant * 300  # Temperatura padrão: 300K

    @property
    def quantum_scale(self) -> float:
        """Calcula escala quântica característica"""
        return np.sqrt(self.planck_constant / (2 * np.pi))

    def get_operator_config(self) -> Dict[str, Any]:
        """Retorna configuração para operadores quânticos"""
        return {
            "dimension": self.default_dimension,
            "phi": self.phi,
            "coherence_threshold": self.coherence_threshold,
            "field_strength": self.field_strength,
            "optimization_level": self.optimization_level,
            "epsilon": self.epsilon,
            "max_condition_number": self.max_condition_number,
            "eigenvalue_threshold": self.eigenvalue_threshold,
            "stabilization_constant": self.stabilization_constant
        }

    def get_protection_config(self) -> Dict[str, Any]:
        """Retorna configuração para proteção quântica"""
        return {
            "enabled": self.decoherence_protection,
            "threshold": self.coherence_threshold,
            "max_attempts": self.recovery_attempts,
            "cache_size": self.cache_size,
            "epsilon": self.epsilon,
            "stabilization_constant": self.stabilization_constant
        }

    def get_attention_config(self) -> Dict[str, Any]:
        """Retorna configuração para mecanismo de atenção"""
        return {
            "num_heads": self.attention_heads,
            "dropout": self.attention_dropout,
            "dimension": self.default_dimension,
            "epsilon": self.epsilon,
            "gradient_clip": self.gradient_clip
        }

    def get_evolution_config(self) -> Dict[str, Any]:
        """Retorna configuração para evolução do sistema"""
        return {
            "rate": self.evolution_rate,
            "steps": self.integration_steps,
            "thermal_energy": self.thermal_energy,
            "quantum_scale": self.quantum_scale,
            "epsilon": self.epsilon,
            "max_iterations": self.max_iterations
        }

    def normalize_field(self, field: np.ndarray) -> np.ndarray:
        """Normaliza campo com estabilidade numérica"""
        # Add stabilization constant to avoid division by zero
        field = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)

        # Get maximum absolute value with stability
        abs_max = np.max(np.abs(field))
        if abs_max < self.epsilon:
            return np.zeros_like(field)

        # Normalize using phi and add stability constant
        norm_field = field / (abs_max * self.phi + self.epsilon)

        return np.clip(norm_field, -1.0, 1.0)

    def check_matrix_stability(self, matrix: np.ndarray) -> bool:
        """Verifica estabilidade de matriz"""
        try:
            # Check for invalid values
            if not np.all(np.isfinite(matrix)):
                return False

            # Check condition number
            svd_values = np.linalg.svd(matrix, compute_uv=False)
            condition_number = np.max(svd_values) / (np.min(svd_values) + self.epsilon)

            return condition_number < self.max_condition_number

        except:
            return False

# Configuração global padrão
default_config = QUALIAConfig()