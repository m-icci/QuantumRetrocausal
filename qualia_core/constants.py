"""
Constantes globais do QUALIA
"""
from typing import List, Dict, Any
import numpy as np

class QualiaConstants:
    """Constantes do sistema QUALIA"""
    
    # Granularidades ativas (Sequência de Fibonacci expandida)
    ACTIVE_GRANULARITIES: List[int] = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
    
    # Classificação de granularidades por modo
    ATOMIC_GRANULARITIES: List[int] = [1, 2]
    BALANCED_GRANULARITIES: List[int] = [5, 8, 13, 21, 34]
    EMERGENT_GRANULARITIES: List[int] = [144, 233, 377]
    
    # Pesos de otimização por granularidade
    GRANULARITY_WEIGHTS: Dict[int, float] = {
        1: 1.4,     # Ultra alta frequência
        2: 1.35,    # Super alta frequência
        3: 1.3,     # Alta frequência
        5: 1.25,    # Média-alta frequência
        8: 1.2,     # Média frequência superior
        13: 1.15,   # Média frequência
        21: 1.1,    # Média frequência inferior
        34: 1.05,   # Baixa-média frequência
        55: 1.0,    # Baixa frequência
        89: 0.95,   # Muito baixa frequência
        144: 0.9,   # Ultra baixa frequência
        233: 0.87,  # Super baixa frequência
        377: 0.85   # Frequência quântica profunda
    }
    
    # Fatores de coerência por granularidade
    COHERENCE_FACTORS: Dict[int, float] = {
        1: 0.99,    # Ultra alta estabilidade
        2: 0.97,    # Super alta estabilidade
        3: 0.95,    # Alta estabilidade
        5: 0.93,    # Média-alta estabilidade
        8: 0.91,    # Média estabilidade superior
        13: 0.89,   # Média estabilidade
        21: 0.87,   # Média estabilidade inferior
        34: 0.85,   # Baixa-média estabilidade
        55: 0.83,   # Baixa estabilidade
        89: 0.81,   # Muito baixa estabilidade
        144: 0.79,  # Ultra baixa estabilidade
        233: 0.78,  # Super baixa estabilidade
        377: 0.77   # Estabilidade quântica profunda
    }
    
    # Constantes geométricas
    PHI: float = 1.618033988749895  # Número áureo
    PHI_INVERSE: float = 0.618033988749895  # Inverso áureo
    
    # Limiares de adaptação
    HARDWARE_LOAD_HIGH: float = 0.8
    HARDWARE_LOAD_LOW: float = 0.3
    MIN_COHERENCE_THRESHOLD: float = 0.85
    
    # Pesos para cálculo de eficiência
    COHERENCE_WEIGHT = 0.3
    RESONANCE_WEIGHT = 0.25
    ENTROPY_WEIGHT = 0.2
    HARDWARE_WEIGHT = 0.25
    
    # Pesos para cálculo de eficiência
    EFFICIENCY_WEIGHTS = {
        "coherence": 0.35,
        "resonance": 0.25,
        "stability": 0.25,
        "execution_time": 0.15
    }
    
    @classmethod
    def get_granularity_weight(cls, g: int) -> float:
        """Retorna peso de otimização para granularidade"""
        return cls.GRANULARITY_WEIGHTS.get(g, 1.0)
    
    @classmethod
    def get_coherence_factor(cls, g: int) -> float:
        """Retorna fator de coerência para granularidade"""
        return cls.COHERENCE_FACTORS.get(g, 0.8)
    
    @classmethod
    def validate_granularity(cls, g: int) -> bool:
        """Valida se granularidade está ativa"""
        return g in cls.ACTIVE_GRANULARITIES
