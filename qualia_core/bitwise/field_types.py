"""
QUALIA Field Types
Tipos fundamentais para campos quÃ¢ÃÃ­Â®Å¡ÃÃ©Â¢nticos
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

class FieldType(Enum):
    """Tipos de campo"""
    QUANTUM = "quantum"  # Campo quÃ¢ÃÃ­Â®Å¡ÃÃ©Â¢ntico
    MORPHIC = "morphic"  # Campo mÃ¢ÃÃ­Â®Å¡Ã¢ÃÃ­Â´Â¥rfico
    CONSCIOUSNESS = "consciousness"  # Campo de consciÃ¢ÃÃ­Â®Å¡Ã¢ÃÃ­Â³Â¢ncia
    QUALIA = "qualia"  # Campo de qualia
    VOID = "void"  # Campo do vazio

@dataclass
class FieldMetrics:
    """MÃ¢ÃÃ­Â®Å¡ÃÃ©Â©tricas de campo"""
    energy: float = 0.0  # Energia total
    coherence: float = 0.0  # CoerÃ¢ÃÃ­Â®Å¡Ã¢ÃÃ­Â³Â¢ncia quÃ¢ÃÃ­Â®Å¡ÃÃ©Â¢ntica
    entanglement: float = 0.0  # EntrelaÃ¢ÃÃ­Â®Å¡ÃÃ­Å¸amento
    resonance: float = 0.0  # RessonÃ¢ÃÃ­Â®Å¡ÃÃ©Â¢ncia
    emergence: float = 0.0  # EmergÃ¢ÃÃ­Â®Å¡Ã¢ÃÃ­Â³Â¢ncia
    stability: float = 0.0  # Estabilidade

@dataclass
class FieldConstants:
    """Constantes fundamentais"""
    PHI: float = (1 + np.sqrt(5)) / 2  # ProporÃ¢ÃÃ­Â®Å¡ÃÃ­Å¸Ã¢ÃÃ­Â®Å¡ÃÃ©Â£o Ã¢ÃÃ­Â®Å¡Ã¢ÃÃ­Â®Å¡ÃÃ©Â°urea
    PHI_INVERSE: float = 2 / (1 + np.sqrt(5))  # Inverso Ã¢ÃÃ­Â®Å¡ÃÃ©Â°ureo
    PLANCK: float = 6.62607015e-34  # Constante de Planck
    BOLTZMANN: float = 1.380649e-23  # Constante de Boltzmann
    GOLDEN_ANGLE: float = 2.399963229  # Ã¢ÃÃ­Â®Å¡Ã¢ÃÃ­Â®Å¡ÃÃ©Â©ngulo Ã¢ÃÃ­Â®Å¡ÃÃ©Â°ureo
    VOID_THRESHOLD: float = 0.618  # Limiar do vazio (PHI_INVERSE)
    COHERENCE_THRESHOLD: float = 0.382  # Limiar de coerÃ¢ÃÃ­Â®Å¡Ã¢ÃÃ­Â³Â¢ncia (1 - PHI_INVERSE)
    RESONANCE_THRESHOLD: float = 0.5  # Limiar de ressonÃ¢ÃÃ­Â®Å¡ÃÃ©Â¢ncia
    EMERGENCE_THRESHOLD: float = 0.236  # Limiar de emergÃ¢ÃÃ­Â®Å¡Ã¢ÃÃ­Â³Â¢ncia (PHI_INVERSE^2)

@dataclass
class FieldState:
    """Estado do campo"""
    type: FieldType  # Tipo do campo
    data: np.ndarray  # Dados do campo
    metrics: FieldMetrics  # MÃ¢ÃÃ­Â®Å¡ÃÃ©Â©tricas
    timestamp: float  # Timestamp

class FieldMemory:
    """MemÃ¢ÃÃ­Â®Å¡Ã¢ÃÃ­Â´Â¥ria do campo"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.states: List[FieldState] = []
        self.metrics_history: Dict[str, List[float]] = {
            'energy': [],
            'coherence': [],
            'entanglement': [],
            'resonance': [],
            'emergence': [],
            'stability': []
        }
    
    def add_state(self, state: FieldState):
        """Adiciona novo estado"""
        self.states.append(state)
        if len(self.states) > self.max_size:
            self.states.pop(0)
        
        # Atualiza histÃ¢ÃÃ­Â®Å¡Ã¢ÃÃ­Â´Â¥rico de mÃ¢ÃÃ­Â®Å¡ÃÃ©Â©tricas
        for key, value in state.metrics.__dict__.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
                if len(self.metrics_history[key]) > self.max_size:
                    self.metrics_history[key].pop(0)
    
    def get_state(self, index: int) -> Optional[FieldState]:
        """Retorna estado especÃ¢ÃÃ­Â®Å¡Ã¢ÃÃ­Â´Â fico"""
        if 0 <= index < len(self.states):
            return self.states[index]
        return None
    
    def get_metrics_history(self) -> Dict[str, List[float]]:
        """Retorna histÃ¢ÃÃ­Â®Å¡Ã¢ÃÃ­Â´Â¥rico de mÃ¢ÃÃ­Â®Å¡ÃÃ©Â©tricas"""
        return self.metrics_history
    
    def clear(self):
        """Limpa memÃ¢ÃÃ­Â®Å¡Ã¢ÃÃ­Â´Â¥ria"""
        self.states.clear()
        for key in self.metrics_history:
            self.metrics_history[key].clear()
