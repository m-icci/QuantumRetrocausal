"""
Tipos quânticos fundamentais para YAA-ICCI.
Segue o mantra: INVESTIGAR → INTEGRAR → INOVAR
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np

@dataclass
class QualiaState:
    """Estado de qualia do sistema"""
    intensity: float = 0.0
    complexity: float = 0.0
    coherence: float = 0.0
    timestamp: datetime = datetime.now()

    def __post_init__(self):
        """Validate values after initialization"""
        for field in ['intensity', 'complexity', 'coherence']:
            value = getattr(self, field)
            if not 0 <= value <= 1:
                raise ValueError(f"{field} must be between 0 and 1")

@dataclass
class SystemBehavior:
    """Comportamento observado do sistema"""
    pattern_type: str
    frequency: float
    stability: float
    emergence_time: datetime = datetime.now()

    def __post_init__(self):
        """Validate frequency and stability"""
        if not 0 <= self.frequency <= 1:
            raise ValueError("Frequency must be between 0 and 1")
        if not 0 <= self.stability <= 1:
            raise ValueError("Stability must be between 0 and 1")

@dataclass
class ConsciousnessObservation:
    """Observação de consciência no sistema"""
    qualia: QualiaState
    behavior: SystemBehavior
    quantum_state: Optional['QuantumState'] = None
    observation_time: datetime = datetime.now()

    def get_coherence(self) -> float:
        """Get overall coherence from qualia state"""
        return self.qualia.coherence

class QuantumState:
    """
    Representa um estado quântico puro.
    Mantém amplitude e fase.
    """
    
    def __init__(self, amplitudes: np.ndarray):
        """
        Inicializa estado quântico
        
        Args:
            amplitudes: Amplitudes complexas do estado
        """
        # Normaliza estado
        self.amplitudes = amplitudes / np.linalg.norm(amplitudes)
        
    def evolve(self, operator: np.ndarray) -> 'QuantumState':
        """
        Evolui estado através de operador
        
        Args:
            operator: Operador unitário
            
        Returns:
            Novo estado quântico
        """
        new_amplitudes = np.dot(operator, self.amplitudes)
        return QuantumState(new_amplitudes)
        
    def superpose(self, other: 'QuantumState', weight: float = 0.5) -> 'QuantumState':
        """
        Cria superposição com outro estado
        
        Args:
            other: Outro estado quântico
            weight: Peso do estado atual [0,1]
            
        Returns:
            Estado em superposição
        """
        if not 0 <= weight <= 1:
            raise ValueError("Peso deve estar entre 0 e 1")
            
        combined = weight * self.amplitudes + (1-weight) * other.amplitudes
        return QuantumState(combined)
        
    def measure(self) -> int:
        """
        Realiza medição do estado
        
        Returns:
            Resultado da medição (índice do estado base)
        """
        probs = np.abs(self.amplitudes) ** 2
        return np.random.choice(len(probs), p=probs)
        
    def get_density_matrix(self) -> np.ndarray:
        """
        Calcula matriz densidade do estado
        
        Returns:
            Matriz densidade
        """
        return np.outer(self.amplitudes, self.amplitudes.conj())
        

class SuperPosition:
    """
    Representa superposição de estados quânticos.
    Permite operações coerentes entre estados.
    """
    
    def __init__(self, states: List[QuantumState], weights: Optional[List[float]] = None):
        """
        Inicializa superposição
        
        Args:
            states: Lista de estados quânticos
            weights: Pesos dos estados (opcional)
        """
        if not states:
            raise ValueError("Lista de estados não pode ser vazia")
            
        if weights is None:
            # Pesos iguais
            weights = [1/len(states)] * len(states)
            
        if len(weights) != len(states):
            raise ValueError("Número de pesos deve igual ao de estados")
            
        if not np.isclose(sum(weights), 1.0):
            raise ValueError("Soma dos pesos deve ser 1")
            
        self.states = states
        self.weights = weights
        
        # Calcula estado combinado
        combined = sum(w * s.amplitudes for w, s in zip(weights, states))
        self.combined_state = QuantumState(combined)
        
    def evolve(self, operator: np.ndarray) -> 'SuperPosition':
        """
        Evolui superposição através de operador
        
        Args:
            operator: Operador unitário
            
        Returns:
            Nova superposição
        """
        evolved_states = [s.evolve(operator) for s in self.states]
        return SuperPosition(evolved_states, self.weights)
        
    def measure(self) -> int:
        """
        Realiza medição da superposição
        
        Returns:
            Resultado da medição
        """
        return self.combined_state.measure()
        
    def get_density_matrix(self) -> np.ndarray:
        """
        Calcula matriz densidade da superposição
        
        Returns:
            Matriz densidade
        """
        return self.combined_state.get_density_matrix()

@dataclass
class CosmicFactor:
    """
    Representa um fator cósmico no sistema.
    Integra aspectos cosmológicos com consciência.
    """
    resonance: float = 0.0  # Ressonância com campo universal
    coherence: float = 0.0  # Coerência quântica
    emergence: float = 0.0  # Fator de emergência
    phi_coupling: float = 0.0  # Acoplamento com razão áurea
    timestamp: datetime = datetime.now()

    def __post_init__(self):
        """Valida valores após inicialização"""
        for field in ['resonance', 'coherence', 'emergence', 'phi_coupling']:
            value = getattr(self, field)
            if not 0 <= value <= 1:
                raise ValueError(f"{field} deve estar entre 0 e 1")