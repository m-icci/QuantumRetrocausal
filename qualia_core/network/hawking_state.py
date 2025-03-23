"""
Gerenciamento de Estados de Partículas de Hawking
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from .quantum_config import QuantumParameters

@dataclass
class HawkingMetrics:
    """Métricas do estado de Hawking"""
    temperature: float
    entropy: float
    energy: float
    coherence: float
    entanglement: float

class HawkingState:
    """
    Gerenciador de Estados de Partículas de Hawking
    
    Implementa a simulação de partículas de Hawking (radiação térmica de buracos negros)
    em um contexto de rede quântica.
    """
    
    def __init__(
        self,
        dimensions: int,
        temperature: float = 0.1,
        entropy: float = 0.5,
        config: Optional[QuantumParameters] = None
    ):
        self.dimensions = dimensions
        self.temperature = temperature
        self.entropy = entropy
        
        # Configuração
        self.config = config or QuantumParameters()
        
        # Estado inicial
        self.state = np.random.random(dimensions)
        self.state = self.state / np.linalg.norm(self.state)
        
        # Histórico de estados
        self.history = []
        self.max_history = 100
        
        # Métricas
        self.metrics = self._calculate_metrics()
    
    def _calculate_metrics(self) -> HawkingMetrics:
        """Calcula métricas do estado"""
        # Temperatura efetiva baseada na energia
        energy = np.mean(np.abs(self.state)**2)
        effective_temp = self.temperature * (1 + energy)
        
        # Entropia de von Neumann
        density_matrix = np.outer(self.state, self.state.conj())
        eigenvalues = np.linalg.eigvals(density_matrix)
        eigenvalues = eigenvalues[eigenvalues > 0]  # Remove autovalores nulos
        von_neumann_entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
        
        # Coerência
        coherence = np.mean(np.abs(self.state))
        
        # Entrelaçamento (simplificado)
        entanglement = np.mean(np.abs(np.fft.fft(self.state)))
        
        return HawkingMetrics(
            temperature=effective_temp,
            entropy=von_neumann_entropy,
            energy=energy,
            coherence=coherence,
            entanglement=entanglement
        )
    
    def evolve(self, time_step: float = 0.01) -> None:
        """Evolui o estado de Hawking"""
        # Atualiza temperatura
        self.temperature *= (1 - time_step * self.config.decoherence_rate)
        
        # Evolui estado
        phase = np.exp(-1j * self.temperature * time_step)
        self.state *= phase
        
        # Normaliza
        self.state = self.state / np.linalg.norm(self.state)
        
        # Atualiza histórico
        self.history.append(self.state.copy())
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        # Atualiza métricas
        self.metrics = self._calculate_metrics()
    
    def entangle(self, other_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Entrelaça com outro estado"""
        # Cria estado de Bell
        bell_state = np.zeros((2, self.dimensions), dtype=complex)
        bell_state[0] = self.state
        bell_state[1] = other_state
        
        # Aplica transformação de Bell
        bell_state = bell_state / np.sqrt(2)
        
        # Atualiza estados
        self.state = bell_state[0]
        other_state = bell_state[1]
        
        return self.state, other_state
    
    def apply_hawking_radiation(self, energy: float) -> None:
        """Aplica efeito de radiação de Hawking"""
        # Simula emissão de partículas
        radiation = np.random.normal(0, np.sqrt(self.temperature), self.dimensions)
        radiation = radiation / np.linalg.norm(radiation)
        
        # Atualiza estado
        self.state = self.state * np.exp(-energy * self.temperature) + radiation * np.sqrt(1 - np.exp(-2 * energy * self.temperature))
        self.state = self.state / np.linalg.norm(self.state)
        
        # Atualiza métricas
        self.metrics = self._calculate_metrics()
    
    def get_retrocausal_echo(self, future_time: float) -> Optional[np.ndarray]:
        """Obtém eco retrocausal do futuro"""
        if not self.history:
            return None
            
        # Usa média ponderada pelo tempo
        weights = np.exp(-np.arange(len(self.history)) / self.config.phi)
        weights /= np.sum(weights)
        
        future_state = np.zeros(self.dimensions, dtype=complex)
        for state, weight in zip(self.history, weights):
            future_state += state * weight
            
        return future_state / np.linalg.norm(future_state)
    
    def to_dict(self) -> dict:
        """Converte para dicionário"""
        return {
            'dimensions': self.dimensions,
            'temperature': self.temperature,
            'entropy': self.entropy,
            'state': self.state.tolist(),
            'metrics': {
                'temperature': self.metrics.temperature,
                'entropy': self.metrics.entropy,
                'energy': self.metrics.energy,
                'coherence': self.metrics.coherence,
                'entanglement': self.metrics.entanglement
            }
        }
    
    @classmethod
    def from_dict(cls, data: dict, config: Optional[QuantumParameters] = None) -> 'HawkingState':
        """Cria a partir de dicionário"""
        instance = cls(
            dimensions=data['dimensions'],
            temperature=data['temperature'],
            entropy=data['entropy'],
            config=config
        )
        instance.state = np.array(data['state'])
        return instance 