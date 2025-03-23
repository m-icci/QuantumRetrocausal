"""
Campo do vazio quântico otimizado com NumPy
"""
import numpy as np
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from datetime import datetime
from scipy import fft, signal, linalg
from ..constants import QualiaConstants

@dataclass
class VoidState:
    """Estado do vazio quântico"""
    potential: np.ndarray
    energy: float = 0.0
    density: float = 0.0
    granularity: int = 21  # Granularidade padrão
    timestamp: datetime = datetime.now()
    coherence: float = 0.0
    resonance: float = 0.0

class QuantumVoid:
    """Implementa vazio quântico com operações NumPy otimizadas"""
    
    def __init__(self, size: int = 64, granularity: int = 21):
        """
        Inicializa vazio quântico
        
        Args:
            size: Tamanho do campo
            granularity: Granularidade (3, 21 ou 42 bits)
        """
        if not QualiaConstants.validate_granularity(granularity):
            raise ValueError(f"Granularidade {granularity} inválida. Use 3, 21 ou 42 bits.")
            
        self.size = size
        self.granularity = granularity
        
        # Inicializa campos com distribuição gaussiana otimizada
        self.state = np.zeros(size, dtype=np.float64)
        self.potential = np.random.normal(0, 0.1, size).astype(np.float64)
        self.potential = self.normalize_array(self.potential)
        
        # Fatores de ajuste dinâmicos
        self.coherence_factor = QualiaConstants.get_coherence_factor(granularity)
        self.adaptation_rate = np.exp(-granularity/42)  # Taxa de adaptação inversamente proporcional à granularidade
        self.resonance_threshold = 0.7  # Limiar de ressonância
        
        # Buffers pré-alocados para análise espectral
        self.fft_buffer = np.zeros(size, dtype=np.complex128)
        self.spectral_density = np.zeros(size, dtype=np.float64)
        self.freq_grid = fft.fftfreq(size)
        self.resonance_filter = self._create_resonance_filter()
        
        # Métricas de coerência
        self.coherence_history = np.zeros(10)
        self.energy_history = np.zeros(10)
        self.history_idx = 0
        
    def normalize_array(self, arr: np.ndarray) -> np.ndarray:
        """Normaliza array com estabilidade numérica usando einsum"""
        norm = np.sqrt(np.einsum('i,i->', arr, arr))
        if norm < 1e-10:
            return np.zeros_like(arr)
        return arr / (norm + 1e-10)
    
    def _create_resonance_filter(self) -> np.ndarray:
        """Cria filtro de ressonância otimizado"""
        return np.exp(-(self.freq_grid * self.size/self.granularity)**2)
        
    def compute_spectral_density(self) -> np.ndarray:
        """Calcula densidade espectral do potencial usando FFT otimizado"""
        np.copyto(self.fft_buffer, self.potential)
        self.fft_buffer = fft.fft(self.fft_buffer)
        np.abs(self.fft_buffer, out=self.spectral_density)
        self.spectral_density **= 2
        self.spectral_density = self.normalize_array(self.spectral_density)
        return self.spectral_density
        
    def apply_resonance_filter(self) -> None:
        """Aplica filtro de ressonância vetorizado"""
        np.multiply(self.fft_buffer, self.resonance_filter, out=self.fft_buffer)
        self.potential = np.real(fft.ifft(self.fft_buffer))
        self.potential = self.normalize_array(self.potential)
        
    def compute_coherence(self) -> float:
        """Calcula coerência usando decomposição SVD"""
        # Cria matriz de correlação
        corr_matrix = np.outer(self.potential, self.potential)
        _, s, _ = linalg.svd(corr_matrix, full_matrices=False)
        return s[0] / (np.sum(s) + 1e-10)
        
    def update_metrics(self, coherence: float, energy: float) -> None:
        """Atualiza métricas de forma vetorizada"""
        self.coherence_history[self.history_idx] = coherence
        self.energy_history[self.history_idx] = energy
        self.history_idx = (self.history_idx + 1) % 10
        
    def evolve(self, market_data: Optional[np.ndarray] = None,
               operator_sequence: Optional[str] = None) -> Dict[str, Any]:
        """
        Evolui estado do vazio usando operações NumPy otimizadas
        
        Args:
            market_data: Dados de mercado opcionais
            operator_sequence: Sequência de operadores
            
        Returns:
            Dict com estado e métricas
        """
        # Atualiza potencial com peso da granularidade de forma vetorizada
        weight = QualiaConstants.get_granularity_weight(self.granularity)
        if market_data is not None:
            np.multiply(self.potential, (1 - self.adaptation_rate), out=self.potential)
            market_influence = weight * market_data * self.adaptation_rate
            np.add(self.potential, market_influence, out=self.potential)
            self.potential = self.normalize_array(self.potential)
            
        # Calcula gradiente e densidade espectral
        self.state = np.gradient(self.potential)
        self.state = self.normalize_array(self.state)
        spectral_density = self.compute_spectral_density()
        
        # Calcula métricas de coerência e energia
        coherence = self.compute_coherence()
        energy = np.sum(self.potential ** 2)
        self.update_metrics(coherence, energy)
        
        # Aplica filtro de ressonância adaptativo
        if coherence > self.resonance_threshold:
            self.apply_resonance_filter()
            
        # Retorna estado atual
        return {
            'state': VoidState(
                potential=self.potential.copy(),
                energy=energy,
                density=np.mean(spectral_density),
                granularity=self.granularity,
                timestamp=datetime.now(),
                coherence=coherence,
                resonance=np.mean(self.resonance_filter)
            ),
            'metrics': {
                'coherence_history': self.coherence_history.copy(),
                'energy_history': self.energy_history.copy(),
                'spectral_density': spectral_density
            }
        }
        
    def peek_future(self, steps: int = 1) -> np.ndarray:
        """
        Prevê estado futuro usando análise espectral
        
        Args:
            steps: Número de passos
            
        Returns:
            Estado previsto
        """
        # Previsão no domínio da frequência
        freq = fft.fftfreq(self.size)
        phase_advance = np.exp(2j * np.pi * freq * steps)
        
        # Evolução espectral
        future_fft = self.fft_buffer * phase_advance
        future_potential = np.real(fft.ifft(future_fft))
        future_potential = self.normalize_array(future_potential)
        
        # Calcula estado futuro com peso da granularidade
        weight = QualiaConstants.get_granularity_weight(self.granularity)
        future_state = np.gradient(future_potential) * weight * self.coherence_factor
        future_state = self.normalize_array(future_state)
        
        return future_state
