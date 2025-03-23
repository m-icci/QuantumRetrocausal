"""
Campo de dança quântica otimizado com NumPy
"""
import numpy as np
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from scipy import fft, signal, linalg
from ..constants import QualiaConstants

@dataclass
class DanceState:
    """Estado da dança quântica"""
    field: np.ndarray
    coherence: float = 0.0
    energy: float = 0.0
    granularity: int = 21  # Granularidade padrão
    timestamp: datetime = datetime.now()
    resonance: float = 0.0
    pattern_strength: float = 0.0

class QuantumDance:
    """Implementa dança quântica com operações NumPy otimizadas"""
    
    def __init__(self, size: int = 64, granularity: int = 21):
        """
        Inicializa dança quântica
        
        Args:
            size: Tamanho do campo
            granularity: Granularidade (3, 21 ou 42 bits)
        """
        if not QualiaConstants.validate_granularity(granularity):
            raise ValueError(f"Granularidade {granularity} inválida. Use 3, 21 ou 42 bits.")
            
        self.size = size
        self.granularity = granularity
        
        # Inicializa campos com distribuição gaussiana otimizada
        self.state = np.random.normal(0, 0.1, size).astype(np.float64)
        self.state = self.normalize_array(self.state)
        self.last_state = self.state.copy()
        
        # Fatores de ajuste dinâmicos
        self.coherence_factor = QualiaConstants.get_coherence_factor(granularity)
        self.adaptation_rate = np.exp(-granularity/42)
        self.resonance_threshold = 0.7
        
        # Buffers pré-alocados
        self.fft_buffer = np.zeros(size, dtype=np.complex128)
        self.pattern_memory = np.zeros((10, size), dtype=np.float64)
        self.pattern_idx = 0
        self.freq_grid = fft.fftfreq(size)
        self.resonance_filter = self._create_resonance_filter()
        
        # Métricas de evolução temporal
        self.coherence_history = np.zeros(10)
        self.energy_history = np.zeros(10)
        self.pattern_strength_history = np.zeros(10)
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
        """Calcula densidade espectral usando FFT otimizado"""
        np.copyto(self.fft_buffer, self.state)
        self.fft_buffer = fft.fft(self.fft_buffer)
        spectral_density = np.abs(self.fft_buffer) ** 2
        return self.normalize_array(spectral_density)
        
    def apply_resonance_filter(self) -> None:
        """Aplica filtro de ressonância vetorizado"""
        np.multiply(self.fft_buffer, self.resonance_filter, out=self.fft_buffer)
        self.state = np.real(fft.ifft(self.fft_buffer))
        self.state = self.normalize_array(self.state)
        
    def update_pattern_memory(self) -> None:
        """Atualiza memória de padrões de forma vetorizada"""
        self.pattern_memory[self.pattern_idx] = self.state
        self.pattern_idx = (self.pattern_idx + 1) % 10
        
    def detect_patterns(self) -> Tuple[float, np.ndarray]:
        """
        Detecta padrões emergentes usando análise espectral e SVD
        
        Returns:
            Tuple[float, np.ndarray]: Força do padrão e padrão dominante
        """
        # Calcula matriz de correlação temporal
        pattern_matrix = self.pattern_memory - np.mean(self.pattern_memory, axis=0)
        u, s, vh = linalg.svd(pattern_matrix, full_matrices=False)
        
        # Extrai padrão dominante e sua força
        pattern_strength = s[0] / (np.sum(s) + 1e-10)
        dominant_pattern = vh[0]
        
        return pattern_strength, dominant_pattern
        
    def compute_coherence(self) -> float:
        """Calcula coerência usando decomposição SVD"""
        # Cria matriz de correlação
        corr_matrix = np.outer(self.state, self.state)
        _, s, _ = linalg.svd(corr_matrix, full_matrices=False)
        return s[0] / (np.sum(s) + 1e-10)
        
    def update_metrics(self, coherence: float, energy: float, pattern_strength: float) -> None:
        """Atualiza métricas de forma vetorizada"""
        self.coherence_history[self.history_idx] = coherence
        self.energy_history[self.history_idx] = energy
        self.pattern_strength_history[self.history_idx] = pattern_strength
        self.history_idx = (self.history_idx + 1) % 10
        
    def evolve(self, void_state: Optional[np.ndarray] = None,
               consciousness_state: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Evolui estado da dança usando operações NumPy otimizadas
        
        Args:
            void_state: Estado do vazio quântico
            consciousness_state: Estado da consciência
            
        Returns:
            Dict com estado e métricas
        """
        # Salva estado anterior
        np.copyto(self.last_state, self.state)
        
        # Aplica influências externas de forma vetorizada
        if void_state is not None:
            void_influence = void_state * self.adaptation_rate
            np.add(self.state, void_influence, out=self.state)
            self.state = self.normalize_array(self.state)
            
        if consciousness_state is not None:
            consciousness_influence = consciousness_state * self.coherence_factor
            np.add(self.state, consciousness_influence, out=self.state)
            self.state = self.normalize_array(self.state)
        
        # Atualiza padrões e calcula métricas
        self.update_pattern_memory()
        pattern_strength, dominant_pattern = self.detect_patterns()
        
        # Calcula métricas de coerência e energia
        coherence = self.compute_coherence()
        energy = np.sum(self.state ** 2)
        self.update_metrics(coherence, energy, pattern_strength)
        
        # Aplica filtro de ressonância adaptativo
        if coherence > self.resonance_threshold:
            self.apply_resonance_filter()
            
        # Retorna estado atual
        return {
            'state': DanceState(
                field=self.state.copy(),
                coherence=coherence,
                energy=energy,
                granularity=self.granularity,
                timestamp=datetime.now(),
                resonance=np.mean(self.resonance_filter),
                pattern_strength=pattern_strength
            ),
            'metrics': {
                'coherence_history': self.coherence_history.copy(),
                'energy_history': self.energy_history.copy(),
                'pattern_strength_history': self.pattern_strength_history.copy(),
                'dominant_pattern': dominant_pattern
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
        future_state = np.real(fft.ifft(future_fft))
        future_state = self.normalize_array(future_state)
        
        # Aplica peso da granularidade
        weight = QualiaConstants.get_granularity_weight(self.granularity)
        future_state *= weight * self.coherence_factor
        future_state = self.normalize_array(future_state)
        
        return future_state
