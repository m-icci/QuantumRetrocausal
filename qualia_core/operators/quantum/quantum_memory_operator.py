"""
Quantum Memory Operator
---------------------
Implementa operador de memória quântica para o inconsciente coletivo.
Integra conceitos de Jung com memória holográfica distribuída.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from quantum.core.state.quantum_state import QuantumState

@dataclass
class ArchetypalPattern:
    """Padrão arquetípico na memória holográfica"""
    symbol: str
    resonance: float
    coherence: float
    field_strength: float
    collective_weight: float

@dataclass 
class HolographicMemoryState:
    """Estado da memória holográfica quântica"""
    personal: np.ndarray      # Memória pessoal
    collective: np.ndarray    # Inconsciente coletivo
    archetypal: np.ndarray   # Padrões arquetípicos
    morphic: np.ndarray      # Campo mórfico

class QuantumMemoryOperator:
    """
    Operador de memória quântica para o inconsciente coletivo.
    Implementa:
    - Memória holográfica distribuída
    - Padrões arquetípicos (Jung)
    - Campos mórficos (Sheldrake)
    - Ressonância mórfica
    """
    
    def __init__(self, dimensions: int = 8):
        """
        Inicializa operador de memória quântica.
        
        Args:
            dimensions: Dimensões do espaço de memória
        """
        self.dimensions = dimensions
        self.phi = (1 + np.sqrt(5)) / 2  # Razão áurea
        
        # Inicializa estados de memória
        self.memory_state = HolographicMemoryState(
            personal=np.zeros((dimensions, dimensions), dtype=np.complex128),
            collective=np.zeros((dimensions, dimensions), dtype=np.complex128),
            archetypal=np.zeros((dimensions, dimensions), dtype=np.complex128),
            morphic=np.zeros((dimensions, dimensions), dtype=np.complex128)
        )
        
        # Histórico de padrões
        self.pattern_history: List[ArchetypalPattern] = []
        
    def store_pattern(self, pattern: np.ndarray, is_collective: bool = False) -> None:
        """
        Armazena padrão na memória holográfica.
        
        Args:
            pattern: Padrão quântico a armazenar
            is_collective: Se True, armazena no inconsciente coletivo
        """
        # Normaliza padrão
        pattern = pattern / np.linalg.norm(pattern)
        
        # Aplica transformada holográfica
        holographic_pattern = self._holographic_transform(pattern)
        
        if is_collective:
            # Integra ao inconsciente coletivo
            self.memory_state.collective += holographic_pattern
            # Atualiza campo mórfico
            self._update_morphic_field(holographic_pattern)
        else:
            # Armazena na memória pessoal
            self.memory_state.personal += holographic_pattern
            
        # Normaliza estados
        self._normalize_states()
        
    def retrieve_pattern(self, query: np.ndarray) -> np.ndarray:
        """
        Recupera padrão da memória holográfica.
        
        Args:
            query: Padrão de consulta
            
        Returns:
            Padrão recuperado
        """
        # Normaliza consulta
        query = query / np.linalg.norm(query)
        
        # Aplica transformada holográfica
        holographic_query = self._holographic_transform(query)
        
        # Calcula ressonância com memórias
        personal_resonance = np.vdot(holographic_query, self.memory_state.personal)
        collective_resonance = np.vdot(holographic_query, self.memory_state.collective)
        archetypal_resonance = np.vdot(holographic_query, self.memory_state.archetypal)
        
        # Pondera contribuições
        pattern = (
            0.3 * self.memory_state.personal +
            0.4 * self.memory_state.collective +
            0.3 * self.memory_state.archetypal
        )
        
        # Aplica campo mórfico
        pattern = self._apply_morphic_field(pattern)
        
        return self._inverse_holographic_transform(pattern)
    
    def _holographic_transform(self, pattern: np.ndarray) -> np.ndarray:
        """Aplica transformada holográfica"""
        return np.fft.fft2(pattern)
    
    def _inverse_holographic_transform(self, pattern: np.ndarray) -> np.ndarray:
        """Aplica transformada holográfica inversa"""
        return np.fft.ifft2(pattern)
    
    def _update_morphic_field(self, pattern: np.ndarray) -> None:
        """Atualiza campo mórfico"""
        # Calcula ressonância mórfica
        resonance = np.abs(np.vdot(pattern, self.memory_state.morphic))
        
        # Atualiza campo com peso baseado na razão áurea
        weight = self.phi / (1 + resonance)
        self.memory_state.morphic = (
            (1 - weight) * self.memory_state.morphic +
            weight * pattern
        )
        
    def _apply_morphic_field(self, pattern: np.ndarray) -> np.ndarray:
        """Aplica influência do campo mórfico"""
        resonance = np.abs(np.vdot(pattern, self.memory_state.morphic))
        return pattern + resonance * self.memory_state.morphic
    
    def _normalize_states(self) -> None:
        """Normaliza todos os estados de memória"""
        for field in [
            self.memory_state.personal,
            self.memory_state.collective,
            self.memory_state.archetypal,
            self.memory_state.morphic
        ]:
            norm = np.linalg.norm(field)
            if norm > 0:
                field /= norm
                
    def analyze_archetypes(self) -> List[ArchetypalPattern]:
        """Analisa padrões arquetípicos ativos"""
        patterns = []
        
        # Extrai componentes principais
        eigenvals, eigenvecs = np.linalg.eigh(self.memory_state.collective)
        
        # Analisa os padrões mais fortes
        for i, (val, vec) in enumerate(
            zip(eigenvals[-3:], eigenvecs.T[-3:])
        ):
            # Calcula métricas
            resonance = np.abs(np.vdot(vec, self.memory_state.morphic))
            coherence = np.abs(np.vdot(vec, self.memory_state.archetypal))
            
            pattern = ArchetypalPattern(
                symbol=f"archetype_{i}",
                resonance=float(resonance),
                coherence=float(coherence),
                field_strength=float(val),
                collective_weight=float(np.abs(val) / np.sum(np.abs(eigenvals)))
            )
            patterns.append(pattern)
            
        self.pattern_history.extend(patterns)
        return patterns
    
    def get_metrics(self) -> Dict[str, float]:
        """Obtém métricas do operador"""
        if not self.pattern_history:
            return {}
            
        recent = self.pattern_history[-10:]
        return {
            "mean_resonance": float(np.mean([p.resonance for p in recent])),
            "mean_coherence": float(np.mean([p.coherence for p in recent])),
            "field_strength": float(np.mean([p.field_strength for p in recent])),
            "collective_influence": float(np.mean([p.collective_weight for p in recent]))
        }
