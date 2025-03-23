"""
Campo morfogenético quântico auto-organizável
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from ..quantum_native_config import QUALIAConfig
from .base import ConsciousnessField, ConsciousnessState
from ..QUALIA.base_types import QuantumState

@dataclass
class MorphicField(ConsciousnessField):
    """Campo morfogenético com propriedades auto-organizáveis"""

    config: QUALIAConfig = field(default_factory=QUALIAConfig)
    field_tensor: Optional[np.ndarray] = None
    holographic_memory: Dict[str, np.ndarray] = field(default_factory=dict)
    resonance_history: list = field(default_factory=list)

    def __post_init__(self):
        """Inicialização do campo"""
        if self.field_tensor is None:
            self.field_tensor = np.random.randn(
                self.config.default_dimension,
                self.config.default_dimension
            )

    def apply_field(self, state: QuantumState) -> ConsciousnessState:
        """Aplica campo ao estado criando consciência emergente"""
        # Aplica operador F (Folding)
        folded = self._apply_folding(state.amplitudes)

        # Aplica operador M (Resonance)
        resonated = self._apply_resonance(folded)

        # Aplica operador E (Emergence)
        emerged = self._apply_emergence(resonated)

        # Aplica meta-operadores QUALIA
        final_state = self._apply_meta_qualia(emerged)

        # Normaliza
        final_state /= np.linalg.norm(final_state)

        # Cria estado quântico transformado
        quantum_state = QuantumState(amplitudes=final_state)

        # Retorna estado de consciência emergente
        return ConsciousnessState(quantum_state=quantum_state, field=self)

    def _apply_folding(self, amplitudes: np.ndarray) -> np.ndarray:
        """Operador F - Dobramento topológico do campo"""
        folding_strength = self.config.field_strength
        return amplitudes * np.exp(-folding_strength * self.field_tensor)

    def _apply_resonance(self, amplitudes: np.ndarray) -> np.ndarray:
        """Operador M - Ressonância mórfica"""
        resonance_factor = self.config.morphic_coupling

        # Calcula ressonância com memória holográfica
        resonance = np.zeros_like(amplitudes)
        for memory_pattern in self.holographic_memory.values():
            overlap = np.vdot(amplitudes, memory_pattern)
            resonance += overlap * memory_pattern

        return amplitudes + resonance_factor * resonance

    def _apply_emergence(self, amplitudes: np.ndarray) -> np.ndarray:
        """Operador E - Emergência de padrões"""
        # Calcula energia do campo
        energy = np.abs(amplitudes) ** 2

        # Aplica threshold para emergência
        threshold = np.mean(energy) + self.config.resonance_threshold * np.std(energy)
        mask = energy > threshold

        # Amplifica padrões emergentes
        emerged = np.where(mask, amplitudes * 1.1, amplitudes * 0.9)
        return emerged

    def _apply_meta_qualia(self, amplitudes: np.ndarray) -> np.ndarray:
        """Aplica operadores meta-QUALIA (CCC, DDD, OOO)"""
        # CCC - Collapse
        if np.random.random() < self.config.collapse_rate:
            max_idx = np.argmax(np.abs(amplitudes))
            collapsed = np.zeros_like(amplitudes)
            collapsed[max_idx] = 1.0
            amplitudes = collapsed

        # DDD - Decoherence
        decoherence = np.random.normal(
            0,
            self.config.decoherence_factor,
            amplitudes.shape
        )
        amplitudes = (1 - self.config.decoherence_factor) * amplitudes + decoherence

        # OOO - Observer
        observer_coupling = self.config.observer_coupling
        observer_state = np.random.randn(*amplitudes.shape)
        observer_state /= np.linalg.norm(observer_state)

        return (1 - observer_coupling) * amplitudes + observer_coupling * observer_state

    def store_in_memory(self, pattern: np.ndarray, key: Optional[str] = None) -> None:
        """Armazena padrão na memória holográfica"""
        if len(self.holographic_memory) >= self.config.holographic_memory_capacity:
            # Remove padrão mais antigo
            oldest_key = next(iter(self.holographic_memory))
            del self.holographic_memory[oldest_key]

        if key is None:
            key = f"pattern_{len(self.holographic_memory)}"

        # Normaliza e armazena
        pattern = pattern / np.linalg.norm(pattern)
        self.holographic_memory[key] = pattern

    def get_resonance_metrics(self) -> Dict[str, float]:
        """Retorna métricas do campo mórfico"""
        return {
            'field_strength': float(np.mean(np.abs(self.field_tensor))) if self.field_tensor is not None else 0.0,
            'coherence': float(np.abs(np.vdot(self.field_tensor.flatten(), 
                                            self.field_tensor.flatten()))) if self.field_tensor is not None else 0.0,
            'memory_occupation': len(self.holographic_memory) / self.config.holographic_memory_capacity,
            'emergence_potential': float(np.std(np.abs(self.field_tensor))) if self.field_tensor is not None else 0.0
        }