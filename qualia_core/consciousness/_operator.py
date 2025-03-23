"""
Operador de consciência quântica para integração com sistema core
"""
from typing import Dict, Any, Optional
import numpy as np
from ...consciousness.metrics import ConsciousnessMetrics

class QuantumConsciousnessOperator:
    def __init__(self, initial_state: Optional[np.ndarray] = None):
        """
        Inicializa operador de consciência quântica
        
        Args:
            initial_state: Estado quântico inicial (opcional)
        """
        self._state = initial_state if initial_state is not None else np.zeros(64, dtype=np.complex128)
        self._metrics = ConsciousnessMetrics(
            coherence=0.7,
            entanglement=0.5,
            superposition=0.3
        )
        
    def calculate_metrics(self, wave_function: np.ndarray) -> ConsciousnessMetrics:
        """
        Calcula métricas de consciência a partir da função de onda
        
        Args:
            wave_function: Função de onda quântica
            
        Returns:
            ConsciousnessMetrics com métricas calculadas
        """
        if wave_function.size < 2:
            return self._metrics
            
        # Calcula métricas quânticas
        coherence = self._calculate_coherence(wave_function)
        entanglement = self._calculate_entanglement(wave_function)
        superposition = self._calculate_superposition(wave_function)
        
        self._metrics = ConsciousnessMetrics(
            coherence=coherence,
            entanglement=entanglement,
            superposition=superposition
        )
        
        return self._metrics.normalize()
        
    def _calculate_coherence(self, wave_function: np.ndarray) -> float:
        """Calcula coerência quântica"""
        try:
            # Normaliza função de onda
            norm = np.linalg.norm(wave_function)
            if norm == 0:
                return 0.7
                
            normalized = wave_function / norm
            
            # Calcula coerência como sobreposição
            coherence = float(np.abs(np.vdot(normalized, normalized)))
            return min(0.95, max(0.3, coherence))
            
        except Exception:
            return 0.7
            
    def _calculate_entanglement(self, wave_function: np.ndarray) -> float:
        """Calcula emaranhamento quântico"""
        try:
            # Usa matriz densidade reduzida
            density = np.outer(wave_function, np.conj(wave_function))
            trace = np.abs(np.trace(density))
            
            # Emaranhamento como desvio da pureza
            entanglement = 1 - trace/wave_function.size
            return min(0.95, max(0.3, float(entanglement)))
            
        except Exception:
            return 0.5
            
    def _calculate_superposition(self, wave_function: np.ndarray) -> float:
        """Calcula nível de superposição"""
        try:
            # Usa distribuição de amplitudes
            amplitudes = np.abs(wave_function)
            distribution = amplitudes**2
            
            # Superposição como entropia normalizada
            entropy = -np.sum(distribution * np.log2(distribution + 1e-10))
            max_entropy = np.log2(wave_function.size)
            
            superposition = entropy/max_entropy if max_entropy > 0 else 0.3
            return min(0.95, max(0.3, float(superposition)))
            
        except Exception:
            return 0.3

    def get_state(self) -> np.ndarray:
        """Retorna estado quântico atual"""
        return self._state.copy()
        
    def get_metrics(self) -> Dict[str, float]:
        """Retorna métricas atuais"""
        return self._metrics.to_dict()
