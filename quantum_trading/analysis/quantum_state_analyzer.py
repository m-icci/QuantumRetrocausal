"""
Analisador de estados quânticos
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
from scipy.stats import entropy

logger = logging.getLogger(__name__)

class QuantumStateAnalyzer:
    """
    Analisa estados quânticos e extrai métricas relevantes
    """
    
    def __init__(self):
        self.metrics_history = []
        self.max_history = 1000
        
    async def analyze(self, state: np.ndarray) -> Dict[str, float]:
        """
        Analisa um estado quântico e retorna métricas
        
        Args:
            state: Array numpy representando o estado quântico
            
        Returns:
            Dict com métricas do estado
        """
        try:
            # Normaliza estado
            state = self._normalize_state(state)
            
            # Calcula métricas básicas
            metrics = {
                'coherence': self._calculate_coherence(state),
                'entropy': self._calculate_entropy(state),
                'energy': self._calculate_energy(state),
                'complexity': self._calculate_complexity(state)
            }
            
            # Calcula métricas avançadas
            metrics.update({
                'phase_alignment': self._calculate_phase_alignment(state),
                'interference_pattern': self._calculate_interference(state),
                'tunneling_probability': self._calculate_tunneling(state),
                'entanglement_measure': self._calculate_entanglement(state)
            })
            
            # Atualiza histórico
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.max_history:
                self.metrics_history.pop(0)
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing quantum state: {str(e)}")
            return {}
            
    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normaliza o estado quântico"""
        try:
            norm = np.linalg.norm(state)
            if norm > 0:
                return state / norm
            return state
        except Exception as e:
            logger.error(f"Error normalizing state: {str(e)}")
            return state
            
    def _calculate_coherence(self, state: np.ndarray) -> float:
        """Calcula coerência do estado"""
        try:
            # Usa matriz densidade
            density_matrix = np.outer(state, state.conj())
            # Coerência como soma dos elementos fora da diagonal
            coherence = np.sum(np.abs(density_matrix - np.diag(np.diag(density_matrix))))
            return float(coherence)
        except Exception as e:
            logger.error(f"Error calculating coherence: {str(e)}")
            return 0.0
            
    def _calculate_entropy(self, state: np.ndarray) -> float:
        """Calcula entropia do estado"""
        try:
            # Usa probabilidades como quadrado das amplitudes
            probs = np.abs(state) ** 2
            # Remove zeros para evitar log(0)
            probs = probs[probs > 0]
            return float(entropy(probs))
        except Exception as e:
            logger.error(f"Error calculating entropy: {str(e)}")
            return 0.0
            
    def _calculate_energy(self, state: np.ndarray) -> float:
        """Calcula energia do estado"""
        try:
            # Energia como média ponderada
            energies = np.arange(len(state))
            probs = np.abs(state) ** 2
            return float(np.sum(energies * probs))
        except Exception as e:
            logger.error(f"Error calculating energy: {str(e)}")
            return 0.0
            
    def _calculate_complexity(self, state: np.ndarray) -> float:
        """Calcula complexidade do estado"""
        try:
            # Usa número de componentes significativas
            significant = np.abs(state) > 0.01
            return float(np.sum(significant))
        except Exception as e:
            logger.error(f"Error calculating complexity: {str(e)}")
            return 0.0
            
    def _calculate_phase_alignment(self, state: np.ndarray) -> float:
        """Calcula alinhamento de fase"""
        try:
            # Usa ângulos das amplitudes complexas
            phases = np.angle(state)
            # Mede dispersão das fases
            return float(1.0 - np.std(phases) / np.pi)
        except Exception as e:
            logger.error(f"Error calculating phase alignment: {str(e)}")
            return 0.0
            
    def _calculate_interference(self, state: np.ndarray) -> float:
        """Calcula padrão de interferência"""
        try:
            # Usa transformada de Fourier
            fft = np.fft.fft(state)
            # Mede força do padrão
            pattern = np.abs(fft) ** 2
            return float(np.max(pattern) / np.mean(pattern))
        except Exception as e:
            logger.error(f"Error calculating interference: {str(e)}")
            return 0.0
            
    def _calculate_tunneling(self, state: np.ndarray) -> float:
        """Calcula probabilidade de tunelamento"""
        try:
            # Usa razão entre regiões separadas
            mid = len(state) // 2
            left = np.sum(np.abs(state[:mid]) ** 2)
            right = np.sum(np.abs(state[mid:]) ** 2)
            return float(min(left, right) / max(left, right))
        except Exception as e:
            logger.error(f"Error calculating tunneling: {str(e)}")
            return 0.0
            
    def _calculate_entanglement(self, state: np.ndarray) -> float:
        """Calcula medida de emaranhamento"""
        try:
            # Usa entropia de von Neumann
            density_matrix = np.outer(state, state.conj())
            eigenvalues = np.linalg.eigvalsh(density_matrix)
            # Remove valores muito pequenos
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            # Entropia de von Neumann
            return float(-np.sum(eigenvalues * np.log2(eigenvalues)))
        except Exception as e:
            logger.error(f"Error calculating entanglement: {str(e)}")
            return 0.0 