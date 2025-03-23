"""
Operador de Emaranhamento Quântico
Implementa medidas de emaranhamento usando apenas numpy
"""

import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass
from quantum.core.state.quantum_state import QuantumState

@dataclass
class EntanglementMetrics:
    """Métricas de emaranhamento"""
    concurrence: float
    negativity: float
    entanglement_entropy: float
    schmidt_rank: int
    geometric_phase: float
    resonance_strength: float

class EntanglementOperator:
    """
    Implementa operações de emaranhamento quântico usando numpy
    - Medidas tradicionais (concorrência, negatividade)
    - Medidas geométricas (fase de Berry, ressonância)
    """
    
    def __init__(self):
        self._metrics_history = []
        self.PHI = (1 + np.sqrt(5)) / 2  # Razão áurea
        
    def apply(self, state: QuantumState) -> Dict[str, Any]:
        """
        Aplica análise de emaranhamento completa
        """
        rho = self._get_density_matrix(state)
        
        metrics = EntanglementMetrics(
            concurrence=self._concurrence(rho),
            negativity=self._negativity(rho),
            entanglement_entropy=self._entanglement_entropy(rho),
            schmidt_rank=self._schmidt_rank(rho),
            geometric_phase=self._geometric_phase(state.state_vector),
            resonance_strength=self._resonance_strength(state.state_vector)
        )
        
        self._metrics_history.append(metrics)
        return self._format_metrics(metrics)

    def _get_density_matrix(self, state: QuantumState) -> np.ndarray:
        """Obtém matriz densidade"""
        if hasattr(state, 'density_matrix'):
            return state.density_matrix
        psi = state.state_vector
        return np.outer(psi, np.conj(psi))

    def _concurrence(self, rho: np.ndarray) -> float:
        """Calcula concorrência para estados de 2 qubits"""
        if rho.shape != (4, 4):
            return 0.0
            
        sigma_y = np.array([[0, -1j], [1j, 0]])
        rho_tilde = np.kron(sigma_y, sigma_y) @ np.conj(rho) @ np.kron(sigma_y, sigma_y)
        R = rho @ rho_tilde
        eigenvals = np.sqrt(np.linalg.eigvalsh(R))
        eigenvals = np.sort(eigenvals)[::-1]
        return max(0.0, float(eigenvals[0] - eigenvals[1] - eigenvals[2] - eigenvals[3]))
        
    def _negativity(self, rho: np.ndarray) -> float:
        """Calcula negatividade"""
        dim = int(np.sqrt(rho.shape[0]))
        rho_pt = self._partial_transpose(rho, [0], [dim, dim])
        eigenvals = np.linalg.eigvalsh(rho_pt)
        return float(np.sum(np.abs(eigenvals) - eigenvals) / 2)
        
    def _entanglement_entropy(self, rho: np.ndarray) -> float:
        """Calcula entropia de emaranhamento"""
        eigenvals = np.linalg.eigvalsh(rho)
        eigenvals = eigenvals[eigenvals > 1e-10]
        return float(-np.sum(eigenvals * np.log2(eigenvals)))
    
    def _schmidt_rank(self, rho: np.ndarray) -> int:
        """Calcula posto de Schmidt"""
        eigenvals = np.linalg.eigvalsh(rho)
        return int(np.sum(eigenvals > 1e-10))
    
    def _geometric_phase(self, state: np.ndarray) -> float:
        """Calcula fase geométrica de Berry"""
        phase = np.angle(np.vdot(state, np.roll(state, 1)))
        return float(np.abs(phase))
    
    def _resonance_strength(self, state: np.ndarray) -> float:
        """Calcula força de ressonância quântica"""
        fourier = np.fft.fft(state)
        return float(np.max(np.abs(fourier)))
    
    def _partial_transpose(self, rho: np.ndarray, sys: List[int], dims: List[int]) -> np.ndarray:
        """Transposta parcial para cálculo de negatividade"""
        return np.reshape(np.transpose(
            np.reshape(rho, dims + dims), 
            tuple(range(len(dims))) + tuple(d + len(dims) for d in range(len(dims)))
        ), rho.shape)
    
    def _format_metrics(self, metrics: EntanglementMetrics) -> Dict[str, float]:
        """Formata métricas para saída"""
        return {
            "concurrence": metrics.concurrence,
            "negativity": metrics.negativity,
            "entanglement_entropy": metrics.entanglement_entropy,
            "schmidt_rank": metrics.schmidt_rank,
            "geometric_phase": metrics.geometric_phase,
            "resonance_strength": metrics.resonance_strength
        }
        
    def get_metrics(self) -> Dict[str, Any]:
        """Obtém resumo das métricas"""
        if not self._metrics_history:
            return {}
            
        metrics = self._metrics_history[-1]
        return self._format_metrics(metrics)
