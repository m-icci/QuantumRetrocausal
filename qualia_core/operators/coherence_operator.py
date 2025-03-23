"""
Quantum Coherence Operator Module
Implements coherence measurement and preservation for quantum states.
"""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
from state.quantum_state import QuantumState

@dataclass
class CoherenceMetrics:
    """Container for coherence-related metrics"""
    l1_norm: float
    relative_entropy: float
    purity: float
    von_neumann_entropy: float

class CoherenceOperator:
    """
    Implements quantum coherence operations and measurements.
    Based on resource theory of quantum coherence.
    """
    
    def __init__(self, precision: float = 1e-15):
        self.precision = precision
        self._metrics_history = []
        
    def apply(self, state: QuantumState) -> Dict[str, float]:
        """
        Apply coherence measurements to quantum state.
        
        Args:
            state: Input quantum state
            
        Returns:
            Dictionary with coherence metrics
        """
        # Get density matrix
        if hasattr(state, 'density_matrix'):
            rho = state.density_matrix
        else:
            # Create density matrix from state vector
            psi = state.state_vector
            rho = np.outer(psi, np.conj(psi))
            
        # Calculate metrics
        metrics = CoherenceMetrics(
            l1_norm=self._l1_norm_coherence(rho),
            relative_entropy=self._relative_entropy_coherence(rho),
            purity=self._purity(rho),
            von_neumann_entropy=self._von_neumann_entropy(rho)
        )
        
        # Store metrics
        self._metrics_history.append(metrics)
        if len(self._metrics_history) > 1000:
            self._metrics_history.pop(0)
            
        return {
            'coherence': np.mean([metrics.l1_norm, metrics.relative_entropy]),
            'purity': metrics.purity,
            'entropy': metrics.von_neumann_entropy
        }
        
    def _l1_norm_coherence(self, rho: np.ndarray) -> float:
        """Calculate l1-norm of coherence"""
        diag_rho = np.diag(np.diag(rho))
        return np.sum(np.abs(rho - diag_rho))
        
    def _relative_entropy_coherence(self, rho: np.ndarray) -> float:
        """Calculate relative entropy of coherence"""
        diag_rho = np.diag(np.diag(rho))
        return self._von_neumann_entropy(diag_rho) - self._von_neumann_entropy(rho)
        
    def _purity(self, rho: np.ndarray) -> float:
        """Calculate state purity"""
        return float(np.real(np.trace(rho @ rho)))
        
    def _von_neumann_entropy(self, rho: np.ndarray) -> float:
        """Calculate von Neumann entropy"""
        eigenvals = np.linalg.eigvalsh(rho)
        eigenvals = eigenvals[eigenvals > self.precision]
        return float(-np.sum(eigenvals * np.log2(eigenvals)))
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get operator metrics summary"""
        if not self._metrics_history:
            return {}
            
        recent_metrics = self._metrics_history[-10:]
        return {
            'mean_coherence': np.mean([m.l1_norm for m in recent_metrics]),
            'mean_purity': np.mean([m.purity for m in recent_metrics]),
            'mean_entropy': np.mean([m.von_neumann_entropy for m in recent_metrics]),
            'coherence_stability': np.std([m.l1_norm for m in recent_metrics])
        }
