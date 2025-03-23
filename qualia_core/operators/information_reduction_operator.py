"""
Quantum Information Reduction Operator Module
Implements controlled collapse and information reduction.
"""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
from state.quantum_state import QuantumState

@dataclass
class ReductionMetrics:
    """Container for reduction-related metrics"""
    collapse_probability: float
    information_loss: float
    reversibility: float
    collapse_threshold: float

class InformationReductionOperator:
    """
    Implements quantum information reduction operations.
    Controls collapse and information preservation.
    """
    
    def __init__(self, collapse_threshold: float = 1e-15):
        self.collapse_threshold = collapse_threshold
        self._metrics_history = []
        
    def should_collapse(self, state: QuantumState) -> bool:
        """
        Check if state should undergo collapse.
        
        Args:
            state: Input quantum state
            
        Returns:
            Boolean indicating if collapse should occur
        """
        if hasattr(state, 'density_matrix'):
            rho = state.density_matrix
            eigenvals = np.linalg.eigvalsh(rho)
            max_eigenval = np.max(np.abs(eigenvals))
        else:
            psi = state.state_vector
            probs = np.abs(psi)**2
            max_prob = np.max(probs)
            
        return max_prob > (1.0 - self.collapse_threshold)
        
    def apply(self, state: QuantumState) -> Dict[str, float]:
        """
        Apply information reduction to quantum state.
        
        Args:
            state: Input quantum state
            
        Returns:
            Dictionary with reduction metrics
        """
        # Get state vector
        psi = state.state_vector
        
        # Calculate initial probabilities
        initial_probs = np.abs(psi)**2
        
        # Find dominant basis state
        max_idx = np.argmax(initial_probs)
        
        # Calculate collapse probability
        collapse_prob = initial_probs[max_idx]
        
        # Calculate information metrics before collapse
        initial_entropy = self._von_neumann_entropy(psi)
        
        # Perform collapse
        new_state = np.zeros_like(psi)
        new_state[max_idx] = 1.0
        
        # Update state
        state.state_vector = new_state
        
        # Calculate final entropy
        final_entropy = self._von_neumann_entropy(new_state)
        
        # Calculate metrics
        metrics = ReductionMetrics(
            collapse_probability=float(collapse_prob),
            information_loss=float(initial_entropy - final_entropy),
            reversibility=float(1.0 - collapse_prob),
            collapse_threshold=self.collapse_threshold
        )
        
        # Store metrics
        self._metrics_history.append(metrics)
        if len(self._metrics_history) > 1000:
            self._metrics_history.pop(0)
            
        return {
            'reduction': collapse_prob,
            'information_loss': metrics.information_loss,
            'reversibility': metrics.reversibility
        }
        
    def _von_neumann_entropy(self, state: np.ndarray) -> float:
        """Calculate von Neumann entropy"""
        if state.ndim == 1:
            # Pure state
            return 0.0
        else:
            # Mixed state
            eigenvals = np.linalg.eigvalsh(state)
            eigenvals = eigenvals[eigenvals > 1e-15]
            return float(-np.sum(eigenvals * np.log2(eigenvals)))
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get operator metrics summary"""
        if not self._metrics_history:
            return {}
            
        recent_metrics = self._metrics_history[-10:]
        return {
            'mean_collapse_prob': np.mean([m.collapse_probability for m in recent_metrics]),
            'mean_info_loss': np.mean([m.information_loss for m in recent_metrics]),
            'mean_reversibility': np.mean([m.reversibility for m in recent_metrics]),
            'reduction_stability': np.std([m.collapse_probability for m in recent_metrics])
        }
