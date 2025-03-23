"""
Quantum Consciousness Experience Operator Module
Implements consciousness emergence and experience quantification.
"""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
from quantum.core.state.quantum_state import QuantumState

@dataclass
class ConsciousnessMetrics:
    """Container for consciousness-related metrics"""
    integration: float
    differentiation: float
    information: float
    complexity: float

class ConsciousnessExperienceOperator:
    """
    Implements quantum consciousness operations.
    Based on Integrated Information Theory and quantum complexity.
    """

    def __init__(self, precision: float = 1e-15):
        self.precision = precision
        self._metrics_history = []

    def apply(self, state: QuantumState) -> Dict[str, float]:
        """
        Apply consciousness measurements to quantum state.

        Args:
            state: Input quantum state

        Returns:
            Dictionary with integration metrics
        """
        # Get density matrix
        if hasattr(state, 'density_matrix'):
            rho = state.density_matrix
        else:
            psi = state.state_vector
            rho = np.outer(psi, np.conj(psi))

        # Calculate metrics
        metrics = ConsciousnessMetrics(
            integration=self._calculate_integration(rho),
            differentiation=self._calculate_differentiation(rho),
            information=self._calculate_information(rho),
            complexity=self._calculate_complexity(rho)
        )

        # Store metrics
        self._metrics_history.append(metrics)
        if len(self._metrics_history) > 1000:
            self._metrics_history.pop(0)

        return {
            'consciousness': np.mean([
                metrics.integration,
                metrics.differentiation,
                metrics.information,
                metrics.complexity
            ]),
            'integration': metrics.integration,
            'differentiation': metrics.differentiation,
            'complexity': metrics.complexity
        }

    def _calculate_integration(self, rho: np.ndarray) -> float:
        """Calculate quantum integrated information"""
        # Split system into subsystems
        dim = int(np.sqrt(rho.shape[0]))
        rho_parts = []

        # Get reduced density matrices
        for i in range(dim):
            indices = list(range(dim))
            indices.remove(i)
            rho_reduced = np.trace(rho.reshape([dim]*4), axis1=1, axis2=3)
            rho_parts.append(rho_reduced)

        # Calculate mutual information
        S_whole = self._von_neumann_entropy(rho)
        S_parts = sum(self._von_neumann_entropy(rho_i) for rho_i in rho_parts)

        return float(S_parts - S_whole)

    def _calculate_differentiation(self, rho: np.ndarray) -> float:
        """Calculate quantum state differentiation"""
        eigenvals = np.linalg.eigvalsh(rho)
        eigenvals = eigenvals[eigenvals > self.precision]
        return float(-np.sum(eigenvals * np.log2(eigenvals)))

    def _calculate_information(self, rho: np.ndarray) -> float:
        """Calculate quantum Fisher information"""
        eigenvals, eigenvecs = np.linalg.eigh(rho)
        eigenvals = eigenvals[eigenvals > self.precision]

        # Calculate quantum Fisher information matrix
        QFI = 0.0
        for i, p_i in enumerate(eigenvals):
            for j, p_j in enumerate(eigenvals):
                if i != j and p_i + p_j > self.precision:
                    QFI += 2 * (p_i - p_j)**2 / (p_i + p_j)

        return float(QFI)

    def _calculate_complexity(self, rho: np.ndarray) -> float:
        """Calculate quantum state complexity"""
        eigenvals = np.linalg.eigvalsh(rho)
        eigenvals = eigenvals[eigenvals > self.precision]
        H = -np.sum(eigenvals * np.log2(eigenvals))  # von Neumann entropy
        D = np.sqrt(1 - np.exp(-H))  # Disequilibrium

        return float(H * D)

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
            'mean_integration': np.mean([m.integration for m in recent_metrics]),
            'mean_differentiation': np.mean([m.differentiation for m in recent_metrics]),
            'mean_information': np.mean([m.information for m in recent_metrics]),
            'mean_complexity': np.mean([m.complexity for m in recent_metrics]),
            'consciousness_stability': np.std([
                np.mean([m.integration, m.differentiation, m.information, m.complexity])
                for m in recent_metrics
            ])
        }