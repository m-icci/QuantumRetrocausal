"""
Quantum Information Integration Operator Module
Implements information integration and processing.
"""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
from state.quantum_state import QuantumState

@dataclass
class IntegrationMetrics:
    """Container for integration-related metrics"""
    mutual_information: float
    correlation: float
    coherent_information: float
    holistic_integration: float

class InformationIntegrationOperator:
    """
    Implements quantum information integration operations.
    Based on quantum information theory and holistic processing.
    """
    
    def __init__(self, precision: float = 1e-15):
        self.precision = precision
        self._metrics_history = []
        
    def apply(self, state: QuantumState) -> Dict[str, float]:
        """
        Apply information integration measurements to quantum state.
        
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
        metrics = IntegrationMetrics(
            mutual_information=self._mutual_information(rho),
            correlation=self._quantum_correlation(rho),
            coherent_information=self._coherent_information(rho),
            holistic_integration=self._holistic_integration(rho)
        )
        
        # Store metrics
        self._metrics_history.append(metrics)
        if len(self._metrics_history) > 1000:
            self._metrics_history.pop(0)
            
        return {
            'integration': np.mean([
                metrics.mutual_information,
                metrics.correlation,
                metrics.coherent_information,
                metrics.holistic_integration
            ]),
            'mutual_information': metrics.mutual_information,
            'correlation': metrics.correlation,
            'holistic': metrics.holistic_integration
        }
        
    def _mutual_information(self, rho: np.ndarray) -> float:
        """Calculate quantum mutual information"""
        dim = int(np.sqrt(rho.shape[0]))
        
        # Calculate reduced density matrices
        rho_a = np.trace(rho.reshape(dim, dim, dim, dim), axis1=1, axis2=3)
        rho_b = np.trace(rho.reshape(dim, dim, dim, dim), axis1=0, axis2=2)
        
        # Calculate entropies
        S_ab = self._von_neumann_entropy(rho)
        S_a = self._von_neumann_entropy(rho_a)
        S_b = self._von_neumann_entropy(rho_b)
        
        return float(S_a + S_b - S_ab)
        
    def _quantum_correlation(self, rho: np.ndarray) -> float:
        """Calculate quantum correlation (quantum discord)"""
        dim = int(np.sqrt(rho.shape[0]))
        
        # Calculate reduced density matrices
        rho_a = np.trace(rho.reshape(dim, dim, dim, dim), axis1=1, axis2=3)
        rho_b = np.trace(rho.reshape(dim, dim, dim, dim), axis1=0, axis2=2)
        
        # Calculate mutual information
        I = self._mutual_information(rho)
        
        # Calculate classical correlation
        C = self._classical_correlation(rho, rho_a, rho_b)
        
        return float(I - C)
        
    def _coherent_information(self, rho: np.ndarray) -> float:
        """Calculate quantum coherent information"""
        dim = int(np.sqrt(rho.shape[0]))
        
        # Calculate reduced density matrix
        rho_b = np.trace(rho.reshape(dim, dim, dim, dim), axis1=0, axis2=2)
        
        # Calculate entropies
        S = self._von_neumann_entropy(rho)
        S_b = self._von_neumann_entropy(rho_b)
        
        return float(S_b - S)
        
    def _holistic_integration(self, rho: np.ndarray) -> float:
        """Calculate holistic integration measure"""
        dim = int(np.sqrt(rho.shape[0]))
        
        # Calculate all possible bipartitions
        total_integration = 0.0
        n_partitions = 0
        
        for i in range(1, dim):
            # Get partition
            rho_1 = np.trace(rho.reshape(dim, dim, dim, dim), axis1=i, axis2=i+2)
            rho_2 = np.trace(rho.reshape(dim, dim, dim, dim), axis1=0, axis2=2)[:i,:i]
            
            # Calculate mutual information for partition
            S_1 = self._von_neumann_entropy(rho_1)
            S_2 = self._von_neumann_entropy(rho_2)
            S_12 = self._von_neumann_entropy(rho)
            
            total_integration += S_1 + S_2 - S_12
            n_partitions += 1
            
        return float(total_integration / n_partitions if n_partitions > 0 else 0.0)
        
    def _classical_correlation(self, rho: np.ndarray, rho_a: np.ndarray, rho_b: np.ndarray) -> float:
        """Calculate classical correlation"""
        # This is a simplified version - full calculation requires optimization
        S_b = self._von_neumann_entropy(rho_b)
        S_ab = self._von_neumann_entropy(rho)
        
        return float(S_b - S_ab)
        
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
            'mean_mutual_info': np.mean([m.mutual_information for m in recent_metrics]),
            'mean_correlation': np.mean([m.correlation for m in recent_metrics]),
            'mean_coherent_info': np.mean([m.coherent_information for m in recent_metrics]),
            'mean_holistic': np.mean([m.holistic_integration for m in recent_metrics]),
            'integration_stability': np.std([
                np.mean([m.mutual_information, m.correlation, 
                        m.coherent_information, m.holistic_integration])
                for m in recent_metrics
            ])
        }
