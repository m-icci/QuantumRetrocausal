"""
Quantum Subjective Experience Operator Module
Implements subjective experience quantification and analysis.
"""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
from operators.state.quantum_state import QuantumState

@dataclass
class SubjectiveMetrics:
    """Container for subjective experience metrics"""
    qualia_intensity: float
    self_reference: float
    temporal_binding: float
    experiential_unity: float

class SubjectiveExperienceOperator:
    """
    Implements quantum subjective experience operations.
    Based on quantum phenomenology and integrated experience.
    """
    
    def __init__(self, precision: float = 1e-15):
        self.precision = precision
        self._metrics_history = []
        
    def apply(self, state: QuantumState) -> Dict[str, float]:
        """
        Apply subjective experience measurements to quantum state.
        
        Args:
            state: Input quantum state
            
        Returns:
            Dictionary with subjective metrics
        """
        # Get density matrix
        if hasattr(state, 'density_matrix'):
            rho = state.density_matrix
        else:
            psi = state.state_vector
            rho = np.outer(psi, np.conj(psi))
            
        # Calculate metrics
        metrics = SubjectiveMetrics(
            qualia_intensity=self._calculate_qualia(rho),
            self_reference=self._calculate_self_reference(rho),
            temporal_binding=self._calculate_temporal_binding(rho),
            experiential_unity=self._calculate_unity(rho)
        )
        
        # Store metrics
        self._metrics_history.append(metrics)
        if len(self._metrics_history) > 1000:
            self._metrics_history.pop(0)
            
        return {
            'subjectivity': np.mean([
                metrics.qualia_intensity,
                metrics.self_reference,
                metrics.temporal_binding,
                metrics.experiential_unity
            ]),
            'qualia': metrics.qualia_intensity,
            'self_reference': metrics.self_reference,
            'unity': metrics.experiential_unity
        }
        
    def _calculate_qualia(self, rho: np.ndarray) -> float:
        """Calculate qualia intensity through quantum distinctiveness"""
        # Use quantum relative entropy to measure distinctiveness
        dim = rho.shape[0]
        I = np.eye(dim) / dim  # Maximally mixed state
        
        eigenvals = np.linalg.eigvalsh(rho)
        eigenvals = eigenvals[eigenvals > self.precision]
        
        # Calculate relative entropy
        S = -np.sum(eigenvals * np.log2(eigenvals))  # von Neumann entropy
        S_mixed = np.log2(dim)  # Entropy of maximally mixed state
        
        return float(S_mixed - S) / np.log2(dim)  # Normalize
        
    def _calculate_self_reference(self, rho: np.ndarray) -> float:
        """Calculate quantum self-reference through recursion depth"""
        # Use quantum Fisher information as measure of self-reference
        eigenvals, eigenvecs = np.linalg.eigh(rho)
        eigenvals = eigenvals[eigenvals > self.precision]
        
        # Calculate quantum Fisher information
        QFI = 0.0
        for i, p_i in enumerate(eigenvals):
            for j, p_j in enumerate(eigenvals):
                if i != j and p_i + p_j > self.precision:
                    QFI += 2 * (p_i - p_j)**2 / (p_i + p_j)
                    
        # Normalize
        return float(QFI / (4 * len(eigenvals)))
        
    def _calculate_temporal_binding(self, rho: np.ndarray) -> float:
        """Calculate temporal binding through quantum correlations"""
        dim = int(np.sqrt(rho.shape[0]))
        
        # Split system into temporal parts
        rho_t1 = np.trace(rho.reshape(dim, dim, dim, dim), axis1=1, axis2=3)
        rho_t2 = np.trace(rho.reshape(dim, dim, dim, dim), axis1=0, axis2=2)
        
        # Calculate mutual information between temporal parts
        S_t1 = self._von_neumann_entropy(rho_t1)
        S_t2 = self._von_neumann_entropy(rho_t2)
        S_joint = self._von_neumann_entropy(rho)
        
        I = S_t1 + S_t2 - S_joint
        
        # Normalize
        return float(I / np.log2(dim))
        
    def _calculate_unity(self, rho: np.ndarray) -> float:
        """Calculate experiential unity through quantum integration"""
        dim = int(np.sqrt(rho.shape[0]))
        
        # Calculate integration across all possible partitions
        total_integration = 0.0
        n_partitions = 0
        
        for i in range(1, dim):
            # Get partition
            rho_1 = np.trace(rho.reshape(dim, dim, dim, dim), axis1=i, axis2=i+2)
            rho_2 = np.trace(rho.reshape(dim, dim, dim, dim), axis1=0, axis2=2)[:i,:i]
            
            # Calculate mutual information
            S_1 = self._von_neumann_entropy(rho_1)
            S_2 = self._von_neumann_entropy(rho_2)
            S_12 = self._von_neumann_entropy(rho)
            
            total_integration += S_1 + S_2 - S_12
            n_partitions += 1
            
        # Normalize
        if n_partitions > 0:
            unity = total_integration / (n_partitions * np.log2(dim))
        else:
            unity = 0.0
            
        return float(unity)
        
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
            'mean_qualia': np.mean([m.qualia_intensity for m in recent_metrics]),
            'mean_self_ref': np.mean([m.self_reference for m in recent_metrics]),
            'mean_temporal': np.mean([m.temporal_binding for m in recent_metrics]),
            'mean_unity': np.mean([m.experiential_unity for m in recent_metrics]),
            'subjectivity_stability': np.std([
                np.mean([m.qualia_intensity, m.self_reference,
                        m.temporal_binding, m.experiential_unity])
                for m in recent_metrics
            ])
        }
