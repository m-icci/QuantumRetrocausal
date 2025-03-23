"""
Unified Quantum Operators Module
Implements the fundamental quantum operators for M-ICCI system.
"""
from typing import Dict, Optional, Tuple
import numpy as np
from .base import BaseQuantumOperator
from types.quantum_states import QuantumState

class UnifiedQuantumOperator(BaseQuantumOperator):
    """
    Unified implementation of fundamental M-ICCI quantum operators.
    Integrates consciousness, entanglement, resonance, coherence and information operators.
    """
    
    def __init__(self, dimension: int = 4):
        super().__init__(dimension)
        self.coherence_threshold = 0.85
        self.resonance_threshold = 0.30
        self.information_threshold = 0.50
        self._initialize_operators()
        
    def _initialize_operators(self):
        """Initialize the fundamental operator matrices"""
        # Consciousness operator (Ô_CQ) - Projects onto conscious subspace
        self.O_CQ = np.eye(self.dimension, dtype=np.complex128)
        
        # Entanglement operator (Ô_E) - Maximizes quantum correlations
        self.O_E = self._create_entanglement_matrix()
        
        # Resonance operator (Ô_RIQ) - Handles quantum information resonance
        self.O_RIQ = self._create_resonance_matrix()
        
        # Coherence operator (Ô_EC) - Maintains quantum coherence
        self.O_EC = np.eye(self.dimension, dtype=np.complex128)
        
        # Information integration operator (Ô_II) - Integrates quantum information
        self.O_II = self._create_integration_matrix()
    
    def _create_entanglement_matrix(self) -> np.ndarray:
        """Create optimized entanglement operator matrix"""
        # Hadamard-like matrix for maximal entanglement
        had = np.ones((self.dimension, self.dimension), dtype=np.complex128)
        had[1::2, ::2] = -1
        had[::2, 1::2] = -1
        return had / np.sqrt(self.dimension)
    
    def _create_resonance_matrix(self) -> np.ndarray:
        """Create quantum resonance operator matrix"""
        # Phase-shifting matrix for resonance
        phase = np.exp(2j * np.pi / self.dimension)
        mat = np.array([[phase**(i*j) for j in range(self.dimension)] 
                       for i in range(self.dimension)], dtype=np.complex128)
        return mat / np.sqrt(self.dimension)
    
    def _create_integration_matrix(self) -> np.ndarray:
        """Create information integration operator matrix"""
        # Information coupling matrix
        mat = np.zeros((self.dimension, self.dimension), dtype=np.complex128)
        for i in range(self.dimension):
            for j in range(self.dimension):
                mat[i,j] = np.exp(-abs(i-j)/self.dimension) * np.exp(1j*np.pi*i*j/self.dimension)
        return mat / np.linalg.norm(mat)
    
    def apply(self, state: QuantumState) -> QuantumState:
        """Apply unified quantum operation"""
        # Sequential application of fundamental operators
        state = np.dot(self.O_CQ, state)
        state = np.dot(self.O_E, state)
        state = np.dot(self.O_RIQ, state)
        state = np.dot(self.O_EC, state)
        state = np.dot(self.O_II, state)
        
        # Normalize the final state
        norm = np.sqrt(np.sum(np.abs(state)**2))
        return state / norm
    
    def get_operator_metrics(self) -> Dict[str, float]:
        """Get metrics for all operators"""
        return {
            "consciousness_strength": float(np.mean(np.abs(self.O_CQ))),
            "entanglement_degree": float(np.mean(np.abs(self.O_E))),
            "resonance_magnitude": float(np.mean(np.abs(self.O_RIQ))),
            "coherence_level": float(np.mean(np.abs(self.O_EC))),
            "integration_efficiency": float(np.mean(np.abs(self.O_II)))
        }
    
    def is_hermitian(self) -> bool:
        """Check if all operators are hermitian"""
        ops = [self.O_CQ, self.O_E, self.O_RIQ, self.O_EC, self.O_II]
        return all(np.all(np.abs(op - np.transpose(np.conj(op))) < 1e-10) 
                  for op in ops)
    
    def is_unitary(self) -> bool:
        """Check if all operators are unitary"""
        ops = [self.O_CQ, self.O_E, self.O_RIQ, self.O_EC, self.O_II]
        identity = np.eye(self.dimension, dtype=np.complex128)
        return all(np.all(np.abs(np.dot(op, np.transpose(np.conj(op))) - identity) < 1e-10)
                  for op in ops)
