"""
Core quantum measurement functionality.
"""

from typing import Dict, Any, Optional
import numpy as np

class QuantumMeasurement:
    """
    Quantum measurement system supporting projective measurements,
    POVM measurements, and expectation value calculations.
    """
    
    def __init__(self, dimension: int):
        """Initialize measurement system.
        
        Args:
            dimension: Hilbert space dimension
        """
        self.dimension = dimension
        self._validate_dimension()
        
    def _validate_dimension(self):
        """Validate measurement dimension."""
        if not isinstance(self.dimension, int) or self.dimension < 1:
            raise ValueError(f"Invalid dimension {self.dimension}")
            
    def projective_measurement(self, state: np.ndarray, basis: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Perform projective measurement in given basis.
        
        Args:
            state: Quantum state vector
            basis: Measurement basis (default: computational basis)
            
        Returns:
            Dictionary containing measurement outcome and probability
        """
        if basis is None:
            # Use computational basis
            basis = np.eye(self.dimension)
            
        # Validate input dimensions
        if state.shape[0] != self.dimension:
            raise ValueError(f"State dimension {state.shape[0]} != system dimension {self.dimension}")
        if basis.shape != (self.dimension, self.dimension):
            raise ValueError(f"Invalid basis dimensions {basis.shape}")
            
        # Calculate measurement probabilities
        probs = np.abs(np.dot(basis.conj().T, state))**2
        
        # Sample outcome based on probabilities
        outcome = np.random.choice(self.dimension, p=probs)
        
        return {
            "outcome": outcome,
            "probability": probs[outcome],
            "state": basis[:, outcome] # Post-measurement state
        }
        
    def povm_measurement(self, state: np.ndarray, operators: np.ndarray) -> Dict[str, Any]:
        """Perform POVM measurement.
        
        Args:
            state: Quantum state vector
            operators: POVM measurement operators
            
        Returns:
            Dictionary containing measurement outcome and probabilities
        """
        # Validate operators form a valid POVM
        if not self._validate_povm(operators):
            raise ValueError("Invalid POVM operators")
            
        # Calculate outcome probabilities
        probs = np.array([
            np.real(np.dot(np.dot(state.conj(), op), state))
            for op in operators
        ])
        
        # Sample outcome
        outcome = np.random.choice(len(operators), p=probs)
        
        return {
            "outcome": outcome,
            "probabilities": probs
        }
        
    def expectation_value(self, state: np.ndarray, operator: np.ndarray) -> float:
        """Calculate expectation value of an operator.
        
        Args:
            state: Quantum state vector
            operator: Quantum operator/observable
            
        Returns:
            Expected value <ψ|A|ψ>
        """
        return np.real(np.dot(np.dot(state.conj(), operator), state))
        
    def _validate_povm(self, operators: np.ndarray) -> bool:
        """Validate that operators form a valid POVM set.
        
        Args:
            operators: Set of POVM operators
            
        Returns:
            True if operators form valid POVM
        """
        # Sum of operators should be identity
        op_sum = sum(op for op in operators)
        return np.allclose(op_sum, np.eye(self.dimension))
