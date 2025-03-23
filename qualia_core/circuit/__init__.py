"""
Core quantum circuit implementations.
"""

from typing import List, Any, Optional, Dict, Tuple
import numpy as np

class QuantumGate:
    """Base class for quantum gates."""
    
    def __init__(self, name: str, num_qubits: int):
        """Initialize quantum gate.
        
        Args:
            name: Name of the gate
            num_qubits: Number of qubits the gate operates on
        """
        self.name = name
        self.num_qubits = num_qubits
        self._matrix = None
        
    @property
    def matrix(self) -> np.ndarray:
        """Get gate matrix representation."""
        if self._matrix is None:
            raise NotImplementedError("Gate matrix not implemented")
        return self._matrix
    
    def apply(self, state: np.ndarray, targets: List[int]) -> np.ndarray:
        """Apply gate to quantum state.
        
        Args:
            state: Quantum state vector
            targets: Target qubit indices
            
        Returns:
            Modified state vector
        """
        raise NotImplementedError("Gate application not implemented")
        
    def validate_targets(self, targets: List[int], total_qubits: int):
        """Validate target qubit indices."""
        if len(targets) != self.num_qubits:
            raise ValueError(f"Gate {self.name} requires {self.num_qubits} qubits, got {len(targets)}")
        if any(t >= total_qubits or t < 0 for t in targets):
            raise ValueError(f"Invalid target indices {targets} for {total_qubits} qubits")

class QuantumCircuit:
    """Base quantum circuit implementation."""
    
    def __init__(self, num_qubits: int):
        """Initialize quantum circuit.
        
        Args:
            num_qubits: Number of qubits in circuit
        """
        self.num_qubits = num_qubits
        self.gates: List[Tuple[QuantumGate, List[int]]] = []
        
    def add_gate(self, gate: QuantumGate, targets: List[int]):
        """Add gate to circuit.
        
        Args:
            gate: Quantum gate to add
            targets: Target qubit indices
        """
        gate.validate_targets(targets, self.num_qubits)
        self.gates.append((gate, targets))
        
    def execute(self, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        """Execute circuit on initial state.
        
        Args:
            initial_state: Initial quantum state (default: |0...0>)
            
        Returns:
            Final quantum state
        """
        if initial_state is None:
            state = np.zeros(2**self.num_qubits)
            state[0] = 1.0
        else:
            state = initial_state.copy()
            
        for gate, targets in self.gates:
            state = gate.apply(state, targets)
            
        return state
        
    def __str__(self) -> str:
        """String representation of circuit."""
        return f"QuantumCircuit({self.num_qubits} qubits, {len(self.gates)} gates)"

# Export classes
__all__ = ['QuantumGate', 'QuantumCircuit']
