"""
Operador Quântico Base
Implementa funcionalidades básicas para operadores quânticos
"""

import numpy as np
from typing import Dict, Any, Optional
from operators.state.quantum_state import QuantumState

class BaseQuantumOperator:
    """Classe base para operadores quânticos"""
    
    def __init__(self, dimensions: int, name: str):
        self.dimensions = dimensions
        self.name = name
        self.matrix = np.eye(dimensions, dtype=np.complex128)
        
    def apply(self, state: QuantumState) -> QuantumState:
        """Aplica operador ao estado quântico"""
        if state.dimensions != self.dimensions:
            raise ValueError(f"Dimensões incompatíveis: {state.dimensions} != {self.dimensions}")
            
        new_state = np.dot(self.matrix, state.state_vector)
        return QuantumState(self.dimensions, new_state)
        
    def compose(self, other: 'BaseQuantumOperator') -> 'BaseQuantumOperator':
        """Compõe dois operadores"""
        if self.dimensions != other.dimensions:
            raise ValueError("Operadores com dimensões diferentes")
            
        result = BaseQuantumOperator(self.dimensions, name=f"{self.name}_{other.name}")
        result.matrix = np.dot(self.matrix, other.matrix)
        return result
        
    def is_unitary(self, tolerance: float = 1e-10) -> bool:
        """Verifica se operador é unitário"""
        identity = np.eye(self.dimensions)
        product = np.dot(self.matrix, self.matrix.conj().T)
        return np.allclose(product, identity, atol=tolerance)
        
    def is_hermitian(self, tolerance: float = 1e-10) -> bool:
        """Verifica se operador é hermitiano"""
        return np.allclose(self.matrix, self.matrix.conj().T, atol=tolerance)
