"""
Implementação dos operadores quânticos fundamentais com validação numérica.
Baseado no YAA Core Quantum Module.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from operators.validation.numerical_validator import NumericalValidator, NumericalValidationConfig
from operators.cache.matrix_cache import MatrixCache

# Configuração de logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

@dataclass
class OperatorMetrics:
    """Métricas de operador quântico"""
    applications: int = 0
    validation_failures: int = 0
    cache_hits: int = 0
    avg_execution_time: float = 0.0

class QuantumOperator:
    """Classe base para operadores quânticos com validação"""
    
    def __init__(self, dimension: int, name: str):
        self.dimension = dimension
        self.name = name
        self.validator = NumericalValidator()
        self.cache = MatrixCache()
        self.metrics = OperatorMetrics()
        
    def _validate_input(self, state: np.ndarray) -> Tuple[bool, str]:
        """Valida estado de entrada"""
        if state.shape[-1] != self.dimension:
            return False, f"Dimensão incompatível: esperado {self.dimension}, recebido {state.shape[-1]}"
        return self.validator.validate_state_vector(state)
        
    def _validate_output(self, state: np.ndarray) -> Tuple[bool, str]:
        """Valida estado de saída"""
        return self.validator.validate_state_vector(state)
        
    def _log_operation(self, input_valid: bool, output_valid: bool, 
                      execution_time: float):
        """Registra métricas da operação"""
        self.metrics.applications += 1
        if not (input_valid and output_valid):
            self.metrics.validation_failures += 1
            
        # Atualiza tempo médio de execução
        n = self.metrics.applications
        self.metrics.avg_execution_time = (
            (self.metrics.avg_execution_time * (n-1) + execution_time) / n
        )
        
        logger.info(
            f"Operador {self.name}: "
            f"Aplicações={self.metrics.applications}, "
            f"Falhas={self.metrics.validation_failures}, "
            f"Tempo médio={self.metrics.avg_execution_time:.6f}s"
        )
        
    def apply(self, state: np.ndarray) -> np.ndarray:
        """Aplica o operador ao estado com validação"""
        import time
        start_time = time.time()
        
        # Valida entrada
        input_valid, input_msg = self._validate_input(state)
        if not input_valid:
            logger.error(f"Validação de entrada falhou: {input_msg}")
            raise ValueError(input_msg)
            
        # Tenta usar cache
        cache_key = f"{self.name}_{hash(state.tobytes())}"
        cached_result = self.cache.get_cached_operation(cache_key)
        
        if cached_result is not None:
            self.metrics.cache_hits += 1
            logger.debug(f"Cache hit para {self.name}")
            result = cached_result
        else:
            # Aplica operação
            try:
                result = self._apply_operation(state)
            except Exception as e:
                logger.error(f"Erro na aplicação do operador {self.name}: {e}")
                raise
                
            # Armazena no cache
            self.cache.cache_operation(state, self.name, result)
            
        # Valida saída
        output_valid, output_msg = self._validate_output(result)
        if not output_valid:
            logger.error(f"Validação de saída falhou: {output_msg}")
            raise ValueError(output_msg)
            
        execution_time = time.time() - start_time
        self._log_operation(input_valid, output_valid, execution_time)
        
        return result
        
    def _apply_operation(self, state: np.ndarray) -> np.ndarray:
        """Implementação específica do operador"""
        raise NotImplementedError

class PauliX(QuantumOperator):
    """Operador Pauli-X (NOT quântico)"""
    def __init__(self):
        super().__init__(dimension=2, name="PauliX")
        self.matrix = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    
    def _apply_operation(self, state: np.ndarray) -> np.ndarray:
        return np.dot(self.matrix, state)

class PauliY(QuantumOperator):
    """Operador Pauli-Y"""
    def __init__(self):
        super().__init__(dimension=2, name="PauliY")
        self.matrix = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    
    def _apply_operation(self, state: np.ndarray) -> np.ndarray:
        return np.dot(self.matrix, state)

class PauliZ(QuantumOperator):
    """Operador Pauli-Z (fase quântica)"""
    def __init__(self):
        super().__init__(dimension=2, name="PauliZ")
        self.matrix = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    
    def _apply_operation(self, state: np.ndarray) -> np.ndarray:
        return np.dot(self.matrix, state)

class Hadamard(QuantumOperator):
    """Operador Hadamard (superposição)"""
    def __init__(self):
        super().__init__(dimension=2, name="Hadamard")
        self.matrix = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    
    def _apply_operation(self, state: np.ndarray) -> np.ndarray:
        return np.dot(self.matrix, state)

class Phase(QuantumOperator):
    """Operador de fase"""
    def __init__(self, phi: float = np.pi/4):
        super().__init__(dimension=2, name="Phase")
        self.phi = phi
        self.matrix = np.array([[1, 0], [0, np.exp(1j*phi)]], dtype=np.complex128)
    
    def _apply_operation(self, state: np.ndarray) -> np.ndarray:
        return np.dot(self.matrix, state)

class CNOT(QuantumOperator):
    """Operador CNOT (NOT controlado)"""
    def __init__(self):
        super().__init__(dimension=4, name="CNOT")
        self.matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=np.complex128)
    
    def _apply_operation(self, state: np.ndarray) -> np.ndarray:
        return np.dot(self.matrix, state)

class Toffoli(QuantumOperator):
    """Operador Toffoli (CCNOT)"""
    def __init__(self):
        super().__init__(dimension=8, name="Toffoli")
        self.matrix = np.eye(8, dtype=np.complex128)
        self.matrix[6:8, 6:8] = np.array([[0, 1], [1, 0]])
    
    def _apply_operation(self, state: np.ndarray) -> np.ndarray:
        return np.dot(self.matrix, state)
