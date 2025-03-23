"""
Otimizador de Matrizes para Operações Quânticas
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class OptimizationParameters:
    """Parâmetros para otimização de matrizes"""
    learning_rate: float = 0.01
    max_iterations: int = 100
    convergence_threshold: float = 1e-6
    regularization: float = 0.001

class MatrixOptimizer:
    """
    Otimizador de matrizes para operações quânticas.
    Implementa técnicas de otimização adaptativa para matrizes complexas.
    """
    
    def __init__(self, parameters: Optional[OptimizationParameters] = None):
        """
        Inicializa otimizador
        
        Args:
            parameters: Parâmetros de otimização
        """
        self.params = parameters or OptimizationParameters()
        self.phi = (1 + np.sqrt(5)) / 2  # Razão áurea
        
    def optimize_matrix(self, matrix: np.ndarray, target_metric: str) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Otimiza matriz para maximizar métrica específica
        
        Args:
            matrix: Matriz a ser otimizada
            target_metric: Métrica alvo ('coherence', 'entanglement', etc)
            
        Returns:
            Tuple contendo matriz otimizada e métricas finais
        """
        current_matrix = matrix.copy()
        best_metric = self._calculate_metric(current_matrix, target_metric)
        best_matrix = current_matrix.copy()
        
        for _ in range(self.params.max_iterations):
            # Calcula gradiente
            gradient = self._calculate_gradient(current_matrix, target_metric)
            
            # Atualiza matriz
            current_matrix += self.params.learning_rate * gradient
            
            # Normaliza
            current_matrix = self._normalize_matrix(current_matrix)
            
            # Calcula nova métrica
            current_metric = self._calculate_metric(current_matrix, target_metric)
            
            # Atualiza melhor resultado
            if current_metric > best_metric:
                best_metric = current_metric
                best_matrix = current_matrix.copy()
                
            # Verifica convergência
            if np.abs(current_metric - best_metric) < self.params.convergence_threshold:
                break
                
        return best_matrix, {
            'final_metric': best_metric,
            'iterations': _,
            'convergence': np.abs(current_metric - best_metric)
        }
        
    def _calculate_metric(self, matrix: np.ndarray, metric: str) -> float:
        """Calcula métrica específica para matriz"""
        if metric == 'coherence':
            return np.abs(np.trace(matrix @ matrix.conj().T)) / matrix.shape[0]
        elif metric == 'entanglement':
            # Calcula entropia de von Neumann como medida de emaranhamento
            eigenvals = np.linalg.eigvalsh(matrix @ matrix.conj().T)
            eigenvals = eigenvals[eigenvals > 1e-10]  # Remove valores muito próximos de zero
            return -np.sum(eigenvals * np.log2(eigenvals))
        else:
            raise ValueError(f"Métrica não suportada: {metric}")
            
    def _calculate_gradient(self, matrix: np.ndarray, metric: str) -> np.ndarray:
        """Calcula gradiente para otimização"""
        eps = 1e-7
        gradient = np.zeros_like(matrix, dtype=np.complex128)
        
        # Calcula gradiente numérico
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                # Perturbação real
                matrix[i,j] += eps
                pos_real = self._calculate_metric(matrix, metric)
                matrix[i,j] -= 2*eps
                neg_real = self._calculate_metric(matrix, metric)
                matrix[i,j] += eps
                
                # Perturbação imaginária
                matrix[i,j] += 1j*eps
                pos_imag = self._calculate_metric(matrix, metric)
                matrix[i,j] -= 2j*eps
                neg_imag = self._calculate_metric(matrix, metric)
                matrix[i,j] += 1j*eps
                
                gradient[i,j] = ((pos_real - neg_real) + 1j*(pos_imag - neg_imag))/(2*eps)
                
        return gradient
        
    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Normaliza matriz mantendo propriedades quânticas"""
        # Garante hermiticidade
        matrix = 0.5 * (matrix + matrix.conj().T)
        
        # Normaliza pelo traço
        trace = np.trace(matrix @ matrix.conj().T)
        if trace > 0:
            matrix /= np.sqrt(trace)
            
        return matrix
