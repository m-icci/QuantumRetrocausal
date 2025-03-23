#!/usr/bin/env python3
"""
QuantumPatternRecognizer: Módulo responsável por reconhecer padrões emergentes no campo quântico.
Analisa e quantifica características como emaranhamento, coerência, complexidade quântica, etc.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
from scipy import linalg

logger = logging.getLogger("quantum_pattern_recognizer")

class QuantumPatternRecognizer:
    """
    Reconhecedor de padrões quânticos emergentes.
    Extrai e quantifica propriedades quânticas fundamentais a partir de campos complexos.
    """
    
    def __init__(self, dimensions: int, num_qubits: int, batch_size: int = 128):
        """
        Inicializa o reconhecedor de padrões quânticos.
        
        Args:
            dimensions: Dimensões do campo a ser analisado
            num_qubits: Número de qubits no sistema
            batch_size: Tamanho do lote para análise de histórico
        """
        self.dimensions = dimensions
        self.num_qubits = num_qubits
        self.batch_size = batch_size
        
        # Constantes internas
        self._pattern_history = []
        self._max_history = batch_size
        
        # Dimensão do espaço de Hilbert
        self.hilbert_dim = 2**num_qubits
        
        logger.info(f"QuantumPatternRecognizer inicializado com dimensões: {dimensions}, qubits: {num_qubits}")
    
    def recognize_patterns(self, field: np.ndarray) -> Dict[str, float]:
        """
        Reconhece padrões quânticos em um campo.
        
        Args:
            field: Campo complexo a ser analisado
            
        Returns:
            Dicionário com métricas quânticas
        """
        if field is None or field.size == 0:
            logger.warning("Campo vazio fornecido para reconhecimento de padrões")
            return {}
        
        # Converter o campo para densidade de probabilidade quântica
        try:
            density_matrix = self._field_to_density_matrix(field)
            
            # Calcular métricas quânticas
            metrics = {}
            
            # Coerência quântica (pureza do estado)
            metrics['coherence'] = self._calculate_coherence(density_matrix)
            
            # Decoerência (complemento da pureza)
            metrics['decoherence'] = 1.0 - metrics['coherence']
            
            # Entropia de von Neumann
            metrics['quantum_entropy'] = self._calculate_von_neumann_entropy(density_matrix)
            
            # Emaranhamento (aproximação via entropia de emaranhamento)
            metrics['entanglement'] = self._calculate_entanglement(density_matrix)
            
            # Complexidade quântica
            metrics['quantum_complexity'] = self._calculate_quantum_complexity(density_matrix)
            
            # Dimensionalidade efetiva
            metrics['effective_dimension'] = self._calculate_effective_dimension(density_matrix)
            
            # Armazenar padrão no histórico
            self._update_pattern_history(metrics)
            
            # Calcular métricas dinâmicas (se histórico disponível)
            if len(self._pattern_history) > 1:
                # Taxa de mudança de coerência
                metrics['coherence_rate'] = self._calculate_rate_of_change('coherence')
                
                # Taxa de mudança de emaranhamento
                metrics['entanglement_rate'] = self._calculate_rate_of_change('entanglement')
                
                # Taxa de mudança de complexidade
                metrics['complexity_rate'] = self._calculate_rate_of_change('quantum_complexity')
            
            logger.debug(f"Padrões quânticos reconhecidos: {len(metrics)} métricas")
            return metrics
            
        except Exception as e:
            logger.error(f"Erro ao reconhecer padrões: {str(e)}")
            return {}
    
    def reset(self) -> None:
        """Reseta o histórico de padrões"""
        self._pattern_history = []
        logger.info("Histórico de padrões resetado")
    
    def _field_to_density_matrix(self, field: np.ndarray) -> np.ndarray:
        """
        Converte um campo complexo para uma matriz densidade quântica.
        
        Args:
            field: Campo complexo
            
        Returns:
            Matriz densidade normalizada
        """
        # Achatar o campo para um vetor
        flattened = field.flatten()
        
        # Limitar tamanho se necessário
        if len(flattened) > self.hilbert_dim**2:
            # Selecionar os elementos mais significativos (maiores em magnitude)
            magnitudes = np.abs(flattened)
            indices = np.argsort(magnitudes)[-self.hilbert_dim**2:]
            flattened = flattened[indices]
        
        # Preencher se necessário
        if len(flattened) < self.hilbert_dim**2:
            padding = np.zeros(self.hilbert_dim**2 - len(flattened), dtype=flattened.dtype)
            flattened = np.concatenate([flattened, padding])
        
        # Remodelar para matriz quadrada
        matrix = flattened.reshape(self.hilbert_dim, self.hilbert_dim)
        
        # Garantir que é hermitiana (requisito para matriz densidade)
        hermitian = (matrix + matrix.conj().T) / 2
        
        # Normalizar (traço = 1)
        trace = np.trace(hermitian)
        if abs(trace) > 1e-10:  # Evitar divisão por zero
            normalized = hermitian / trace
        else:
            # Se traço for muito pequeno, criar uma matriz densidade padrão
            normalized = np.eye(self.hilbert_dim) / self.hilbert_dim
        
        return normalized
    
    def _calculate_coherence(self, density_matrix: np.ndarray) -> float:
        """
        Calcula a coerência quântica como pureza do estado.
        
        Args:
            density_matrix: Matriz densidade
            
        Returns:
            Valor da coerência (entre 0 e 1)
        """
        # Pureza = Tr(ρ²)
        try:
            coherence = np.real(np.trace(density_matrix @ density_matrix))
            
            # Normalizar para [0,1]
            coherence = min(1.0, max(0.0, coherence))
            return float(coherence)
        except:
            return 0.5  # Valor padrão em caso de erro
    
    def _calculate_von_neumann_entropy(self, density_matrix: np.ndarray) -> float:
        """
        Calcula a entropia de von Neumann.
        
        Args:
            density_matrix: Matriz densidade
            
        Returns:
            Entropia de von Neumann
        """
        try:
            # Calcular autovalores da matriz densidade
            eigenvalues = np.real(linalg.eigvals(density_matrix))
            
            # Filtrar autovalores pequenos para evitar problemas numéricos
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            
            # Entropia de von Neumann: -Tr(ρ log ρ) = -Σ λᵢ log λᵢ
            entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
            
            # Normalizar para [0,1] usando dimensão do espaço
            max_entropy = np.log2(self.hilbert_dim)
            if max_entropy > 0:
                normalized_entropy = entropy / max_entropy
            else:
                normalized_entropy = 0
                
            return min(1.0, max(0.0, float(normalized_entropy)))
        except:
            return 0.5  # Valor padrão em caso de erro
    
    def _calculate_entanglement(self, density_matrix: np.ndarray) -> float:
        """
        Calcula uma métrica de emaranhamento.
        
        Args:
            density_matrix: Matriz densidade
            
        Returns:
            Valor do emaranhamento (entre 0 e 1)
        """
        try:
            # Particionar o sistema em dois subsistemas (divisão mais igual possível)
            n_subsystem_a = self.num_qubits // 2
            dim_a = 2**n_subsystem_a
            dim_b = self.hilbert_dim // dim_a
            
            # Remodelar matriz para facilitar o traço parcial
            if self.hilbert_dim == density_matrix.shape[0]:
                rho_reshaped = density_matrix.reshape(dim_a, dim_b, dim_a, dim_b)
                
                # Calcular traço parcial sobre o subsistema B
                rho_a = np.zeros((dim_a, dim_a), dtype=complex)
                for i in range(dim_b):
                    rho_a += rho_reshaped[:, i, :, i]
                
                # Calcular entropia de von Neumann do subsistema A
                eigenvalues_a = np.real(linalg.eigvals(rho_a))
                eigenvalues_a = eigenvalues_a[eigenvalues_a > 1e-10]
                entropy_a = -np.sum(eigenvalues_a * np.log2(eigenvalues_a + 1e-10))
                
                # Normalizar para [0,1]
                max_entropy_a = np.log2(dim_a)
                if max_entropy_a > 0:
                    entanglement = entropy_a / max_entropy_a
                else:
                    entanglement = 0
                    
                return min(1.0, max(0.0, float(entanglement)))
            else:
                # Fallback se as dimensões não corresponderem
                return self._calculate_von_neumann_entropy(density_matrix)
        except:
            # Fallback em caso de erro
            coherence = self._calculate_coherence(density_matrix)
            return 1.0 - coherence
    
    def _calculate_quantum_complexity(self, density_matrix: np.ndarray) -> float:
        """
        Calcula a complexidade quântica do estado.
        
        Args:
            density_matrix: Matriz densidade
            
        Returns:
            Valor da complexidade (entre 0 e 1)
        """
        try:
            # Calcular autovalores e autovetores
            eigenvalues, eigenvectors = linalg.eigh(density_matrix)
            
            # Ordenar em ordem decrescente
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Filtrar valores pequenos
            eigenvalues = np.real(eigenvalues[eigenvalues > 1e-10])
            
            # Complexidade baseada na distribuição dos autovalores
            # Usa entropia da distribuição e desvio da uniformidade
            if len(eigenvalues) > 1:
                # Normalizar os autovalores
                eigenvalues_norm = eigenvalues / np.sum(eigenvalues)
                
                # Entropia da distribuição
                entropy = -np.sum(eigenvalues_norm * np.log2(eigenvalues_norm + 1e-10))
                
                # Desvio da distribuição uniforme
                uniform_dist = np.ones_like(eigenvalues_norm) / len(eigenvalues_norm)
                deviation = np.sum(np.abs(eigenvalues_norm - uniform_dist)) / 2
                
                # Combinar métricas para complexidade total
                # Estados de máxima complexidade têm alta entropia mas não são completamente uniformes
                complexity = entropy * (1 - deviation) / np.log2(len(eigenvalues))
                
                return min(1.0, max(0.0, float(complexity)))
            else:
                return 0.0  # Estado puro tem complexidade zero
        except:
            # Valor padrão baseado na coerência como fallback
            coherence = self._calculate_coherence(density_matrix)
            return 1.0 - coherence**2
    
    def _calculate_effective_dimension(self, density_matrix: np.ndarray) -> float:
        """
        Calcula a dimensão efetiva do estado quântico.
        
        Args:
            density_matrix: Matriz densidade
            
        Returns:
            Dimensão efetiva normalizada
        """
        try:
            # Calcular autovalores da matriz densidade
            eigenvalues = np.real(linalg.eigvals(density_matrix))
            
            # Filtrar valores pequenos
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            
            if len(eigenvalues) > 0:
                # Normalizar
                eigenvalues = eigenvalues / np.sum(eigenvalues)
                
                # Dimensão efetiva: 1/Σλᵢ²
                effective_dimension = 1.0 / np.sum(eigenvalues**2)
                
                # Normalizar para [0,1]
                normalized_dimension = effective_dimension / self.hilbert_dim
                
                return min(1.0, max(0.0, float(normalized_dimension)))
            else:
                return 0.0
        except:
            return 0.5  # Valor padrão em caso de erro
    
    def _update_pattern_history(self, metrics: Dict[str, float]) -> None:
        """
        Atualiza o histórico de padrões.
        
        Args:
            metrics: Métricas quânticas atuais
        """
        # Adicionar ao histórico
        self._pattern_history.append(metrics)
        
        # Limitar tamanho do histórico
        if len(self._pattern_history) > self._max_history:
            self._pattern_history.pop(0)
    
    def _calculate_rate_of_change(self, metric_name: str) -> float:
        """
        Calcula a taxa de mudança de uma métrica ao longo do tempo.
        
        Args:
            metric_name: Nome da métrica
            
        Returns:
            Taxa de mudança normalizada
        """
        if len(self._pattern_history) < 2 or metric_name not in self._pattern_history[-1]:
            return 0.0
        
        try:
            # Obter valores recentes
            current = self._pattern_history[-1].get(metric_name, 0)
            previous = self._pattern_history[-2].get(metric_name, 0)
            
            # Calcular taxa de mudança
            rate = current - previous
            
            # Normalizar para [-1,1]
            max_change = 0.1  # Assume mudança máxima de 10% por iteração
            normalized_rate = rate / max_change
            
            return max(-1.0, min(1.0, normalized_rate))
        except:
            return 0.0 