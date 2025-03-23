"""
QUALIA Quantum Merge - Utilitários
---------------------------------

Funções utilitárias para operações de merge quântico.
"""

import numpy as np
from typing import List, Dict, Any, Union
from ..state.quantum_state import QuantumState

def calculate_quantum_coherence(state: np.ndarray) -> float:
    """
    Calcula a coerência quântica de um estado

    Args:
        state: Estado quântico

    Returns:
        float: Valor de coerência
    """
    if state.size == 0:
        return 0.0
    
    # Normalizar estado
    norm = np.linalg.norm(state)
    if norm == 0:
        return 0.0
        
    normalized_state = state / norm
    
    # Calcular matriz densidade
    density_matrix = np.outer(normalized_state, normalized_state.conj())
    
    # Calcular coerência como traço da matriz densidade
    coherence = np.abs(np.trace(density_matrix))
    
    return float(coherence)

def calculate_phase_coherence(state1: np.ndarray, state2: np.ndarray) -> float:
    """
    Calcula a coerência de fase entre dois estados quânticos

    Args:
        state1: Primeiro estado
        state2: Segundo estado

    Returns:
        float: Coerência de fase
    """
    if state1.size == 0 or state2.size == 0:
        return 0.0
        
    # Normalizar estados
    norm1 = np.linalg.norm(state1)
    norm2 = np.linalg.norm(state2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    state1_norm = state1 / norm1
    state2_norm = state2 / norm2
    
    # Calcular produto interno
    overlap = np.abs(np.vdot(state1_norm, state2_norm))
    
    return float(overlap)

def calculate_entropy(state: np.ndarray) -> float:
    """
    Calcula a entropia de von Neumann de um estado quântico

    Args:
        state: Estado quântico

    Returns:
        float: Entropia do estado
    """
    if state.size == 0:
        return 0.0
        
    # Normalizar estado
    norm = np.linalg.norm(state)
    if norm == 0:
        return 0.0
        
    normalized_state = state / norm
    
    # Calcular matriz densidade
    density_matrix = np.outer(normalized_state, normalized_state.conj())
    
    # Calcular autovalores
    eigenvalues = np.linalg.eigvalsh(density_matrix)
    
    # Remover autovalores muito próximos de zero
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    
    # Calcular entropia
    entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
    
    return float(entropy)

def merge_potential(state1: np.ndarray, state2: np.ndarray) -> Dict[str, float]:
    """
    Calcula métricas de potencial de merge entre dois estados

    Args:
        state1: Primeiro estado
        state2: Segundo estado

    Returns:
        Dict com métricas de potencial de merge
    """
    coherence1 = calculate_quantum_coherence(state1)
    coherence2 = calculate_quantum_coherence(state2)
    phase_coherence = calculate_phase_coherence(state1, state2)
    entropy1 = calculate_entropy(state1)
    entropy2 = calculate_entropy(state2)
    
    return {
        'coherence1': coherence1,
        'coherence2': coherence2,
        'phase_coherence': phase_coherence,
        'entropy1': entropy1,
        'entropy2': entropy2,
        'merge_potential': phase_coherence * (1 - max(entropy1, entropy2))
    }
