"""
Operadores Quânticos para o Sistema de Trading Holográfico

Este módulo define operadores fundamentais para manipulação de campos quânticos:
- F: Folding (dobramento)
- M: Morphic Resonance (ressonância morfológica)
- E: Emergence (emergência)
- DDD: Decoherence (decoerência)
- ZZZ: Retrocausalidade
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

def apply_folding(field: np.ndarray, damping_factor: float = 0.9) -> np.ndarray:
    """
    Aplica dobramento com amortecimento adequado, evitando erros numéricos.
    
    Args:
        field (np.ndarray): Campo quântico de entrada
        damping_factor (float): Fator de amortecimento
    
    Returns:
        np.ndarray: Campo após dobramento
    """
    field *= damping_factor
    field[np.abs(field) < 1e-10] = 0  # Evita valores subnormais
    return np.clip(field, -1, 1)  # Limita amplitude

def apply_resonance(field: np.ndarray, resonance_factor: float = 1.05) -> np.ndarray:
    """
    Aplica ressonância considerando estabilidade numérica.
    
    Args:
        field (np.ndarray): Campo quântico de entrada
        resonance_factor (float): Fator de ressonância
    
    Returns:
        np.ndarray: Campo após ressonância
    """
    field *= resonance_factor
    return np.clip(field, -1, 1)  # Evita saturação extrema

def apply_emergence(field: np.ndarray, threshold_factor: float = 1.1) -> np.ndarray:
    """
    Destaca padrões emergentes sem desestabilizar o campo.
    
    Args:
        field (np.ndarray): Campo quântico de entrada
        threshold_factor (float): Fator de amplificação
    
    Returns:
        np.ndarray: Campo após emergência
    """
    threshold = np.mean(field) + np.std(field)
    field[field > threshold] *= threshold_factor
    return np.clip(field, -1, 1)  # Garante estabilidade

def apply_decoherence(field: np.ndarray, gamma: float = 0.02) -> np.ndarray:
    """
    Aplica decoerência com modelo de Lindblad simplificado.
    
    Args:
        field (np.ndarray): Campo quântico de entrada
        gamma (float): Taxa de decoerência
    
    Returns:
        np.ndarray: Campo quântico após decoerência
    """
    try:
        noise = np.random.normal(0, gamma, field.shape)
        field = field * (1 - gamma) + noise
        return np.clip(field, 0, 1)  # Normaliza
    except Exception as e:
        logger.error(f"Erro no apply_decoherence: {e}")
        return field

def apply_retrocausality(field: np.ndarray, past_field: np.ndarray, beta: float = 0.1) -> np.ndarray:
    """
    Incorpora padrões do passado ao presente com ajuste de entropia.
    
    Args:
        field (np.ndarray): Campo quântico atual
        past_field (np.ndarray): Campo quântico do passo anterior
        beta (float): Fator de retroalimentação
    
    Returns:
        np.ndarray: Campo quântico após retrocausalidade
    """
    try:
        # Calcula entropia para ajuste
        def calculate_entropy(matrix: np.ndarray) -> float:
            eigenvals = np.linalg.eigvalsh(matrix)
            eigenvals = eigenvals[eigenvals > 1e-10]
            return -np.sum(eigenvals * np.log2(eigenvals)) if len(eigenvals) > 0 else 0
        
        entropy_ratio = calculate_entropy(field) / (calculate_entropy(past_field) + 1e-10)
        field += beta * (past_field - field) * entropy_ratio
        return np.clip(field, 0, 1)
    except Exception as e:
        logger.error(f"Erro no apply_retrocausality: {e}")
        return field

def calculate_coherence(field: np.ndarray) -> float:
    """
    Calcula coerência baseada na matriz densidade reduzida.
    
    Args:
        field (np.ndarray): Campo quântico
    
    Returns:
        float: Valor de coerência
    """
    try:
        diag_elements = np.diag(field)
        coherence = np.sum(diag_elements**2) / (np.sum(field**2) + 1e-10)
        return coherence
    except Exception as e:
        logger.error(f"Erro no calculate_coherence: {e}")
        return 0.0

def calculate_entropy(field: np.ndarray) -> float:
    """
    Entropia de von Neumann adaptada ao campo quântico.
    
    Args:
        field (np.ndarray): Campo quântico
    
    Returns:
        float: Valor de entropia
    """
    try:
        eigenvals = np.linalg.eigvalsh(field)
        eigenvals = eigenvals[eigenvals > 1e-10]  # Evita log(0)
        return -np.sum(eigenvals * np.log2(eigenvals)) if len(eigenvals) > 0 else 0.0
    except Exception as e:
        logger.error(f"Erro no calculate_entropy: {e}")
        return 0.0
