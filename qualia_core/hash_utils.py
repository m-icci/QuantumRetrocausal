"""
Utilitários de Hash QUALIA - Módulo Consolidado
-------------------------------------
Implementa funções otimizadas para hashing e verificação de dificuldade.
"""

import hashlib
import math
import logging
import struct
import time
from typing import Tuple, Optional, Dict, Any, Union, List

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_hash(header: str, nonce: Union[int, bytes]) -> str:
    """
    Calcula o hash SHA-256 de um cabeçalho com nonce
    
    Args:
        header: Cabeçalho da transação/bloco
        nonce: Valor do nonce (int ou bytes)
        
    Returns:
        Hash hexadecimal
    """
    # Converter nonce para bytes se for inteiro
    if isinstance(nonce, int):
        nonce_bytes = nonce.to_bytes(8, byteorder='little')
    else:
        nonce_bytes = nonce
    
    # Converter header para bytes se for string
    if isinstance(header, str):
        header_bytes = header.encode()
    else:
        header_bytes = header
    
    # Concatenar e calcular hash
    data = header_bytes + nonce_bytes
    hash_obj = hashlib.sha256(data)
    return hash_obj.hexdigest()

def calculate_double_hash(header: str, nonce: Union[int, bytes]) -> str:
    """
    Calcula o hash duplo SHA-256 (usado em Bitcoin)
    
    Args:
        header: Cabeçalho da transação/bloco
        nonce: Valor do nonce (int ou bytes)
        
    Returns:
        Hash hexadecimal duplo
    """
    # Calcular primeiro hash
    first_hash = calculate_hash(header, nonce)
    # Calcular segundo hash
    hash_obj = hashlib.sha256(bytes.fromhex(first_hash))
    return hash_obj.hexdigest()

def calculate_target_hex(difficulty_trilhoes: float) -> str:
    """
    Calcula o target hexadecimal baseado na dificuldade em trilhões
    
    Args:
        difficulty_trilhoes: Dificuldade em trilhões
        
    Returns:
        String hexadecimal representando o target
    """
    difficulty_bits = int(math.log2(difficulty_trilhoes * 10**12))
    adjusted_difficulty = max(1, 256 - difficulty_bits)
    # Criar string de zeros para a parte mais significativa
    target_hex = '0' * (adjusted_difficulty // 4)
    # Preencher o resto com 'f'
    return target_hex.ljust(64, 'f')

def check_hash_difficulty(hash_hex: str, target_hex: str) -> bool:
    """
    Verifica se o hash atende à dificuldade especificada
    
    Args:
        hash_hex: Hash em formato hexadecimal
        target_hex: Target em formato hexadecimal
        
    Returns:
        True se o hash é menor que o target (atende à dificuldade)
    """
    return int(hash_hex, 16) < int(target_hex, 16)

def calculate_hash_rate(hashes: int, elapsed_time: float) -> float:
    """
    Calcula taxa de hash em hashes por segundo
    
    Args:
        hashes: Número de hashes calculados
        elapsed_time: Tempo decorrido em segundos
        
    Returns:
        Taxa de hash (hashes/segundo)
    """
    if elapsed_time > 0:
        return hashes / elapsed_time
    return 0.0

def verify_hash_with_difficulty(nonce: Union[int, bytes], 
                                header: str, 
                                difficulty_trilhoes: float) -> Tuple[bool, int]:
    """
    Verifica se um nonce é válido para a dificuldade especificada
    
    Args:
        nonce: Nonce a verificar
        header: Cabeçalho do bloco
        difficulty_trilhoes: Dificuldade em trilhões
        
    Returns:
        Tupla (válido, valor_inteiro_do_hash)
    """
    # Calcular hash
    hash_hex = calculate_hash(header, nonce)
    # Calcular target hex
    target_hex = calculate_target_hex(difficulty_trilhoes)
    
    # Converter valores para inteiros
    hash_int = int(hash_hex, 16)
    target_int = int(target_hex, 16)
    
    # Verificar se hash atende ao target
    is_valid = hash_int < target_int
    
    # Retornar resultado e valor numérico do hash para análise
    return is_valid, hash_int

def format_hash_rate(hash_rate: float) -> str:
    """
    Formata taxa de hash em unidades legíveis
    
    Args:
        hash_rate: Taxa de hash em hashes/segundo
        
    Returns:
        String formatada (ex: "1.23 KH/s")
    """
    units = ['H/s', 'KH/s', 'MH/s', 'GH/s', 'TH/s', 'PH/s']
    unit_index = 0
    
    while hash_rate >= 1000 and unit_index < len(units) - 1:
        hash_rate /= 1000
        unit_index += 1
    
    return f"{hash_rate:.2f} {units[unit_index]}"

def compute_proximity(hash_int: int, target_int: int) -> float:
    """
    Calcula a proximidade de um hash ao target
    
    Args:
        hash_int: Valor inteiro do hash
        target_int: Valor inteiro do target
        
    Returns:
        Valor de proximidade entre 0 e 1 (1 = hash exatamente igual ao target)
    """
    if hash_int >= target_int:
        # Hash não atende à dificuldade, calcular proximidade relativa
        max_value = 2**256 - 1
        distance = hash_int - target_int
        max_distance = max_value - target_int
        
        # Normalizar para 0-1, valores próximos ao target têm proximidade maior
        proximity = 1.0 - (distance / max_distance)
        # Ajustar escala para valores mais significativos
        return max(0.0, proximity ** 2)
    else:
        # Hash atende à dificuldade, calcular proximidade relativa ao zero
        return 1.0 - (hash_int / target_int)

def estimate_time_to_find(hash_rate: float, difficulty_trilhoes: float) -> float:
    """
    Estima o tempo necessário para encontrar um hash válido
    
    Args:
        hash_rate: Taxa de hash em hashes/segundo
        difficulty_trilhoes: Dificuldade em trilhões
        
    Returns:
        Tempo estimado em segundos
    """
    # Probabilidade aproximada de encontrar um hash válido
    probability = 1.0 / (difficulty_trilhoes * 10**12)
    
    # Tempo estimado = 1 / (taxa_hash * probabilidade)
    if hash_rate > 0 and probability > 0:
        return 1.0 / (hash_rate * probability)
    
    # Evitar divisão por zero
    return float('inf')
"""
