"""
Utilitários para o sistema QUALIA.
"""

import os
import json
import time
import hashlib
import numpy as np
from typing import Dict, List, Union, Optional
from datetime import datetime
from .logging_config import get_logger

logger = get_logger(__name__)

def calculate_quantum_metrics(data: np.ndarray) -> Dict[str, float]:
    """
    Calcula métricas quânticas a partir dos dados fornecidos.
    
    Args:
        data (np.ndarray): Dados para análise
        
    Returns:
        Dict[str, float]: Dicionário com métricas calculadas
    """
    try:
        # Normaliza os dados
        normalized_data = (data - np.mean(data)) / np.std(data)
        
        # Calcula coerência quântica
        coherence = np.abs(np.mean(np.exp(1j * normalized_data)))
        
        # Calcula entropia do campo
        field_entropy = -np.sum(normalized_data**2 * np.log(normalized_data**2 + 1e-10))
        
        # Calcula razão de energia escura
        dark_ratio = np.sum(np.abs(normalized_data[normalized_data < 0])) / np.sum(np.abs(normalized_data))
        
        return {
            'coherence': float(coherence),
            'field_entropy': float(field_entropy),
            'dark_ratio': float(dark_ratio)
        }
    except Exception as e:
        logger.error(f"Erro ao calcular métricas quânticas: {e}")
        return {
            'coherence': 0.0,
            'field_entropy': 0.0,
            'dark_ratio': 0.0
        }

def calculate_morphic_resonance(
    current_state: np.ndarray,
    historical_states: np.ndarray,
    threshold: float = 0.6
) -> Dict[str, float]:
    """
    Calcula a ressonância mórfica entre estados.
    
    Args:
        current_state (np.ndarray): Estado atual
        historical_states (np.ndarray): Estados históricos
        threshold (float): Limiar de ressonância
        
    Returns:
        Dict[str, float]: Métricas de ressonância
    """
    try:
        # Normaliza os estados
        current_norm = current_state / np.linalg.norm(current_state)
        historical_norm = historical_states / np.linalg.norm(historical_states, axis=1)[:, None]
        
        # Calcula ressonância
        resonance = np.abs(np.dot(historical_norm, current_norm))
        max_resonance = np.max(resonance)
        avg_resonance = np.mean(resonance)
        
        # Calcula campo mórfico
        morphic_field = np.sum(resonance > threshold) / len(resonance)
        
        return {
            'max_resonance': float(max_resonance),
            'avg_resonance': float(avg_resonance),
            'morphic_field': float(morphic_field)
        }
    except Exception as e:
        logger.error(f"Erro ao calcular ressonância mórfica: {e}")
        return {
            'max_resonance': 0.0,
            'avg_resonance': 0.0,
            'morphic_field': 0.0
        }

def validate_trade_parameters(
    symbol: str,
    side: str,
    amount: float,
    price: Optional[float] = None
) -> bool:
    """
    Valida parâmetros de trade.
    
    Args:
        symbol (str): Par de trading
        side (str): Lado da operação (buy/sell)
        amount (float): Quantidade
        price (float, optional): Preço
        
    Returns:
        bool: True se parâmetros são válidos
    """
    try:
        # Valida símbolo
        if not isinstance(symbol, str) or '/' not in symbol:
            logger.error(f"Símbolo inválido: {symbol}")
            return False
            
        # Valida lado
        if side not in ['buy', 'sell']:
            logger.error(f"Lado inválido: {side}")
            return False
            
        # Valida quantidade
        if not isinstance(amount, (int, float)) or amount <= 0:
            logger.error(f"Quantidade inválida: {amount}")
            return False
            
        # Valida preço se fornecido
        if price is not None and (not isinstance(price, (int, float)) or price <= 0):
            logger.error(f"Preço inválido: {price}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Erro ao validar parâmetros: {e}")
        return False

def calculate_position_size(
    available_balance: float,
    current_price: float,
    risk_percentage: float,
    max_position_size: float
) -> float:
    """
    Calcula tamanho da posição baseado em gerenciamento de risco.
    
    Args:
        available_balance (float): Saldo disponível
        current_price (float): Preço atual
        risk_percentage (float): Percentual de risco (0-1)
        max_position_size (float): Tamanho máximo da posição
        
    Returns:
        float: Tamanho da posição calculado
    """
    try:
        # Calcula posição baseada no risco
        position_size = available_balance * risk_percentage
        
        # Converte para unidades do ativo
        units = position_size / current_price
        
        # Limita ao tamanho máximo
        units = min(units, max_position_size)
        
        return float(units)
        
    except Exception as e:
        logger.error(f"Erro ao calcular tamanho da posição: {e}")
        return 0.0

def generate_trade_id(
    symbol: str,
    side: str,
    timestamp: Optional[int] = None
) -> str:
    """
    Gera ID único para trade.
    
    Args:
        symbol (str): Par de trading
        side (str): Lado da operação
        timestamp (int, optional): Timestamp do trade
        
    Returns:
        str: ID único do trade
    """
    if timestamp is None:
        timestamp = int(time.time() * 1000)
        
    # Concatena dados
    data = f"{symbol}-{side}-{timestamp}"
    
    # Gera hash
    trade_id = hashlib.sha256(data.encode()).hexdigest()[:12]
    
    return trade_id

def save_trade_history(
    trade_data: Dict,
    filename: str = 'trade_history.json'
) -> bool:
    """
    Salva histórico de trades em arquivo JSON.
    
    Args:
        trade_data (Dict): Dados do trade
        filename (str): Nome do arquivo
        
    Returns:
        bool: True se salvou com sucesso
    """
    try:
        # Carrega histórico existente
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                history = json.load(f)
        else:
            history = []
            
        # Adiciona novo trade
        trade_data['timestamp'] = int(time.time() * 1000)
        history.append(trade_data)
        
        # Salva arquivo
        with open(filename, 'w') as f:
            json.dump(history, f, indent=2)
            
        return True
        
    except Exception as e:
        logger.error(f"Erro ao salvar histórico: {e}")
        return False

def format_trade_log(
    trade_data: Dict,
    include_metrics: bool = False
) -> str:
    """
    Formata dados do trade para log.
    
    Args:
        trade_data (Dict): Dados do trade
        include_metrics (bool): Incluir métricas detalhadas
        
    Returns:
        str: Mensagem formatada
    """
    try:
        # Formata dados básicos
        msg = (
            f"Trade {trade_data['id']} - "
            f"{trade_data['symbol']} - "
            f"{trade_data['side'].upper()} - "
            f"Quantidade: {trade_data['amount']:.8f} - "
            f"Preço: {trade_data['price']:.8f}"
        )
        
        # Adiciona métricas se solicitado
        if include_metrics and 'metrics' in trade_data:
            metrics = trade_data['metrics']
            msg += (
                f" - Coerência: {metrics.get('coherence', 0):.2f} - "
                f"Ressonância: {metrics.get('resonance', 0):.2f} - "
                f"Campo Mórfico: {metrics.get('morphic_field', 0):.2f}"
            )
            
        return msg
        
    except Exception as e:
        logger.error(f"Erro ao formatar log: {e}")
        return str(trade_data)

# Exemplo de uso
if __name__ == '__main__':
    # Testa cálculo de métricas quânticas
    data = np.random.randn(1000)
    metrics = calculate_quantum_metrics(data)
    print("Métricas Quânticas:", metrics)
    
    # Testa cálculo de ressonância
    current = np.random.randn(10)
    historical = np.random.randn(100, 10)
    resonance = calculate_morphic_resonance(current, historical)
    print("Ressonância Mórfica:", resonance)
    
    # Testa validação de parâmetros
    valid = validate_trade_parameters("BTC/USDT", "buy", 0.1, 50000)
    print("Parâmetros Válidos:", valid)
    
    # Testa cálculo de posição
    size = calculate_position_size(10000, 50000, 0.01, 0.1)
    print("Tamanho da Posição:", size)
    
    # Testa geração de ID
    trade_id = generate_trade_id("BTC/USDT", "buy")
    print("ID do Trade:", trade_id) 