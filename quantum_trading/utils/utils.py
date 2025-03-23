"""
Funções Utilitárias
=================

Funções auxiliares utilizadas em todo o sistema QUALIA.
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from .logging_config import logger

def load_json(filepath: str) -> Dict[str, Any]:
    """Carrega dados de um arquivo JSON."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Erro ao carregar JSON {filepath}: {e}")
        return {}

def save_json(data: Dict[str, Any], filepath: str) -> bool:
    """Salva dados em um arquivo JSON."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Erro ao salvar JSON {filepath}: {e}")
        return False

def normalize_data(data: np.ndarray) -> np.ndarray:
    """Normaliza dados para o intervalo [0, 1]."""
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        return np.zeros_like(data)
    return (data - min_val) / (max_val - min_val)

def calculate_metrics(
    data: np.ndarray,
    window: int = 20
) -> Tuple[float, float, float]:
    """Calcula métricas básicas de uma série temporal."""
    if len(data) < window:
        return 0.0, 0.0, 0.0
        
    # Volatilidade
    returns = np.diff(data) / data[:-1]
    volatility = np.std(returns) * np.sqrt(252)
    
    # Momentum
    momentum = (data[-1] / data[-window]) - 1
    
    # Força relativa
    gains = np.sum(returns > 0)
    losses = np.sum(returns < 0)
    rsi = 100 - (100 / (1 + gains/max(losses, 1)))
    
    return volatility, momentum, rsi

def format_timestamp(
    timestamp: Optional[Union[int, float, str]] = None,
    fmt: str = "%Y-%m-%d %H:%M:%S"
) -> str:
    """Formata timestamp para string."""
    if timestamp is None:
        timestamp = time.time()
    elif isinstance(timestamp, str):
        return timestamp
        
    return datetime.fromtimestamp(float(timestamp)).strftime(fmt)

def parse_timeframe(timeframe: str) -> int:
    """Converte timeframe em minutos."""
    units = {
        's': 1/60,
        'm': 1,
        'h': 60,
        'd': 1440,
        'w': 10080,
        'M': 43200
    }
    
    unit = timeframe[-1]
    if unit not in units:
        raise ValueError(f"Unidade de tempo inválida: {unit}")
        
    try:
        value = int(timeframe[:-1])
        return int(value * units[unit])
    except:
        raise ValueError(f"Timeframe inválido: {timeframe}")

def retry(
    func: callable,
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0
) -> Any:
    """Executa uma função com retry em caso de erro."""
    attempt = 0
    while attempt < max_attempts:
        try:
            return func()
        except Exception as e:
            attempt += 1
            if attempt == max_attempts:
                raise e
                
            wait = delay * (backoff ** (attempt - 1))
            logger.warning(f"Tentativa {attempt} falhou. Aguardando {wait:.1f}s...")
            time.sleep(wait)

def validate_pair(pair: str) -> bool:
    """Valida formato do par de trading."""
    try:
        base, quote = pair.split('/')
        return bool(base and quote)
    except:
        return False 