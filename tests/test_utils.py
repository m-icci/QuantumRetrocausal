"""
Testes para funções utilitárias
============================

Testes unitários para as funções do módulo utils.
"""

import json
import os
import tempfile
from datetime import datetime

import numpy as np
import pytest

from quantum_trading.utils.utils import (
    load_json,
    save_json,
    normalize_data,
    calculate_metrics,
    format_timestamp,
    parse_timeframe,
    retry,
    validate_pair
)

def test_load_json():
    """Testa carregamento de arquivo JSON."""
    # Cria arquivo temporário
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        data = {"test": "data"}
        json.dump(data, f)
        filepath = f.name
    
    # Testa carregamento
    assert load_json(filepath) == data
    
    # Testa arquivo inválido
    assert load_json("invalid.json") == {}
    
    # Limpa
    os.unlink(filepath)

def test_save_json():
    """Testa salvamento de arquivo JSON."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        filepath = f.name
    
    data = {"test": "data"}
    assert save_json(data, filepath)
    
    with open(filepath) as f:
        assert json.load(f) == data
    
    os.unlink(filepath)

def test_normalize_data():
    """Testa normalização de dados."""
    data = np.array([1, 2, 3, 4, 5])
    normalized = normalize_data(data)
    
    assert np.allclose(normalized, [0, 0.25, 0.5, 0.75, 1])
    assert np.allclose(normalize_data([1, 1, 1]), [0, 0, 0])

def test_calculate_metrics():
    """Testa cálculo de métricas."""
    data = np.array([100, 101, 99, 102, 98, 103])
    vol, mom, rsi = calculate_metrics(data, window=3)
    
    assert isinstance(vol, float)
    assert isinstance(mom, float)
    assert isinstance(rsi, float)
    assert 0 <= rsi <= 100

def test_format_timestamp():
    """Testa formatação de timestamp."""
    now = datetime.now()
    timestamp = now.timestamp()
    
    assert format_timestamp(timestamp) == now.strftime("%Y-%m-%d %H:%M:%S")
    assert format_timestamp("2021-01-01") == "2021-01-01"

def test_parse_timeframe():
    """Testa parsing de timeframe."""
    assert parse_timeframe("1m") == 1
    assert parse_timeframe("1h") == 60
    assert parse_timeframe("1d") == 1440
    
    with pytest.raises(ValueError):
        parse_timeframe("invalid")

def test_retry():
    """Testa função de retry."""
    count = 0
    def fail_twice():
        nonlocal count
        count += 1
        if count < 3:
            raise ValueError("Erro")
        return "success"
    
    assert retry(fail_twice, max_attempts=3) == "success"
    assert count == 3
    
    with pytest.raises(ValueError):
        retry(fail_twice, max_attempts=2)

def test_validate_pair():
    """Testa validação de par de trading."""
    assert validate_pair("BTC/USDT")
    assert not validate_pair("invalid")
    assert not validate_pair("BTC")

if __name__ == '__main__':
    pytest.main([__file__]) 