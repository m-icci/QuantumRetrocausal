"""
Testes da análise de mercado.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from ..trading_config import TradingConfig
from ..exchange_integration import ExchangeIntegration
from ..market_analysis import MarketAnalysis

@pytest.fixture
def config():
    """Cria configuração."""
    return TradingConfig(
        exchange='binance',
        api_key='test_key',
        api_secret='test_secret',
        symbol='BTC/USDT',
        timeframe='1h',
        leverage=1,
        max_positions=3,
        daily_trades_limit=10,
        daily_loss_limit=0.02,
        min_confidence=0.7,
        position_size=0.1,
        min_position_size=0.01,
        max_position_size=0.5,
        stop_loss=0.02,
        take_profit=0.04,
        risk_per_trade=0.01,
        rsi_period=14,
        rsi_oversold=30,
        rsi_overbought=70,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        bb_period=20,
        bb_std=2,
        atr_period=14
    )

@pytest.fixture
def exchange(config):
    """Cria integração com exchange."""
    return Mock(spec=ExchangeIntegration)

@pytest.fixture
def analysis(config, exchange):
    """Cria análise de mercado."""
    return MarketAnalysis(config, exchange)

@pytest.mark.asyncio
async def test_analyze_market(analysis, exchange):
    """Testa análise de mercado."""
    # Configura mocks
    candles = [
        {'timestamp': 0, 'open': 1.0, 'high': 2.0, 'low': 0.5, 'close': 1.5, 'volume': 1.0},
        {'timestamp': 1, 'open': 1.5, 'high': 2.5, 'low': 1.0, 'close': 2.0, 'volume': 2.0}
    ]
    exchange.get_candles = Mock(return_value=candles)
    
    # Analisa mercado
    result = await analysis.analyze_market()
    
    # Verifica resultado
    assert 'indicators' in result
    assert 'patterns' in result
    assert 'signals' in result

@pytest.mark.asyncio
async def test_get_signals(analysis):
    """Testa obtenção de sinais."""
    # Configura mocks
    analysis._signals_cache = (0, {'buy': True, 'sell': False})
    
    # Obtém sinais
    result = await analysis.get_signals()
    
    # Verifica resultado
    assert result == {'buy': True, 'sell': False}

def test_calculate_indicators(analysis):
    """Testa cálculo de indicadores."""
    # Cria dados de teste
    data = pd.DataFrame({
        'timestamp': [0, 1],
        'open': [1.0, 1.5],
        'high': [2.0, 2.5],
        'low': [0.5, 1.0],
        'close': [1.5, 2.0],
        'volume': [1.0, 2.0]
    })
    
    # Calcula indicadores
    result = analysis._calculate_indicators(data)
    
    # Verifica resultado
    assert 'rsi' in result
    assert 'macd' in result
    assert 'bb' in result
    assert 'atr' in result

def test_identify_patterns(analysis):
    """Testa identificação de padrões."""
    # Cria dados de teste
    data = pd.DataFrame({
        'timestamp': [0, 1],
        'open': [1.0, 1.5],
        'high': [2.0, 2.5],
        'low': [0.5, 1.0],
        'close': [1.5, 2.0],
        'volume': [1.0, 2.0]
    })
    
    # Identifica padrões
    result = analysis._identify_patterns(data)
    
    # Verifica resultado
    assert isinstance(result, dict)

def test_generate_signals(analysis):
    """Testa geração de sinais."""
    # Cria dados de teste
    indicators = {
        'rsi': np.array([30.0, 70.0]),
        'macd': {
            'macd': np.array([0.1, 0.2]),
            'signal': np.array([0.0, 0.1]),
            'hist': np.array([0.1, 0.1])
        },
        'bb': {
            'upper': np.array([2.0, 2.5]),
            'middle': np.array([1.5, 2.0]),
            'lower': np.array([1.0, 1.5])
        },
        'atr': np.array([0.1, 0.2])
    }
    patterns = {'doji': True, 'hammer': False}
    
    # Gera sinais
    result = analysis._generate_signals(indicators, patterns)
    
    # Verifica resultado
    assert 'buy' in result
    assert 'sell' in result 