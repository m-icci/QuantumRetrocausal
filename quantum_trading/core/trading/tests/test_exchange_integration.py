"""
Testes da integração com a exchange.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch

from ..trading_config import TradingConfig
from ..exchange_integration import ExchangeIntegration

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
    return ExchangeIntegration(config)

@pytest.mark.asyncio
async def test_start(exchange):
    """Testa inicialização da integração."""
    # Configura mocks
    exchange._session = Mock()
    
    # Inicia integração
    await exchange.start()
    
    # Verifica estado
    assert exchange._running
    assert exchange._update_task is not None

@pytest.mark.asyncio
async def test_stop(exchange):
    """Testa parada da integração."""
    # Configura mocks
    exchange._session = Mock()
    exchange._session.close = Mock()
    
    # Inicia integração
    exchange._running = True
    exchange._update_task = Mock()
    
    # Para integração
    await exchange.stop()
    
    # Verifica chamadas
    exchange._session.close.assert_called_once()
    
    # Verifica estado
    assert not exchange._running

@pytest.mark.asyncio
async def test_get_orderbook(exchange):
    """Testa obtenção do livro de ordens."""
    # Configura mocks
    orderbook = {'bids': [[1.0, 1.0]], 'asks': [[2.0, 1.0]]}
    exchange._orderbook_cache = {'BTC/USDT': (0, orderbook)}
    
    # Obtém livro de ordens
    result = await exchange.get_orderbook()
    
    # Verifica resultado
    assert result == orderbook

@pytest.mark.asyncio
async def test_get_trades(exchange):
    """Testa obtenção dos trades."""
    # Configura mocks
    trades = [{'price': 1.0, 'amount': 1.0}]
    exchange._trades_cache = {'BTC/USDT': (0, trades)}
    
    # Obtém trades
    result = await exchange.get_trades()
    
    # Verifica resultado
    assert result == trades

@pytest.mark.asyncio
async def test_get_candles(exchange):
    """Testa obtenção dos candles."""
    # Configura mocks
    candles = [{'timestamp': 0, 'open': 1.0, 'high': 2.0, 'low': 0.5, 'close': 1.5, 'volume': 1.0}]
    exchange._candles_cache = {'BTC/USDT': (0, candles)}
    
    # Obtém candles
    result = await exchange.get_candles()
    
    # Verifica resultado
    assert result == candles

@pytest.mark.asyncio
async def test_get_ticker(exchange):
    """Testa obtenção do ticker."""
    # Configura mocks
    ticker = {'bid': 1.0, 'ask': 2.0, 'last': 1.5}
    exchange._ticker_cache = {'BTC/USDT': (0, ticker)}
    
    # Obtém ticker
    result = await exchange.get_ticker()
    
    # Verifica resultado
    assert result == ticker

@pytest.mark.asyncio
async def test_get_balance(exchange):
    """Testa obtenção do saldo."""
    # Configura mocks
    balance = {'BTC': {'free': 1.0, 'used': 0.0, 'total': 1.0}}
    exchange._balance_cache = (0, balance)
    
    # Obtém saldo
    result = await exchange.get_balance()
    
    # Verifica resultado
    assert result == balance

@pytest.mark.asyncio
async def test_place_order(exchange):
    """Testa colocação de ordem."""
    # Configura mocks
    order = {'id': '1', 'symbol': 'BTC/USDT', 'type': 'limit', 'side': 'buy', 'price': 1.0, 'amount': 1.0}
    exchange._session = Mock()
    exchange._session.post = Mock(return_value=Mock(json=Mock(return_value=order)))
    
    # Coloca ordem
    result = await exchange.place_order('buy', 1.0, 1.0)
    
    # Verifica resultado
    assert result == order

@pytest.mark.asyncio
async def test_cancel_order(exchange):
    """Testa cancelamento de ordem."""
    # Configura mocks
    order = {'id': '1', 'symbol': 'BTC/USDT', 'type': 'limit', 'side': 'buy', 'price': 1.0, 'amount': 1.0}
    exchange._session = Mock()
    exchange._session.delete = Mock(return_value=Mock(json=Mock(return_value=order)))
    
    # Cancela ordem
    result = await exchange.cancel_order('1')
    
    # Verifica resultado
    assert result == order 