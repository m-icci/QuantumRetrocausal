"""
Testes para módulo de API
======================

Testes unitários para os componentes de API de mercado.
"""

import pytest
from unittest.mock import Mock, patch

from quantum_trading.api.market_api import MarketAPI
from quantum_trading.exceptions import APIError, ValidationError

@pytest.fixture
def api():
    """Fixture com API configurada."""
    return MarketAPI(
        exchange="binance",
        api_key="test_key",
        api_secret="test_secret"
    )

def test_api_initialization(api):
    """Testa inicialização da API."""
    assert api.exchange == "binance"
    assert api.api_key == "test_key"
    assert api.api_secret == "test_secret"

@patch("ccxt.binance")
def test_get_ticker(mock_exchange):
    """Testa obtenção de ticker."""
    # Configura mock
    mock_exchange.return_value.fetch_ticker.return_value = {
        "symbol": "BTC/USDT",
        "bid": 50000.0,
        "ask": 50100.0,
        "last": 50050.0,
        "baseVolume": 100.0,
        "timestamp": 1234567890000
    }
    
    api = MarketAPI("binance", "test_key", "test_secret")
    
    # Testa chamada
    ticker = api.get_ticker("BTC/USDT")
    assert isinstance(ticker, dict)
    assert ticker["symbol"] == "BTC/USDT"
    assert ticker["bid"] == 50000.0
    assert ticker["ask"] == 50100.0
    
    # Testa erro
    mock_exchange.return_value.fetch_ticker.side_effect = Exception("API error")
    with pytest.raises(APIError):
        api.get_ticker("BTC/USDT")

@patch("ccxt.binance")
def test_get_balance(mock_exchange):
    """Testa obtenção de saldo."""
    # Configura mock
    mock_exchange.return_value.fetch_balance.return_value = {
        "USDT": {"free": 10000.0, "used": 0.0, "total": 10000.0},
        "BTC": {"free": 1.0, "used": 0.0, "total": 1.0}
    }
    
    api = MarketAPI("binance", "test_key", "test_secret")
    
    # Testa chamada
    balance = api.get_balance()
    assert isinstance(balance, dict)
    assert "USDT" in balance
    assert "BTC" in balance
    assert balance["USDT"]["free"] == 10000.0
    
    # Testa erro
    mock_exchange.return_value.fetch_balance.side_effect = Exception("API error")
    with pytest.raises(APIError):
        api.get_balance()

@patch("ccxt.binance")
def test_create_order(mock_exchange):
    """Testa criação de ordem."""
    # Configura mock
    mock_exchange.return_value.create_order.return_value = {
        "id": "test123",
        "symbol": "BTC/USDT",
        "type": "limit",
        "side": "buy",
        "price": 50000.0,
        "amount": 0.1,
        "filled": 0.1,
        "status": "closed"
    }
    
    api = MarketAPI("binance", "test_key", "test_secret")
    
    # Testa ordem válida
    order = api.create_order(
        symbol="BTC/USDT",
        order_type="limit",
        side="buy",
        amount=0.1,
        price=50000.0
    )
    assert isinstance(order, dict)
    assert order["status"] == "closed"
    assert order["filled"] == 0.1
    
    # Testa validação
    with pytest.raises(ValidationError):
        api.create_order(
            symbol="INVALID",
            order_type="invalid",
            side="invalid",
            amount=-1,
            price=-1
        )
    
    # Testa erro de API
    mock_exchange.return_value.create_order.side_effect = Exception("API error")
    with pytest.raises(APIError):
        api.create_order(
            symbol="BTC/USDT",
            order_type="limit",
            side="buy",
            amount=0.1,
            price=50000.0
        )

@patch("ccxt.binance")
def test_get_ohlcv(mock_exchange):
    """Testa obtenção de dados OHLCV."""
    # Configura mock
    mock_data = [[
        1234567890000,  # timestamp
        50000.0,        # open
        50100.0,        # high
        49900.0,        # low
        50050.0,        # close
        100.0           # volume
    ] for _ in range(100)]
    
    mock_exchange.return_value.fetch_ohlcv.return_value = mock_data
    
    api = MarketAPI("binance", "test_key", "test_secret")
    
    # Testa chamada
    ohlcv = api.get_ohlcv("BTC/USDT", timeframe="1m", limit=100)
    assert isinstance(ohlcv, list)
    assert len(ohlcv) == 100
    assert len(ohlcv[0]) == 6
    
    # Testa erro
    mock_exchange.return_value.fetch_ohlcv.side_effect = Exception("API error")
    with pytest.raises(APIError):
        api.get_ohlcv("BTC/USDT", timeframe="1m", limit=100)

@patch("ccxt.binance")
def test_cancel_order(mock_exchange):
    """Testa cancelamento de ordem."""
    # Configura mock
    mock_exchange.return_value.cancel_order.return_value = {
        "id": "test123",
        "status": "canceled"
    }
    
    api = MarketAPI("binance", "test_key", "test_secret")
    
    # Testa cancelamento
    result = api.cancel_order("test123", "BTC/USDT")
    assert isinstance(result, dict)
    assert result["status"] == "canceled"
    
    # Testa erro
    mock_exchange.return_value.cancel_order.side_effect = Exception("API error")
    with pytest.raises(APIError):
        api.cancel_order("test123", "BTC/USDT") 