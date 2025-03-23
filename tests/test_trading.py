"""
Testes para módulo de trading
=========================

Testes unitários para os componentes de trading.
"""

import pytest
from unittest.mock import Mock, patch

from quantum_trading.core.real_time_trader import RealTimeTrader
from quantum_trading.api.market_api import MarketAPI
from quantum_trading.exceptions import TradingError, MarketError, APIError

@pytest.fixture
def market_api():
    """Fixture com API de mercado mockada."""
    api = Mock(spec=MarketAPI)
    
    # Configura respostas mock
    api.get_ticker.return_value = {
        "symbol": "BTC/USDT",
        "bid": 50000.0,
        "ask": 50100.0,
        "last": 50050.0,
        "volume": 100.0
    }
    
    api.get_balance.return_value = {
        "USDT": {"free": 10000.0, "used": 0.0, "total": 10000.0},
        "BTC": {"free": 1.0, "used": 0.0, "total": 1.0}
    }
    
    api.create_order.return_value = {
        "id": "test123",
        "symbol": "BTC/USDT",
        "type": "limit",
        "side": "buy",
        "price": 50000.0,
        "amount": 0.1,
        "filled": 0.1,
        "status": "closed"
    }
    
    return api

@pytest.fixture
def trader(market_api):
    """Fixture com trader configurado."""
    return RealTimeTrader(
        market_api=market_api,
        max_position_size=1.0,
        min_confidence=0.8,
        stop_loss=0.02,
        take_profit=0.05
    )

def test_trader_initialization(trader):
    """Testa inicialização do trader."""
    assert trader.market_api is not None
    assert trader.max_position_size == 1.0
    assert trader.min_confidence == 0.8
    assert trader.stop_loss == 0.02
    assert trader.take_profit == 0.05

def test_analyze_opportunity(trader):
    """Testa análise de oportunidade de trading."""
    # Configura dados mock
    market_data = {
        "symbol": "BTC/USDT",
        "timeframe": "1m",
        "ohlcv": [[1, 50000, 50100, 49900, 50050, 100] for _ in range(100)]
    }
    
    # Testa análise
    opportunity = trader.analyze_opportunity(market_data)
    assert isinstance(opportunity, dict)
    assert "symbol" in opportunity
    assert "side" in opportunity
    assert "confidence" in opportunity
    assert "price" in opportunity
    assert "amount" in opportunity

def test_execute_trade(trader):
    """Testa execução de trade."""
    # Configura trade mock
    trade = {
        "symbol": "BTC/USDT",
        "side": "buy",
        "price": 50000.0,
        "amount": 0.1,
        "confidence": 0.9
    }
    
    # Testa execução
    result = trader.execute_trade(trade)
    assert isinstance(result, dict)
    assert result["status"] == "closed"
    assert result["filled"] == 0.1

def test_validate_trade(trader):
    """Testa validação de trade."""
    # Trade válido
    valid_trade = {
        "symbol": "BTC/USDT",
        "side": "buy",
        "price": 50000.0,
        "amount": 0.1,
        "confidence": 0.9
    }
    assert trader.validate_trade(valid_trade)
    
    # Trade inválido (confiança baixa)
    invalid_trade = valid_trade.copy()
    invalid_trade["confidence"] = 0.5
    with pytest.raises(TradingError):
        trader.validate_trade(invalid_trade)

def test_calculate_position_size(trader):
    """Testa cálculo de tamanho da posição."""
    # Configura parâmetros
    balance = 10000.0
    price = 50000.0
    risk = 0.01
    
    # Testa cálculo
    size = trader.calculate_position_size(balance, price, risk)
    assert 0 < size <= trader.max_position_size
    assert size * price <= balance

def test_error_handling(trader):
    """Testa tratamento de erros."""
    # Simula erro de API
    trader.market_api.get_ticker.side_effect = APIError("API indisponível")
    with pytest.raises(MarketError):
        trader.get_market_data("BTC/USDT")
    
    # Simula erro de mercado
    trader.market_api.create_order.side_effect = MarketError("Liquidez insuficiente")
    with pytest.raises(TradingError):
        trader.execute_trade({
            "symbol": "BTC/USDT",
            "side": "buy",
            "price": 50000.0,
            "amount": 1000.0
        })

def test_risk_management(trader):
    """Testa gerenciamento de risco."""
    # Configura posição atual
    position = {
        "symbol": "BTC/USDT",
        "side": "buy",
        "entry_price": 50000.0,
        "amount": 0.1,
        "stop_loss": 49000.0,
        "take_profit": 52500.0
    }
    
    # Testa stop loss
    current_price = 48000.0
    assert trader.should_close_position(position, current_price)
    
    # Testa take profit
    current_price = 53000.0
    assert trader.should_close_position(position, current_price)
    
    # Testa preço dentro dos limites
    current_price = 50000.0
    assert not trader.should_close_position(position, current_price) 