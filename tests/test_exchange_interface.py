"""
Test suite for exchange interface with comprehensive coverage
"""
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime
from qualia.core.exchange_interface import ExchangeInterface, ExchangeStatus

@pytest.fixture(autouse=True)
def mock_environment(monkeypatch):
    """Setup test environment variables"""
    monkeypatch.setenv('KRAKEN_API_KEY', 'test_key')
    monkeypatch.setenv('KRAKEN_API_SECRET', 'test_secret')

@pytest.fixture
def mock_ccxt():
    """Mock CCXT exchange with comprehensive responses"""
    with patch('ccxt.kraken') as mock_exchange:
        mock_instance = MagicMock()
        mock_instance.load_markets.return_value = True
        mock_instance.fetch_balance.return_value = {'BTC': {'free': 1.0}}
        mock_instance.fetch_time.return_value = 1612137600000

        # Configure fetch_ohlcv to return valid data always
        def mock_fetch_ohlcv(symbol, *args, **kwargs):
            if symbol == 'INVALID/PAIR':
                return []
            return [
                [1612137600000, 33000, 34000, 32000, 33500, 100],
                [1612137900000, 33500, 35000, 33000, 34000, 150]
            ]

        mock_instance.fetch_ohlcv = mock_fetch_ohlcv
        mock_exchange.return_value = mock_instance
        yield mock_exchange

def test_exchange_initialization(mock_ccxt):
    """Test exchange initialization with proper config"""
    exchange = ExchangeInterface(
        simulation_mode=False,
        api_key='test_key',
        api_secret='test_secret',
        kraken_enabled=True
    )

    assert exchange.simulation_mode is False
    assert exchange.quantum_dimension == 64
    assert exchange.quantum_validation is True
    assert exchange.state.is_sandbox is False
    assert exchange.exchange_id == 'kraken'
    assert exchange.kraken_enabled is True

def test_market_data_retrieval(mock_ccxt):
    """Test market data retrieval functionality"""
    exchange = ExchangeInterface(
        simulation_mode=False,
        api_key='test_key',
        api_secret='test_secret',
        kraken_enabled=True
    )

    data = exchange.fetch_market_data('BTC/USD')
    assert data is not None
    assert isinstance(data, np.ndarray)
    assert data.shape == (2, 6)
    assert not np.isnan(data).any()

def test_trade_execution_live(mock_ccxt):
    """Test trade execution in live mode"""
    exchange = ExchangeInterface(
        simulation_mode=False,
        api_key='test_key',
        api_secret='test_secret',
        kraken_enabled=True
    )

    # Setup mock to return a successful order response
    mock_response = {
        'id': 'test_order_id',
        'status': 'closed',
        'symbol': 'BTC/USD',
        'type': 'market',
        'side': 'buy',
        'price': 45000.0,
        'amount': 0.1,
        'info': {'status': 'executed'},
        'timestamp': int(datetime.now().timestamp() * 1000)
    }

    # Configure mock to return proper response
    mock_ccxt.return_value.has = {'createMarketOrder': True}
    mock_ccxt.return_value.create_market_buy_order.return_value = mock_response
    mock_ccxt.return_value.markets = {'BTC/USD': {'active': True}}

    result = exchange.create_market_buy_order('BTC/USD', 0.1)
    assert result is not None
    assert result['status'] == 'executed'
    assert result['symbol'] == 'BTC/USD'
    assert result['amount'] == 0.1
    assert 'order_details' in result

def test_connection_verification(mock_ccxt):
    """Test connection verification"""
    exchange = ExchangeInterface(
        simulation_mode=False,
        api_key='test_key',
        api_secret='test_secret',
        kraken_enabled=True
    )

    # Configure mock for successful connection test
    mock_ccxt.return_value.fetch_time.return_value = int(datetime.now().timestamp() * 1000)

    status = exchange.verify_connection()
    assert status is not None
    assert isinstance(status, ExchangeStatus)
    assert status.status == 'READY'
    assert 'connection operational' in status.message.lower()
    assert status.details['mode'] == 'LIVE'
    assert status.details['kraken_enabled'] is True

def test_error_handling(mock_ccxt):
    """Test error handling with invalid operations"""
    exchange = ExchangeInterface(
        simulation_mode=False,
        api_key='test_key',
        api_secret='test_secret',
        kraken_enabled=True
    )

    # Test empty data for invalid symbol
    data = exchange.fetch_market_data('INVALID/PAIR')
    assert data is None

    # Test order with invalid parameters
    result = exchange.create_market_buy_order('BTC/USD', -1)
    assert result is not None
    assert result['status'] == 'error'
    assert 'Invalid amount specified' in result['message']

    # Test with missing credentials in simulation mode
    exchange = ExchangeInterface(simulation_mode=True)
    result = exchange.create_market_buy_order('BTC/USD', 0.1)
    assert result['status'] == 'simulated'
    assert 'Order simulated successfully' in result['message']

if __name__ == '__main__':
    pytest.main([__file__])