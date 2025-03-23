"""
Tests for the data loading component.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from quantum_trading.data.data_loader import DataLoader

@pytest.fixture
def config():
    """Test configuration fixture."""
    return {
        'exchange': {
            'name': 'binance',
            'trading_type': 'spot',
            'rate_limit': {
                'requests_per_minute': 1200,
                'order_rate_limit': 100
            }
        },
        'data': {
            'timeframes': ['1m', '5m', '15m', '1h'],
            'symbols': ['BTC/USDT'],
            'cache_duration': {
                'ticker': 1,      # 1 second
                'orderbook': 1,   # 1 second
                'trades': 60,     # 1 minute
                'candles': 300,   # 5 minutes
                'balance': 5      # 5 seconds
            },
            'history_limit': 1000
        },
        'simulation': {
            'enabled': False,
            'data_source': 'live',
            'price_data': 'real'
        }
    }

@pytest.fixture
def mock_exchange():
    """Mock exchange fixture."""
    exchange = AsyncMock()
    
    # Mock fetch_ticker
    exchange.fetch_ticker.return_value = {
        'symbol': 'BTC/USDT',
        'timestamp': datetime.now().timestamp() * 1000,
        'last': 50000.0,
        'bid': 49900.0,
        'ask': 50100.0,
        'volume': 1000.0
    }
    
    # Mock fetch_order_book
    exchange.fetch_order_book.return_value = {
        'symbol': 'BTC/USDT',
        'timestamp': datetime.now().timestamp() * 1000,
        'bids': [[49900.0, 1.0], [49800.0, 2.0]],
        'asks': [[50100.0, 1.0], [50200.0, 2.0]]
    }
    
    # Mock fetch_trades
    trades = []
    for i in range(100):
        trades.append({
            'id': str(i),
            'timestamp': (datetime.now() - timedelta(seconds=i)).timestamp() * 1000,
            'price': 50000.0 + np.sin(i/10)*100,
            'amount': 1.0 + np.random.random(),
            'side': 'buy' if i % 2 == 0 else 'sell'
        })
    exchange.fetch_trades.return_value = trades
    
    # Mock fetch_ohlcv
    candles = []
    for i in range(100):
        timestamp = (datetime.now() - timedelta(minutes=i)).timestamp() * 1000
        candles.append([
            timestamp,
            50000.0 + np.sin(i/10)*100,  # open
            50100.0 + np.sin(i/10)*100,  # high
            49900.0 + np.sin(i/10)*100,  # low
            50000.0 + np.sin(i/10)*100,  # close
            1000.0 + np.random.random()*100  # volume
        ])
    exchange.fetch_ohlcv.return_value = candles
    
    # Mock fetch_balance
    exchange.fetch_balance.return_value = {
        'total': {'USDT': 100000.0, 'BTC': 1.0},
        'free': {'USDT': 90000.0, 'BTC': 0.8},
        'used': {'USDT': 10000.0, 'BTC': 0.2}
    }
    
    return exchange

@pytest.mark.asyncio
async def test_loader_initialization(config):
    """Test data loader initialization."""
    loader = DataLoader(config)
    assert loader.config == config
    assert loader.exchange_name == config['exchange']['name']
    assert loader.trading_type == config['exchange']['trading_type']
    assert not loader.is_connected

@pytest.mark.asyncio
async def test_connection_management(config, mock_exchange):
    """Test connection management."""
    with patch('ccxt.async_support.binance', return_value=mock_exchange):
        loader = DataLoader(config)
        
        # Test connection
        await loader.connect()
        assert loader.is_connected
        assert loader.exchange is not None
        
        # Test disconnection
        await loader.disconnect()
        assert not loader.is_connected
        assert loader.exchange is None

@pytest.mark.asyncio
async def test_current_price_fetching(config, mock_exchange):
    """Test current price fetching."""
    with patch('ccxt.async_support.binance', return_value=mock_exchange):
        loader = DataLoader(config)
        await loader.connect()
        
        # Test real mode
        price = await loader.get_current_price('BTC/USDT')
        assert isinstance(price, float)
        assert price > 0
        
        # Test simulation mode
        loader.config['simulation']['enabled'] = True
        price = await loader.get_current_price('BTC/USDT')
        assert isinstance(price, float)
        assert price > 0

@pytest.mark.asyncio
async def test_order_book_fetching(config, mock_exchange):
    """Test order book fetching."""
    with patch('ccxt.async_support.binance', return_value=mock_exchange):
        loader = DataLoader(config)
        await loader.connect()
        
        # Get order book
        order_book = await loader.get_order_book('BTC/USDT')
        assert isinstance(order_book, dict)
        assert 'bids' in order_book
        assert 'asks' in order_book
        assert len(order_book['bids']) > 0
        assert len(order_book['asks']) > 0
        assert all(isinstance(bid, list) for bid in order_book['bids'])
        assert all(isinstance(ask, list) for ask in order_book['asks'])

@pytest.mark.asyncio
async def test_trades_fetching(config, mock_exchange):
    """Test trades fetching."""
    with patch('ccxt.async_support.binance', return_value=mock_exchange):
        loader = DataLoader(config)
        await loader.connect()
        
        # Get trades
        trades = await loader.get_trades('BTC/USDT')
        assert isinstance(trades, pd.DataFrame)
        assert not trades.empty
        assert 'timestamp' in trades.columns
        assert 'price' in trades.columns
        assert 'amount' in trades.columns
        assert 'side' in trades.columns

@pytest.mark.asyncio
async def test_candles_fetching(config, mock_exchange):
    """Test candles fetching."""
    with patch('ccxt.async_support.binance', return_value=mock_exchange):
        loader = DataLoader(config)
        await loader.connect()
        
        # Get candles
        candles = await loader.get_candles('BTC/USDT', '1m')
        assert isinstance(candles, pd.DataFrame)
        assert not candles.empty
        assert 'timestamp' in candles.columns
        assert 'open' in candles.columns
        assert 'high' in candles.columns
        assert 'low' in candles.columns
        assert 'close' in candles.columns
        assert 'volume' in candles.columns

@pytest.mark.asyncio
async def test_balance_fetching(config, mock_exchange):
    """Test balance fetching."""
    with patch('ccxt.async_support.binance', return_value=mock_exchange):
        loader = DataLoader(config)
        await loader.connect()
        
        # Get balance
        balance = await loader.get_balance()
        assert isinstance(balance, dict)
        assert 'total' in balance
        assert 'free' in balance
        assert 'used' in balance
        assert 'USDT' in balance['total']
        assert 'BTC' in balance['total']

@pytest.mark.asyncio
async def test_cache_management(config, mock_exchange):
    """Test cache management."""
    with patch('ccxt.async_support.binance', return_value=mock_exchange):
        loader = DataLoader(config)
        await loader.connect()
        
        # Test cache initialization
        assert hasattr(loader, '_cache')
        assert 'ticker' in loader._cache
        assert 'orderbook' in loader._cache
        assert 'trades' in loader._cache
        assert 'candles' in loader._cache
        
        # Test cache update
        await loader.get_current_price('BTC/USDT')
        assert 'BTC/USDT' in loader._cache['ticker']
        
        # Test cache cleaning
        await loader._clean_cache()
        assert all(len(cache) == 0 for cache in loader._cache.values())

@pytest.mark.asyncio
async def test_rate_limiting(config, mock_exchange):
    """Test rate limiting."""
    with patch('ccxt.async_support.binance', return_value=mock_exchange):
        loader = DataLoader(config)
        await loader.connect()
        
        # Test multiple requests
        for _ in range(5):
            await loader.get_current_price('BTC/USDT')
            await loader.get_order_book('BTC/USDT')
        
        # Verify rate limit tracking
        assert hasattr(loader, '_request_count')
        assert loader._request_count > 0

@pytest.mark.asyncio
async def test_error_handling(config, mock_exchange):
    """Test error handling."""
    with patch('ccxt.async_support.binance', return_value=mock_exchange):
        loader = DataLoader(config)
        
        # Test connection error
        mock_exchange.fetch_ticker.side_effect = Exception("Network error")
        with pytest.raises(Exception):
            await loader.get_current_price('BTC/USDT')
        
        # Test invalid symbol
        with pytest.raises(ValueError):
            await loader.get_current_price('INVALID/PAIR')

@pytest.mark.asyncio
async def test_simulation_mode(config, mock_exchange):
    """Test simulation mode."""
    config['simulation']['enabled'] = True
    with patch('ccxt.async_support.binance', return_value=mock_exchange):
        loader = DataLoader(config)
        await loader.connect()
        
        # Test simulated price
        price = await loader.get_current_price('BTC/USDT')
        assert isinstance(price, float)
        assert price > 0
        
        # Test simulated order book
        order_book = await loader.get_order_book('BTC/USDT')
        assert isinstance(order_book, dict)
        assert 'bids' in order_book
        assert 'asks' in order_book

if __name__ == '__main__':
    pytest.main(['-v', __file__]) 