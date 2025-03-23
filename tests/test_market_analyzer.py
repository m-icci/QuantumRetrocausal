"""
Tests for the market analysis component.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from quantum_trading.analysis.market_analyzer import MarketAnalyzer
from quantum_trading.data.data_loader import DataLoader

@pytest.fixture
def config():
    """Test configuration fixture."""
    return {
        'analysis': {
            'timeframes': ['1m', '5m', '15m', '1h'],
            'indicators': {
                'rsi': {'period': 14},
                'macd': {'fast': 12, 'slow': 26, 'signal': 9},
                'bollinger': {'period': 20, 'std_dev': 2}
            },
            'volume_profile': {
                'num_bins': 50,
                'min_volume_threshold': 0.1
            },
            'order_flow': {
                'depth': 10,
                'min_size': 0.1,
                'update_interval': 1
            }
        },
        'quantum': {
            'entanglement_threshold': 0.7,
            'coherence_window': 100,
            'decoherence_rate': 0.1
        }
    }

@pytest.fixture
def mock_data_loader():
    """Mock DataLoader fixture."""
    loader = AsyncMock(spec=DataLoader)
    
    # Mock current price
    loader.get_current_price.return_value = 50000.0
    
    # Mock order book
    loader.get_order_book.return_value = {
        'bids': [[49900.0, 1.0], [49800.0, 2.0]],
        'asks': [[50100.0, 1.0], [50200.0, 2.0]]
    }
    
    # Mock trades
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='1min')
    loader.get_trades.return_value = pd.DataFrame({
        'timestamp': dates[:1000],
        'price': [50000.0 + np.sin(i/100)*1000 for i in range(1000)],
        'amount': [1.0 + np.random.random() for _ in range(1000)],
        'side': ['buy', 'sell'] * 500
    })
    
    # Mock candles
    loader.get_candles.return_value = pd.DataFrame({
        'timestamp': dates[:1000],
        'open': [50000.0 + np.sin(i/100)*1000 for i in range(1000)],
        'high': [50100.0 + np.sin(i/100)*1000 for i in range(1000)],
        'low': [49900.0 + np.sin(i/100)*1000 for i in range(1000)],
        'close': [50000.0 + np.sin(i/100)*1000 for i in range(1000)],
        'volume': [1000.0 + np.random.random()*100 for _ in range(1000)]
    })
    
    return loader

@pytest.mark.asyncio
async def test_analyzer_initialization(config):
    """Test market analyzer initialization."""
    analyzer = MarketAnalyzer(config)
    assert analyzer.config == config
    assert analyzer.timeframes == config['analysis']['timeframes']
    assert analyzer.indicators == config['analysis']['indicators']

@pytest.mark.asyncio
async def test_technical_analysis(config, mock_data_loader):
    """Test technical analysis calculations."""
    analyzer = MarketAnalyzer(config)
    analyzer.data_loader = mock_data_loader
    
    # Get candles
    candles = await mock_data_loader.get_candles()
    
    # Calculate indicators
    indicators = analyzer.calculate_indicators(candles)
    assert isinstance(indicators, dict)
    assert 'rsi' in indicators
    assert 'macd' in indicators
    assert 'bollinger' in indicators
    
    # Verify indicator values
    assert len(indicators['rsi']) == len(candles)
    assert all(0 <= v <= 100 for v in indicators['rsi'] if not np.isnan(v))
    assert all(isinstance(v, dict) for v in indicators['macd'])
    assert all(isinstance(v, dict) for v in indicators['bollinger'])

@pytest.mark.asyncio
async def test_volume_profile_analysis(config, mock_data_loader):
    """Test volume profile analysis."""
    analyzer = MarketAnalyzer(config)
    analyzer.data_loader = mock_data_loader
    
    # Get trades
    trades = await mock_data_loader.get_trades()
    
    # Calculate volume profile
    profile = analyzer.analyze_volume_profile(trades)
    assert isinstance(profile, dict)
    assert 'price_levels' in profile
    assert 'volumes' in profile
    assert 'poc' in profile  # Point of Control
    assert len(profile['price_levels']) == config['analysis']['volume_profile']['num_bins']

@pytest.mark.asyncio
async def test_order_flow_analysis(config, mock_data_loader):
    """Test order flow analysis."""
    analyzer = MarketAnalyzer(config)
    analyzer.data_loader = mock_data_loader
    
    # Get order book
    order_book = await mock_data_loader.get_order_book()
    
    # Analyze order flow
    flow = analyzer.analyze_order_flow(order_book)
    assert isinstance(flow, dict)
    assert 'buy_pressure' in flow
    assert 'sell_pressure' in flow
    assert 'imbalance' in flow
    assert -1 <= flow['imbalance'] <= 1

@pytest.mark.asyncio
async def test_market_microstructure(config, mock_data_loader):
    """Test market microstructure analysis."""
    analyzer = MarketAnalyzer(config)
    analyzer.data_loader = mock_data_loader
    
    # Get trades and order book
    trades = await mock_data_loader.get_trades()
    order_book = await mock_data_loader.get_order_book()
    
    # Analyze microstructure
    structure = analyzer.analyze_microstructure(trades, order_book)
    assert isinstance(structure, dict)
    assert 'spread' in structure
    assert 'depth' in structure
    assert 'trade_flow' in structure
    assert 'liquidity_score' in structure

@pytest.mark.asyncio
async def test_trend_analysis(config, mock_data_loader):
    """Test trend analysis."""
    analyzer = MarketAnalyzer(config)
    analyzer.data_loader = mock_data_loader
    
    # Get candles
    candles = await mock_data_loader.get_candles()
    
    # Analyze trend
    trend = analyzer.analyze_trend(candles)
    assert isinstance(trend, dict)
    assert 'direction' in trend
    assert 'strength' in trend
    assert 'support_levels' in trend
    assert 'resistance_levels' in trend
    assert trend['direction'] in ['up', 'down', 'sideways']
    assert 0 <= trend['strength'] <= 1

@pytest.mark.asyncio
async def test_volatility_analysis(config, mock_data_loader):
    """Test volatility analysis."""
    analyzer = MarketAnalyzer(config)
    analyzer.data_loader = mock_data_loader
    
    # Get candles
    candles = await mock_data_loader.get_candles()
    
    # Analyze volatility
    volatility = analyzer.analyze_volatility(candles)
    assert isinstance(volatility, dict)
    assert 'current' in volatility
    assert 'historical' in volatility
    assert 'forecast' in volatility
    assert all(v >= 0 for v in volatility.values())

@pytest.mark.asyncio
async def test_pattern_recognition(config, mock_data_loader):
    """Test pattern recognition."""
    analyzer = MarketAnalyzer(config)
    analyzer.data_loader = mock_data_loader
    
    # Get candles
    candles = await mock_data_loader.get_candles()
    
    # Detect patterns
    patterns = analyzer.detect_patterns(candles)
    assert isinstance(patterns, list)
    for pattern in patterns:
        assert isinstance(pattern, dict)
        assert 'type' in pattern
        assert 'confidence' in pattern
        assert 'start_index' in pattern
        assert 'end_index' in pattern

@pytest.mark.asyncio
async def test_liquidity_analysis(config, mock_data_loader):
    """Test liquidity analysis."""
    analyzer = MarketAnalyzer(config)
    analyzer.data_loader = mock_data_loader
    
    # Get order book
    order_book = await mock_data_loader.get_order_book()
    
    # Analyze liquidity
    liquidity = analyzer.analyze_liquidity(order_book)
    assert isinstance(liquidity, dict)
    assert 'bid_liquidity' in liquidity
    assert 'ask_liquidity' in liquidity
    assert 'spread' in liquidity
    assert liquidity['spread'] >= 0

@pytest.mark.asyncio
async def test_market_regime(config, mock_data_loader):
    """Test market regime detection."""
    analyzer = MarketAnalyzer(config)
    analyzer.data_loader = mock_data_loader
    
    # Get candles and trades
    candles = await mock_data_loader.get_candles()
    trades = await mock_data_loader.get_trades()
    
    # Detect regime
    regime = analyzer.detect_market_regime(candles, trades)
    assert isinstance(regime, dict)
    assert 'type' in regime
    assert 'confidence' in regime
    assert regime['type'] in ['trending', 'ranging', 'volatile']
    assert 0 <= regime['confidence'] <= 1

@pytest.mark.asyncio
async def test_error_handling(config, mock_data_loader):
    """Test error handling."""
    analyzer = MarketAnalyzer(config)
    analyzer.data_loader = mock_data_loader
    
    # Test with invalid data
    with pytest.raises(ValueError):
        analyzer.calculate_indicators(pd.DataFrame())
    
    with pytest.raises(ValueError):
        analyzer.analyze_volume_profile(pd.DataFrame())

if __name__ == '__main__':
    pytest.main(['-v', __file__]) 