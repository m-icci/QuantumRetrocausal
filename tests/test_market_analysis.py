"""
Tests for the market analysis component.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from quantum_trading.core.trading.market_analysis import MarketAnalysis
from quantum_trading.data.data_loader import DataLoader

@pytest.fixture
def config():
    """Test configuration fixture."""
    return {
        'trading': {
            'symbol': 'BTC/USDT',
            'timeframe': '1m'
        },
        'scalping': {
            'min_volume': 1000,
            'max_spread': 0.0002,
            'micro_trend': {
                'window_size': 20,
                'momentum_threshold': 0.0001
            }
        },
        'quantum': {
            'analysis': {
                'entanglement_threshold': 0.8,
                'coherence_window': 100,
                'decoherence_rate': 0.01
            }
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
        'bids': [[49990.0, 1.0], [49980.0, 2.0]],
        'asks': [[50010.0, 1.0], [50020.0, 2.0]]
    }
    
    # Mock trades
    loader.get_trades.return_value = pd.DataFrame({
        'timestamp': [datetime.now() - timedelta(seconds=i) for i in range(100)],
        'price': [50000.0 + np.sin(i/10)*10 for i in range(100)],
        'amount': [1.0 + np.random.random() for _ in range(100)]
    })
    
    # Mock candles
    loader.get_candles.return_value = pd.DataFrame({
        'timestamp': [datetime.now() - timedelta(minutes=i) for i in range(100)],
        'open': [50000.0 + np.sin(i/10)*10 for i in range(100)],
        'high': [50010.0 + np.sin(i/10)*10 for i in range(100)],
        'low': [49990.0 + np.sin(i/10)*10 for i in range(100)],
        'close': [50000.0 + np.sin(i/10)*10 for i in range(100)],
        'volume': [1000.0 + np.random.random()*100 for _ in range(100)]
    })
    
    return loader

@pytest.mark.asyncio
async def test_market_analysis_initialization(config, mock_data_loader):
    """Test market analysis initialization."""
    analysis = MarketAnalysis(config)
    assert analysis.config == config
    assert analysis.symbol == config['trading']['symbol']
    assert analysis.timeframe == config['trading']['timeframe']

@pytest.mark.asyncio
async def test_volume_profile_analysis(config, mock_data_loader):
    """Test volume profile analysis."""
    analysis = MarketAnalysis(config)
    
    # Test with sufficient volume
    mock_data_loader.get_trades.return_value['amount'] = [100.0] * 100
    result = await analysis.check_volume_profile(mock_data_loader)
    assert result is True
    
    # Test with insufficient volume
    mock_data_loader.get_trades.return_value['amount'] = [0.1] * 100
    result = await analysis.check_volume_profile(mock_data_loader)
    assert result is False

@pytest.mark.asyncio
async def test_order_book_analysis(config, mock_data_loader):
    """Test order book analysis."""
    analysis = MarketAnalysis(config)
    
    # Test with acceptable spread
    mock_data_loader.get_order_book.return_value = {
        'bids': [[49990.0, 1.0]],
        'asks': [[50010.0, 1.0]]
    }
    result = await analysis.check_order_book(mock_data_loader)
    assert result is True
    
    # Test with wide spread
    mock_data_loader.get_order_book.return_value = {
        'bids': [[49900.0, 1.0]],
        'asks': [[50100.0, 1.0]]
    }
    result = await analysis.check_order_book(mock_data_loader)
    assert result is False

@pytest.mark.asyncio
async def test_micro_movement_analysis(config, mock_data_loader):
    """Test micro-movement analysis."""
    analysis = MarketAnalysis(config)
    
    # Test with strong trend
    prices = [50000.0 + i for i in range(100)]
    mock_data_loader.get_trades.return_value['price'] = prices
    result = await analysis.check_micro_movements(mock_data_loader)
    assert result is True
    
    # Test with sideways movement
    prices = [50000.0 + np.random.random() for _ in range(100)]
    mock_data_loader.get_trades.return_value['price'] = prices
    result = await analysis.check_micro_movements(mock_data_loader)
    assert result is False

@pytest.mark.asyncio
async def test_quantum_analysis(config, mock_data_loader):
    """Test quantum analysis features."""
    analysis = MarketAnalysis(config)
    
    # Test entanglement detection
    trades_df = mock_data_loader.get_trades.return_value
    entanglement = analysis._calculate_entanglement(trades_df['price'].values)
    assert 0 <= entanglement <= 1
    
    # Test coherence calculation
    coherence = analysis._calculate_coherence(trades_df['price'].values)
    assert 0 <= coherence <= 1
    
    # Test decoherence rate
    decoherence = analysis._calculate_decoherence(trades_df['price'].values)
    assert 0 <= decoherence <= 1

@pytest.mark.asyncio
async def test_pattern_recognition(config, mock_data_loader):
    """Test pattern recognition capabilities."""
    analysis = MarketAnalysis(config)
    
    # Generate test data with a known pattern
    t = np.linspace(0, 10, 100)
    pattern = 50000.0 + 100 * np.sin(t)
    mock_data_loader.get_trades.return_value['price'] = pattern
    
    # Test pattern detection
    patterns = await analysis.detect_patterns(mock_data_loader)
    assert len(patterns) > 0
    assert all(0 <= p['confidence'] <= 1 for p in patterns)

@pytest.mark.asyncio
async def test_market_state_analysis(config, mock_data_loader):
    """Test market state analysis."""
    analysis = MarketAnalysis(config)
    
    # Test market state classification
    state = await analysis.analyze_market_state(mock_data_loader)
    assert 'volatility' in state
    assert 'trend' in state
    assert 'liquidity' in state
    
    # Verify state values are within expected ranges
    assert 0 <= state['volatility'] <= 1
    assert -1 <= state['trend'] <= 1
    assert 0 <= state['liquidity'] <= 1

if __name__ == '__main__':
    pytest.main(['-v', __file__]) 