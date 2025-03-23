"""
Tests for the trading system component.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from quantum_trading.core.trading.trading_system import TradingSystem
from quantum_trading.data.data_loader import DataLoader
from quantum_trading.core.trading.risk_manager import RiskManager
from quantum_trading.analysis.quantum_pattern_analyzer import QuantumPatternAnalyzer

@pytest.fixture
def config():
    """Test configuration fixture."""
    return {
        'trading': {
            'symbol': 'BTC/USDT',
            'timeframe': '1m',
            'mode': 'simulated',
            'max_active_positions': 3,
            'position_sizing': {
                'method': 'risk_based',
                'risk_per_trade': 0.01
            }
        },
        'risk': {
            'max_position_size': 0.1,
            'max_drawdown': 0.05,
            'max_daily_loss': 0.02,
            'max_leverage': 3.0,
            'min_profit_ratio': 1.5,
            'max_correlation': 0.8
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
    
    # Mock balance
    loader.get_balance.return_value = {
        'total': {'USDT': 100000.0, 'BTC': 1.0},
        'free': {'USDT': 90000.0, 'BTC': 0.8},
        'used': {'USDT': 10000.0, 'BTC': 0.2}
    }
    
    return loader

@pytest.fixture
def mock_risk_manager():
    """Mock RiskManager fixture."""
    manager = AsyncMock(spec=RiskManager)
    
    # Mock position size calculation
    manager.calculate_position_size.return_value = 0.1
    
    # Mock risk limit checks
    manager.check_risk_limits.return_value = True
    
    # Mock performance metrics
    manager.calculate_risk_metrics.return_value = {
        'sharpe_ratio': 1.5,
        'max_drawdown': 0.03,
        'win_rate': 0.6,
        'profit_factor': 1.8
    }
    
    return manager

@pytest.fixture
def mock_quantum_analyzer():
    """Mock QuantumPatternAnalyzer fixture."""
    analyzer = AsyncMock(spec=QuantumPatternAnalyzer)
    
    # Mock pattern detection
    analyzer.detect_patterns.return_value = [{
        'type': 'bullish',
        'confidence': 0.8,
        'start_index': 0,
        'end_index': 10
    }]
    
    # Mock quantum metrics
    analyzer.calculate_quantum_metrics.return_value = {
        'entanglement': 0.7,
        'coherence': 0.8,
        'decoherence': 0.1
    }
    
    return analyzer

@pytest.mark.asyncio
async def test_trading_system_initialization(config):
    """Test trading system initialization."""
    trading_system = TradingSystem(config)
    assert trading_system.config == config
    assert trading_system.symbol == config['trading']['symbol']
    assert trading_system.timeframe == config['trading']['timeframe']
    assert trading_system.mode == config['trading']['mode']

@pytest.mark.asyncio
async def test_market_analysis(config, mock_data_loader, mock_quantum_analyzer):
    """Test market analysis."""
    trading_system = TradingSystem(config)
    trading_system.data_loader = mock_data_loader
    trading_system.quantum_analyzer = mock_quantum_analyzer
    
    # Analyze market
    analysis = await trading_system.analyze_market()
    assert isinstance(analysis, dict)
    assert 'patterns' in analysis
    assert 'quantum_metrics' in analysis
    assert 'market_state' in analysis

@pytest.mark.asyncio
async def test_position_management(config, mock_data_loader, mock_risk_manager):
    """Test position management."""
    trading_system = TradingSystem(config)
    trading_system.data_loader = mock_data_loader
    trading_system.risk_manager = mock_risk_manager
    
    # Open position
    position = await trading_system.open_position('buy', 0.1, 50000.0)
    assert isinstance(position, dict)
    assert 'id' in position
    assert 'side' in position
    assert 'amount' in position
    assert 'price' in position
    
    # Close position
    result = await trading_system.close_position(position['id'])
    assert isinstance(result, dict)
    assert 'success' in result
    assert 'profit' in result

@pytest.mark.asyncio
async def test_order_execution(config, mock_data_loader):
    """Test order execution."""
    trading_system = TradingSystem(config)
    trading_system.data_loader = mock_data_loader
    
    # Execute market order
    order = await trading_system.execute_order('buy', 0.1, order_type='market')
    assert isinstance(order, dict)
    assert 'id' in order
    assert 'status' in order
    assert order['type'] == 'market'
    
    # Execute limit order
    order = await trading_system.execute_order('sell', 0.1, price=51000.0, order_type='limit')
    assert isinstance(order, dict)
    assert 'id' in order
    assert 'status' in order
    assert order['type'] == 'limit'

@pytest.mark.asyncio
async def test_risk_management(config, mock_data_loader, mock_risk_manager):
    """Test risk management integration."""
    trading_system = TradingSystem(config)
    trading_system.data_loader = mock_data_loader
    trading_system.risk_manager = mock_risk_manager
    
    # Check risk limits
    risk_check = await trading_system.check_risk_limits()
    assert isinstance(risk_check, bool)
    
    # Calculate position size
    size = await trading_system.calculate_position_size(50000.0, 49000.0)
    assert isinstance(size, float)
    assert size > 0

@pytest.mark.asyncio
async def test_performance_tracking(config, mock_data_loader, mock_risk_manager):
    """Test performance tracking."""
    trading_system = TradingSystem(config)
    trading_system.data_loader = mock_data_loader
    trading_system.risk_manager = mock_risk_manager
    
    # Get performance metrics
    metrics = await trading_system.get_performance_metrics()
    assert isinstance(metrics, dict)
    assert 'sharpe_ratio' in metrics
    assert 'max_drawdown' in metrics
    assert 'win_rate' in metrics
    assert 'profit_factor' in metrics

@pytest.mark.asyncio
async def test_quantum_integration(config, mock_data_loader, mock_quantum_analyzer):
    """Test quantum analysis integration."""
    trading_system = TradingSystem(config)
    trading_system.data_loader = mock_data_loader
    trading_system.quantum_analyzer = mock_quantum_analyzer
    
    # Get quantum analysis
    analysis = await trading_system.get_quantum_analysis()
    assert isinstance(analysis, dict)
    assert 'entanglement' in analysis
    assert 'coherence' in analysis
    assert 'decoherence' in analysis

@pytest.mark.asyncio
async def test_error_handling(config, mock_data_loader):
    """Test error handling."""
    trading_system = TradingSystem(config)
    trading_system.data_loader = mock_data_loader
    
    # Test invalid order
    with pytest.raises(ValueError):
        await trading_system.execute_order('invalid', 0.1)
    
    # Test insufficient balance
    mock_data_loader.get_balance.return_value['free']['USDT'] = 0
    with pytest.raises(ValueError):
        await trading_system.execute_order('buy', 1.0)

@pytest.mark.asyncio
async def test_position_tracking(config, mock_data_loader):
    """Test position tracking."""
    trading_system = TradingSystem(config)
    trading_system.data_loader = mock_data_loader
    
    # Open multiple positions
    position1 = await trading_system.open_position('buy', 0.1, 50000.0)
    position2 = await trading_system.open_position('buy', 0.2, 50000.0)
    
    # Get active positions
    positions = trading_system.get_active_positions()
    assert isinstance(positions, list)
    assert len(positions) == 2
    assert all('id' in p for p in positions)
    
    # Close position
    await trading_system.close_position(position1['id'])
    positions = trading_system.get_active_positions()
    assert len(positions) == 1

@pytest.mark.asyncio
async def test_system_state(config, mock_data_loader):
    """Test system state management."""
    trading_system = TradingSystem(config)
    trading_system.data_loader = mock_data_loader
    
    # Get system state
    state = trading_system.get_system_state()
    assert isinstance(state, dict)
    assert 'mode' in state
    assert 'active_positions' in state
    assert 'balance' in state
    assert 'risk_level' in state

if __name__ == '__main__':
    pytest.main(['-v', __file__]) 