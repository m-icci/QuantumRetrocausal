"""
Tests for the scalping system component.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from quantum_trading.scalping import ScalpingSystem
from quantum_trading.data.data_loader import DataLoader
from quantum_trading.core.trading.market_analysis import MarketAnalysis
from quantum_trading.core.trading.order_executor import OrderExecutor
from quantum_trading.core.trading.risk_manager import RiskManager

@pytest.fixture
def config():
    """Test configuration fixture."""
    return {
        'trading': {
            'symbol': 'BTC/USDT',
            'timeframe': '1m',
            'mode': 'simulated'
        },
        'scalping': {
            'enabled': True,
            'min_profit': 0.0005,
            'max_loss': 0.0003,
            'max_position_time': 300,
            'min_volume': 1000,
            'max_spread': 0.0002,
            'micro_trend': {
                'window_size': 20,
                'momentum_threshold': 0.0001
            }
        },
        'risk_management': {
            'max_position_size': 0.1,
            'max_daily_loss': 0.02,
            'max_drawdown': 0.05,
            'position_sizing': {
                'method': 'fixed_risk',
                'risk_per_trade': 0.01
            }
        },
        'costs': {
            'exchange_fee': 0.0004,
            'slippage': 0.0001,
            'min_trade_size': 0.001
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
        'amount': [1.0 + np.random.random() for _ in range(100)],
        'side': ['buy', 'sell'] * 50
    })
    
    # Mock balance
    loader.get_balance.return_value = {
        'total': {'USDT': 10000.0, 'BTC': 0.5},
        'free': {'USDT': 9000.0, 'BTC': 0.4},
        'used': {'USDT': 1000.0, 'BTC': 0.1}
    }
    
    return loader

@pytest.fixture
def mock_market_analysis():
    """Mock MarketAnalysis fixture."""
    analysis = AsyncMock(spec=MarketAnalysis)
    analysis.check_volume_profile.return_value = True
    analysis.check_order_book.return_value = True
    analysis.check_micro_movements.return_value = True
    return analysis

@pytest.fixture
def mock_order_executor():
    """Mock OrderExecutor fixture."""
    executor = AsyncMock(spec=OrderExecutor)
    executor.execute_order.return_value = {
        'id': '123',
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'price': 50000.0,
        'amount': 0.1,
        'timestamp': datetime.now()
    }
    return executor

@pytest.fixture
def mock_risk_manager():
    """Mock RiskManager fixture."""
    manager = AsyncMock(spec=RiskManager)
    manager.check_risk_limits.return_value = True
    manager.calculate_position_size.return_value = 0.1
    return manager

@pytest.mark.asyncio
async def test_scalping_system_initialization(config, mock_data_loader, 
                                           mock_market_analysis, mock_order_executor, 
                                           mock_risk_manager):
    """Test scalping system initialization."""
    system = ScalpingSystem(
        config=config,
        data_loader=mock_data_loader,
        market_analysis=mock_market_analysis,
        order_executor=mock_order_executor,
        risk_manager=mock_risk_manager
    )
    
    assert system.config == config
    assert system.data_loader == mock_data_loader
    assert system.market_analysis == mock_market_analysis
    assert system.order_executor == mock_order_executor
    assert system.risk_manager == mock_risk_manager
    assert system.min_profit == config['scalping']['min_profit']
    assert system.max_loss == config['scalping']['max_loss']

@pytest.mark.asyncio
async def test_entry_conditions(config, mock_data_loader, 
                              mock_market_analysis, mock_order_executor, 
                              mock_risk_manager):
    """Test entry conditions checking."""
    system = ScalpingSystem(
        config=config,
        data_loader=mock_data_loader,
        market_analysis=mock_market_analysis,
        order_executor=mock_order_executor,
        risk_manager=mock_risk_manager
    )
    
    # Test with all conditions met
    should_enter = await system.check_entry_conditions()
    assert should_enter is True
    
    # Test with insufficient volume
    mock_market_analysis.check_volume_profile.return_value = False
    should_enter = await system.check_entry_conditions()
    assert should_enter is False
    
    # Test with wide spread
    mock_market_analysis.check_volume_profile.return_value = True
    mock_market_analysis.check_order_book.return_value = False
    should_enter = await system.check_entry_conditions()
    assert should_enter is False
    
    # Test with no trend
    mock_market_analysis.check_order_book.return_value = True
    mock_market_analysis.check_micro_movements.return_value = False
    should_enter = await system.check_entry_conditions()
    assert should_enter is False

@pytest.mark.asyncio
async def test_position_management(config, mock_data_loader, 
                                mock_market_analysis, mock_order_executor, 
                                mock_risk_manager):
    """Test position management."""
    system = ScalpingSystem(
        config=config,
        data_loader=mock_data_loader,
        market_analysis=mock_market_analysis,
        order_executor=mock_order_executor,
        risk_manager=mock_risk_manager
    )
    
    # Open position
    position = await system.open_position('buy')
    assert position['side'] == 'buy'
    assert position['amount'] == 0.1
    assert position['price'] == 50000.0
    
    # Check position status
    is_open = system.has_open_position()
    assert is_open is True
    
    # Close position
    closed = await system.close_position(reason='take_profit')
    assert closed['side'] == 'sell'
    assert closed['amount'] == position['amount']
    
    # Verify position closed
    is_open = system.has_open_position()
    assert is_open is False

@pytest.mark.asyncio
async def test_profit_loss_calculation(config, mock_data_loader, 
                                     mock_market_analysis, mock_order_executor, 
                                     mock_risk_manager):
    """Test profit/loss calculation."""
    system = ScalpingSystem(
        config=config,
        data_loader=mock_data_loader,
        market_analysis=mock_market_analysis,
        order_executor=mock_order_executor,
        risk_manager=mock_risk_manager
    )
    
    # Open position
    position = await system.open_position('buy')
    
    # Test profit calculation
    mock_data_loader.get_current_price.return_value = 50100.0
    profit_pct = await system.calculate_current_pnl()
    assert profit_pct > 0
    
    # Test loss calculation
    mock_data_loader.get_current_price.return_value = 49900.0
    loss_pct = await system.calculate_current_pnl()
    assert loss_pct < 0

@pytest.mark.asyncio
async def test_position_timeout(config, mock_data_loader, 
                              mock_market_analysis, mock_order_executor, 
                              mock_risk_manager):
    """Test position timeout."""
    system = ScalpingSystem(
        config=config,
        data_loader=mock_data_loader,
        market_analysis=mock_market_analysis,
        order_executor=mock_order_executor,
        risk_manager=mock_risk_manager
    )
    
    # Open position
    position = await system.open_position('buy')
    position['timestamp'] = datetime.now() - timedelta(seconds=config['scalping']['max_position_time'] + 1)
    
    # Check timeout
    should_close = await system.check_position_timeout()
    assert should_close is True

@pytest.mark.asyncio
async def test_risk_limits(config, mock_data_loader, 
                          mock_market_analysis, mock_order_executor, 
                          mock_risk_manager):
    """Test risk limit enforcement."""
    system = ScalpingSystem(
        config=config,
        data_loader=mock_data_loader,
        market_analysis=mock_market_analysis,
        order_executor=mock_order_executor,
        risk_manager=mock_risk_manager
    )
    
    # Test with risk limits exceeded
    mock_risk_manager.check_risk_limits.return_value = False
    should_enter = await system.check_entry_conditions()
    assert should_enter is False
    
    # Test with insufficient position size
    mock_risk_manager.calculate_position_size.return_value = 0.0
    position = await system.open_position('buy')
    assert position is None

if __name__ == '__main__':
    pytest.main(['-v', __file__]) 