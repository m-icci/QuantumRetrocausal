"""
Tests for the risk management component.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from quantum_trading.core.trading.risk_manager import RiskManager
from quantum_trading.data.data_loader import DataLoader

@pytest.fixture
def config():
    """Test configuration fixture."""
    return {
        'risk': {
            'max_position_size': 0.1,  # 10% of account balance
            'max_drawdown': 0.05,      # 5% maximum drawdown
            'max_daily_loss': 0.02,    # 2% maximum daily loss
            'max_leverage': 3.0,       # 3x maximum leverage
            'min_profit_ratio': 1.5,   # Minimum profit/loss ratio
            'max_correlation': 0.8,    # Maximum market correlation
            'dynamic_limits': {
                'enabled': True,
                'adjustment_factor': 0.1,
                'min_trades': 30
            }
        },
        'metrics': {
            'sharpe_ratio_threshold': 1.5,
            'max_consecutive_losses': 5,
            'volatility_threshold': 0.02
        }
    }

@pytest.fixture
def mock_data_loader():
    """Mock DataLoader fixture."""
    loader = AsyncMock(spec=DataLoader)
    
    # Mock trade history
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='1min')
    loader.get_trade_history.return_value = pd.DataFrame({
        'timestamp': dates[:1000],
        'type': ['buy', 'sell'] * 500,
        'price': [50000.0 + np.sin(i/100)*1000 for i in range(1000)],
        'amount': [0.1 + np.random.random()*0.1 for _ in range(1000)],
        'profit': [np.random.normal(100, 50) for _ in range(1000)]
    })
    
    # Mock daily history
    loader.get_daily_history.return_value = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', end='2023-12-31', freq='D'),
        'balance': [100000.0 + np.cumsum(np.random.normal(100, 50, 365))[i] for i in range(365)],
        'equity': [100000.0 + np.cumsum(np.random.normal(100, 50, 365))[i] for i in range(365)],
        'profit': [np.random.normal(100, 50) for _ in range(365)]
    })
    
    return loader

@pytest.mark.asyncio
async def test_risk_manager_initialization(config):
    """Test risk manager initialization."""
    risk_manager = RiskManager(config)
    assert risk_manager.config == config
    assert risk_manager.max_position_size == config['risk']['max_position_size']
    assert risk_manager.max_drawdown == config['risk']['max_drawdown']
    assert risk_manager.max_daily_loss == config['risk']['max_daily_loss']

@pytest.mark.asyncio
async def test_position_size_calculation(config):
    """Test position size calculation."""
    risk_manager = RiskManager(config)
    
    # Test with different account balances and risk levels
    balance = 100000.0
    risk_level = 0.01  # 1% risk per trade
    
    position_size = risk_manager.calculate_position_size(balance, risk_level)
    assert isinstance(position_size, float)
    assert position_size <= balance * config['risk']['max_position_size']
    assert position_size > 0

@pytest.mark.asyncio
async def test_drawdown_calculation(config, mock_data_loader):
    """Test drawdown calculation."""
    risk_manager = RiskManager(config)
    
    # Get daily history
    history = await mock_data_loader.get_daily_history()
    
    # Calculate drawdown
    drawdown = risk_manager.calculate_drawdown(history)
    assert isinstance(drawdown, float)
    assert 0 <= drawdown <= 1
    
    # Test max drawdown limit
    assert drawdown <= config['risk']['max_drawdown']

@pytest.mark.asyncio
async def test_daily_loss_check(config, mock_data_loader):
    """Test daily loss checking."""
    risk_manager = RiskManager(config)
    
    # Get daily history
    history = await mock_data_loader.get_daily_history()
    
    # Check daily loss
    daily_loss = risk_manager.check_daily_loss(history)
    assert isinstance(daily_loss, dict)
    assert 'current_loss' in daily_loss
    assert 'limit_exceeded' in daily_loss
    assert isinstance(daily_loss['limit_exceeded'], bool)

@pytest.mark.asyncio
async def test_leverage_check(config):
    """Test leverage checking."""
    risk_manager = RiskManager(config)
    
    # Test with different position sizes and account balances
    position_size = 10000.0
    balance = 5000.0
    margin = 2000.0
    
    leverage = risk_manager.check_leverage(position_size, balance, margin)
    assert isinstance(leverage, dict)
    assert 'current_leverage' in leverage
    assert 'limit_exceeded' in leverage
    assert leverage['current_leverage'] <= config['risk']['max_leverage']

@pytest.mark.asyncio
async def test_profit_ratio_analysis(config, mock_data_loader):
    """Test profit ratio analysis."""
    risk_manager = RiskManager(config)
    
    # Get trade history
    trades = await mock_data_loader.get_trade_history()
    
    # Calculate profit ratio
    ratio = risk_manager.analyze_profit_ratio(trades)
    assert isinstance(ratio, dict)
    assert 'profit_ratio' in ratio
    assert 'meets_minimum' in ratio
    assert isinstance(ratio['meets_minimum'], bool)

@pytest.mark.asyncio
async def test_correlation_analysis(config, mock_data_loader):
    """Test market correlation analysis."""
    risk_manager = RiskManager(config)
    
    # Get market data
    market_data = await mock_data_loader.get_daily_history()
    
    # Calculate correlation
    correlation = risk_manager.analyze_market_correlation(market_data)
    assert isinstance(correlation, float)
    assert -1 <= correlation <= 1
    assert abs(correlation) <= config['risk']['max_correlation']

@pytest.mark.asyncio
async def test_dynamic_risk_limits(config, mock_data_loader):
    """Test dynamic risk limit adjustment."""
    risk_manager = RiskManager(config)
    
    # Get trade history
    trades = await mock_data_loader.get_trade_history()
    
    # Adjust risk limits
    adjusted_limits = risk_manager.adjust_risk_limits(trades)
    assert isinstance(adjusted_limits, dict)
    assert 'max_position_size' in adjusted_limits
    assert 'max_drawdown' in adjusted_limits
    assert adjusted_limits['max_position_size'] > 0
    assert 0 < adjusted_limits['max_drawdown'] < 1

@pytest.mark.asyncio
async def test_risk_metrics_calculation(config, mock_data_loader):
    """Test risk metrics calculation."""
    risk_manager = RiskManager(config)
    
    # Get trade history
    trades = await mock_data_loader.get_trade_history()
    
    # Calculate metrics
    metrics = risk_manager.calculate_risk_metrics(trades)
    assert isinstance(metrics, dict)
    assert 'sharpe_ratio' in metrics
    assert 'max_drawdown' in metrics
    assert 'win_rate' in metrics
    assert 'profit_factor' in metrics
    assert all(isinstance(v, float) for v in metrics.values())

@pytest.mark.asyncio
async def test_risk_limit_validation(config):
    """Test risk limit validation."""
    risk_manager = RiskManager(config)
    
    # Test with valid values
    valid_limits = {
        'position_size': 0.05,
        'drawdown': 0.03,
        'daily_loss': 0.01
    }
    validation = risk_manager.validate_risk_limits(valid_limits)
    assert validation['valid'] is True
    
    # Test with invalid values
    invalid_limits = {
        'position_size': -0.1,
        'drawdown': 1.5,
        'daily_loss': 0
    }
    validation = risk_manager.validate_risk_limits(invalid_limits)
    assert validation['valid'] is False

@pytest.mark.asyncio
async def test_error_handling(config):
    """Test error handling."""
    risk_manager = RiskManager(config)
    
    # Test with invalid input
    with pytest.raises(ValueError):
        risk_manager.calculate_position_size(-1000, 0.01)
    
    with pytest.raises(ValueError):
        risk_manager.check_leverage(-5000, 1000, 500)

@pytest.mark.asyncio
async def test_performance_monitoring(config, mock_data_loader):
    """Test performance monitoring."""
    risk_manager = RiskManager(config)
    
    # Get trade history
    trades = await mock_data_loader.get_trade_history()
    
    # Monitor performance
    performance = risk_manager.monitor_performance(trades)
    assert isinstance(performance, dict)
    assert 'current_drawdown' in performance
    assert 'daily_profit_loss' in performance
    assert 'risk_level' in performance
    assert all(isinstance(v, float) for v in performance.values())

if __name__ == '__main__':
    pytest.main(['-v', __file__])