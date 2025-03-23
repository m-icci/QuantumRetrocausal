"""
Tests for the order execution component.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from quantum_trading.core.trading.order_executor import OrderExecutor
from quantum_trading.data.data_loader import DataLoader

@pytest.fixture
def config():
    """Test configuration fixture."""
    return {
        'trading': {
            'symbol': 'BTC/USDT',
            'timeframe': '1m',
            'mode': 'simulated',
            'max_slippage': 0.001,  # 0.1% maximum slippage
            'min_order_size': 0.001  # Minimum order size in BTC
        },
        'execution': {
            'retry_attempts': 3,
            'retry_delay': 1,
            'timeout': 30,
            'smart_routing': {
                'enabled': True,
                'min_liquidity': 1.0,
                'max_impact': 0.001
            }
        },
        'costs': {
            'maker_fee': 0.001,  # 0.1% maker fee
            'taker_fee': 0.002   # 0.2% taker fee
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
    
    # Mock balance
    loader.get_balance.return_value = {
        'total': {'USDT': 100000.0, 'BTC': 1.0},
        'free': {'USDT': 90000.0, 'BTC': 0.8},
        'used': {'USDT': 10000.0, 'BTC': 0.2}
    }
    
    return loader

@pytest.mark.asyncio
async def test_executor_initialization(config):
    """Test order executor initialization."""
    executor = OrderExecutor(config)
    assert executor.config == config
    assert executor.symbol == config['trading']['symbol']
    assert executor.mode == config['trading']['mode']
    assert executor.max_slippage == config['trading']['max_slippage']

@pytest.mark.asyncio
async def test_market_order_execution(config, mock_data_loader):
    """Test market order execution."""
    executor = OrderExecutor(config)
    executor.data_loader = mock_data_loader
    
    # Execute buy market order
    order = await executor.execute_order(
        side='buy',
        amount=0.1,
        order_type='market'
    )
    assert isinstance(order, dict)
    assert order['status'] == 'filled'
    assert order['side'] == 'buy'
    assert order['amount'] == 0.1
    assert 'price' in order
    assert 'cost' in order
    assert 'fee' in order
    
    # Execute sell market order
    order = await executor.execute_order(
        side='sell',
        amount=0.1,
        order_type='market'
    )
    assert isinstance(order, dict)
    assert order['status'] == 'filled'
    assert order['side'] == 'sell'

@pytest.mark.asyncio
async def test_limit_order_execution(config, mock_data_loader):
    """Test limit order execution."""
    executor = OrderExecutor(config)
    executor.data_loader = mock_data_loader
    
    # Place buy limit order
    order = await executor.execute_order(
        side='buy',
        amount=0.1,
        order_type='limit',
        price=49000.0
    )
    assert isinstance(order, dict)
    assert order['status'] == 'open'
    assert order['type'] == 'limit'
    assert order['price'] == 49000.0
    
    # Place sell limit order
    order = await executor.execute_order(
        side='sell',
        amount=0.1,
        order_type='limit',
        price=51000.0
    )
    assert isinstance(order, dict)
    assert order['status'] == 'open'
    assert order['type'] == 'limit'

@pytest.mark.asyncio
async def test_order_cancellation(config, mock_data_loader):
    """Test order cancellation."""
    executor = OrderExecutor(config)
    executor.data_loader = mock_data_loader
    
    # Place limit order
    order = await executor.execute_order(
        side='buy',
        amount=0.1,
        order_type='limit',
        price=49000.0
    )
    
    # Cancel order
    result = await executor.cancel_order(order['id'])
    assert isinstance(result, dict)
    assert result['success'] is True
    assert result['order_id'] == order['id']

@pytest.mark.asyncio
async def test_smart_order_routing(config, mock_data_loader):
    """Test smart order routing."""
    executor = OrderExecutor(config)
    executor.data_loader = mock_data_loader
    
    # Execute order with smart routing
    order = await executor.execute_smart_order(
        side='buy',
        amount=1.0,
        max_slippage=0.001
    )
    assert isinstance(order, dict)
    assert 'splits' in order
    assert 'total_cost' in order
    assert 'average_price' in order
    assert order['total_amount'] == 1.0

@pytest.mark.asyncio
async def test_cost_calculation(config, mock_data_loader):
    """Test trading cost calculation."""
    executor = OrderExecutor(config)
    executor.data_loader = mock_data_loader
    
    # Calculate costs for market order
    costs = executor.calculate_costs(
        amount=0.1,
        price=50000.0,
        order_type='market'
    )
    assert isinstance(costs, dict)
    assert 'fee' in costs
    assert 'slippage' in costs
    assert 'total' in costs
    assert costs['fee'] == 0.1 * 50000.0 * config['costs']['taker_fee']

@pytest.mark.asyncio
async def test_slippage_estimation(config, mock_data_loader):
    """Test slippage estimation."""
    executor = OrderExecutor(config)
    executor.data_loader = mock_data_loader
    
    # Estimate slippage
    slippage = await executor.estimate_slippage(
        side='buy',
        amount=1.0,
        current_price=50000.0
    )
    assert isinstance(slippage, float)
    assert 0 <= slippage <= config['trading']['max_slippage']

@pytest.mark.asyncio
async def test_order_validation(config, mock_data_loader):
    """Test order validation."""
    executor = OrderExecutor(config)
    executor.data_loader = mock_data_loader
    
    # Test valid order
    validation = await executor.validate_order(
        side='buy',
        amount=0.1,
        order_type='market'
    )
    assert validation['valid'] is True
    
    # Test invalid amount
    validation = await executor.validate_order(
        side='buy',
        amount=0.0,
        order_type='market'
    )
    assert validation['valid'] is False
    
    # Test invalid side
    validation = await executor.validate_order(
        side='invalid',
        amount=0.1,
        order_type='market'
    )
    assert validation['valid'] is False

@pytest.mark.asyncio
async def test_balance_check(config, mock_data_loader):
    """Test balance checking."""
    executor = OrderExecutor(config)
    executor.data_loader = mock_data_loader
    
    # Check sufficient balance
    check = await executor.check_balance(
        side='buy',
        amount=0.1,
        price=50000.0
    )
    assert check['sufficient'] is True
    
    # Check insufficient balance
    check = await executor.check_balance(
        side='buy',
        amount=100.0,
        price=50000.0
    )
    assert check['sufficient'] is False

@pytest.mark.asyncio
async def test_order_tracking(config, mock_data_loader):
    """Test order tracking."""
    executor = OrderExecutor(config)
    executor.data_loader = mock_data_loader
    
    # Place order
    order = await executor.execute_order(
        side='buy',
        amount=0.1,
        order_type='market'
    )
    
    # Track order
    status = await executor.track_order(order['id'])
    assert isinstance(status, dict)
    assert 'status' in status
    assert 'filled_amount' in status
    assert 'remaining_amount' in status

@pytest.mark.asyncio
async def test_error_handling(config, mock_data_loader):
    """Test error handling."""
    executor = OrderExecutor(config)
    executor.data_loader = mock_data_loader
    
    # Test invalid order type
    with pytest.raises(ValueError):
        await executor.execute_order(
            side='buy',
            amount=0.1,
            order_type='invalid'
        )
    
    # Test invalid price for limit order
    with pytest.raises(ValueError):
        await executor.execute_order(
            side='buy',
            amount=0.1,
            order_type='limit',
            price=-1
        )

@pytest.mark.asyncio
async def test_retry_mechanism(config, mock_data_loader):
    """Test order retry mechanism."""
    executor = OrderExecutor(config)
    executor.data_loader = mock_data_loader
    
    # Mock failed execution
    mock_data_loader.execute_order = AsyncMock(side_effect=[
        Exception("Network error"),
        Exception("Timeout"),
        {'status': 'filled', 'id': '123'}
    ])
    
    # Execute order with retries
    order = await executor.execute_with_retry(
        side='buy',
        amount=0.1,
        order_type='market'
    )
    assert isinstance(order, dict)
    assert order['status'] == 'filled'
    assert mock_data_loader.execute_order.call_count == 3

if __name__ == '__main__':
    pytest.main(['-v', __file__]) 