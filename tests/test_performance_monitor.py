"""
Test suite for enhanced PerformanceMonitor
Validates real-time feedback and adaptation mechanisms
"""
import pytest
import numpy as np
from datetime import datetime, timedelta

from qualia.core.performance_monitor import (
    PerformanceMonitor,
    TradeMetrics,
    AdaptiveParameters
)

@pytest.fixture
def performance_monitor():
    """Create a fresh performance monitor instance for each test"""
    monitor = PerformanceMonitor()
    # Initialize with some default parameters
    monitor.adaptive_params = AdaptiveParameters(alpha=0.5, beta=0.5, gamma=0.5)
    return monitor

def test_adaptive_parameters():
    """Test adaptive parameters validation"""
    params = AdaptiveParameters(alpha=0.5, beta=0.5, gamma=0.5)
    assert params.validate()

    # Test bounds
    params = AdaptiveParameters(alpha=1.1, beta=0.5, gamma=0.5)
    assert not params.validate()

def test_add_trade():
    """Test trade recording with execution metrics"""
    monitor = PerformanceMonitor()

    trade = {
        'symbol': 'BTC/USDT',
        'action': 'BUY',
        'expected_price': 50000.0,
        'executed_price': 50100.0,
        'requested_size': 1.0,
        'filled_size': 0.95,
        'order_time': datetime.now(),
        'execution_time': datetime.now() + timedelta(seconds=1),
        'profit': 100.0
    }

    monitor.add_trade(trade)
    assert len(monitor.trade_history) == 1
    assert len(monitor.execution_metrics) == 1
    assert 'slippage' in monitor.execution_metrics[0]
    assert 'fill_rate' in monitor.execution_metrics[0]
    assert 'response_time' in monitor.execution_metrics[0]

def test_calculate_metrics():
    """Test comprehensive metrics calculation"""
    monitor = PerformanceMonitor()

    # Add sample trades with varying execution quality
    for i in range(10):
        profit = 100 if i % 2 == 0 else -50
        # Improve execution quality for each trade
        slippage = 0.002 - (i * 0.0001)  # Decreasing slippage
        fill_rate = 0.95 + (i * 0.005)   # Increasing fill rate

        trade = {
            'symbol': 'BTC/USDT',
            'action': 'BUY',
            'expected_price': 50000.0,
            'executed_price': 50000.0 * (1 + slippage),
            'requested_size': 1.0,
            'filled_size': fill_rate,
            'order_time': datetime.now(),
            'execution_time': datetime.now() + timedelta(seconds=1),
            'profit': profit
        }
        monitor.add_trade(trade)

    metrics = monitor.calculate_metrics()
    assert isinstance(metrics, TradeMetrics)
    assert 0 <= metrics.win_rate <= 1
    assert metrics.profit_factor > 0
    assert metrics.execution_quality > 0  # Should be positive due to improving execution
    assert 0 <= metrics.fill_rate <= 1

def test_parameter_adaptation():
    """Test dynamic parameter adaptation"""
    monitor = PerformanceMonitor()

    # Store initial parameters with explicit deep copy
    initial_alpha = float(monitor.adaptive_params.alpha)
    initial_beta = float(monitor.adaptive_params.beta)
    initial_gamma = float(monitor.adaptive_params.gamma)

    print(f"\nInitial parameters: α={initial_alpha:.4f}, β={initial_beta:.4f}, γ={initial_gamma:.4f}")

    # Add trades with purposefully poor execution quality to force adaptation
    for i in range(3):  # Reduced iterations with stronger signals
        trade = {
            'symbol': 'BTC/USDT',
            'action': 'BUY',
            'expected_price': 50000.0,
            'executed_price': 50000.0 * 1.02,  # 2% slippage to force adaptation
            'requested_size': 1.0,
            'filled_size': 0.8,  # Poor fill rate
            'order_time': datetime.now(),
            'execution_time': datetime.now() + timedelta(seconds=2),
            'profit': -100.0  # Significant loss to trigger adaptation
        }
        monitor.add_trade(trade)
        monitor._adapt_parameters()  # Force parameter update

    # Get final values
    final_alpha = monitor.adaptive_params.alpha
    final_beta = monitor.adaptive_params.beta
    final_gamma = monitor.adaptive_params.gamma

    print(f"Final parameters: α={final_alpha:.4f}, β={final_beta:.4f}, γ={final_gamma:.4f}")

    # Verify significant parameter changes with detailed messages
    min_change = 0.05  # Expected minimum change

    for param_name, initial, final in [
        ('Alpha', initial_alpha, final_alpha),
        ('Beta', initial_beta, final_beta),
        ('Gamma', initial_gamma, final_gamma)
    ]:
        change = abs(final - initial)
        assert change > min_change, (
            f"{param_name} did not change enough.\n"
            f"Initial: {initial:.6f}\n"
            f"Final: {final:.6f}\n"
            f"Change: {change:.6f} (minimum required: {min_change})"
        )

    # Verify parameters remain within valid bounds
    for param_name, value in [
        ('Alpha', final_alpha),
        ('Beta', final_beta),
        ('Gamma', final_gamma)
    ]:
        assert 0 <= value <= 1, f"{param_name} out of bounds: {value:.6f}"

def test_performance_summary():
    """Test performance summary generation"""
    monitor = PerformanceMonitor()

    # Add sample trades
    for i in range(5):
        trade = {
            'symbol': 'BTC/USDT',
            'action': 'BUY',
            'expected_price': 50000.0,
            'executed_price': 50100.0,
            'requested_size': 1.0,
            'filled_size': 0.95,
            'order_time': datetime.now(),
            'execution_time': datetime.now() + timedelta(seconds=1),
            'profit': 100.0
        }
        monitor.add_trade(trade)

    summary = monitor.get_performance_summary()
    assert isinstance(summary, dict)
    assert 'current_metrics' in summary
    assert 'trade_count' in summary
    assert 'execution_quality_trend' in summary
    assert 'adaptive_parameters' in summary
    assert isinstance(summary['metrics_history'], list)

if __name__ == '__main__':
    pytest.main([__file__])