"""
Test configuration and fixtures for QUALIA Trading System
"""
import pytest
import numpy as np
from pathlib import Path
from typing import Dict, Any
import logging
from datetime import datetime
import sys
import os
import asyncio

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from qualia.utils.logging import setup_logger
from qualia.quantum_trader import QuantumTrader
from qualia.quantum_state_manager import QuantumStateManager
from qualia.core.holographic_memory import HolographicMemory
from qualia.core.market_data import MarketState

@pytest.fixture
def test_logger():
    """Fixture for test logger with temporary log directory"""
    log_dir = Path("test_logs")
    logger = setup_logger(
        "test_logger",
        level=logging.DEBUG,
        log_dir=str(log_dir),
        structured=True
    )
    yield logger

    # Cleanup logs after tests
    for log_file in log_dir.glob("*.log"):
        log_file.unlink()
    log_dir.rmdir()

@pytest.fixture
def quantum_state():
    """Fixture for test quantum state"""
    class MockQuantumState:
        def __init__(self):
            self.dimension = 64
            self.state_vector = np.random.rand(64) + 1j * np.random.rand(64)
            self.state_vector /= np.linalg.norm(self.state_vector)

    return MockQuantumState()

@pytest.fixture
def market_data():
    """Fixture for test market data"""
    time_points = 100
    close_prices = np.random.rand(time_points) * 1000 + 30000  # Bitcoin-like prices
    volumes = np.random.rand(time_points) * 100
    timestamps = np.arange(time_points)

    ohlcv_data = []
    for i in range(time_points):
        # [timestamp, open, high, low, close, volume]
        ohlcv_data.append([
            timestamps[i],
            close_prices[i] * 0.99,  # Open slightly lower
            close_prices[i] * 1.02,  # High slightly higher
            close_prices[i] * 0.98,  # Low slightly lower
            close_prices[i],         # Close
            volumes[i]
        ])

    return np.array(ohlcv_data)

@pytest.fixture
def mock_market_state(market_data):
    """Fixture for mock market state with enhanced ICCI consciousness"""
    return MarketState(
        timestamp=float(datetime.now().timestamp()),  # Fixed timestamp type
        ohlcv=market_data,
        quantum_features=np.random.rand(64, 64),
        market_phase='accumulation',
        entropy=0.5,
        complexity=1.0
    )

@pytest.fixture
def quantum_trader(mocker, mock_market_state):
    """Fixture for QuantumTrader instance with mocked dependencies"""
    # Mock the exchange interface
    mock_exchange = mocker.MagicMock()
    mock_exchange.fetch_balance.return_value = {
        'USDT': {'free': 10000.0, 'used': 0.0, 'total': 10000.0},
        'BTC': {'free': 0.1, 'used': 0.0, 'total': 0.1}
    }
    mock_exchange.create_market_buy_order.return_value = {
        'id': '1234',
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'price': 45000.0,
        'amount': 0.1,
        'cost': 4500.0,
        'timestamp': datetime.now().timestamp()
    }
    mock_exchange.create_market_sell_order.return_value = {
        'id': '1235',
        'symbol': 'BTC/USDT',
        'side': 'sell',
        'price': 45000.0,
        'amount': 0.1,
        'cost': 4500.0,
        'timestamp': datetime.now().timestamp()
    }
    mock_exchange.validate_market_state.return_value = True

    # Create trader with mocked dependencies
    trader = QuantumTrader(
        exchange_id='test_exchange',
        trading_pairs=['BTC/USDT'],
        quantum_dimension=64
    )

    # Patch market data provider
    mock_market_data = mocker.MagicMock()
    mock_market_data.get_market_data.return_value = mock_market_state
    mocker.patch.object(trader, 'market_data', mock_market_data)

    # Patch exchange interface
    mocker.patch.object(trader, 'exchange', mock_exchange)

    # Initialize trade history
    trader._trade_history = []  # Use protected attribute to avoid type issues

    return trader

@pytest.fixture
def state_manager():
    """Fixture for QuantumStateManager"""
    return QuantumStateManager(dimension=64)

@pytest.fixture
def holographic_memory():
    """Fixture for HolographicMemory with M-ICCI configuration"""
    return HolographicMemory(
        dimension=64,
        memory_capacity=1000,
        phi=1.618033988749895,  # Golden ratio
        decay_rate=0.01,
        resonance_threshold=0.8
    )

@pytest.fixture
def test_pattern():
    """Fixture for test quantum pattern"""
    pattern = np.random.randn(64) + 1j * np.random.randn(64)
    return pattern / np.linalg.norm(pattern)

@pytest.fixture
def event_loop():
    """Create event loop for tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def config() -> Dict:
    """Test configuration fixture."""
    return {
        'mining': {
            'algorithm': 'randomx',
            'threads': 4,
            'intensity': 0.8,
            'batch_size': 256,
            'optimization': {
                'auto_tune': True,
                'target_temperature': 75,
                'power_limit': 100
            }
        },
        'quantum': {
            'qubits': 20,
            'circuit_depth': 5,
            'measurement_basis': 'computational',
            'optimization': {
                'quantum_annealing': True,
                'annealing_cycles': 1000
            }
        },
        'network': {
            'pool': 'monero.pool.com:3333',
            'wallet': '4xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
            'worker_id': 'worker1',
            'password': 'x'
        }
    }