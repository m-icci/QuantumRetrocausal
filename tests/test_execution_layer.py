"""
Test suite for execution layer functionality
"""
import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch
from typing import Dict, List, Any, Optional

from qualia.execution_layer import (
    ExecutionLayer,
    ExecutionResult,
    OrderType,
    ExecutionStatus,
    GeoRestrictionError
)

@pytest.fixture
def execution_layer() -> ExecutionLayer:
    """Create ExecutionLayer instance for testing"""
    layer = ExecutionLayer(
        symbols=["BTC-USDT", "ETH-USDT"],
        timeframe="1m",
        risk_threshold=0.7,
        quantum_dimension=64,
        kucoin_enabled=False  # Use simulation mode for testing
    )
    return layer

@pytest.fixture
def mock_market_data() -> np.ndarray:
    """Generate mock market data for testing"""
    timestamps = np.linspace(
        datetime.now().timestamp(),
        datetime.now().timestamp() + 3600,
        100
    )
    prices = 40000 + 1000 * np.sin(np.linspace(0, 4*np.pi, 100))
    volumes = 10 + 5 * np.random.rand(100)

    return np.column_stack([timestamps, prices, volumes])

def test_initialization(execution_layer):
    """Test execution layer initialization"""
    assert execution_layer.quantum_dimension == 64
    assert execution_layer.risk_threshold == 0.7
    assert not execution_layer.kucoin_enabled
    assert execution_layer.exchange is not None
    assert execution_layer.quantum_field is not None
    assert execution_layer.validator is not None

def test_execute_market_order(execution_layer):
    """Test market order execution"""
    result = execution_layer.execute_quantum_trade(
        symbol='BTC-USDT',
        quantum_state=np.random.randn(64, 64),
        risk_metrics={'coherence': 0.8, 'phi_resonance': 0.7}
    )

    assert isinstance(result, ExecutionResult)
    assert result.status == 'simulated'  # Since we're in simulation mode
    assert result.metrics is not None
    assert result.consciousness_metrics is not None

def test_execute_limit_order(execution_layer):
    """Test limit order execution in simulation mode"""
    result = execution_layer.execute_quantum_trade(
        symbol='BTC-USDT',
        quantum_state=np.random.randn(64, 64),
        risk_metrics={'coherence': 0.8, 'phi_resonance': 0.7}
    )

    assert isinstance(result, ExecutionResult)
    assert result.status == 'simulated'
    assert result.metrics is not None
    assert result.consciousness_metrics is not None

def test_order_validation(execution_layer):
    """Test order validation"""
    # Test invalid symbol
    result = execution_layer.execute_quantum_trade(
        symbol='INVALID-PAIR',
        quantum_state=np.random.randn(64, 64),
        risk_metrics={'coherence': 0.8, 'phi_resonance': 0.7}
    )
    assert result.status == 'error'
    assert 'Invalid symbol' in result.error

    # Test invalid quantum state
    result = execution_layer.execute_quantum_trade(
        symbol='BTC-USDT',
        quantum_state=np.zeros((32, 32)),  # Wrong dimension
        risk_metrics={'coherence': 0.8, 'phi_resonance': 0.7}
    )
    assert result.status == 'error'
    assert 'quantum state' in result.error.lower()

def test_risk_checks(execution_layer):
    """Test risk management checks"""
    # Test low coherence rejection
    result = execution_layer.execute_quantum_trade(
        symbol='BTC-USDT',
        quantum_state=np.random.randn(64, 64),
        risk_metrics={'coherence': 0.2, 'phi_resonance': 0.2}  # Low values
    )
    assert result.status == 'rejected'
    assert 'risk' in result.error.lower()

def test_quantum_metrics(execution_layer):
    """Test quantum metrics calculation during execution"""
    result = execution_layer.execute_quantum_trade(
        symbol='BTC-USDT',
        quantum_state=np.random.randn(64, 64),
        risk_metrics={'coherence': 0.8, 'phi_resonance': 0.7}
    )

    assert result.consciousness_metrics is not None
    assert isinstance(result.consciousness_metrics, dict)
    assert 'coherence' in result.consciousness_metrics
    assert 'morphic_resonance' in result.consciousness_metrics
    assert all(0 <= v <= 1 for v in result.consciousness_metrics.values())

def test_market_impact(execution_layer):
    """Test market impact handling"""
    result = execution_layer.execute_quantum_trade(
        symbol='BTC-USDT',
        quantum_state=np.random.randn(64, 64),
        risk_metrics={
            'coherence': 0.8,
            'phi_resonance': 0.7,
            'market_impact': 0.05
        }
    )
    assert result.status == 'simulated'
    assert result.metrics is not None
    assert isinstance(result.metrics, dict)

def test_execution_timing(execution_layer):
    """Test execution timing constraints"""
    start_time = datetime.now()
    result = execution_layer.execute_quantum_trade(
        symbol='BTC-USDT',
        quantum_state=np.random.randn(64, 64),
        risk_metrics={'coherence': 0.8, 'phi_resonance': 0.7}
    )
    end_time = datetime.now()

    assert result.status in ['simulated', 'executed']
    assert isinstance(result.timestamp, datetime)
    assert start_time <= result.timestamp <= end_time

def test_error_handling(execution_layer):
    """Test error handling in execution layer"""
    # Test None quantum state
    result = execution_layer.execute_quantum_trade(
        symbol='BTC-USDT',
        quantum_state=None,
        risk_metrics={'coherence': 0.8, 'phi_resonance': 0.7}
    )
    assert result.status == 'error'
    assert result.error is not None
    assert 'quantum state' in result.error.lower()

    # Test missing risk metrics
    result = execution_layer.execute_quantum_trade(
        symbol='BTC-USDT',
        quantum_state=np.random.randn(64, 64),
        risk_metrics=None
    )
    assert result.status == 'error'
    assert result.error is not None
    assert 'risk metrics' in result.error.lower()