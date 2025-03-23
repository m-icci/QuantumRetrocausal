"""
Unit tests for QuantumTrader functionality with enhanced numerical stability testing
"""
import pytest
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock, patch

from qualia.quantum_trader import QuantumTrader, TradeSignal  # Updated import path
from qualia.core.market_data import MarketDataProvider, MarketState
from qualia.analysis.risk_metrics import RiskMetrics

@pytest.fixture
def mock_market_data():
    """Create realistic market data for testing"""
    time_points = 100
    timestamps = np.arange(time_points) * 60  # 1-minute intervals
    prices = 40000 + 1000 * np.sin(np.arange(time_points) * 0.1) + np.random.randn(time_points) * 100
    volumes = np.exp(np.random.randn(time_points) * 0.5) * 10

    ohlcv_data = []
    for i in range(time_points):
        base_price = prices[i]
        ohlcv_data.append([
            timestamps[i],
            base_price * (1 + np.random.randn() * 0.001),  # Open
            base_price * (1 + np.random.randn() * 0.002),  # High
            base_price * (1 - np.random.randn() * 0.002),  # Low
            base_price,                                     # Close
            volumes[i]                                      # Volume
        ])

    return np.array(ohlcv_data)

@pytest.fixture
def mock_market_state(mock_market_data):
    """Create mock market state with valid quantum features"""
    # Create normalized quantum features
    quantum_features = np.random.rand(64)  # 1D array for state vector
    quantum_features = quantum_features / np.linalg.norm(quantum_features)

    # Create valid market state
    market_state = MarketState(
        timestamp=datetime.now(),
        ohlcv=mock_market_data,
        quantum_features=quantum_features,
        market_phase='accumulation',
        entropy=0.5,
        complexity=1.0
    )

    return market_state

@pytest.fixture
def mock_quantum_metrics():
    """Create valid quantum metrics"""
    return {
        'coherence': 0.8,
        'entropy': 0.3,
        'market_stability': 0.7,
        'quantum_alignment': 0.75,
        'consciousness': 0.8,
        'morphic_resonance': 0.7
    }

@pytest.fixture
def mock_consciousness_metrics():
    """Create valid consciousness metrics"""
    return {
        'consciousness_level': 0.85,
        'coherence': 0.9,
        'morphic_resonance': 0.8
    }

@pytest.fixture
def mock_risk_metrics():
    """Create valid risk metrics"""
    return {
        'risk_level': 0.3,
        'stability_index': 0.75,
        'market_volatility': 0.4
    }

def test_market_analysis_execution(quantum_trader, mock_market_state, mock_quantum_metrics, mock_consciousness_metrics, mock_risk_metrics, mocker):
    """Test market analysis execution with realistic data"""
    # Mock dependencies
    mocker.patch(
        'qualia.core.market_data.MarketDataProvider.get_market_data',
        return_value=mock_market_state
    )

    mocker.patch.object(
        quantum_trader.quantum_analyzer,
        'calculate_metrics',
        return_value=mock_quantum_metrics
    )

    mocker.patch.object(
        quantum_trader.memory,
        'calculate_consciousness_metrics',
        return_value=mock_consciousness_metrics
    )

    mocker.patch.object(
        quantum_trader.risk_analyzer,
        'calculate_risk_metrics',
        return_value=mock_risk_metrics
    )

    # Test analysis
    signal = quantum_trader.analyze_market('BTC/USDT')

    # Validate signal
    assert signal is not None
    assert isinstance(signal, TradeSignal)
    assert signal.symbol == 'BTC/USDT'
    assert isinstance(signal.timestamp, datetime)
    assert signal.action in ['BUY', 'SELL', 'HOLD']
    assert 0 <= signal.confidence <= 1
    assert np.isfinite(signal.confidence)
    assert signal.price > 0
    assert np.isfinite(signal.price)

    # Validate quantum metrics
    required_quantum_metrics = {'coherence', 'entropy', 'market_stability'}
    assert all(metric in signal.quantum_metrics for metric in required_quantum_metrics)
    assert all(isinstance(signal.quantum_metrics[metric], (int, float)) 
              for metric in required_quantum_metrics)
    assert all(0 <= float(signal.quantum_metrics[metric]) <= 1 
              for metric in required_quantum_metrics)
    assert all(np.isfinite(signal.quantum_metrics[metric]) 
              for metric in required_quantum_metrics)

    # Validate risk metrics
    risk_metrics = signal.get_risk_metrics()
    required_risk_metrics = {'risk_level', 'stability_index', 'market_volatility'}
    assert all(metric in risk_metrics for metric in required_risk_metrics)
    assert all(isinstance(risk_metrics[metric], (int, float)) 
              for metric in required_risk_metrics)
    assert all(0 <= float(risk_metrics[metric]) <= 1 
              for metric in required_risk_metrics)
    assert all(np.isfinite(risk_metrics[metric]) 
              for metric in required_risk_metrics)

def test_numerical_stability_operations(quantum_trader, mock_quantum_metrics, mock_consciousness_metrics, mock_risk_metrics, mocker):
    """Test numerical stability of quantum operations"""
    # Create edge case market data
    edge_case_data = np.array([
        [0.0, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10],  # Very small values
        [1.0, 1e10, 1e10, 1e10, 1e10, 1e10],       # Very large values
        [2.0, np.nan, np.inf, -np.inf, 0.0, 0.0]   # Invalid values
    ])

    # Create normalized quantum features for edge cases
    quantum_features = np.ones(64) / np.sqrt(64)  # Normalized state vector

    # Create market state with edge cases
    market_state = MarketState(
        timestamp=datetime.now(),
        ohlcv=edge_case_data,
        quantum_features=quantum_features,
        market_phase='accumulation',
        entropy=0.5,
        complexity=1.0
    )

    # Mock dependencies
    mocker.patch(
        'qualia.core.market_data.MarketDataProvider.get_market_data',
        return_value=market_state
    )

    mocker.patch.object(
        quantum_trader.quantum_analyzer,
        'calculate_metrics',
        return_value=mock_quantum_metrics
    )

    mocker.patch.object(
        quantum_trader.memory,
        'calculate_consciousness_metrics',
        return_value=mock_consciousness_metrics
    )

    mocker.patch.object(
        quantum_trader.risk_analyzer,
        'calculate_risk_metrics',
        return_value=mock_risk_metrics
    )

    # Test signal generation with edge cases
    signal = quantum_trader.analyze_market('BTC/USDT')

    # Verify signal stability
    assert signal is not None
    assert isinstance(signal, TradeSignal)
    assert signal.action in ['BUY', 'SELL', 'HOLD']
    assert 0 <= signal.confidence <= 1
    assert np.isfinite(signal.confidence)

    # Verify metrics stability
    assert all(isinstance(v, (int, float)) for v in signal.quantum_metrics.values())
    assert all(0 <= float(v) <= 1 for v in signal.quantum_metrics.values())
    assert all(np.isfinite(v) for v in signal.quantum_metrics.values())

    risk_metrics = signal.get_risk_metrics()
    required_risk_metrics = {'risk_level', 'stability_index', 'market_volatility'}
    assert all(metric in risk_metrics for metric in required_risk_metrics)
    assert all(isinstance(risk_metrics[metric], (int, float)) for metric in required_risk_metrics)
    assert all(0 <= float(risk_metrics[metric]) <= 1 for metric in required_risk_metrics)
    assert all(np.isfinite(risk_metrics[metric]) for metric in required_risk_metrics)

def test_trading_metrics_tracking(quantum_trader, mock_quantum_metrics):
    """Test trading metrics tracking with enhanced stability"""
    # Initialize trade history with stable metrics
    quantum_trader.trade_history = [
        {
            'timestamp': datetime.now(),
            'symbol': 'BTC/USDT',
            'action': 'BUY',
            'price': 45000.0,
            'quantum_metrics': mock_quantum_metrics,
            'consciousness_metrics': {
                'consciousness_level': 0.85,
                'coherence': 0.9,
                'morphic_resonance': 0.8
            },
            'risk_metrics': {
                'risk_level': 0.3,
                'stability_index': 0.7,
                'market_volatility': 0.4
            }
        }
        for _ in range(5)
    ]

    # Get trading metrics
    metrics = quantum_trader.get_trading_metrics()

    # Validate metrics structure and values
    assert isinstance(metrics, dict)
    assert metrics['total_trades'] == 5
    assert all(key in metrics for key in ['quantum_metrics', 'consciousness_metrics', 'risk_metrics'])

    # Validate metric ranges and stability
    for metric_group in [metrics['quantum_metrics'], metrics['risk_metrics']]:
        for value in metric_group.values():
            assert isinstance(value, float)
            assert 0 <= value <= 1
            assert np.isfinite(value)

def test_quantum_metrics_validation(quantum_trader):
    """Test validation of quantum metrics"""
    metrics = quantum_trader.get_trading_metrics()

    # Validate metric structure
    required_quantum_metrics = {'coherence', 'entropy', 'market_stability'}
    required_consciousness_metrics = {'consciousness_level', 'coherence', 'morphic_resonance'}
    required_risk_metrics = {'risk_level', 'stability_index', 'market_volatility'}

    assert all(metric in metrics['quantum_metrics'] for metric in required_quantum_metrics)
    assert all(metric in metrics['consciousness_metrics'] for metric in required_consciousness_metrics)
    assert all(metric in metrics['risk_metrics'] for metric in required_risk_metrics)

    # Validate metric ranges and stability
    assert all(0 <= v <= 1 for v in metrics['quantum_metrics'].values())
    assert all(0 <= v <= 1 for v in metrics['consciousness_metrics'].values())
    assert all(0 <= v <= 1 for v in metrics['risk_metrics'].values())
    assert all(np.isfinite(v) for v in metrics['quantum_metrics'].values())
    assert all(np.isfinite(v) for v in metrics['consciousness_metrics'].values())
    assert all(np.isfinite(v) for v in metrics['risk_metrics'].values())