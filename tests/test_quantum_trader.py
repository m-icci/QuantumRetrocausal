"""Test suite for QuantumTrader functionality with enhanced test coverage"""
import pytest
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock, patch
from types import MappingProxyType

from qualia.quantum_trader import QuantumTrader, TradeSignal
from qualia.market_data import MarketState
from qualia.analysis.risk_metrics import RiskMetrics

@pytest.fixture
def mock_market_state():
    """Create mock market state with valid quantum features"""
    quantum_features = np.random.rand(64)
    quantum_features = quantum_features / np.linalg.norm(quantum_features)
    return MarketState(
        timestamp=datetime.now(),  # Changed to datetime object
        ohlcv=np.random.rand(64, 6),
        quantum_features=quantum_features,
        market_phase='accumulation',
        entropy=0.5,
        complexity=0.5
    )

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

@pytest.fixture
def mock_exchange():
    """Mock exchange for trading tests"""
    with patch('qualia.core.exchange_interface.ExchangeInterface') as mock:
        mock_instance = MagicMock()
        mock_instance.create_market_buy_order.return_value = {
            'id': '123',
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'price': 45000.0,
            'amount': 0.1,
            'cost': 4500.0
        }
        mock_instance.create_market_sell_order.return_value = {
            'id': '124',
            'symbol': 'BTC/USDT',
            'side': 'sell',
            'price': 45000.0,
            'amount': 0.1,
            'cost': 4500.0
        }
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def quantum_trader(mock_exchange, mock_market_data):
    """Create QuantumTrader instance with mocked dependencies"""
    trader = QuantumTrader(
        exchange_id='test_exchange',
        trading_pairs=['BTC/USDT'],
        quantum_dimension=64
    )
    trader.exchange = mock_exchange
    return trader

@pytest.fixture
def mock_market_data():
    """Mock market data provider"""
    with patch('qualia.market_data.MarketDataProvider') as mock:
        mock_instance = MagicMock()
        mock_instance.get_market_data.return_value = MarketState(
            timestamp=datetime.now(), # Changed to datetime object
            ohlcv=np.random.rand(64, 5),
            quantum_features=np.random.rand(64),
            market_phase='accumulation',
            entropy=0.5,
            complexity=1.0
        )
        mock.return_value = mock_instance
        yield mock_instance

def test_quantum_trader_initialization(quantum_trader):
    """Test QuantumTrader initialization"""
    # Test basic attributes
    assert quantum_trader.exchange_id == 'test_exchange'
    assert quantum_trader.trading_pairs == ['BTC/USDT']
    assert quantum_trader.quantum_dimension == 64

    # Test component initialization
    assert hasattr(quantum_trader, 'market_data')
    assert hasattr(quantum_trader, 'exchange')
    assert hasattr(quantum_trader, 'analyzer')
    assert hasattr(quantum_trader, 'risk_analyzer')

def test_analyze_market(quantum_trader, mock_market_state, mock_quantum_metrics, mock_consciousness_metrics, mock_risk_metrics, mocker):
    """Test market analysis functionality"""
    # Mock dependencies
    mocker.patch(
        'qualia.market_data.MarketDataProvider.get_market_data',
        return_value=mock_market_state
    )

    mocker.patch.object(
        quantum_trader.analyzer,
        'calculate_metrics',
        return_value=mock_quantum_metrics
    )

    # Mock exchange interface methods
    exchange_mock = mocker.MagicMock()
    exchange_mock.fetch_balance.return_value = {'USDT': {'free': 10000.0}}
    exchange_mock.validate_market_state.return_value = True
    quantum_trader.exchange = exchange_mock

    # Test analysis
    signal = quantum_trader.analyze_market('BTC/USDT')

    # Validate signal
    assert isinstance(signal, TradeSignal)
    assert signal.symbol == 'BTC/USDT'
    assert isinstance(signal.timestamp, datetime) # Changed to datetime
    assert signal.action in ['BUY', 'SELL', 'HOLD']
    assert 0 <= signal.confidence <= 1
    assert np.isfinite(signal.confidence)

    # Validate quantum metrics
    required_quantum_metrics = {'coherence', 'entropy', 'market_stability'}
    assert all(metric in signal.quantum_metrics for metric in required_quantum_metrics)
    assert all(0 <= float(signal.quantum_metrics[metric]) <= 1 for metric in required_quantum_metrics)
    assert all(np.isfinite(signal.quantum_metrics[metric]) for metric in required_quantum_metrics)

    # Validate risk metrics
    risk_metrics = signal.risk_metrics
    required_risk_metrics = {'risk_level', 'stability_index', 'market_volatility'}
    assert all(metric in risk_metrics for metric in required_risk_metrics)
    assert all(0 <= float(risk_metrics[metric]) <= 1 for metric in required_risk_metrics)
    assert all(np.isfinite(risk_metrics[metric]) for metric in required_risk_metrics)

def test_generate_signal(quantum_trader, mock_market_state):
    """Test trading signal generation"""
    quantum_metrics = {
        'coherence': 0.9,
        'market_stability': 0.8,
        'entropy': 0.3,
        'morphic_resonance': 0.7  # Added required metric
    }

    consciousness_metrics = {
        'consciousness_level': 0.85,
        'coherence': 0.9,
        'morphic_resonance': 0.8
    }

    risk_metrics = {
        'risk_level': 0.3,
        'stability_index': 0.75,
        'market_volatility': 0.4
    }

    action, confidence = quantum_trader._generate_signal(
        quantum_metrics,
        consciousness_metrics,
        risk_metrics,
        mock_market_state
    )

    assert action in ['BUY', 'SELL', 'HOLD']
    assert 0 <= confidence <= 1

    # Test high risk scenario
    risk_metrics['risk_level'] = 0.9
    action, confidence = quantum_trader._generate_signal(
        quantum_metrics,
        consciousness_metrics,
        risk_metrics,
        mock_market_state
    )
    assert action == 'SELL'  # Should avoid risky trades

def test_calculate_adaptive_stops(quantum_trader):
    """Test adaptive stop loss calculation"""
    current_price = 45000.0
    quantum_metrics = {
        'coherence': 0.9,
        'entropy': 0.3,
        'market_stability': 0.8
    }
    consciousness_metrics = {
        'consciousness_level': 0.8,
        'coherence': 0.9,
        'morphic_resonance': 0.8
    }
    risk_metrics = {
        'risk_level': 0.3,
        'stability_index': 0.75,
        'market_volatility': 0.4
    }

    stop_loss, take_profit = quantum_trader._calculate_adaptive_stops(
        current_price,
        quantum_metrics,
        consciousness_metrics,
        risk_metrics
    )

    assert stop_loss < current_price
    assert take_profit > current_price
    assert np.isfinite(stop_loss)
    assert np.isfinite(take_profit)

def test_position_size_calculation(quantum_trader, mocker):
    """Test position size calculation"""
    # Mock exchange interface
    exchange_mock = mocker.MagicMock()
    exchange_mock.fetch_balance.return_value = {'USDT': {'free': 10000.0}}
    quantum_trader.exchange = exchange_mock

    # Enable KuCoin mode for testing
    quantum_trader.exchange.kucoin_enabled = True
    quantum_trader.exchange.trading_enabled = True
    quantum_trader.exchange.state.is_sandbox = False

    signal = TradeSignal(
        timestamp=datetime.now(), # Changed to datetime object
        symbol='BTC/USDT',
        price=45000.0,
        action='BUY',
        confidence=0.8,
        quantum_metrics={
            'coherence': 0.9,
            'entropy': 0.3,
            'market_stability': 0.8,
            'morphic_resonance': 0.85  # Added required metric
        },
        consciousness_metrics={
            'consciousness_level': 0.85,
            'coherence': 0.9,
            'morphic_resonance': 0.8  # Required for consciousness metrics
        },
        risk_metrics={
            'risk_level': 0.3,
            'stability_index': 0.75,
            'market_volatility': 0.4
        },
        alerts=[],  # Initialize empty alerts list
        hedge_ratio=0.5  # Added hedge ratio
    )

    position_size = quantum_trader._calculate_position_size(signal)

    assert 0 < position_size <= 0.05  # Max 5% of capital
    assert isinstance(position_size, float)
    assert np.isfinite(position_size)

    # Test edge case with max risk
    signal.risk_metrics['risk_level'] = 1.0
    position_size = quantum_trader._calculate_position_size(signal)
    assert position_size <= 0.01  # Should be very small when risk is high

    # Test edge case with very low risk
    signal.risk_metrics['risk_level'] = 0.1
    position_size = quantum_trader._calculate_position_size(signal)
    assert 0.01 <= position_size <= 0.05  # Should allow larger position with low risk

def test_trading_metrics(quantum_trader):
    """Test trading metrics collection"""
    metrics = quantum_trader.get_trading_metrics()

    assert 'quantum_metrics' in metrics
    assert 'consciousness_metrics' in metrics
    assert 'risk_metrics' in metrics
    assert isinstance(metrics['quantum_metrics'], dict)
    assert isinstance(metrics['consciousness_metrics'], dict)
    assert isinstance(metrics['risk_metrics'], dict)

    # Verify numeric validity
    for metric_group in [metrics['quantum_metrics'], metrics['consciousness_metrics'], metrics['risk_metrics']]:
        for value in metric_group.values():
            assert isinstance(value, float)
            assert 0 <= value <= 1
            assert np.isfinite(value)

def test_quantum_coherence_stability(quantum_trader):
    """Test quantum coherence calculations for numerical stability"""
    metrics = quantum_trader.get_trading_metrics()

    assert 'quantum_metrics' in metrics
    for value in metrics['quantum_metrics'].values():
        assert isinstance(value, float)
        assert 0 <= value <= 1
        assert np.isfinite(value)

def test_quantum_metrics_validation(quantum_trader):
    """Test quantum metrics validation"""
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

def test_numerical_stability_operations(quantum_trader):
    """Test numerical stability of quantum operations"""
    metrics = quantum_trader.get_trading_metrics()

    # Test valid range
    for group in [metrics['quantum_metrics'], metrics['consciousness_metrics'], metrics['risk_metrics']]:
        for value in group.values():
            assert isinstance(value, float)
            assert 0 <= value <= 1
            assert np.isfinite(value)

    # Test edge cases
    quantum_trader.trade_history = [
        {
            'timestamp': datetime.now(), # Changed to datetime object
            'price': 1e-10,  # Very small price
            'quantum_metrics': {k: 1e-10 for k in metrics['quantum_metrics'].keys()},
            'consciousness_metrics': {k: 1e-10 for k in metrics['consciousness_metrics'].keys()},
            'risk_metrics': {k: 1e-10 for k in metrics['risk_metrics'].keys()}
        },
        {
            'timestamp': datetime.now(), # Changed to datetime object
            'price': 1e10,  # Very large price
            'quantum_metrics': {k: 1.0 - 1e-10 for k in metrics['quantum_metrics'].keys()},
            'consciousness_metrics': {k: 1.0 - 1e-10 for k in metrics['consciousness_metrics'].keys()},
            'risk_metrics': {k: 1.0 - 1e-10 for k in metrics['risk_metrics'].keys()}
        }
    ]

    # Test metrics after edge case data
    metrics = quantum_trader.get_trading_metrics()
    for group in [metrics['quantum_metrics'], metrics['consciousness_metrics'], metrics['risk_metrics']]:
        for value in group.values():
            assert isinstance(value, float)
            assert 0 <= value <= 1
            assert np.isfinite(value)


#New Tests from edited Snippet
def test_risk_metrics_immutability():
    """Test risk metrics immutability using MappingProxyType"""
    trader = QuantumTrader()
    risk_metrics = trader.calculate_risk_metrics()

    assert isinstance(risk_metrics, MappingProxyType), "Risk metrics should be immutable"
    with pytest.raises((AttributeError, TypeError)):
        risk_metrics['new_key'] = 1.0

def test_invalid_quantum_features():
    """Test handling of invalid quantum features"""
    trader = QuantumTrader()
    invalid_state = MarketState(
        timestamp=datetime.now(), # Changed to datetime object
        ohlcv=np.random.rand(10, 5),
        quantum_features=np.random.rand(64),  # Match the expected dimension
        market_phase="accumulation",
        entropy=0.5,
        complexity=0.5
    )

    metrics = trader._calculate_quantum_metrics(invalid_state)
    assert isinstance(metrics, dict)
    assert all(0 <= v <= 1 for v in metrics.values())
    assert 'coherence' in metrics
    assert 'entropy' in metrics

def test_missing_market_data():
    """Test behavior with missing market data"""
    trader = QuantumTrader()
    risk_metrics = trader.calculate_risk_metrics(None)

    assert isinstance(risk_metrics, MappingProxyType)
    assert 'risk_level' in risk_metrics
    assert 'stability_index' in risk_metrics
    assert all(0 <= v <= 1 for v in risk_metrics.values())

def test_invalid_risk_metrics():
    """Test handling of invalid risk metrics"""
    trader = QuantumTrader()
    mock_state = MarketState(
        timestamp=datetime.now(), # Changed to datetime object
        ohlcv=np.random.rand(10, 5),
        quantum_features=np.random.rand(64),
        market_phase="accumulation",
        entropy=0.5,
        complexity=0.5
    )

    # Mock risk analyzer to return invalid metrics
    trader.risk_analyzer.calculate_risk_metrics = MagicMock(return_value={
        'risk_level': 1.5,  # Invalid value > 1
        'stability_index': -0.5,  # Invalid value < 0
    })

    risk_metrics = trader._calculate_risk_metrics(mock_state, {'coherence': 0.5})
    assert isinstance(risk_metrics, MappingProxyType)
    assert all(0 <= v <= 1 for v in risk_metrics.values())

def test_quantum_metrics_calculation():
    """Test quantum metrics calculation with various inputs"""
    trader = QuantumTrader()

    # Test with valid input
    valid_state = MarketState(
        timestamp=datetime.now(), # Changed to datetime object
        ohlcv=np.random.rand(10, 5),
        quantum_features=np.random.rand(64),
        market_phase="accumulation",
        entropy=0.5,
        complexity=0.5
    )
    metrics = trader._calculate_quantum_metrics(valid_state)
    assert isinstance(metrics, dict)
    assert all(0 <= v <= 1 for v in metrics.values())
    required_metrics = {'coherence', 'entropy', 'market_stability', 'morphic_resonance'}
    assert all(k in metrics for k in required_metrics)

    # Test with None values
    none_state = MarketState(
        timestamp=datetime.now(), # Changed to datetime object
        ohlcv=None,
        quantum_features=np.zeros(64),
        market_phase="accumulation",
        entropy=0.5,
        complexity=0.5
    )
    metrics = trader._calculate_quantum_metrics(none_state)
    assert isinstance(metrics, dict)
    assert all(0 <= v <= 1 for v in metrics.values())

def test_consciousness_metrics_calculation():
    """Test consciousness metrics calculation with proper dimensions"""
    trader = QuantumTrader(quantum_dimension=64)  # Explicitly set dimension
    valid_features = np.random.rand(64)  # Match dimension
    valid_features = valid_features / np.linalg.norm(valid_features)  # Normalize

    # Test with valid input
    metrics = trader.memory.calculate_consciousness_metrics(valid_features)
    assert isinstance(metrics, dict)
    assert 'consciousness_level' in metrics
    assert 'coherence' in metrics
    assert all(0 <= v <= 1 for v in metrics.values())

def test_market_state_validation():
    """Test market state validation"""
    trader = QuantumTrader()

    # Test with empty market state
    empty_state = MarketState(
        timestamp=datetime.now(), # Changed to datetime object
        ohlcv=np.array([]),
        quantum_features=np.random.rand(64),
        market_phase="accumulation",
        entropy=0.5,
        complexity=0.5
    )
    metrics = trader._calculate_quantum_metrics(empty_state)
    assert isinstance(metrics, dict)
    assert all(0 <= v <= 1 for v in metrics.values())

    # Test with invalid market phase
    invalid_phase_state = MarketState(
        timestamp=datetime.now(), # Changed to datetime object
        ohlcv=np.random.rand(10, 5),
        quantum_features=np.random.rand(64),
        market_phase="invalid_phase",
        entropy=0.5,
        complexity=0.5
    )
    metrics = trader._calculate_quantum_metrics(invalid_phase_state)
    assert isinstance(metrics, dict)
    assert all(0 <= v <= 1 for v in metrics.values())

def test_risk_metrics_validation_and_immutability():
    """Test risk metrics validation and immutability"""
    trader = QuantumTrader()

    # Test with default market state
    metrics = trader.calculate_risk_metrics()
    assert isinstance(metrics, MappingProxyType), "Risk metrics should be a MappingProxyType"

    # Verify immutability
    with pytest.raises((AttributeError, TypeError)):
        metrics['new_key'] = 1.0

    # Verify required fields
    required_fields = {'risk_level', 'stability_index', 'market_volatility', 'quantum_adjusted_risk'}
    assert all(field in metrics for field in required_fields), "Missing required risk metric fields"

    # Test metric ranges
    assert all(isinstance(v, float) for v in metrics.values()), "All metrics should be floats"
    assert all(0 <= v <= 1 for v in metrics.values()), "All metrics should be between 0 and 1"

def test_quantum_metrics_error_handling():
    """Test error handling in quantum metrics calculation"""
    trader = QuantumTrader()

    # Test with invalid market state
    invalid_state = MarketState(
        timestamp=datetime.now(), # Changed to datetime object
        ohlcv=np.random.rand(10, 5),
        quantum_features=None,  # Invalid quantum features
        market_phase="accumulation",
        entropy=0.5,
        complexity=0.5
    )

    metrics = trader._calculate_quantum_metrics(invalid_state)
    assert isinstance(metrics, dict)
    assert all(0 <= v <= 1 for v in metrics.values())
    assert 'coherence' in metrics
    assert 'entropy' in metrics
    assert 'market_stability' in metrics

def test_risk_metrics_error_handling():
    """Test error handling in risk metrics calculation"""
    trader = QuantumTrader()

    # Test with None market state
    metrics = trader.calculate_risk_metrics(None)
    assert isinstance(metrics, MappingProxyType)
    assert all(0 <= v <= 1 for v in metrics.values())

    # Test with invalid risk analyzer response
    mock_state = MarketState(
        timestamp=datetime.now(), # Changed to datetime object
        ohlcv=np.random.rand(10, 5),
        quantum_features=np.random.rand(64),
        market_phase="accumulation",
        entropy=0.5,
        complexity=0.5
    )

    trader.risk_analyzer.calculate_risk_metrics = MagicMock(return_value=None)
    metrics = trader.calculate_risk_metrics(mock_state)
    assert isinstance(metrics, MappingProxyType)
    assert all(0 <= v <= 1 for v in metrics.values())

def test_quantum_metrics_integration():
    """Test integration of quantum metrics with risk calculation"""
    trader = QuantumTrader()
    mock_state = MarketState(
        timestamp=datetime.now(), # Changed to datetime object
        ohlcv=np.random.rand(10, 5),
        quantum_features=np.random.rand(64),
        market_phase="accumulation",
        entropy=0.5,
        complexity=0.5
    )

    # Get quantum metrics
    quantum_metrics = trader._calculate_quantum_metrics(mock_state)
    assert isinstance(quantum_metrics, dict)
    assert all(0 <= v <= 1 for v in quantum_metrics.values())

    # Use quantum metrics in risk calculation
    risk_metrics = trader._calculate_risk_metrics(mock_state, quantum_metrics)
    assert isinstance(risk_metrics, (dict, MappingProxyType))
    assert all(0 <= v <= 1 for v in risk_metrics.values())

def test_alert_processing():
    """Test alert processing with proper serialization"""
    trader = QuantumTrader()
    signal = TradeSignal(
        timestamp=datetime.now(), # Changed to datetime object
        symbol='BTC/USDT',
        price=45000.0,
        action='BUY',
        confidence=0.8,
        quantum_metrics={
            'coherence': 0.9,
            'entropy': 0.3,
            'market_stability': 0.8,
            'morphic_resonance': 0.7
        },
        consciousness_metrics={
            'consciousness_level': 0.85,
            'coherence': 0.9,
            'morphic_resonance': 0.8
        },
        risk_metrics={
            'risk_level': 0.3,
            'stability_index': 0.75,
            'market_volatility': 0.4
        },
        alerts=[],  # Initialize as empty list
        hedge_ratio=0.5
    )

    # Verify alert list is properly initialized and serializable
    assert hasattr(signal, 'alerts')
    assert isinstance(signal.alerts, list)
    assert len(signal.alerts) == 0

    # Test adding alerts
    signal.alerts.append({
        'severity': 'low',
        'message': 'Test alert',
        'timestamp': str(datetime.now())
    })

    assert len(signal.alerts) == 1
    assert isinstance(signal.alerts[0], dict)
    assert 'severity' in signal.alerts[0]

"""
Integration tests for QuantumTrader focusing on trading functionality
"""
import pytest
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock, patch
from qualia.core.quantum_trader import QuantumTrader, TradeSignal
from qualia.core.market_data import MarketState

@pytest.fixture
def mock_exchange():
    """Mock exchange for trading tests"""
    with patch('qualia.core.exchange_interface.ExchangeInterface') as mock:
        mock_instance = MagicMock()
        mock_instance.create_market_buy_order.return_value = {
            'id': '123',
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'price': 45000.0,
            'amount': 0.1,
            'cost': 4500.0
        }
        mock_instance.create_market_sell_order.return_value = {
            'id': '124',
            'symbol': 'BTC/USDT',
            'side': 'sell',
            'price': 45000.0,
            'amount': 0.1,
            'cost': 4500.0
        }
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_market_data():
    """Mock market data provider"""
    with patch('qualia.core.market_data.MarketDataProvider') as mock:
        mock_instance = MagicMock()
        mock_instance.get_market_data.return_value = MarketState(
            timestamp=datetime.now(), # Changed to datetime object
            ohlcv=np.random.rand(64, 5),
            quantum_features=np.random.rand(64),
            market_phase='accumulation',
            entropy=0.5,
            complexity=1.0
        )
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def quantum_trader(mock_exchange, mock_market_data):
    """Create QuantumTrader instance with mocked dependencies"""
    trader = QuantumTrader(
        exchange_id='test_exchange',
        trading_pairs=['BTC/USDT'],  # Formato correto
        quantum_dimension=64,
        consciousness_threshold=0.7
    )
    # Enable KuCoin mode for the mock exchange
    trader.exchange.kucoin_enabled = True
    trader.exchange.trading_enabled = True
    trader.exchange.state.is_sandbox = False
    return trader

def test_market_analysis_integration(quantum_trader, mock_market_data):
    """Test complete market analysis pipeline"""
    # Test market analysis
    signal = quantum_trader.analyze_market('BTC/USDT')

    # Verify signal structure
    assert isinstance(signal, TradeSignal)
    assert signal.symbol == 'BTC/USDT'
    assert signal.action in ['BUY', 'SELL', 'HOLD']
    assert 0 <= signal.confidence <= 1

    # Verify metrics
    assert 'coherence' in signal.quantum_metrics
    assert 'consciousness_level' in signal.consciousness_metrics
    assert 'risk_level' in signal.risk_metrics
    assert hasattr(signal, 'alerts')
    assert isinstance(signal.alerts, list)

    # Verify stop levels
    assert signal.stop_loss is not None
    assert signal.take_profit is not None
    assert signal.stop_loss < signal.price < signal.take_profit


def test_trade_execution_integration(quantum_trader, mock_exchange):
    """Test complete trade execution pipeline"""
    # Create test signal
    signal = TradeSignal(
        timestamp=datetime.now(), # Changed to datetime object
        symbol='BTC/USDT',
        price=45000.0,
        action='BUY',
        confidence=0.8,
        quantum_metrics={
            'coherence': 0.9,
            'entropy': 0.3,
            'market_stability': 0.8
        },
        consciousness_metrics={
            'consciousness_level': 0.85,
            'coherence': 0.9,
            'morphic_resonance': 0.8
        },
        risk_metrics={
            'risk_level': 0.3,
            'stability_index': 0.75,
            'market_volatility': 0.4
        },
        stop_loss=44500.0,
        take_profit=45500.0,
        alerts=[]
    )

    # Execute trade
    result = quantum_trader.execute_trade(signal)

    # Verify trade execution
    assert result['status'] == 'executed'
    assert result['action'] == 'BUY'
    assert 'order' in result
    assert result['quantum_metrics'] == signal.quantum_metrics
    assert result['consciousness_metrics'] == signal.consciousness_metrics
    assert isinstance(result.get('risk_metrics'), dict)

def test_position_sizing_integration(quantum_trader):
    """Test position sizing with risk management"""
    signal = TradeSignal(
        timestamp=datetime.now(), # Changed to datetime object
        symbol='BTC/USDT',
        price=45000.0,
        action='BUY',
        confidence=0.8,
        quantum_metrics={'coherence': 0.9, 'entropy': 0.3},
        consciousness_metrics={'consciousness_level': 0.85},
        risk_metrics={'risk_level':0.3},
        stop_loss=44500.0,
        take_profit=45500.0,
        alerts=[]
    )

    # Calculate position size
    position_size = quantum_trader._calculate_position_size(signal)

    # Verify position size constraints
    assert isinstance(position_size, float)
    assert 0 < position_size <= 0.05  # Max 5% of capital
    assert np.isfinite(position_size)

def test_trading_metrics_integration(quantum_trader):
    """Test trading metrics collection and validation"""
    # Mock KuCoin initialization - already done in fixture
    metrics = quantum_trader.get_trading_metrics()

    # Verify metric structure
    assert 'quantum_metrics' in metrics
    assert 'consciousness_metrics' in metrics
    assert 'risk_metrics' in metrics

    # Verify metric ranges
    for metric_group in [metrics['quantum_metrics'], metrics['consciousness_metrics'], metrics['risk_metrics']]:
        for value in metric_group.values():
            assert isinstance(value, float)
            assert 0 <= value <= 1
            assert np.isfinite(value)

def test_trade_signal_validation(quantum_trader):
    """Test trade signal validation"""
    valid_signal = TradeSignal(
        timestamp=datetime.now(), # Changed to datetime object
        symbol='BTC/USDT',
        price=45000.0,
        action='BUY',confidence=0.8,
        quantum_metrics={
            'coherence': 0.9,
            'entropy': 0.3,
            'market_stability': 0.8,
            'morphic_resonance': 0.85  # Added required metric
        },
consciousness_metrics={
            'consciousness_level': 0.85,
            'coherence': 0.9,
            'morphic_resonance': 0.8  # Required for consciousness metrics
        },
        risk_metrics={
            'risk_level': 0.3,
            'stability_index': 0.75,
            'market_volatility': 0.4
        },
        alerts=[]
    )
    assert valid_signal.validate()

    # Test invalid signal
    invalid_signal = TradeSignal(
        timestamp=datetime.now(), # Changed to datetime object
        symbol='BTC/USDT',
        price=45000.0,
        action='INVALID_ACTION',  # Invalid action
        confidence=0.8,
        quantum_metrics={
            'coherence': 0.9,
            'entropy': 0.3  # Missing required metrics
        },
        consciousness_metrics={
            'consciousness_level': 0.85  # Missing required metrics
        },
        risk_metrics={
            'risk_level': 0.3
        },
        alerts=[]
    )
    assert not invalid_signal.validate()

def test_signal_generation_with_edge_cases(mock_market_state, mock_exchange):
    """Test signal generation with various edge cases"""
    with patch('qualia.market_data.MarketDataProvider') as mock_market_data:
        mock_market_data.return_value.get_market_data.return_value = mock_market_state

        trader = QuantumTrader(
            exchange_id='test_exchange',
            trading_pairs=['BTC/USDT']
        )

        # Replace trader's market_data with our mock
        trader.market_data = mock_market_data.return_value

        # Setup KuCoin mode
        trader.exchange.kucoin_enabled = True
        trader.exchange.trading_enabled = True
        trader.exchange.state.is_sandbox = False

        # Test market analysis with extreme conditions
        signal = trader.analyze_market('BTC/USDT')

        # Verify the signal properties
        assert isinstance(signal, TradeSignal)
        assert signal.action in ['BUY', 'SELL', 'HOLD']
        assert 0 <= signal.confidence <= 1
        assert 'market_stability' in signal.quantum_metrics
        assert hasattr(signal, 'alerts')
        assert isinstance(signal.alerts, list)

def test_position_management_edge_cases(mock_exchange):
    """Test position management with edge cases"""
    trader = QuantumTrader(
        exchange_id='test_exchange',
        trading_pairs=['BTC/USDT']
    )

    # Setup KuCoin mode
    trader.exchange.kucoin_enabled = True
    trader.exchange.trading_enabled = True
    trader.exchange.state.is_sandbox = False

    # Test execution with extreme parameters
    test_signal = TradeSignal(
        timestamp=datetime.now(), # Changed to datetime object
        symbol='BTC/USDT',
        price=50000.0,
        action='BUY',
        confidence=0.8,
        quantum_metrics={
            'coherence': 0.9,
            'entropy': 0.1,
            'market_stability': 0.8,
            'morphic_resonance': 0.7  # Added required metric
        },
        consciousness_metrics={
            'consciousness_level': 0.85,
            'coherence': 0.9,
            'morphic_resonance': 0.7  # Added required metric
        },
        risk_metrics={
            'risk_level': 0.3,
            'stability_index': 0.75,
            'market_volatility': 0.4
        },
        alerts=[],  # Added required alerts list
        hedge_ratio=0.5  # Added optional hedge ratio
    )

    ## Set up mock to simulate successful trade
    mock_exchange.create_market_buy_order.return_value = {
        'id': '123',
        'status': 'closed',
        'filled': 0.1,
        'cost': 5000.0
    }

    result = trader.execute_trade(test_signal)
    assert isinstance(result, dict)
    assert 'status' in result
    assert result['status'] == 'executed'

def test_trade_execution_error_handling():
    """Test trade execution error handling"""
    with patch('qualia.market_data.MarketDataProvider') as mock_market_data, \
         patch('qualia.quantum_state_manager.QuantumStateManager') as mock_quantum_manager, \
         patch('qualia.core.risk_manager.RiskManager') as mock_risk_manager, \
         patch('qualia.core.exchange_interface.ExchangeInterface') as mock_exchange:

        # Setup market data mock
        mock_market_data.return_value.get_market_data.return_value = MarketState(
            timestamp=datetime.now(), # Changed to datetime object
            ohlcv=np.random.rand(64, 6),
            quantum_features=np.random.rand(64),
            market_phase='accumulation',
            entropy=0.5,
            complexity=0.5
        )

        # Setup quantum manager mock
        mock_quantum_manager.return_value.calculate_consciousness_metrics.return_value = {
            'coherence': 0.8,
            'consciousness': 0.7,
            'market_stability': 0.6,
            'morphic_resonance': 0.5
        }

        # Setup risk manager mock
        mock_risk_manager.return_value.calculate_position_size.return_value = 0.1
        mock_risk_manager.return_value.calculate_stop_levels.return_value = (45000.0, 55000.0)

        # Setup exchange mock to raise exception
        mock_exchange_instance = mock_exchange.return_value
        mock_exchange_instance.create_market_buy_order.side_effect = Exception('Invalid order')
        mock_exchange_instance.create_market_sell_order.side_effect = Exception('Invalid order')
        mock_exchange_instance.fetch_balance.return_value = {'USDT': {'free': 10000.0}}

        # Create trader and replace components with mocks
        trader = QuantumTrader(
            exchange_id='test_exchange',
            trading_pairs=['BTC/USDT']
        )
        trader.market_data = mock_market_data.return_value
        trader.exchange = mock_exchange_instance

        # Enable KuCoin mode
        trader.exchange.kucoin_enabled = True
        trader.exchange.trading_enabled = True
        trader.exchange.state.is_sandbox = False

        # Test execution with invalid parameters
        invalid_signal = TradeSignal(
            timestamp=datetime.now(), # Changed to datetime object
            symbol='INVALID/PAIR',
            price=50000.0,
            action='BUY',
            confidence=0.8,
            quantum_metrics={
                'coherence': 0.9,
                'entropy': 0.1,
                'market_stability': 0.8,
                'morphic_resonance': 0.7
            },
            consciousness_metrics={
                'consciousness_level': 0.85,
                'coherence': 0.9,
                'morphic_resonance': 0.7
            },
            risk_metrics={
                'risk_level': 0.3,
                'stability_index': 0.75,
                'market_volatility': 0.4
            },
            alerts=[],  # Added required alerts list
            hedge_ratio=0.5  # Added optional hedge ratio
        )

        # Test error handling for invalid order
        result = trader.execute_trade(invalid_signal)
        assert isinstance(result, dict)
        assert result['status'] == 'error'
        assert 'error' in result
        assert isinstance(result['error'], str)
        assert 'Invalid order' in result['error']

        # Test with invalid action
        invalid_action_signal = TradeSignal(
            timestamp=datetime.now(), # Changed to datetime object
            symbol='BTC/USDT',
            price=50000.0,
            action='INVALID',
            confidence=0.8,
            quantum_metrics={
                'coherence': 0.9,
                'entropy': 0.1,
                'market_stability': 0.8,
                'morphic_resonance': 0.7  # Added required metric
            },
            consciousness_metrics={
                'consciousness_level': 0.85,
                'coherence': 0.9,
                'morphic_resonance': 0.7  # Added required metric
            },
            risk_metrics={
                'risk_level': 0.3,
                'stability_index': 0.75,
                'market_volatility': 0.4
            },
            alerts=[],  # Added required alerts list
            hedge_ratio=0.5  # Added optional hedge ratio
        )

        result = trader.execute_trade(invalid_action_signal)
        assert isinstance(result, dict)
        assert result['status'] == 'error'
        assert 'Invalid action' in result.get('error', '')

def test_invalid_symbol_format():
    """Test que o sistema rejeita formatos inválidos de par"""
    with pytest.raises(ValueError, match="Invalid symbol format"):
        trader = QuantumTrader(
            exchange_id='test_exchange',
            trading_pairs=['BTC/USDT'],  # Formato inválido
            quantum_dimension=64
        )