"""
Tests for quantum state dimensionality validation with enhanced coverage
"""
import numpy as np
import pytest
from datetime import datetime, timezone
from qualia.quantum_state_manager import QuantumStateManager
from qualia.quantum_trader import QuantumTrader
from qualia.core.market_data import MarketState
from qualia.core.holographic_memory import HolographicMemory, HolographicPattern

def test_quantum_state_dimensions():
    """Test quantum state dimensionality"""
    dimension = 64
    qsm = QuantumStateManager(dimension=dimension)

    # Test density matrix dimensions
    assert qsm.rho.shape == (dimension, dimension), \
        f"Density matrix shape mismatch: expected ({dimension}, {dimension}), got {qsm.rho.shape}"

    # Test operator dimensions
    assert qsm.C.shape == (dimension, dimension), "Consciousness operator dimension mismatch"
    assert qsm.M.shape == (dimension, dimension), "Morphic operator dimension mismatch"
    assert qsm.I.shape == (dimension, dimension), "Integration operator dimension mismatch"

    # Test state normalization
    trace = np.abs(np.trace(qsm.rho))
    np.testing.assert_almost_equal(trace, 1.0, decimal=6, err_msg="State not properly normalized")

def test_matrix_operations_dimensionality():
    """Test matrix operations dimensionality"""
    dimension = 64
    qsm = QuantumStateManager(dimension=dimension)

    # Test consciousness calculation
    consciousness_val = np.abs(np.trace(qsm.C @ qsm.rho))
    assert isinstance(consciousness_val, (float, np.float64)), "Invalid consciousness value type"

    # Test morphic resonance calculation
    resonance_val = np.abs(np.trace(qsm.M @ qsm.rho))
    assert isinstance(resonance_val, (float, np.float64)), "Invalid resonance value type"

    # Test integration calculation
    integration_val = np.abs(np.trace(qsm.I @ qsm.rho))
    assert isinstance(integration_val, (float, np.float64)), "Invalid integration value type"

def test_quantum_trader_dimensions():
    """Test quantum trader dimensionality consistency"""
    trader = QuantumTrader()

    # Test quantum metrics calculation
    metrics = trader.get_quantum_metrics()
    assert isinstance(metrics, dict), "Metrics should be a dictionary"
    assert 'coherence' in metrics, "Coherence metric missing"
    assert 'entropy' in metrics, "Entropy metric missing"
    assert 'consciousness' in metrics, "Consciousness metric missing"
    assert all(isinstance(v, float) for v in metrics.values()), "All metric values should be floats"
    assert all(0 <= v <= 1 for v in metrics.values()), "All metrics should be normalized between 0 and 1"

def test_risk_metrics_immutability():
    """Test risk metrics immutability"""
    trader = QuantumTrader()

    # Create a valid market state
    mock_state = MarketState(
        timestamp=datetime.now(timezone.utc).timestamp(),
        ohlcv=np.random.rand(10, 5),
        quantum_features=np.random.rand(64),
        market_phase="accumulation",
        entropy=0.5,
        complexity=0.5
    )

    # Get initial risk metrics
    risk_metrics = trader.calculate_risk_metrics(mock_state)
    assert isinstance(risk_metrics, dict), "Risk metrics should be a dictionary"

    # Verify expected structure
    assert 'risk_level' in risk_metrics, "Missing risk_level in metrics"
    assert 'stability_index' in risk_metrics, "Missing stability_index in metrics"
    assert 'market_volatility' in risk_metrics, "Missing market_volatility in metrics"
    assert 'quantum_adjusted_risk' in risk_metrics, "Missing quantum_adjusted_risk in metrics"

    # Test value ranges
    assert all(isinstance(v, float) for v in risk_metrics.values()), "All metrics should be floats"
    assert all(0 <= v <= 1 for v in risk_metrics.values()), "All metrics should be between 0 and 1"

def test_enhanced_risk_validation():
    """Test enhanced risk validation with edge cases"""
    trader = QuantumTrader()

    # Test with empty market state
    empty_state = MarketState(
        timestamp=datetime.now(timezone.utc).timestamp(),
        ohlcv=np.array([]),
        quantum_features=np.zeros(64),
        market_phase="accumulation",
        entropy=0.5,
        complexity=0.5
    )

    metrics = trader.calculate_risk_metrics(empty_state)
    assert isinstance(metrics, dict), "Should return dictionary"
    assert all(0 <= v <= 1 for v in metrics.values()), "Values should be normalized"

    # Test with None market state
    metrics = trader.calculate_risk_metrics(None)
    assert isinstance(metrics, dict), "Should return dictionary"
    assert all(0 <= v <= 1 for v in metrics.values()), "Values should be normalized"

    # Test with invalid market state
    invalid_state = MarketState(
        timestamp=datetime.now(timezone.utc).timestamp(),
        ohlcv=None,
        quantum_features=None,
        market_phase="invalid",
        entropy=-1,
        complexity=2
    )

    metrics = trader.calculate_risk_metrics(invalid_state)
    assert isinstance(metrics, dict), "Should return dictionary"
    assert all(0 <= v <= 1 for v in metrics.values()), "Values should be normalized"

def test_holographic_memory_dimensions():
    """Test holographic memory dimensionality consistency"""
    memory = HolographicMemory()
    dimension = 64  # Default dimension

    # Test pattern storage and retrieval
    # Create a test pattern
    pattern = np.random.rand(dimension)
    pattern_id = memory.store_pattern(pattern)
    assert pattern_id is not None, "Failed to store pattern"

    # Test pattern retrieval dimensions
    retrieved = memory.retrieve_pattern(pattern_id)
    assert retrieved is not None, "Failed to retrieve pattern"
    assert isinstance(retrieved, HolographicPattern), "Retrieved object is not a HolographicPattern"
    assert retrieved.pattern.shape in [(dimension,), (dimension, 1)], \
        f"Retrieved pattern dimension mismatch: got {retrieved.pattern.shape}"

    # Test dimension validation
    assert memory.validate_dimensions(pattern), "Dimension validation failed for valid pattern"
    assert not memory.validate_dimensions(np.random.rand(dimension + 1)), \
        "Dimension validation passed for invalid pattern"

def test_pattern_storage_validation():
    """Test pattern storage with dimension validation"""
    memory = HolographicMemory()

    # Test valid pattern storage
    valid_pattern = np.random.rand(64)
    pattern_id = memory.store_pattern(valid_pattern)
    assert pattern_id is not None, "Failed to store valid pattern"

    # Test invalid pattern handling
    invalid_pattern = np.random.rand(128)
    with pytest.raises(Exception):
        memory.store_pattern(invalid_pattern)

    # Test operator dimension consistency
    assert memory._validate_operators(), "Operator dimensions mismatch"

def test_quantum_metrics_validation():
    """Test quantum metrics validation and normalization"""
    trader = QuantumTrader()

    # Test metrics with invalid market state
    metrics = trader.get_quantum_metrics()
    required_metrics = {'coherence', 'entropy', 'consciousness', 'market_stability', 'morphic_resonance'}
    assert all(metric in metrics for metric in required_metrics), "Missing required metrics"
    assert all(isinstance(v, float) for v in metrics.values()), "Invalid metric types"
    assert all(0 <= v <= 1 for v in metrics.values()), "Metrics not properly normalized"

def test_consciousness_metrics():
    """Test consciousness metrics calculation"""
    trader = QuantumTrader()
    memory = HolographicMemory()

    # Test consciousness metrics calculation
    mock_state = MarketState(
        timestamp=datetime.now(timezone.utc).timestamp(),
        ohlcv=np.random.rand(10, 5),
        quantum_features=np.random.rand(64),
        market_phase="accumulation",
        entropy=0.5,
        complexity=0.5
    )
    metrics = memory.calculate_consciousness_metrics(mock_state.quantum_features)

    assert isinstance(metrics, dict), "Invalid consciousness metrics format"
    assert 'consciousness_level' in metrics, "Missing consciousness level"
    assert 'coherence' in metrics, "Missing coherence metric"
    assert all(0 <= v <= 1 for v in metrics.values()), "Consciousness metrics not normalized"

def test_quantum_metrics_error_handling():
    """Test quantum metrics error handling"""
    trader = QuantumTrader()

    # Test with None quantum features
    invalid_state = MarketState(
        timestamp=datetime.now(timezone.utc).timestamp(),
        ohlcv=np.random.rand(10, 5),
        quantum_features=None,
        market_phase="accumulation",
        entropy=0.5,
        complexity=0.5
    )

    metrics = trader._calculate_quantum_metrics(invalid_state)
    assert isinstance(metrics, dict), "Should return dict for invalid input"
    assert all(0 <= v <= 1 for v in metrics.values()), "Values should be normalized"

@pytest.mark.parametrize("dimension", [32, 64, 128])
def test_dimension_scaling(dimension):
    """Test system behavior with different dimensions"""
    qsm = QuantumStateManager(dimension=dimension)

    # Test basic properties
    assert qsm.dimension == dimension, f"Dimension mismatch: expected {dimension}, got {qsm.dimension}"
    assert qsm.rho.shape == (dimension, dimension), \
        f"Density matrix shape mismatch for dimension {dimension}"

    # Test operator shapes
    assert qsm.C.shape == (dimension, dimension), \
        f"Consciousness operator shape mismatch for dimension {dimension}"
    assert qsm.M.shape == (dimension, dimension), \
        f"Morphic operator shape mismatch for dimension {dimension}"
    assert qsm.I.shape == (dimension, dimension), \
        f"Integration operator shape mismatch for dimension {dimension}"

    # Calculate metrics to ensure no dimension-related errors
    metrics = qsm._calculate_quantum_metrics()
    assert isinstance(metrics, dict), "Metrics calculation failed"
    assert all(isinstance(v, float) for v in metrics.values()), "Invalid metric types"


def test_error_handling():
    """Test error handling for dimension-related issues"""
    trader = QuantumTrader()
    memory = HolographicMemory()

    # Test invalid pattern retrieval
    assert memory.retrieve_pattern("nonexistent_id") is None

    # Test metrics calculation with invalid state
    metrics = trader.get_quantum_metrics()
    assert all(0 <= v <= 1 for v in metrics.values()), "Metrics should be normalized between 0 and 1"


def test_trader_initialization():
    """Test quantum trader initialization and configuration"""
    # Test with default parameters
    trader = QuantumTrader()
    assert trader.quantum_dimension == 64, "Invalid default quantum dimension"
    assert trader.consciousness_threshold == 0.7, "Invalid default consciousness threshold"

    # Test with custom parameters
    custom_trader = QuantumTrader(
        quantum_dimension=128,
        consciousness_threshold=0.8,
        trading_pairs=['ETH/USDT']
    )
    assert custom_trader.quantum_dimension == 128, "Custom quantum dimension not set"
    assert custom_trader.consciousness_threshold == 0.8, "Custom consciousness threshold not set"
    assert 'ETH/USDT' in custom_trader.trading_pairs, "Trading pairs not properly set"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])