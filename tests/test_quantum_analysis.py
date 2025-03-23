"""
Unit tests for quantum analysis functionality with focus on error handling
"""
import pytest
import numpy as np
import json
from qualia.analysis.quantum_analysis import QuantumAnalyzer, QuantumState

@pytest.fixture
def quantum_analyzer():
    """Create a quantum analyzer instance for testing"""
    return QuantumAnalyzer(
        dimension=64,
        planck_constant=1.0,
        memory_capacity=1000,
        phi=1.618033988749895
    )

@pytest.fixture
def mock_market_data():
    """Create mock market data for testing"""
    return np.random.rand(64, 6)  # 64 time points, 6 features (OHLCV)

def test_quantum_analyzer_initialization(quantum_analyzer):
    """Test quantum analyzer initialization"""
    assert quantum_analyzer.dimension == 64
    assert quantum_analyzer.planck_constant == 1.0
    assert quantum_analyzer.phi == 1.618033988749895
    assert quantum_analyzer.epsilon == 1e-10

    # Test operator initialization
    assert 'field' in quantum_analyzer.operators
    assert 'morphic' in quantum_analyzer.operators
    assert 'emergence' in quantum_analyzer.operators

    # Test operators dimensions
    for op in quantum_analyzer.operators.values():
        assert op.shape == (64, 64)
        assert np.allclose(op @ op.conj().T, op @ op.conj().T)  # Hermiticity check

def test_complex_number_serialization(quantum_analyzer):
    """Test proper serialization of complex numbers"""
    # Create a complex quantum state
    complex_state = np.random.rand(64) + 1j * np.random.rand(64)

    # Test serialization of single complex number
    complex_num = 1 + 2j
    serialized = quantum_analyzer._serialize_complex(complex_num)
    assert isinstance(serialized, dict)
    assert all(k in serialized for k in ['real', 'imag', 'magnitude', 'phase'])
    assert serialized['real'] == 1.0
    assert serialized['imag'] == 2.0

    # Test serialization of complex array
    serialized_array = quantum_analyzer._serialize_complex(complex_state)
    assert isinstance(serialized_array, dict)
    assert all(k in serialized_array for k in ['real', 'imag', 'magnitude', 'phase'])

    # Verify JSON serialization works
    try:
        json.dumps(serialized_array)
        json_serializable = True
    except:
        json_serializable = False
    assert json_serializable

def test_matrix_dimension_validation(quantum_analyzer):
    """Test matrix dimension validation in quantum operations"""
    # Test with correct dimensions
    valid_state = np.random.rand(64, 64) + 1j * np.random.rand(64, 64)
    valid_state = valid_state / np.trace(valid_state)  # Normalize

    metrics = quantum_analyzer.calculate_metrics(valid_state)
    assert isinstance(metrics, dict)
    assert all(isinstance(v, float) for v in metrics.values())
    assert all(0 <= v <= 1 for v in metrics.values())

    # Test with incorrect dimensions
    invalid_state = np.random.rand(32, 32)  # Wrong dimension
    metrics = quantum_analyzer.calculate_metrics(invalid_state)
    assert isinstance(metrics, dict)  # Should return safe metrics
    assert 'coherence' in metrics
    assert 'entropy' in metrics

def test_quantum_metrics_json_serialization(quantum_analyzer):
    """Test that quantum metrics are JSON serializable"""
    metrics = quantum_analyzer.get_quantum_metrics()

    try:
        serialized = json.dumps(metrics)
        deserialized = json.loads(serialized)
        json_serializable = True
    except:
        json_serializable = False

    assert json_serializable
    assert isinstance(deserialized, dict)
    assert all(isinstance(v, float) for v in deserialized.values())
    assert all(0 <= v <= 1 for v in deserialized.values())

def test_error_handling_invalid_quantum_state(quantum_analyzer):
    """Test error handling for invalid quantum states"""
    # Test with None state
    metrics = quantum_analyzer.calculate_metrics(None)
    assert isinstance(metrics, dict)
    assert all(0 <= v <= 1 for v in metrics.values())

    # Test with invalid complex values
    invalid_state = np.full((64,), np.inf + 1j * np.inf)
    metrics = quantum_analyzer.calculate_metrics(invalid_state)
    assert isinstance(metrics, dict)
    assert all(0 <= v <= 1 for v in metrics.values())

    # Test with NaN values
    nan_state = np.full((64,), np.nan + 1j * np.nan)
    metrics = quantum_analyzer.calculate_metrics(nan_state)
    assert isinstance(metrics, dict)
    assert all(0 <= v <= 1 for v in metrics.values())
    assert all(np.isfinite(v) for v in metrics.values())

def test_holographic_memory_integration(quantum_analyzer, mock_market_data):
    """Test integration with holographic memory"""
    state = quantum_analyzer.evolve_state(mock_market_data)
    assert isinstance(state, QuantumState)
    assert state.state_vector.shape == (64,)
    assert state.density_matrix.shape == (64, 64)

    # Test pattern storage and retrieval
    pattern = state.state_vector
    quantum_analyzer.holographic_memory.store_pattern(pattern)

    # Verify metrics calculation with stored pattern
    metrics = quantum_analyzer.calculate_metrics(pattern)
    assert isinstance(metrics, dict)
    assert all(isinstance(v, float) for v in metrics.values())
    assert all(0 <= v <= 1 for v in metrics.values())

def test_quantum_state_evolution(quantum_analyzer, mock_market_data):
    """Test quantum state evolution with market data"""
    evolved_state = quantum_analyzer.evolve_state(mock_market_data)
    
    assert isinstance(evolved_state, QuantumState)
    assert evolved_state.state_vector.shape == (64,)
    assert evolved_state.density_matrix.shape == (64, 64)
    assert 0 <= evolved_state.coherence <= 1
    assert 0 <= evolved_state.entropy <= np.log2(64)
    assert np.all(np.isfinite(evolved_state.density_matrix))
    assert np.all(np.isfinite(evolved_state.state_vector))

def test_quantum_metrics_calculation(quantum_analyzer, mock_market_data):
    """Test quantum metrics calculation"""
    metrics = quantum_analyzer.calculate_metrics(mock_market_data)
    
    required_metrics = {
        'coherence', 'entropy', 'market_stability',
        'quantum_alignment', 'entanglement', 'morphic_resonance'
    }
    
    assert all(metric in metrics for metric in required_metrics)
    assert all(isinstance(metrics[metric], float) for metric in required_metrics)
    assert all(0 <= metrics[metric] <= 1 for metric in required_metrics)
    assert all(np.isfinite(metrics[metric]) for metric in required_metrics)

def test_edge_case_handling(quantum_analyzer):
    """Test handling of edge cases in quantum analysis"""
    # Test with zero state
    zero_state = np.zeros(64)
    metrics = quantum_analyzer.calculate_metrics(zero_state)
    assert all(0 <= metrics[key] <= 1 for key in metrics)
    assert all(np.isfinite(metrics[key]) for key in metrics)
    
    # Test with very small values
    small_state = np.full(64, 1e-10)
    metrics = quantum_analyzer.calculate_metrics(small_state)
    assert all(0 <= metrics[key] <= 1 for key in metrics)
    assert all(np.isfinite(metrics[key]) for key in metrics)
    
    # Test with very large values
    large_state = np.full(64, 1e10)
    metrics = quantum_analyzer.calculate_metrics(large_state)
    assert all(0 <= metrics[key] <= 1 for key in metrics)
    assert all(np.isfinite(metrics[key]) for key in metrics)

def test_current_metrics(quantum_analyzer):
    """Test getting current metrics"""
    metrics = quantum_analyzer.get_current_metrics()
    
    required_metrics = {
        'coherence', 'entropy', 'market_stability',
        'consciousness', 'entanglement', 'morphic_resonance'
    }
    
    assert all(metric in metrics for metric in required_metrics)
    assert all(isinstance(metrics[metric], float) for metric in required_metrics)
    assert all(0 <= metrics[metric] <= 1 for metric in required_metrics)
    assert all(np.isfinite(metrics[metric]) for metric in required_metrics)

def test_quantum_resonance_calculation(quantum_analyzer):
    """Test quantum resonance calculation"""
    state = np.random.randn(64) + 1j * np.random.randn(64)
    state = state / np.linalg.norm(state)
    resonance = quantum_analyzer._calculate_resonance(np.outer(state, state.conj()))
    
    assert isinstance(resonance, float)
    assert 0 <= resonance <= 1
    assert np.isfinite(resonance)

def test_entanglement_calculation(quantum_analyzer):
    """Test quantum entanglement calculation"""
    # Create a maximally entangled state
    state = np.zeros((64, 64), dtype=complex)
    for i in range(0, 64, 2):
        state[i, i] = 1/np.sqrt(2)
        state[i+1, i+1] = 1/np.sqrt(2)
    
    entanglement = quantum_analyzer._calculate_entanglement(state)
    
    assert isinstance(entanglement, float)
    assert 0 <= entanglement <= 1
    assert np.isfinite(entanglement)