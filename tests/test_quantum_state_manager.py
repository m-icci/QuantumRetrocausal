"""
Enhanced test suite for QuantumStateManager with comprehensive coverage of edge cases and error handling
"""
import numpy as np
import pytest
from datetime import datetime
from qualia.quantum_state_manager import QuantumStateManager, CONSCIOUSNESS_THRESHOLD

@pytest.fixture
def quantum_manager():
    """Fixture for creating a quantum state manager instance"""
    return QuantumStateManager(dimension=4)

def test_quantum_state_manager_initialization(quantum_manager):
    """Test proper initialization of QuantumStateManager"""
    # Test basic properties
    assert quantum_manager.dimension == 4
    assert quantum_manager.epsilon > 0
    assert quantum_manager.beta > 0

    # Test density matrix properties
    assert quantum_manager.rho.shape == (4, 4)
    assert np.allclose(quantum_manager.rho.conj().T, quantum_manager.rho)  # Hermitian
    assert np.isclose(np.trace(quantum_manager.rho), 1.0, atol=1e-6)  # Normalized

    # Test operator shapes
    operators = {
        'hamiltonian': quantum_manager.hamiltonian,
        'morphic': quantum_manager.M,
        'integration': quantum_manager.I,
        'consciousness': quantum_manager.C,
        'coherence': quantum_manager.CI
    }

    for name, op in operators.items():
        assert op.shape == (4, 4), f"{name} operator has wrong shape"
        assert np.all(np.isfinite(op)), f"{name} operator has non-finite values"
        if name in ['hamiltonian', 'consciousness', 'coherence']:
            assert np.allclose(op, op.conj().T), f"{name} operator not Hermitian"

def test_create_consciousness_hamiltonian(quantum_manager):
    """Test creation of consciousness Hamiltonian"""
    h = quantum_manager._create_consciousness_hamiltonian()

    # Test shape and Hermiticity
    assert h.shape == (4, 4)
    assert np.allclose(h, h.conj().T)

    # Test normalization
    trace_norm = np.abs(np.trace(h @ h.conj().T))
    assert np.isclose(trace_norm, 1.0, atol=1e-6)

    # Test consciousness coupling
    eigenvals = np.linalg.eigvalsh(h)
    assert np.all(np.isfinite(eigenvals))

def test_create_morphic_operator(quantum_manager):
    """Test creation of Morphic operator"""
    m = quantum_manager._create_morphic_operator()

    # Test shape and numerical properties
    assert m.shape == (4, 4)
    assert np.all(np.isfinite(m))

    # Test normalization
    norm = np.sqrt(np.trace(m @ m.conj().T))
    assert np.isclose(norm, 1.0, atol=1e-6)

def test_create_integration_operator(quantum_manager):
    """Test creation of Integration operator"""
    i = quantum_manager._create_integration_operator()

    # Test shape and numerical properties
    assert i.shape == (4, 4)
    assert np.all(np.isfinite(i))

    # Test normalization
    norm = np.sqrt(np.sum(np.abs(i)**2))
    assert np.isclose(norm, 1.0, atol=1e-6)

def test_create_consciousness_operator(quantum_manager):
    """Test creation of Consciousness operator"""
    c = quantum_manager._create_consciousness_operator()

    # Test shape and Hermiticity
    assert c.shape == (4, 4)
    assert np.allclose(c, c.conj().T)

    # Test normalization
    norm = np.sqrt(np.sum(np.abs(c)**2))
    assert np.isclose(norm, 1.0, atol=1e-6)

def test_create_coherence_operator(quantum_manager):
    """Test creation of Coherence operator"""
    ci = quantum_manager._create_coherence_operator()

    # Test shape and numerical properties
    assert ci.shape == (4, 4)
    assert np.all(np.isfinite(ci))

    # Test diagonal elements decay
    diag = np.diag(ci)
    assert np.all(diag[:-1] >= diag[1:])  # Check monotonic decay

def test_evolve_quantum_state(quantum_manager):
    """Test quantum state evolution"""
    initial_state = quantum_manager.rho.copy()

    # Evolution should maintain trace = 1
    quantum_manager.evolve_quantum_state(dt=0.01)
    assert np.isclose(np.trace(quantum_manager.rho), 1.0, atol=1e-6)

    # Evolution should change the state
    assert not np.allclose(quantum_manager.rho, initial_state)

    # State should remain Hermitian
    assert np.allclose(quantum_manager.rho, quantum_manager.rho.conj().T)

def test_calculate_consciousness_metrics(quantum_manager):
    """Test calculation of consciousness metrics"""
    metrics = quantum_manager.calculate_consciousness_metrics()

    expected_metrics = {
        'entropy',
        'coherence',
        'consciousness',
        'morphic_resonance',
        'integration_index',
        'phi_coupling'
    }

    # Check metrics completeness and ranges
    assert set(metrics.keys()) == expected_metrics
    for key, value in metrics.items():
        if key != 'phi_coupling':
            assert 0 <= value <= 1, f"{key} should be between 0 and 1"
        assert isinstance(value, float)

def test_matrix_regularization(quantum_manager):
    """Test matrix regularization functionality"""
    # Test with random matrix
    test_matrix = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
    reg_matrix = quantum_manager._regularize_matrix(test_matrix)

    # Check properties
    assert reg_matrix.shape == (4, 4)
    assert np.allclose(reg_matrix, reg_matrix.conj().T)

    # Check positive definiteness
    eigenvals = np.linalg.eigvalsh(reg_matrix)
    assert np.all(eigenvals >= -1e-10)

def test_error_handling(quantum_manager):
    """Test error handling and recovery"""
    # Test invalid state handling
    quantum_manager.rho = np.zeros((4, 4))
    quantum_manager.evolve_quantum_state(dt=0.01)

    # Should recover to valid state
    assert np.isclose(np.trace(quantum_manager.rho), 1.0, atol=1e-6)
    assert np.allclose(quantum_manager.rho, quantum_manager.rho.conj().T)

def test_get_state_serialization(quantum_manager):
    """Test state serialization and retrieval"""
    state_data = quantum_manager.get_state()

    assert 'density_matrix' in state_data
    assert 'metrics' in state_data
    assert 'timestamp' in state_data

    density_matrix = state_data['density_matrix']
    assert 'real' in density_matrix
    assert 'imag' in density_matrix
    assert 'shape' in density_matrix
    assert density_matrix['shape'] == [4, 4]

@pytest.mark.parametrize("dimension", [2, 4, 8])
def test_different_dimensions(dimension):
    """Test QuantumStateManager with different dimensions"""
    manager = QuantumStateManager(dimension=dimension)

    # Test initialization
    assert manager.rho.shape == (dimension, dimension)
    assert np.isclose(np.trace(manager.rho), 1.0, atol=1e-6)

    # Test evolution
    manager.evolve_quantum_state(dt=0.01)
    metrics = manager.calculate_consciousness_metrics()

    # Verify metrics
    assert all(isinstance(v, float) for v in metrics.values())
    assert all(0 <= v <= 1 for v in metrics.values() if v != metrics['phi_coupling'])


def test_quantum_state_manager_initialization():
    """Test proper initialization of QuantumStateManager"""
    dimension = 64
    manager = QuantumStateManager(dimension=dimension)
    
    # Test basic properties
    assert manager.dimension == dimension
    assert manager.epsilon > 0
    assert manager.beta > 0
    
    # Test density matrix properties
    assert manager.rho.shape == (dimension, dimension)
    assert np.allclose(manager.rho.conj().T, manager.rho)  # Hermitian
    assert np.isclose(np.trace(manager.rho), 1.0, atol=1e-6)  # Normalized
    
    # Test operator shapes
    operators = [
        manager.hamiltonian,
        manager.M,
        manager.I,
        manager.C,
        manager.CI
    ]
    for op in operators:
        assert op.shape == (dimension, dimension)

def test_operator_validation():
    """Test validation of quantum operators"""
    manager = QuantumStateManager(dimension=4)
    assert manager._validate_operators()
    
    # Test with invalid operator
    original_hamiltonian = manager.hamiltonian.copy()
    manager.hamiltonian = np.ones((4, 4)) * float('inf')
    assert not manager._validate_operators()
    
    # Restore valid state
    manager.hamiltonian = original_hamiltonian
    assert manager._validate_operators()

def test_quantum_state_evolution():
    """Test quantum state evolution"""
    manager = QuantumStateManager(dimension=4)
    initial_state = manager.rho.copy()
    
    # Evolution should maintain trace = 1
    manager.evolve_quantum_state(dt=0.01)
    assert np.isclose(np.trace(manager.rho), 1.0, atol=1e-6)
    
    # Evolution should change the state
    assert not np.allclose(manager.rho, initial_state)
    
    # State should remain Hermitian
    assert np.allclose(manager.rho, manager.rho.conj().T)

def test_consciousness_metrics():
    """Test calculation of consciousness metrics"""
    manager = QuantumStateManager(dimension=4)
    metrics = manager.calculate_consciousness_metrics()
    
    expected_metrics = {
        'entropy',
        'coherence',
        'consciousness',
        'morphic_resonance',
        'integration_index',
        'phi_coupling'
    }
    
    # Check metrics completeness
    assert set(metrics.keys()) == expected_metrics
    
    # Check metrics ranges
    for key, value in metrics.items():
        if key != 'phi_coupling':
            assert 0 <= value <= 1, f"{key} should be between 0 and 1"
        assert isinstance(value, float)

def test_error_handling():
    """Test error handling and safe state initialization"""
    manager = QuantumStateManager(dimension=4)
    
    # Test safe state initialization
    manager._initialize_safe_state()
    assert np.allclose(manager.rho, np.eye(4) / 4)
    
    # Test recovery from invalid state
    manager.rho = np.zeros((4, 4))
    manager.evolve_quantum_state(dt=0.01)
    assert np.isclose(np.trace(manager.rho), 1.0, atol=1e-6)

def test_complex_number_conversion(quantum_manager):
    """Test safe conversion of complex numbers"""
    # Test with pure real number
    assert quantum_manager._safe_complex_to_real(1.0) == 1.0

    # Test with complex number
    complex_val = 1 + 2j
    real_val = quantum_manager._safe_complex_to_real(complex_val)
    assert isinstance(real_val, float)
    assert np.isclose(real_val, complex_val.real)

    # Test with invalid input
    assert quantum_manager._safe_complex_to_real(None) == 0.5

@pytest.mark.parametrize("dimension", [2, 4, 8, 16])
def test_different_dimensions(dimension):
    """Test QuantumStateManager with different dimensions"""
    manager = QuantumStateManager(dimension=dimension)
    
    # Basic initialization checks
    assert manager.rho.shape == (dimension, dimension)
    assert np.isclose(np.trace(manager.rho), 1.0, atol=1e-6)
    
    # Evolution check
    manager.evolve_quantum_state(dt=0.01)
    metrics = manager.calculate_consciousness_metrics()
    
    # Verify metrics still work
    assert all(isinstance(v, float) for v in metrics.values())


def test_invalid_dimension_handling():
    """Test handling of invalid dimensions"""
    with pytest.raises(ValueError):
        QuantumStateManager(dimension=0)

    with pytest.raises(ValueError):
        QuantumStateManager(dimension=-1)

def test_epsilon_handling(quantum_manager):
    """Test epsilon threshold handling in matrix operations"""
    # Test with near-zero matrix
    test_matrix = np.zeros((4, 4)) + 1e-12
    reg_matrix = quantum_manager._regularize_matrix(test_matrix)

    # Should be regularized to avoid numerical instability
    assert np.all(np.abs(reg_matrix) >= quantum_manager.epsilon)
    assert np.allclose(reg_matrix, reg_matrix.conj().T)

def test_consciousness_metrics_edge_cases(quantum_manager):
    """Test consciousness metrics calculation with edge cases"""
    # Set density matrix to near-zero
    quantum_manager.rho = np.zeros((4, 4)) + 1e-12
    metrics = quantum_manager.calculate_consciousness_metrics()

    # Check metrics stay within valid ranges
    for key, value in metrics.items():
        if key != 'phi_coupling':
            assert 0 <= value <= 1, f"{key} should be between 0 and 1"
            assert value > 0, f"{key} should be greater than 0"

    # Test with invalid density matrix
    quantum_manager.rho = np.ones((4, 4)) * float('inf')
    metrics = quantum_manager.calculate_consciousness_metrics()
    assert all(isinstance(v, float) for v in metrics.values())

def test_quantum_evolution_stability(quantum_manager):
    """Test stability of quantum evolution under various conditions"""
    # Evolution with normal state
    initial_state = quantum_manager.rho.copy()
    quantum_manager.evolve_quantum_state(dt=0.01)
    assert np.isclose(np.trace(quantum_manager.rho), 1.0, atol=1e-6)

    # Evolution with near-zero state
    quantum_manager.rho = np.zeros((4, 4)) + 1e-12
    quantum_manager.evolve_quantum_state(dt=0.01)
    assert np.isclose(np.trace(quantum_manager.rho), 1.0, atol=1e-6)

    # Evolution with large time step
    quantum_manager.evolve_quantum_state(dt=10.0)
    assert np.isclose(np.trace(quantum_manager.rho), 1.0, atol=1e-6)

def test_operator_validation_edge_cases(quantum_manager):
    """Test validation of quantum operators under edge conditions"""
    # Test with invalid Hamiltonian
    original_hamiltonian = quantum_manager.hamiltonian.copy()
    quantum_manager.hamiltonian = np.ones((4, 4)) * float('inf')
    assert not quantum_manager._validate_operators()

    # Test with non-Hermitian operator
    quantum_manager.hamiltonian = np.random.randn(4, 4)  # Non-Hermitian
    assert not quantum_manager._validate_operators()

    # Restore valid state and verify
    quantum_manager.hamiltonian = original_hamiltonian
    assert quantum_manager._validate_operators()

def test_metrics_history_tracking(quantum_manager):
    """Test metrics history initialization and updates"""
    # Check initial history structure
    assert all(isinstance(quantum_manager.metrics_history[k], list) for k in [
        'coherence', 'entropy', 'consciousness', 'morphic_resonance',
        'integration_index', 'phi_coupling'
    ])

    # Evolve and check history updates
    for _ in range(3):
        quantum_manager.evolve_quantum_state(dt=0.01)
        metrics = quantum_manager.calculate_consciousness_metrics()

        # Verify all metrics are properly tracked
        for key, value in metrics.items():
            assert isinstance(value, float)
            assert not np.isnan(value)
            assert not np.isinf(value)

def test_state_serialization_robustness(quantum_manager):
    """Test robustness of state serialization"""
    # Test normal state serialization
    state_data = quantum_manager.get_state()
    assert 'density_matrix' in state_data
    assert 'metrics' in state_data
    assert 'timestamp' in state_data

    # Test serialization with invalid state
    quantum_manager.rho = np.ones((4, 4)) * float('inf')
    state_data = quantum_manager.get_state()
    assert state_data['density_matrix'] is None
    assert all(isinstance(v, float) for v in state_data['metrics'].values())

def test_safe_matrix_operations(quantum_manager):
    """Test safe matrix operations with edge cases"""
    # Test matrix logarithm with small eigenvalues
    test_matrix = np.eye(4) * 1e-12
    log_result = quantum_manager._safe_matrix_log(test_matrix)
    assert np.all(np.isfinite(log_result))

    # Test matrix exponential with large values
    test_matrix = np.eye(4) * 100
    exp_result = quantum_manager._safe_matrix_exp(test_matrix)
    assert np.all(np.isfinite(exp_result))

def test_consciousness_threshold_compliance(quantum_manager):
    """Test compliance with consciousness threshold"""
    metrics = quantum_manager.calculate_consciousness_metrics()
    consciousness_value = metrics['consciousness']

    # Value should be normalized between 0 and 1
    assert 0 <= consciousness_value <= 1

    # Test relationship with threshold
    if consciousness_value >= CONSCIOUSNESS_THRESHOLD:
        # Higher consciousness should correlate with higher coherence
        assert metrics['coherence'] > 0.5