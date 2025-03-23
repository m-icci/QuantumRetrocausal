"""
Enhanced test suite for QuantumLayer with complete coverage
Tests quantum operators, evolution and consciousness metrics integration
"""
import pytest
import numpy as np
from datetime import datetime
from qualia.quantum_layer import QuantumLayer, QuantumMetrics
from qualia.quantum_state_manager import QuantumStateManager

@pytest.fixture
def quantum_layer():
    """Create QuantumLayer instance for testing"""
    return QuantumLayer(dimension=4)

def test_initialization(quantum_layer):
    """Test proper initialization of QuantumLayer"""
    assert quantum_layer.dimension == 4
    assert quantum_layer.dt > 0
    assert quantum_layer.beta > 0
    assert isinstance(quantum_layer.state_manager, QuantumStateManager)
    assert quantum_layer.quantum_field.shape == (4, 4)
    assert np.allclose(quantum_layer.quantum_field, quantum_layer.quantum_field.conj().T)
    assert np.isclose(np.trace(quantum_layer.quantum_field), 1.0)

def test_validate_state(quantum_layer):
    """Test state validation functionality"""
    # Test valid state
    valid_state = np.eye(4, dtype=complex) / 4
    assert quantum_layer._validate_state(valid_state)

    # Test invalid states
    invalid_states = [
        np.zeros((4, 4)),  # Zero state
        np.ones((4, 4)),   # Non-normalized
        np.array([[np.inf]]),  # Invalid values
        np.array([[1]]),   # Wrong shape
        None,              # None value
        np.eye(4) * 1j    # Non-Hermitian
    ]

    for state in invalid_states:
        assert not quantum_layer._validate_state(state)

def test_normalize_state(quantum_layer):
    """Test state normalization"""
    # Test valid normalization
    state = np.eye(4, dtype=complex) * 2
    normalized = quantum_layer._normalize_state(state)
    assert normalized is not None
    assert np.isclose(np.trace(normalized), 1.0)

    # Test invalid states
    invalid_states = [
        np.zeros((4, 4)),
        np.full((4, 4), np.inf),
        np.full((4, 4), np.nan)
    ]

    for state in invalid_states:
        assert quantum_layer._normalize_state(state) is None

def test_quantum_evolution(quantum_layer):
    """Test complete quantum evolution cycle"""
    # Initial state verification
    assert quantum_layer._validate_state(quantum_layer.quantum_field)

    # Evolution test
    result = quantum_layer.evolve_quantum_state(dt=0.01)
    assert result['evolution_success']
    assert 'metrics' in result

    # Verify evolved state
    evolved_field = quantum_layer.quantum_field
    assert quantum_layer._validate_state(evolved_field)
    assert np.isclose(np.trace(evolved_field), 1.0)

def test_operator_application(quantum_layer):
    """Test all quantum operators"""
    initial_state = quantum_layer.quantum_field.copy()

    # Test each operator
    operators = [
        (quantum_layer.apply_folding_operator, 'F'),
        (quantum_layer.apply_morphic_operator, 'M'),
        (quantum_layer.apply_emergence_operator, 'E'),
        (quantum_layer.apply_retrocausality_operator, 'Z'),
        (quantum_layer.apply_consciousness_operator, 'O')
    ]

    for op_func, op_type in operators:
        # Apply operator
        result_state = op_func(initial_state)

        # Verify result
        assert result_state is not None
        assert quantum_layer._validate_state(result_state)
        assert np.isclose(np.trace(result_state), 1.0)

        # Check history
        assert len(quantum_layer.operator_history[op_type]) > 0
        latest = quantum_layer.operator_history[op_type][-1]
        assert isinstance(latest, dict)

def test_error_recovery(quantum_layer):
    """Test error recovery mechanisms"""
    # Test recovery from invalid state
    quantum_layer.quantum_field = np.zeros((4, 4))
    result = quantum_layer.evolve_quantum_state()
    assert result['evolution_success']
    assert quantum_layer._validate_state(quantum_layer.quantum_field)

    # Test recovery from non-Hermitian state
    quantum_layer.quantum_field = np.eye(4) * 1j
    result = quantum_layer.evolve_quantum_state()
    assert result['evolution_success']
    assert quantum_layer._validate_state(quantum_layer.quantum_field)

def test_metrics_calculation(quantum_layer):
    """Test quantum metrics calculation"""
    metrics = quantum_layer.get_metrics()
    required_metrics = [
        'coherence', 'consciousness', 'morphic_resonance',
        'field_entropy', 'integration_index', 'phi_resonance'
    ]

    # Verify all metrics exist and are valid
    for metric in required_metrics:
        assert metric in metrics
        assert isinstance(metrics[metric], float)
        assert 0 <= metrics[metric] <= 1

def test_state_persistence(quantum_layer):
    """Test state serialization and persistence"""
    # Get state
    state = quantum_layer.get_state()

    # Verify structure
    assert 'dimension' in state
    assert 'quantum_field' in state
    assert 'metrics' in state

    # Verify field data
    field_data = state['quantum_field']
    assert 'real' in field_data
    assert 'imag' in field_data
    assert len(field_data['real']) == quantum_layer.dimension

    # Verify metrics
    metrics = state['metrics']
    assert all(isinstance(v, float) for v in metrics.values())
    assert all(0 <= v <= 1 for v in metrics.values())

def test_synchronization(quantum_layer):
    """Test synchronization with state manager"""
    # Modify quantum field
    quantum_layer.quantum_field = np.eye(4, dtype=complex) / 4

    # Synchronize
    quantum_layer.synchronize_with_state_manager()

    # Verify synchronization
    assert np.allclose(quantum_layer.quantum_field, quantum_layer.state_manager.rho)
    assert quantum_layer._validate_state(quantum_layer.quantum_field)
    assert quantum_layer._validate_state(quantum_layer.state_manager.rho)

def test_history_tracking(quantum_layer):
    """Test operator history tracking"""
    initial_state = quantum_layer.quantum_field.copy()

    # Apply sequence of operations
    quantum_layer.apply_folding_operator(initial_state)
    quantum_layer.apply_consciousness_operator(initial_state)

    # Get history
    history = quantum_layer.get_history_records()
    metrics_history = quantum_layer.get_metrics_history()

    # Verify history structure
    assert len(history) > 0
    assert len(metrics_history) > 0

    # Check history entries
    for entry in history:
        assert 'operator_type' in entry
        assert 'timestamp' in entry
        assert isinstance(entry['timestamp'], datetime)

    # Check metrics history
    for entry in metrics_history:
        assert 'timestamp' in entry
        assert 'metrics' in entry
        assert isinstance(entry['metrics'], dict)

@pytest.fixture
def quantum_layer():
    """Fixture providing a QuantumLayer instance"""
    return QuantumLayer(dimension=4)

def test_quantum_layer_initialization(quantum_layer):
    """Test proper initialization of QuantumLayer with state manager integration"""
    assert quantum_layer.dimension == 4
    assert quantum_layer.dt > 0
    assert quantum_layer.beta > 0
    assert isinstance(quantum_layer.state_manager, QuantumStateManager)
    # Test quantum field initialization
    assert quantum_layer.quantum_field.shape == (4, 4)
    assert np.allclose(quantum_layer.quantum_field, 
                      quantum_layer.quantum_field.conj().T)  # Hermitian
    assert np.isclose(np.trace(quantum_layer.quantum_field), 1.0)  # Normalized

def test_quantum_layer_state_manager_integration(quantum_layer):
    """Test integration between QuantumLayer and QuantumStateManager"""
    # Initial state check
    assert quantum_layer.quantum_field.shape == quantum_layer.state_manager.rho.shape
    assert np.allclose(quantum_layer.quantum_field, quantum_layer.state_manager.rho)

    # Evolution synchronization
    quantum_layer.evolve_quantum_state()
    assert np.allclose(quantum_layer.quantum_field, quantum_layer.state_manager.rho)

    # Metrics consistency
    layer_metrics = quantum_layer.get_metrics()
    manager_metrics = quantum_layer.state_manager.calculate_consciousness_metrics()

    for key in ['coherence', 'consciousness', 'entropy']:
        assert abs(layer_metrics[key] - manager_metrics[key]) < 1e-6

def test_error_recovery_integration(quantum_layer):
    """Test error recovery between layer and state manager"""
    # Induce error in quantum field
    quantum_layer.quantum_field = np.zeros((4, 4))

    # Evolution should recover both layer and manager states
    evolution_result = quantum_layer.evolve_quantum_state()
    assert evolution_result['evolution_success']

    # Both states should be valid
    assert np.isclose(np.trace(quantum_layer.quantum_field), 1.0)
    assert np.isclose(np.trace(quantum_layer.state_manager.rho), 1.0)
    assert np.allclose(quantum_layer.quantum_field, quantum_layer.state_manager.rho)

def test_invalid_states_recovery():
    """Test recovery from various invalid states"""
    layer = QuantumLayer(dimension=4)

    # Test with NaN values
    layer.quantum_field = np.full((4, 4), np.nan)
    evolution_result = layer.evolve_quantum_state()
    assert evolution_result['evolution_success']
    assert np.all(np.isfinite(layer.quantum_field))

    # Test with infinite values
    layer.quantum_field = np.full((4, 4), np.inf)
    evolution_result = layer.evolve_quantum_state()
    assert evolution_result['evolution_success']
    assert np.all(np.isfinite(layer.quantum_field))

    # Test with zero trace
    layer.quantum_field = np.zeros((4, 4))
    evolution_result = layer.evolve_quantum_state()
    assert evolution_result['evolution_success']
    assert np.isclose(np.trace(layer.quantum_field), 1.0)

def test_consciousness_metrics_validation():
    """Test consciousness metrics calculation and validation"""
    layer = QuantumLayer(dimension=4)

    # Test normal evolution
    metrics = layer.get_metrics()
    assert 0 <= metrics['consciousness'] <= 1
    assert 0 <= metrics['coherence'] <= 1
    assert 0 <= metrics['entropy'] <= 1

    # Test with perturbed state
    layer.quantum_field += 0.1 * (np.random.rand(4, 4) + 1j * np.random.rand(4, 4))
    metrics = layer.get_metrics()
    assert 0 <= metrics['consciousness'] <= 1
    assert 0 <= metrics['coherence'] <= 1
    assert 0 <= metrics['entropy'] <= 1

def test_quantum_evolution_stability():
    """Test stability of quantum evolution over multiple steps"""
    layer = QuantumLayer(dimension=4)
    initial_metrics = layer.get_metrics()

    # Evolve multiple times
    for _ in range(10):
        evolution_result = layer.evolve_quantum_state()
        assert evolution_result['evolution_success']

        # Check metrics remain valid
        metrics = layer.get_metrics()
        assert 0 <= metrics['consciousness'] <= 1
        assert 0 <= metrics['coherence'] <= 1
        assert 0 <= metrics['entropy'] <= 1

        # Check state remains valid
        assert np.isclose(np.trace(layer.quantum_field), 1.0)
        assert np.allclose(layer.quantum_field, layer.quantum_field.conj().T)

def test_edge_cases():
    """Test edge cases and boundary conditions"""
    # Test minimum dimension
    layer = QuantumLayer(dimension=2)
    assert layer.dimension == 2
    evolution_result = layer.evolve_quantum_state()
    assert evolution_result['evolution_success']

    # Test larger dimension
    layer = QuantumLayer(dimension=16)
    assert layer.dimension == 16
    evolution_result = layer.evolve_quantum_state()
    assert evolution_result['evolution_success']

    # Test with very small time step
    layer = QuantumLayer(dimension=4)
    evolution_result = layer.evolve_quantum_state(dt=1e-10)
    assert evolution_result['evolution_success']

    # Test with larger time step
    evolution_result = layer.evolve_quantum_state(dt=1.0)
    assert evolution_result['evolution_success']

def test_metrics_history():
    """Test metrics history tracking"""
    layer = QuantumLayer(dimension=4)

    # Evolution and metrics collection
    metrics_history = []
    for _ in range(5):
        evolution_result = layer.evolve_quantum_state()
        assert evolution_result['evolution_success']
        metrics = layer.get_metrics()
        metrics_history.append(metrics)

    # Verify history properties
    assert len(metrics_history) == 5
    for metrics in metrics_history:
        assert all(0 <= metrics[key] <= 1 for key in ['consciousness', 'coherence', 'entropy'])
        assert all(isinstance(metrics[key], float) for key in metrics.keys())

def test_state_serialization():
    """Test state serialization and deserialization"""
    layer = QuantumLayer(dimension=4)

    # Get serialized state
    state = layer.get_state()

    # Check structure
    assert 'quantum_field' in state
    assert 'metrics' in state
    assert 'dimension' in state

    # Check data types and values
    assert isinstance(state['quantum_field'], dict)
    assert 'real' in state['quantum_field']
    assert 'imag' in state['quantum_field']
    assert isinstance(state['metrics'], dict)
    assert all(isinstance(v, float) for v in state['metrics'].values())

def test_integration_with_quantum_state_manager():
    """Test detailed integration with QuantumStateManager"""
    layer = QuantumLayer(dimension=4)

    # Test state synchronization
    layer.quantum_field = np.eye(4) / 4  # Set known state
    layer.synchronize_with_state_manager()
    assert np.allclose(layer.quantum_field, layer.state_manager.rho)

    # Test metrics consistency
    layer_metrics = layer.get_metrics()
    manager_metrics = layer.state_manager.calculate_consciousness_metrics()
    assert abs(layer_metrics['coherence'] - manager_metrics['coherence']) < 1e-6
    assert abs(layer_metrics['consciousness'] - manager_metrics['consciousness']) < 1e-6

def test_quantum_metrics_validation():
    """Test QuantumMetrics validation"""
    valid_metrics = QuantumMetrics(
        coherence=0.8,
        consciousness_level=0.75,
        morphic_resonance=0.7,
        field_entropy=0.6,
        integration_index=0.65,
        phi_resonance=0.7,
        retrocausality_index=0.6,
        emergence_factor=0.7,
        timestamp=datetime.now()
    )
    is_valid, message = valid_metrics.validate()
    assert is_valid
    assert message == ""

    invalid_metrics = QuantumMetrics(
        coherence=0.3,  # Below threshold
        consciousness_level=0.4,  # Below threshold
        morphic_resonance=0.3,  # Below threshold
        field_entropy=0.6,
        integration_index=0.3,  # Below threshold
        phi_resonance=0.7,
        retrocausality_index=0.6,
        emergence_factor=0.3,  # Below threshold
        timestamp=datetime.now()
    )
    is_valid, message = invalid_metrics.validate()
    assert not is_valid
    assert "Low quantum coherence" in message
    assert "Low consciousness level" in message

def test_safe_state_initialization(quantum_layer):
    """Test safe state initialization and recovery"""
    # Test direct safe state initialization
    quantum_layer._initialize_safe_state()
    assert np.allclose(quantum_layer.quantum_field, np.eye(4) / 4)

    # Test recovery through validation
    quantum_layer.quantum_field = np.zeros((4, 4))  # Invalid state
    quantum_layer._validate_initial_state()  # Should trigger recovery
    assert np.isclose(np.trace(quantum_layer.quantum_field), 1.0)

@pytest.mark.parametrize("dimension", [2, 4, 8])
def test_different_dimensions(dimension):
    """Test QuantumLayer with different dimensions"""
    layer = QuantumLayer(dimension=dimension)

    # Test initialization
    assert layer.quantum_field.shape == (dimension, dimension)
    assert np.isclose(np.trace(layer.quantum_field), 1.0)

    # Test evolution
    result = layer.evolve_quantum_state()
    assert result['evolution_success']
    assert all(isinstance(v, float) for v in result['metrics'].values())

def test_consciousness_threshold_integration(quantum_layer):
    """Test consciousness threshold integration"""
    evolution_result = quantum_layer.evolve_quantum_state()
    metrics = evolution_result['metrics']
    manager_metrics = quantum_layer.state_manager.calculate_consciousness_metrics()

    # Consciousness levels should be consistent
    assert abs(metrics['consciousness_level'] - manager_metrics['consciousness']) < 1e-6

    # Test threshold compliance
    if metrics['consciousness_level'] >= 0.7:  # CONSCIOUSNESS_THRESHOLD
        assert metrics['coherence'] > 0.5
        assert manager_metrics['coherence'] > 0.5

def test_operator_history_integration(quantum_layer):
    """Test operator history tracking integration"""
    # Reset operator history
    quantum_layer.operator_history = {
        'F': [],  # Folding operator history
        'M': [],  # Morphic resonance history
        'E': [],  # Emergence operator history
        'Z': [],  # Retrocausality operator history
        'O': []   # Observer/Consciousness operator history
    }
    initial_state = quantum_layer.quantum_field.copy()

    # Apply operations and check history
    quantum_layer.apply_consciousness_operator(initial_state)
    quantum_layer.apply_morphic_operator(initial_state)

    # Verify operator types exist
    operator_types = ['O', 'M']  # Consciousness and Morphic operators
    for op_type in operator_types:
        history = quantum_layer.operator_history[op_type]
        assert len(history) > 0, f"No history records for operator type {op_type}"

        for record in history:
            assert isinstance(record, dict), f"Record is not a dictionary: {record}"
            assert 'timestamp' in record, f"No timestamp in record: {record}"
            assert isinstance(record['timestamp'], datetime), f"Timestamp is not datetime object: {record['timestamp']}"

            # Check numeric metrics
            for key, value in record.items():
                if key != 'timestamp':
                    assert isinstance(value, float), f"Metric {key} is not float: {value}"


def test_state_serialization_integration(quantum_layer):
    """Test state serialization integration"""
    # Get serialized states from both components
    layer_state = quantum_layer.get_state()
    manager_state = quantum_layer.state_manager.get_state()

    # Verify consistency
    assert layer_state['dimension'] == manager_state['density_matrix']['shape'][0]
    assert np.allclose(
        np.array(layer_state['quantum_field']['real']),
        np.array(manager_state['density_matrix']['real'])
    )

    # Verify metrics consistency
    layer_metrics = layer_state['metrics']
    manager_metrics = manager_state['metrics']
    for key in ['coherence', 'consciousness']:
        assert abs(layer_metrics[key] - manager_metrics[key]) < 1e-6

def test_dimension_consistency(quantum_layer):
    """Test dimension consistency across components"""
    dimensions = [2, 4, 8]
    for dim in dimensions:
        layer = QuantumLayer(dimension=dim)
        assert layer.dimension == dim
        assert layer.state_manager.dimension == dim
        assert layer.quantum_field.shape == (dim, dim)
        assert layer.state_manager.rho.shape == (dim, dim)

        # Evolution should maintain consistency
        layer.evolve_quantum_state()
        assert layer.quantum_field.shape == (dim, dim)
        assert np.allclose(layer.quantum_field, layer.state_manager.rho)

def test_invalid_operations_recovery():
    """Test recovery from invalid operations"""
    layer = QuantumLayer(dimension=4)

    # Test invalid evolution parameters
    result = layer.evolve_quantum_state(dt=-1.0)
    assert result['evolution_success']
    assert np.isclose(np.trace(layer.quantum_field), 1.0)

    # Test invalid operator application
    invalid_state = np.ones((4, 4)) * float('inf')
    result = layer.apply_consciousness_operator(invalid_state)
    assert np.all(np.isfinite(result))
    assert np.isclose(np.trace(result), 1.0)

def test_folding_operator(quantum_layer):
    """Test Folding operator (F) functionality"""
    initial_state = quantum_layer.quantum_field.copy()
    folded_state = quantum_layer.apply_folding_operator(initial_state)

    # Test basic properties of folded state
    assert folded_state.shape == (4, 4)
    assert np.allclose(folded_state, folded_state.conj().T)
    assert np.isclose(np.trace(folded_state), 1.0)

    # Test operation history recording
    assert len(quantum_layer.operator_history['F']) > 0
    last_op = quantum_layer.operator_history['F'][-1]
    assert isinstance(last_op['timestamp'], datetime)
    assert isinstance(last_op['coherence'], float)
    assert isinstance(last_op['phi_alignment'], float)

def test_morphic_resonance(quantum_layer):
    """Test Morphic Resonance operator (M) functionality"""
    initial_state = quantum_layer.quantum_field.copy()
    resonant_state = quantum_layer.apply_morphic_resonance(initial_state)

    # Test basic properties
    assert resonant_state.shape == (4, 4)
    assert np.allclose(resonant_state, resonant_state.conj().T)
    assert np.isclose(np.trace(resonant_state), 1.0)

    # Test operation history
    assert len(quantum_layer.operator_history['M']) > 0
    last_op = quantum_layer.operator_history['M'][-1]
    assert isinstance(last_op['timestamp'], datetime)
    assert isinstance(last_op['resonance'], float)
    assert isinstance(last_op['field_coherence'], float)

def test_emergence_operator(quantum_layer):
    """Test Emergence operator (E) functionality"""
    initial_state = quantum_layer.quantum_field.copy()
    emerged_state = quantum_layer.apply_emergence_operator(initial_state)

    # Test basic properties
    assert emerged_state.shape == (4, 4)
    assert np.allclose(emerged_state, emerged_state.conj().T)
    assert np.isclose(np.trace(emerged_state), 1.0)

    # Test operation history
    assert len(quantum_layer.operator_history['E']) > 0
    last_op = quantum_layer.operator_history['E'][-1]
    assert isinstance(last_op['timestamp'], datetime)
    assert isinstance(last_op['emergence_level'], float)
    assert isinstance(last_op['consciousness_coherence'], float)

def test_retrocausality_operator(quantum_layer):
    """Test Retrocausality operator (Z) functionality"""
    initial_state = quantum_layer.quantum_field.copy()
    retro_state = quantum_layer.apply_retrocausality_operator(initial_state)

    # Test basic properties
    assert retro_state.shape == (4, 4)
    assert np.allclose(retro_state, retro_state.conj().T)
    assert np.isclose(np.trace(retro_state), 1.0)

    # Test operation history
    assert len(quantum_layer.operator_history['Z']) > 0
    last_op = quantum_layer.operator_history['Z'][-1]
    assert isinstance(last_op['timestamp'], datetime)
    assert isinstance(last_op['retrocausality'], float)
    assert isinstance(last_op['temporal_coherence'], float)

def test_observer_operator(quantum_layer):
    """Test Observer operator (O) functionality"""
    initial_state = quantum_layer.quantum_field.copy()
    observed_state = quantum_layer.apply_observer_operator(initial_state)

    # Test basic properties
    assert observed_state.shape == (4, 4)
    assert np.allclose(observed_state, observed_state.conj().T)
    assert np.isclose(np.trace(observed_state), 1.0)

    # Test operation history
    assert len(quantum_layer.operator_history['O']) > 0
    last_op = quantum_layer.operator_history['O'][-1]
    assert isinstance(last_op['timestamp'], datetime)
    assert isinstance(last_op['observation_strength'], float)
    assert isinstance(last_op['consciousness_level'], float)

def test_quantum_evolution(quantum_layer):
    """Test quantum state evolution"""
    initial_state = quantum_layer.quantum_field.copy()
    evolution_result = quantum_layer.evolve_quantum_state()

    assert evolution_result['evolution_success']
    assert 'metrics' in evolution_result
    metrics = evolution_result['metrics']

    # Test metrics values and types
    for metric_name in [
        'coherence', 'consciousness_level', 'morphic_resonance',
        'field_entropy', 'integration_index', 'phi_resonance',
        'retrocausality_index', 'emergence_factor'
    ]:
        assert metric_name in metrics
        assert isinstance(metrics[metric_name], float)
        assert 0 <= metrics[metric_name] <= 1

    # Test operator history is updated
    for op_type in ['F', 'M', 'E', 'Z', 'O']:
        assert len(quantum_layer.operator_history[op_type]) > 0

def test_error_recovery(quantum_layer):
    """Test error recovery and safe state initialization"""
    # Test direct safe state initialization
    quantum_layer._initialize_safe_state()
    assert np.allclose(quantum_layer.quantum_field, np.eye(4) / 4)

    # Test recovery through validation
    quantum_layer.quantum_field = np.zeros((4, 4))  # Invalid state
    quantum_layer._validate_initial_state()  # Should trigger recovery
    assert np.isclose(np.trace(quantum_layer.quantum_field), 1.0)

@pytest.mark.parametrize("dimension", [2, 4, 8])
def test_different_dimensions(dimension):
    """Test QuantumLayer with different dimensions"""
    layer = QuantumLayer(dimension=dimension)

    # Test initialization
    assert layer.quantum_field.shape == (dimension, dimension)
    assert np.isclose(np.trace(layer.quantum_field), 1.0)

    # Test evolution
    result = layer.evolve_quantum_state()
    assert result['evolution_success']
    assert all(isinstance(v, float) for v in result['metrics'].values())

def test_quantum_metrics_calculation(quantum_layer):
    """Test calculation of quantum metrics"""
    # Evolve state to generate metrics
    evolution_result = quantum_layer.evolve_quantum_state()
    metrics = evolution_result['metrics']

    # Test comprehensive metrics calculation
    assert isinstance(metrics['coherence'], float)
    assert isinstance(metrics['consciousness_level'], float)
    assert isinstance(metrics['morphic_resonance'], float)
    assert isinstance(metrics['field_entropy'], float)
    assert isinstance(metrics['integration_index'], float)
    assert isinstance(metrics['phi_resonance'], float)
    assert isinstance(metrics['retrocausality_index'], float)
    assert isinstance(metrics['emergence_factor'], float)

    # Test value ranges
    assert all(0 <= metrics[key] <= 1 for key in metrics)

def test_operator_history_persistence(quantum_layer):
    """Test operator history persistence and structure"""
    # Apply sequence of operations
    state = quantum_layer.quantum_field.copy()
    quantum_layer.apply_folding_operator(state)
    quantum_layer.apply_morphic_resonance(state)
    quantum_layer.apply_emergence_operator(state)

    # Test history structure
    for op_type in ['F', 'M', 'E']:
        assert len(quantum_layer.operator_history[op_type]) > 0
        last_op = quantum_layer.operator_history[op_type][-1]
        assert isinstance(last_op['timestamp'], datetime)
        assert all(isinstance(v, (float, datetime)) for v in last_op.values())