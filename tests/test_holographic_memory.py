"""
Test suite for HolographicMemory implementation
Validates M-ICCI principles and holographic operators
"""
import pytest
import numpy as np
from datetime import datetime

from qualia.core.holographic_memory import HolographicMemory, HolographicPattern

@pytest.fixture
def holographic_memory():
    """Fixture for HolographicMemory instance"""
    return HolographicMemory(dimension=64)

@pytest.fixture
def test_pattern():
    """Fixture for test quantum pattern"""
    pattern = np.random.randn(64) + 1j * np.random.randn(64)
    return pattern / np.linalg.norm(pattern)

def test_holographic_operators_initialization(holographic_memory):
    """Test initialization of holographic operators"""
    # Verify operators existence
    assert hasattr(holographic_memory.operators, 'OHR')
    assert hasattr(holographic_memory.operators, 'OHW')
    assert hasattr(holographic_memory.operators, 'OHM')
    assert hasattr(holographic_memory.operators, 'OHI')
    assert hasattr(holographic_memory.operators, 'OHP')
    assert hasattr(holographic_memory.operators, 'OHS')

def test_operator_application(holographic_memory, test_pattern):
    """Test application of holographic operators"""
    try:
        # Test OHR operator
        resonant_pattern = holographic_memory.operators.OHR(test_pattern)
        assert resonant_pattern.shape == test_pattern.shape
        assert np.all(np.isfinite(resonant_pattern))

        # Test OHW operator with proper shape
        test_pattern_reshaped = test_pattern.reshape(-1, 1)
        wavelet_pattern = holographic_memory.operators.OHW(test_pattern_reshaped)
        assert wavelet_pattern.shape == test_pattern_reshaped.shape
        assert np.all(np.isfinite(wavelet_pattern))

        # Test OHM operator
        memory_pattern = holographic_memory.operators.OHM(test_pattern_reshaped)
        assert memory_pattern.shape == test_pattern_reshaped.shape
        assert np.all(np.isfinite(memory_pattern))

        # Test operator chain with proper shape
        evolved_pattern = test_pattern_reshaped
        for operator in [
            holographic_memory.operators.OHR,
            holographic_memory.operators.OHW,
            holographic_memory.operators.OHM,
            holographic_memory.operators.OHI,
            holographic_memory.operators.OHP,
            holographic_memory.operators.OHS
        ]:
            evolved_pattern = operator(evolved_pattern)
            assert evolved_pattern.shape == test_pattern_reshaped.shape
            assert np.all(np.isfinite(evolved_pattern))

    except Exception as e:
        pytest.fail(f"Operator application failed: {str(e)}")

def test_pattern_storage_and_retrieval(holographic_memory, test_pattern):
    """Test holographic pattern storage and retrieval"""
    try:
        # Reshape pattern for storage
        test_pattern_reshaped = test_pattern.reshape(-1, 1)

        # Store pattern
        pattern_id = holographic_memory.store_pattern(test_pattern_reshaped)
        assert pattern_id is not None
        assert pattern_id in holographic_memory.patterns

        # Verify stored pattern
        stored_pattern = holographic_memory.patterns[pattern_id]
        assert isinstance(stored_pattern, HolographicPattern)
        assert stored_pattern.pattern.shape == test_pattern_reshaped.shape
        assert stored_pattern.quantum_signature is not None
        assert stored_pattern.consciousness_field is not None
        assert stored_pattern.morphic_field is not None

        # Test retrieval
        retrieved_pattern = holographic_memory.retrieve_pattern(pattern_id)
        assert retrieved_pattern is not None
        assert np.allclose(retrieved_pattern.pattern, stored_pattern.pattern)

    except Exception as e:
        pytest.fail(f"Pattern storage and retrieval failed: {str(e)}")

def test_resonant_pattern_search(holographic_memory, test_pattern):
    """Test resonant pattern search functionality"""
    # Store multiple patterns
    patterns = []
    test_pattern_reshaped = test_pattern.reshape(-1, 1)

    for _ in range(5):
        pattern = np.random.randn(64) + 1j * np.random.randn(64)
        pattern = pattern.reshape(-1, 1) / np.linalg.norm(pattern)
        pattern_id = holographic_memory.store_pattern(pattern)
        patterns.append(pattern_id)

    # Search for resonant patterns
    resonances = holographic_memory.find_resonant_patterns(test_pattern_reshaped)
    assert len(resonances) >= 0
    if len(resonances) > 0:
        assert all(0 <= score <= 1 for _, score in resonances)

def test_quantum_metrics_calculation(holographic_memory):
    """Test quantum metrics calculation"""
    field_state = holographic_memory.get_field_state()
    required_metrics = {
        'field_energy',
        'consciousness_energy',
        'field_coherence',
        'consciousness_coherence',
        'memory_usage',
        'average_resonance',
        'pattern_count'
    }
    assert all(metric in field_state for metric in required_metrics)
    assert all(0 <= field_state[m] <= 1 for m in ['field_coherence', 'consciousness_coherence'])

def test_memory_capacity_management(holographic_memory):
    """Test memory capacity management"""
    initial_count = len(holographic_memory.patterns)

    # Store patterns up to capacity
    for _ in range(holographic_memory.memory_capacity + 5):
        pattern = np.random.randn(64) + 1j * np.random.randn(64)
        pattern = pattern.reshape(-1, 1) / np.linalg.norm(pattern)
        holographic_memory.store_pattern(pattern)

    # Verify memory doesn't exceed capacity
    assert len(holographic_memory.patterns) <= holographic_memory.memory_capacity

def test_consciousness_field_evolution(holographic_memory, test_pattern):
    """Test consciousness field evolution"""
    initial_field = holographic_memory.consciousness_field.copy()

    # Store pattern and verify field update
    test_pattern_reshaped = test_pattern.reshape(-1, 1)
    holographic_memory.store_pattern(test_pattern_reshaped)

    # Field should change after pattern storage
    assert not np.allclose(initial_field, holographic_memory.consciousness_field)

    # Verify field properties
    assert np.all(np.isfinite(holographic_memory.consciousness_field))
    field_energy = np.abs(np.trace(holographic_memory.consciousness_field))
    assert 0 <= field_energy <= holographic_memory.dimension