"""
Test suite for quantum utility functions
"""
import pytest
import numpy as np
from qualia.utils.quantum_utils import (
    calculate_quantum_coherence,
    calculate_dark_finance_metrics,
    calculate_entropy,
    calculate_quantum_fidelity,
    calculate_field_energy,
    calculate_morphic_resonance
)

@pytest.fixture
def test_state():
    """Fixture for test quantum state"""
    state = np.random.randn(64) + 1j * np.random.randn(64)
    return state / np.linalg.norm(state)

@pytest.fixture
def test_density_matrix(test_state):
    """Fixture for test density matrix"""
    state = test_state.reshape(-1, 1)
    return state @ state.conj().T

def test_calculate_coherence(test_density_matrix):
    """Test quantum coherence calculation"""
    coherence = calculate_quantum_coherence(test_density_matrix)
    assert 0 <= coherence <= 1
    assert isinstance(coherence, float)

def test_calculate_dark_finance_metrics(test_density_matrix):
    """Test dark finance metrics calculation"""
    metrics = calculate_dark_finance_metrics(test_density_matrix)

    # Verify all required metrics are present
    required_metrics = {
        'dark_ratio',
        'dark_coupling',
        'dark_coherence',
        'dark_phi_resonance'
    }
    assert all(metric in metrics for metric in required_metrics)

    # Verify metric values are in valid range
    assert all(0 <= metrics[m] <= 1 for m in metrics)
    assert all(isinstance(metrics[m], float) for m in metrics)

def test_calculate_entropy(test_density_matrix):
    """Test von Neumann entropy calculation"""
    entropy = calculate_entropy(test_density_matrix)
    assert 0 <= entropy <= 1
    assert isinstance(entropy, float)

def test_calculate_quantum_fidelity():
    """Test quantum fidelity calculation"""
    # Create two test states
    state1 = np.random.randn(64) + 1j * np.random.randn(64)
    state1 = state1 / np.linalg.norm(state1)

    state2 = np.random.randn(64) + 1j * np.random.randn(64)
    state2 = state2 / np.linalg.norm(state2)

    fidelity = calculate_quantum_fidelity(state1, state2)
    assert 0 <= fidelity <= 1
    assert isinstance(fidelity, float)

def test_calculate_field_energy(test_density_matrix):
    """Test quantum field energy calculation"""
    energy = calculate_field_energy(test_density_matrix)
    assert 0 <= energy <= 1
    assert isinstance(energy, float)

def test_calculate_morphic_resonance(test_density_matrix):
    """Test morphic resonance calculation"""
    resonance = calculate_morphic_resonance(test_density_matrix)
    assert 0 <= resonance <= 1
    assert isinstance(resonance, float)

def test_edge_cases():
    """Test edge cases and error handling"""
    # Test with zero state
    zero_state = np.zeros(64)
    metrics = calculate_dark_finance_metrics(zero_state)
    assert all(0 <= metrics[m] <= 1 for m in metrics)

    # Test with invalid dimensions
    invalid_state = np.random.randn(32)
    metrics = calculate_dark_finance_metrics(invalid_state)
    assert all(0 <= metrics[m] <= 1 for m in metrics)

    # Test with non-normalized state
    large_state = np.random.randn(64) * 1000
    metrics = calculate_dark_finance_metrics(large_state)
    assert all(0 <= metrics[m] <= 1 for m in metrics)

def test_numerical_stability():
    """Test numerical stability of calculations"""
    # Test with very small numbers
    small_state = np.random.randn(64) * 1e-10
    metrics = calculate_dark_finance_metrics(small_state)
    assert all(np.isfinite(metrics[m]) for m in metrics)

    # Test with very large numbers
    large_state = np.random.randn(64) * 1e10
    metrics = calculate_dark_finance_metrics(large_state)
    assert all(np.isfinite(metrics[m]) for m in metrics)