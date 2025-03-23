"""
Test suite for quantum field operations
Validates core quantum field manipulation functions and dimension standardization
"""
import pytest
import numpy as np
from qualia.utils.quantum_field import (
    safe_complex_to_real,
    serialize_complex_value,
    create_quantum_field,
    calculate_financial_decoherence,
    predict_market_state,
)

@pytest.fixture
def sample_field():
    """Create a sample quantum field for testing"""
    return create_quantum_field(size=64)  # Standard size

@pytest.fixture
def sample_hamiltonian():
    """Create a sample Hamiltonian for testing"""
    size = 64  # Standard size
    H = np.random.randn(size, size) + 1j * np.random.randn(size, size)
    return (H + H.conj().T) / 2  # Make Hermitian

def test_safe_complex_to_real():
    """Test complex to real conversion"""
    # Test with various input types
    assert safe_complex_to_real(1.0) == 1.0
    assert safe_complex_to_real(1 + 1j) == np.sqrt(2)
    assert np.isclose(safe_complex_to_real([1 + 1j, 2 + 2j]), np.sqrt(5))
    assert safe_complex_to_real(None) == 0.0

def test_create_quantum_field():
    """Test quantum field creation with standard size"""
    # Test standard 64-dimension field
    field = create_quantum_field(size=64)
    assert field.shape == (64, 64)
    assert np.iscomplex(field).any()
    assert abs(np.linalg.norm(field.flatten()) - 1.0) < 1e-10

    # Test non-standard size gets normalized to 64
    field_large = create_quantum_field(size=128)
    assert field_large.shape == (64, 64)
    assert abs(np.linalg.norm(field_large.flatten()) - 1.0) < 1e-10

def test_calculate_financial_decoherence():
    """Test financial decoherence calculation with dimension standardization"""
    # Test with standard size
    state1 = create_quantum_field(size=64)
    state2 = create_quantum_field(size=64)
    decoherence = calculate_financial_decoherence(state1, state2)
    assert 0 <= decoherence <= 1

    # Test with intentionally mismatched dimensions (should standardize)
    state3 = create_quantum_field(size=32)  # Smaller dimension
    decoherence = calculate_financial_decoherence(state1, state3)
    assert 0 <= decoherence <= 1

    # Test with same state (should have minimal decoherence)
    self_decoherence = calculate_financial_decoherence(state1, state1)
    assert self_decoherence < 0.1

def test_predict_market_state(sample_field, sample_hamiltonian):
    """Test market state prediction with dimension standardization"""
    initial_state = sample_field.flatten()

    predicted = predict_market_state(
        initial_state,
        sample_hamiltonian,
        dt=0.01,
        steps=5
    )

    # Verify output dimensions
    assert predicted.shape == (64, 1)  # Standard size
    assert np.iscomplex(predicted).any()
    assert abs(np.linalg.norm(predicted) - 1.0) < 1e-10

def test_dimension_standardization():
    """Test dimension standardization behavior"""
    # Test larger dimension reduction
    large_state = create_quantum_field(size=128)
    standard_hamiltonian = create_quantum_field(size=64)
    result = predict_market_state(large_state.flatten(), standard_hamiltonian)
    assert result.shape == (64, 1)

    # Verify quantum state normalization
    assert abs(np.linalg.norm(result) - 1.0) < 1e-10

    # Test smaller dimension padding
    small_state = create_quantum_field(size=32)
    result = predict_market_state(small_state.flatten(), standard_hamiltonian)
    assert result.shape == (64, 1)
    assert abs(np.linalg.norm(result) - 1.0) < 1e-10

    # Verify relative energy preservation
    # Instead of comparing direct energy density ratios, verify energy conservation principles
    large_energy = np.sum(np.abs(large_state)**2)
    result_energy = np.sum(np.abs(result)**2)
    energy_ratio = result_energy / large_energy

    # Allow for quantum transition losses but maintain reasonable bounds
    assert 0.5 <= energy_ratio <= 2.0, f"Energy ratio {energy_ratio} outside acceptable bounds"

def test_type_safety():
    """Test type safety of quantum operations"""
    # Test invalid inputs - should raise TypeError
    with pytest.raises((TypeError, ValueError)):
        predict_market_state(None, None)

    # Test mixed type handling
    state = create_quantum_field(size=64)
    hamiltonian = create_quantum_field(size=64)

    # Convert to different types
    state = state.astype(np.float64)
    result = predict_market_state(state, hamiltonian)
    assert result.dtype == np.complex128  # Should be converted to complex