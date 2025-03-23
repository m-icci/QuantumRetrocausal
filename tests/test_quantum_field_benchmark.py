"""
Benchmark tests for quantum field operations
"""
import pytest
import numpy as np
from qualia.utils.quantum_field import (
    create_quantum_field,
    standardize_dimensions,
    predict_market_state,
    calculate_financial_decoherence,
    reset_performance_metrics
)

@pytest.fixture(autouse=True)
def clear_metrics():
    """Reset performance metrics before each test"""
    reset_performance_metrics()

def test_create_quantum_field_benchmark(benchmark):
    """Benchmark quantum field creation"""
    def create_field():
        return create_quantum_field(size=64)
    
    result = benchmark(create_field)
    assert result.shape == (64, 64)
    assert np.iscomplex(result).any()

def test_standardize_dimensions_benchmark(benchmark):
    """Benchmark dimension standardization with various input sizes"""
    # Test with larger state reduction
    large_state = create_quantum_field(size=128).flatten()
    
    def standardize():
        return standardize_dimensions(large_state, target_size=64)
    
    result = benchmark(standardize)
    assert result.size == 64
    assert np.iscomplex(result).any()

def test_predict_market_state_benchmark(benchmark):
    """Benchmark market state prediction"""
    state = create_quantum_field(size=64).flatten()
    hamiltonian = create_quantum_field(size=64)
    
    def predict():
        return predict_market_state(state, hamiltonian, dt=0.01, steps=10)
    
    result = benchmark(predict)
    assert result.shape == (64, 1)
    assert np.iscomplex(result).any()

def test_financial_decoherence_benchmark(benchmark):
    """Benchmark financial decoherence calculation"""
    state1 = create_quantum_field(size=64)
    state2 = create_quantum_field(size=64)
    
    def calc_decoherence():
        return calculate_financial_decoherence(state1, state2)
    
    result = benchmark(calc_decoherence)
    assert 0 <= result <= 1

@pytest.mark.parametrize("input_size", [32, 64, 128, 256])
def test_dimension_scaling_benchmark(benchmark, input_size):
    """Benchmark dimension standardization scaling performance"""
    input_state = create_quantum_field(size=input_size).flatten()
    
    def standardize():
        return standardize_dimensions(input_state, target_size=64)
    
    result = benchmark(standardize)
    assert result.size == 64
    assert np.iscomplex(result).any()
