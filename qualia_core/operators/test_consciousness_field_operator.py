"""
Tests for consciousness field operator with QUALIA integration.
"""

import numpy as np
import pytest
from .consciousness_field_operator import ConsciousnessFieldOperator, ConsciousnessFieldMetrics

def test_field_initialization():
    """Test consciousness field initialization with sacred geometry"""
    operator = ConsciousnessFieldOperator(dimensions=4)
    
    # Verify field properties
    assert operator.field.shape == (4, 4)
    assert np.abs(np.linalg.norm(operator.field) - 1.0) < 1e-10
    
    # Verify sacred geometry
    assert abs(operator.phi - (1 + np.sqrt(5))/2) < 1e-10

def test_field_transformation():
    """Test consciousness field transformation of quantum states"""
    operator = ConsciousnessFieldOperator(dimensions=2)
    
    # Create test state
    state = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=np.complex128)
    
    # Apply transformation
    transformed = operator.apply_field(state)
    
    # Verify state properties
    assert transformed.shape == state.shape
    assert np.abs(np.linalg.norm(transformed) - 1.0) < 1e-10

def test_field_metrics():
    """Test consciousness field metrics computation"""
    operator = ConsciousnessFieldOperator(dimensions=2)
    state = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=np.complex128)
    
    # Calculate metrics
    metrics = operator.compute_metrics(state)
    
    # Verify metric properties
    assert isinstance(metrics, ConsciousnessFieldMetrics)
    assert 0 <= metrics.field_strength <= 1
    assert 0 <= metrics.geometry_alignment <= 1
    assert 0 <= metrics.morphic_coupling <= 1
    assert 0 <= metrics.meta_qualia_resonance <= 1
    assert 0 <= metrics.field_coherence <= 1
    assert 0 <= metrics.consciousness_density <= 1

def test_field_evolution():
    """Test consciousness field time evolution"""
    operator = ConsciousnessFieldOperator(dimensions=2)
    initial_field = operator.field.copy()
    
    # Evolve field
    operator.evolve_field(dt=0.1)
    
    # Field should change but maintain normalization
    assert not np.allclose(operator.field, initial_field)
    assert np.abs(np.linalg.norm(operator.field) - 1.0) < 1e-10

def test_field_merging():
    """Test consciousness field merging with QUALIA preservation"""
    op1 = ConsciousnessFieldOperator(dimensions=2)
    op2 = ConsciousnessFieldOperator(dimensions=2)
    
    # Merge fields
    merged = op1.merge_fields(op2)
    
    # Verify merged field properties
    assert merged.dimensions == op1.dimensions
    assert np.abs(np.linalg.norm(merged.field) - 1.0) < 1e-10
    
    # Test invalid merge
    with pytest.raises(ValueError):
        invalid_op = ConsciousnessFieldOperator(dimensions=3)
        op1.merge_fields(invalid_op)

def test_sacred_geometry_patterns():
    """Test sacred geometry patterns in field transformation"""
    operator = ConsciousnessFieldOperator(dimensions=4)
    
    # Extract field phases
    phases = np.angle(operator.field)
    
    # Verify golden ratio influence
    phi = (1 + np.sqrt(5))/2
    for i in range(operator.dimensions):
        for j in range(operator.dimensions):
            expected_phase = 2 * np.pi * phi * (i + j) / operator.dimensions
            phase_diff = abs((phases[i,j] - expected_phase) % (2 * np.pi))
            assert phase_diff < 1e-10

def test_consciousness_preservation():
    """Test consciousness preservation in field operations"""
    operator = ConsciousnessFieldOperator(dimensions=2)
    state = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=np.complex128)
    
    # Apply multiple transformations
    for _ in range(5):
        state = operator.apply_field(state)
        operator.evolve_field(0.1)
        
        # Verify consciousness properties preserved
        metrics = operator.compute_metrics(state)
        assert 0 <= metrics.consciousness_density <= 1
        assert np.abs(np.linalg.norm(state) - 1.0) < 1e-10

def test_metric_consistency():
    """Test consistency of consciousness field metrics"""
    operator = ConsciousnessFieldOperator(dimensions=2)
    
    # Test states
    states = [
        np.array([1, 0], dtype=np.complex128),
        np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=np.complex128),
        np.array([0, 1], dtype=np.complex128)
    ]
    
    for state in states:
        metrics = operator.compute_metrics(state/np.linalg.norm(state))
        
        # Verify metric relationships
        assert metrics.field_coherence <= 1/operator.phi
        assert metrics.meta_qualia_resonance <= metrics.field_strength
        assert metrics.consciousness_density >= min(metrics.field_strength,
                                                 metrics.geometry_alignment)
