"""
Unified tests for QUALIA module
"""
import unittest
import numpy as np

from quantum.core.QUALIA.config import QUALIAConfig
from quantum.core.QUALIA.base_types import (
    QuantumState,
    QuantumPattern,
    QuantumMetric,
    QuantumOperator
)

from quantum.core.QUALIA.operators.folding import apply_folding
from quantum.core.QUALIA.operators.resonance import apply_resonance
from quantum.core.QUALIA.operators.emergence import apply_emergence
from quantum.core.QUALIA.metrics import get_metrics

class TestQUALIAUnified(unittest.TestCase):
    """Test cases for unified QUALIA implementation"""
    
    def setUp(self):
        """Setup test environment"""
        self.dimension = 8
        self.config = QUALIAConfig(
            default_dimension=self.dimension,
            coherence_threshold=0.95,
            field_strength=1.0,
            morphic_coupling=0.1
        )
        
        # Base state |0⟩
        base_amplitudes = np.zeros(self.dimension, dtype=np.complex128)
        base_amplitudes[0] = 1.0
        self.base_state = QuantumState(base_amplitudes, self.dimension)
        
        # Superposition (|0⟩ + |1⟩)/√2
        super_amplitudes = np.zeros(self.dimension, dtype=np.complex128)
        super_amplitudes[0] = 1/np.sqrt(2)
        super_amplitudes[1] = 1/np.sqrt(2)
        self.super_state = QuantumState(super_amplitudes, self.dimension)
    
    def test_state_validity(self):
        """Test quantum state validation"""
        self.assertTrue(self.base_state.is_valid)
        self.assertTrue(self.super_state.is_valid)
    
    def test_folding_operator(self):
        """Test folding operator"""
        result = apply_folding(self.base_state, self.config)
        self.assertTrue(isinstance(result, QuantumState))
        self.assertTrue(result.is_valid)
        
        # Test superposition
        result = apply_folding(self.super_state, self.config)
        self.assertTrue(isinstance(result, QuantumState))
        self.assertTrue(result.is_valid)
    
    def test_resonance_operator(self):
        """Test resonance operator"""
        result = apply_resonance(self.base_state, self.config)
        self.assertTrue(isinstance(result, QuantumState))
        self.assertTrue(result.is_valid)
        
        # Test superposition
        result = apply_resonance(self.super_state, self.config)
        self.assertTrue(isinstance(result, QuantumState))
        self.assertTrue(result.is_valid)
    
    def test_emergence_operator(self):
        """Test emergence operator"""
        result = apply_emergence(self.base_state, self.config)
        self.assertTrue(isinstance(result, QuantumState))
        self.assertTrue(result.is_valid)
        
        # Test superposition
        result = apply_emergence(self.super_state, self.config)
        self.assertTrue(isinstance(result, QuantumState))
        self.assertTrue(result.is_valid)
    
    def test_metrics(self):
        """Test metrics calculation"""
        metrics = get_metrics(self.super_state, self.config)
        
        # Verify all required metrics are present
        required_metrics = {
            'coherence',
            'resonance',
            'emergence',
            'implied_order',
            'integration',
            'consciousness',
            'coherence_time'
        }
        
        metric_names = {m.name for m in metrics}
        self.assertTrue(required_metrics.issubset(metric_names))
        
        # Verify metric values are in valid range [0,1]
        for metric in metrics:
            self.assertGreaterEqual(metric.value, 0.0)
            self.assertLessEqual(metric.value, 1.0)

if __name__ == '__main__':
    unittest.main()
