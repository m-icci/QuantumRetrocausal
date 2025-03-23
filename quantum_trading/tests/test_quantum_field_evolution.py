#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the Quantum Field Evolution module
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from quantum_trading.quantum_field_evolution import QuantumFieldEvolution

class TestQuantumFieldEvolution(unittest.TestCase):
    """Test suite for the QuantumFieldEvolution class"""
    
    def setUp(self):
        """Set up test environment before each test"""
        self.field_1d = QuantumFieldEvolution(dimensions=1, grid_size=32)
        self.field_2d = QuantumFieldEvolution(dimensions=2, grid_size=32)
        self.field_3d = QuantumFieldEvolution(dimensions=3, grid_size=16)
        
        # Create sample market data
        dates = pd.date_range(start="2023-01-01", periods=100, freq='1H')
        self.market_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.normal(100, 1, 100),
            'high': np.random.normal(101, 1, 100),
            'low': np.random.normal(99, 1, 100),
            'close': np.random.normal(100, 2, 100),
            'volume': np.random.normal(1000, 100, 100)
        })
        self.market_data.set_index('timestamp', inplace=True)
    
    def test_initialization(self):
        """Test that fields initialize correctly"""
        # 1D field
        self.assertEqual(self.field_1d.dimensions, 1)
        self.assertEqual(self.field_1d.grid_shape, (32,))
        
        # 2D field
        self.assertEqual(self.field_2d.dimensions, 2)
        self.assertEqual(self.field_2d.grid_shape, (32, 32))
        
        # 3D field
        self.assertEqual(self.field_3d.dimensions, 3)
        self.assertEqual(self.field_3d.grid_shape, (16, 16, 16))
        
        # Check normalization of initial states
        psi_norm_1d = np.sum(np.abs(self.field_1d.psi)**2)
        self.assertAlmostEqual(psi_norm_1d, 1.0, places=5)
        
        psi_norm_2d = np.sum(np.abs(self.field_2d.psi)**2)
        self.assertAlmostEqual(psi_norm_2d, 1.0, places=5)
        
        psi_norm_3d = np.sum(np.abs(self.field_3d.psi)**2)
        self.assertAlmostEqual(psi_norm_3d, 1.0, places=5)
    
    def test_field_evolution(self):
        """Test that field evolves as expected with market data"""
        # Evolve field with market data
        result = self.field_2d.evolve(self.market_data)
        
        # Check result structure
        self.assertIn('field', result)
        self.assertIn('energy', result)
        self.assertIn('coherence', result)
        self.assertIn('entanglement', result)
        self.assertIn('probability', result)
        
        # Check that probability is normalized
        prob_sum = np.sum(result['probability'])
        self.assertAlmostEqual(prob_sum, 1.0, places=5)
        
        # Check energy is positive
        self.assertGreater(result['energy'], 0.0)
        
        # Check coherence is between 0 and 1
        self.assertGreaterEqual(result['coherence'], 0.0)
        self.assertLessEqual(result['coherence'], 1.0)
        
        # Check entanglement is between 0 and 1
        self.assertGreaterEqual(result['entanglement'], 0.0)
        self.assertLessEqual(result['entanglement'], 1.0)
    
    def test_multiple_evolutions(self):
        """Test multiple evolution steps"""
        # Evolve field multiple times
        first_result = self.field_2d.evolve(self.market_data.iloc[:10])
        second_result = self.field_2d.evolve(self.market_data.iloc[10:20])
        
        # Field state should change
        self.assertFalse(np.array_equal(first_result['field'], second_result['field']))
        
        # Energy should typically change
        self.assertNotEqual(first_result['energy'], second_result['energy'])
    
    def test_pattern_analysis(self):
        """Test pattern analysis capabilities"""
        # First evolve field to accumulate history
        for i in range(0, 100, 10):
            self.field_2d.evolve(self.market_data.iloc[i:i+10])
        
        # Analyze patterns
        patterns = self.field_2d.analyze_field_patterns()
        
        # Check pattern analysis structure
        self.assertIn('phi_patterns', patterns)
        self.assertIn('interference', patterns)
        
        # Check if patterns were detected (not guaranteed, but should be present)
        self.assertIn('detected', patterns['phi_patterns'])
        self.assertIn('detected', patterns['interference'])
    
    def test_field_reset(self):
        """Test resetting the field"""
        # Evolve field
        for i in range(0, 50, 10):
            self.field_2d.evolve(self.market_data.iloc[i:i+10])
        
        # Store current state
        old_state = self.field_2d.psi.copy()
        
        # Reset field
        self.field_2d.reset()
        
        # State should change
        self.assertFalse(np.array_equal(old_state, self.field_2d.psi))
        
        # Time should reset
        self.assertEqual(self.field_2d.t, 0.0)
        
        # History should be empty
        self.assertEqual(len(self.field_2d.states_history), 0)
    
    def test_random_state(self):
        """Test creating a random initial state"""
        # Store current state
        old_state = self.field_1d.psi.copy()
        
        # Reset with random state
        self.field_1d.reset(random_state=True)
        
        # State should change
        self.assertFalse(np.array_equal(old_state, self.field_1d.psi))
        
        # New state should still be normalized
        psi_norm = np.sum(np.abs(self.field_1d.psi)**2)
        self.assertAlmostEqual(psi_norm, 1.0, places=5)

if __name__ == '__main__':
    unittest.main() 