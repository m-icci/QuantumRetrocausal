#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the Retrocausal Analysis module
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from quantum_trading.analysis.retrocausal_analysis import RetrocausalAnalyzer

class TestRetrocausalAnalyzer(unittest.TestCase):
    """Test suite for the RetrocausalAnalyzer class"""
    
    def setUp(self):
        """Set up test environment before each test"""
        # Create analyzer with smaller windows for testing
        self.analyzer = RetrocausalAnalyzer(
            lookback_window=20,
            time_scales=[2, 5, 10],
            confidence_threshold=0.6,
            causality_window=10
        )
        
        # Generate test timestamps
        self.timestamps = [
            datetime.now() - timedelta(minutes=i)
            for i in range(50, 0, -1)
        ]
        
        # Generate fake price data
        self.prices = []
        price = 100.0
        for i in range(50):
            # Add some trends, oscillations and random movements
            if i < 15:
                # Uptrend
                change = np.random.normal(0.05, 0.02)
            elif i < 30:
                # Downtrend
                change = np.random.normal(-0.04, 0.02)
            else:
                # Oscillation
                change = 0.1 * np.sin(i/3) + np.random.normal(0, 0.01)
                
            price *= (1 + change)
            self.prices.append(price)
        
        # Generate fake field metrics with some patterns
        self.field_metrics = []
        for i in range(50):
            # Create some phi-resonant patterns in coherence
            coherence = 0.5 + 0.2 * np.sin(i * 0.618) + 0.1 * np.sin(i * 0.382)
            
            # Energy with some correlation to prices
            if i > 5:
                rel_price_change = (self.prices[i] - self.prices[i-5]) / self.prices[i-5]
                energy = 0.5 + 0.3 * rel_price_change + 0.1 * np.random.random()
            else:
                energy = 0.5 + 0.1 * np.random.random()
                
            # Entanglement with some lead/lag relationship
            if i > 2:
                entanglement = 0.4 + 0.3 * np.sin(i/4 + np.pi/2)
            else:
                entanglement = 0.4 + 0.1 * np.random.random()
                
            self.field_metrics.append({
                'energy': energy,
                'coherence': coherence,
                'entanglement': entanglement,
                'phase': i * 0.1 % (2 * np.pi),
                'probability': np.random.random()
            })
    
    def test_initialization(self):
        """Test initialization of the analyzer"""
        self.assertEqual(self.analyzer.lookback_window, 20)
        self.assertEqual(self.analyzer.time_scales, [2, 5, 10])
        self.assertEqual(self.analyzer.confidence_threshold, 0.6)
        self.assertEqual(len(self.analyzer.metrics_by_scale), 3)
    
    def test_add_data_point(self):
        """Test adding data points"""
        # Add a few data points
        for i in range(10):
            price_data = {
                'open': self.prices[i] * 0.99,
                'high': self.prices[i] * 1.01,
                'low': self.prices[i] * 0.98,
                'close': self.prices[i],
                'volume': 1000 + 100 * np.random.random()
            }
            
            self.analyzer.add_data_point(
                timestamp=self.timestamps[i],
                price_data=price_data,
                field_metrics=self.field_metrics[i]
            )
        
        # Check that data was stored
        self.assertEqual(len(self.analyzer.price_history), 10)
        self.assertEqual(len(self.analyzer.field_metrics_history), 10)
        
        # Check that metrics were calculated for each time scale
        for scale in self.analyzer.time_scales:
            self.assertGreater(len(self.analyzer.metrics_by_scale[scale]), 0)
    
    def test_full_analysis_with_insufficient_data(self):
        """Test analysis with insufficient data"""
        # Add just a few data points
        for i in range(5):
            self.analyzer.add_data_point(
                timestamp=self.timestamps[i],
                price_data={'close': self.prices[i]},
                field_metrics=self.field_metrics[i]
            )
        
        # Perform analysis
        result = self.analyzer.analyze()
        
        # Should return no signal due to insufficient data
        self.assertFalse(result.get('signal_detected', False))
        self.assertEqual(result.get('explanation'), "Insufficient data for analysis")
    
    def test_full_analysis_cycle(self):
        """Test a full analysis cycle with sufficient data"""
        # Add enough data points
        for i in range(30):
            price_data = {
                'open': self.prices[i] * 0.99,
                'high': self.prices[i] * 1.01,
                'low': self.prices[i] * 0.98,
                'close': self.prices[i],
                'volume': 1000 + 100 * np.random.random()
            }
            
            self.analyzer.add_data_point(
                timestamp=self.timestamps[i],
                price_data=price_data,
                field_metrics=self.field_metrics[i]
            )
        
        # Run analysis
        result = self.analyzer.analyze()
        
        # Check result structure
        self.assertIn('signal_detected', result)
        self.assertIn('confidence', result)
        self.assertIn('direction', result)
        self.assertIn('time_scale', result)
        self.assertIn('explanation', result)
        
        # Check components
        self.assertIn('components', result)
        self.assertIn('anomalies', result['components'])
        self.assertIn('causality', result['components'])
        self.assertIn('synchronicity', result['components'])
        self.assertIn('phi_recursion', result['components'])
    
    def test_get_signal(self):
        """Test getting a trading signal"""
        # Add data points
        for i in range(30):
            self.analyzer.add_data_point(
                timestamp=self.timestamps[i],
                price_data={'close': self.prices[i]},
                field_metrics=self.field_metrics[i]
            )
        
        # Get signal
        signal = self.analyzer.get_signal()
        
        # Check signal structure
        self.assertIn('signal', signal)
        self.assertIn('confidence', signal)
        self.assertIn('direction', signal)
        self.assertIn('time_scale', signal)
        self.assertIn('explanation', signal)
        
        # Signal might be true or false based on threshold
        if signal['signal']:
            self.assertGreaterEqual(signal['confidence'], self.analyzer.confidence_threshold)
            self.assertIn(signal['direction'], [-1, 0, 1])
            self.assertIn(signal['time_scale'], self.analyzer.time_scales)
        else:
            # If no signal, confidence should be low
            self.assertLess(signal['confidence'], self.analyzer.confidence_threshold)
    
    def test_reset(self):
        """Test resetting the analyzer"""
        # Add data points
        for i in range(15):
            self.analyzer.add_data_point(
                timestamp=self.timestamps[i],
                price_data={'close': self.prices[i]},
                field_metrics=self.field_metrics[i]
            )
        
        # Reset analyzer
        self.analyzer.reset()
        
        # Check that data structures are cleared
        self.assertEqual(len(self.analyzer.price_history), 0)
        self.assertEqual(len(self.analyzer.field_metrics_history), 0)
        self.assertEqual(len(self.analyzer.retrocausal_events), 0)
        
        # Check that metrics for each time scale are cleared
        for scale in self.analyzer.time_scales:
            self.assertEqual(len(self.analyzer.metrics_by_scale[scale]), 0)
        
        # Check that causal graph is reset
        self.assertEqual(len(self.analyzer.causal_graph.nodes), 0)
        self.assertEqual(len(self.analyzer.causal_graph.edges), 0)
        
        # Check sync patterns cleared
        self.assertEqual(len(self.analyzer.sync_patterns), 0)

if __name__ == '__main__':
    unittest.main() 