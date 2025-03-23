"""
Testes para integração QUALIA-Scalping
"""
import unittest
import numpy as np
from datetime import datetime, timedelta

from quantum.core.portfolio.scalping.qualia_scalping import QualiaScalpingIntegration
from quantum.core.QUALIA import QualiaState

class TestQualiaScalping(unittest.TestCase):
    """Testes para integração QUALIA com scalping"""
    
    def setUp(self):
        """Setup para testes"""
        self.qualia = QualiaScalpingIntegration(
            field_dimensions=8,
            coherence_threshold=0.75,
            resonance_threshold=0.7
        )
        
        # Dados de mercado simulados - tendência de alta
        self.market_data = {
            'opens': [100.0 + i * 0.1 for i in range(100)],
            'highs': [100.0 + i * 0.1 + 0.05 for i in range(100)],
            'lows': [100.0 + i * 0.1 - 0.05 for i in range(100)],
            'closes': [100.0 + i * 0.1 for i in range(100)],
            'volumes': [1000.0 for _ in range(100)]
        }
        
    def test_market_analysis(self):
        """Testa análise de mercado"""
        state = self.qualia.analyze_market(self.market_data)
        
        self.assertIsInstance(state, QualiaState)
        self.assertGreaterEqual(state.geometric_coherence, 0.0)
        self.assertLessEqual(state.geometric_coherence, 1.0)
        self.assertGreaterEqual(state.philosophical_resonance, 0.0)
        self.assertLessEqual(state.philosophical_resonance, 1.0)
        
    def test_trading_signal(self):
        """Testa geração de sinal"""
        state = self.qualia.analyze_market(self.market_data)
        signal = self.qualia.get_trading_signal(state)
        
        self.assertIsInstance(signal, dict)
        self.assertIn('direction', signal)
        self.assertIn('confidence', signal)
        self.assertIn('geometric_coherence', signal)
        self.assertIn('philosophical_resonance', signal)
        
        self.assertIn(signal['direction'], [-1, 0, 1])
        self.assertGreaterEqual(signal['confidence'], 0.0)
        self.assertLessEqual(signal['confidence'], 1.0)
        
    def test_market_tensor(self):
        """Testa conversão para tensor"""
        tensor = self.qualia._market_to_tensor(self.market_data)
        
        self.assertEqual(tensor.shape, (8, 8))
        self.assertTrue(np.all(tensor >= 0))
        self.assertTrue(np.all(tensor <= 1.618))  # PHI
        
if __name__ == '__main__':
    unittest.main()
