"""
Tests for quantum merge implementation
"""
import unittest
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
from scipy import special

from quantum.core.merge.morphic_field import (
    MorphicField, MorphicPattern, FieldMetrics,
    apply_folding, apply_resonance, apply_emergence
)

class TestQuantumMerge(unittest.TestCase):
    """Testes para merge quântico"""
    
    def setUp(self):
        """Setup para testes"""
        self.merge = MorphicField(
            field_dimensions=8,
            coherence_threshold=0.75,
            resonance_threshold=0.7,
            max_history=1000
        )
        
        # Dados de mercado de teste
        self.market_data = {
            'prices': [100.0, 101.0, 102.0, 101.5, 102.5],
            'volumes': [1000, 1100, 900, 1200, 1000],
            'highs': [102.0, 103.0, 104.0, 103.5, 104.5],
            'lows': [99.0, 100.0, 101.0, 100.5, 101.5],
            'close': [100.0, 101.0, 102.0, 101.5, 102.5]
        }
        
    def test_field_evolution(self):
        """Testa evolução do campo mórfico"""
        metrics = self.merge.update(self.market_data)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('strength', metrics)
        self.assertIn('coherence', metrics)
        self.assertIn('resonance', metrics)
        
        # Verifica valores
        self.assertGreaterEqual(metrics['strength'], 0.0)
        self.assertGreaterEqual(metrics['coherence'], 0.0)
        self.assertGreaterEqual(metrics['resonance'], 0.0)
        
    def test_pattern_detection(self):
        """Testa detecção de padrões"""
        # Evolui campo várias vezes
        for _ in range(10):
            self.merge.evolve_field(self.market_data)
            
        # Obtém padrões
        patterns = self.merge.get_patterns()
        
        self.assertIsInstance(patterns, list)
        if patterns:
            pattern = patterns[0]
            self.assertIsInstance(pattern, MorphicPattern)
            self.assertGreaterEqual(pattern.field_strength, 0.0)
            self.assertGreaterEqual(pattern.coherence, 0.0)
            self.assertGreaterEqual(pattern.resonance, 0.0)
            
    def test_metrics_filtering(self):
        """Testa filtragem de métricas"""
        # Evolui campo várias vezes
        for _ in range(10):
            self.merge.evolve_field(self.market_data)
            
        # Obtém métricas
        metrics = self.merge.get_field_metrics()
        
        self.assertIsInstance(metrics, list)
        if metrics:
            metric = metrics[0]
            self.assertIsInstance(metric, FieldMetrics)
            self.assertGreaterEqual(metric.strength, 0.0)
            self.assertGreaterEqual(metric.coherence, 0.0)
            self.assertGreaterEqual(metric.resonance, 0.0)
            
    def test_evolution_analysis(self):
        """Testa análise de evolução"""
        # Evolui campo várias vezes
        for _ in range(10):
            self.merge.evolve_field(self.market_data)
            
        # Analisa evolução
        evolution = self.merge.analyze_field_evolution()
        
        self.assertIsInstance(evolution, dict)
        self.assertIn('mean_strength', evolution)
        self.assertIn('mean_coherence', evolution)
        self.assertIn('mean_resonance', evolution)
        
    def test_clear_history(self):
        """Testa limpeza de histórico"""
        # Evolui campo várias vezes
        for _ in range(10):
            self.merge.evolve_field(self.market_data)
            
        # Limpa histórico
        self.merge.clear_history()
        
        self.assertEqual(len(self.merge.patterns), 0)
        self.assertEqual(len(self.merge.metrics_history), 0)
        
if __name__ == '__main__':
    unittest.main()
