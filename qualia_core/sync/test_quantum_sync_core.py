#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Testes para o QuantumSyncCore
============================
Testa a funcionalidade do núcleo de sincronização quântica.
"""

import unittest
import numpy as np
from datetime import datetime
from pathlib import Path
import tempfile
import json

from quantum_sync_core import (
    UnifiedQuantumField,
    CycleSynchronizer,
    GlobalOptimizer,
    QuantumSyncCore
)

class TestUnifiedQuantumField(unittest.TestCase):
    """Testa o campo quântico unificado"""
    
    def setUp(self):
        """Prepara o ambiente de teste"""
        self.field = UnifiedQuantumField(field_dimension=32)
    
    def test_initialization(self):
        """Testa inicialização do campo"""
        self.assertEqual(self.field.field_dimension, 32)
        self.assertEqual(len(self.field.morphic_fields), 3)
        self.assertTrue('main' in self.field.morphic_fields)
        self.assertTrue('black_hole' in self.field.morphic_fields)
        self.assertTrue('retrocausal' in self.field.morphic_fields)
    
    def test_field_synchronization(self):
        """Testa sincronização de campos"""
        coherence = self.field.synchronize_fields()
        self.assertIsInstance(coherence, float)
        self.assertGreaterEqual(coherence, 0.0)
        self.assertLessEqual(coherence, 1.0)
    
    def test_emergence_patterns(self):
        """Testa análise de padrões emergentes"""
        self.field.synchronize_fields()
        patterns = self.field.emergence_patterns
        
        self.assertIn('resonance', patterns)
        self.assertIn('entropy', patterns)
        self.assertIn('complexity', patterns)
        
        for value in patterns.values():
            self.assertIsInstance(value, float)
            self.assertGreaterEqual(value, 0.0)

class TestCycleSynchronizer(unittest.TestCase):
    """Testa o sincronizador de ciclos"""
    
    def setUp(self):
        """Prepara o ambiente de teste"""
        self.sync = CycleSynchronizer()
    
    def test_initialization(self):
        """Testa inicialização do sincronizador"""
        self.assertEqual(self.sync.evolution_phase, 0.0)
        self.assertEqual(self.sync.mining_phase, 0.0)
        self.assertEqual(self.sync.sync_factor, 0.0)
        self.assertEqual(len(self.sync.phase_history), 0)
    
    def test_phase_synchronization(self):
        """Testa sincronização de fases"""
        sync_factor = self.sync.synchronize_phases(
            evolution_rate=0.1,
            mining_rate=0.1
        )
        
        self.assertIsInstance(sync_factor, float)
        self.assertGreaterEqual(sync_factor, 0.0)
        self.assertLessEqual(sync_factor, 1.0)
        self.assertEqual(len(self.sync.phase_history), 1)
    
    def test_phase_alignment(self):
        """Testa cálculo de alinhamento de fases"""
        # Simula algumas fases
        for _ in range(10):
            self.sync.synchronize_phases(0.1, 0.1)
        
        alignment = self.sync.get_phase_alignment()
        self.assertIsInstance(alignment, float)
        self.assertGreaterEqual(alignment, 0.0)
        self.assertLessEqual(alignment, 1.0)

class TestGlobalOptimizer(unittest.TestCase):
    """Testa o otimizador global"""
    
    def setUp(self):
        """Prepara o ambiente de teste"""
        self.optimizer = GlobalOptimizer()
    
    def test_initialization(self):
        """Testa inicialização do otimizador"""
        self.assertEqual(len(self.optimizer.coherence_history), 0)
        self.assertEqual(len(self.optimizer.optimization_state), 0)
        self.assertEqual(len(self.optimizer.adaptation_metrics), 0)
    
    def test_system_optimization(self):
        """Testa otimização do sistema"""
        adjustments = self.optimizer.optimize_system(
            coherence=0.8,
            sync_factor=0.9,
            mining_metrics={'hash_rate': 100.0, 'volatility': 0.1}
        )
        
        self.assertIsInstance(adjustments, dict)
        self.assertIn('field_strength', adjustments)
        self.assertIn('coherence_target', adjustments)
        self.assertIn('sync_rate', adjustments)
        
        for value in adjustments.values():
            self.assertIsInstance(value, float)
            self.assertGreaterEqual(value, 0.1)
            self.assertLessEqual(value, 1.0)
    
    def test_trend_analysis(self):
        """Testa análise de tendências"""
        # Simula histórico de coerência
        self.optimizer.coherence_history = [0.5, 0.6, 0.7, 0.8, 0.9]
        
        trends = self.optimizer._analyze_trends()
        self.assertIn('slope', trends)
        self.assertIn('volatility', trends)
        
        self.assertGreater(trends['slope'], 0.0)  # Tendência positiva
        self.assertGreaterEqual(trends['volatility'], 0.0)

class TestQuantumSyncCore(unittest.TestCase):
    """Testa o núcleo de sincronização quântica"""
    
    def setUp(self):
        """Prepara o ambiente de teste"""
        self.core = QuantumSyncCore(field_dimension=32)
    
    def test_initialization(self):
        """Testa inicialização do núcleo"""
        self.assertEqual(self.core.field_dimension, 32)
        self.assertEqual(len(self.core.system_state), 0)
        self.assertEqual(len(self.core.sync_history), 0)
    
    def test_system_synchronization(self):
        """Testa sincronização do sistema"""
        mining_metrics = {
            'hash_rate': 100.0,
            'valid_shares': 10,
            'difficulty': 1.0,
            'efficiency': 0.8
        }
        
        state = self.core.synchronize_all_systems(
            mining_metrics=mining_metrics,
            evolution_rate=0.1
        )
        
        self.assertIn('coherence', state)
        self.assertIn('sync_factor', state)
        self.assertIn('mining_metrics', state)
        self.assertIn('evolution_rate', state)
        self.assertIn('adjustments', state)
        self.assertIn('timestamp', state)
    
    def test_mining_feedback(self):
        """Testa processamento de feedback de mineração"""
        mining_results = {
            'hash_rate': 100.0,
            'valid_shares': 10,
            'difficulty': 1.0,
            'efficiency': 0.8,
            'evolution_rate': 0.1
        }
        
        feedback = self.core.process_mining_feedback(mining_results)
        
        self.assertIn('field_strength', feedback)
        self.assertIn('coherence_target', feedback)
        self.assertIn('adaptation_rate', feedback)
    
    def test_evolution_state(self):
        """Testa atualização de estado evolutivo"""
        evolution_data = {
            'evolution_rate': 0.1,
            'mining_metrics': {
                'hash_rate': 100.0,
                'efficiency': 0.8
            }
        }
        
        state = self.core.update_evolution_state(evolution_data)
        
        self.assertIn('sync_state', state)
        self.assertIn('optimization', state)
        self.assertIn('evolution_data', state)
    
    def test_state_persistence(self):
        """Testa persistência de estado"""
        # Cria arquivo temporário
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            filepath = tmp.name
        
        try:
            # Simula estado
            self.core.system_state = {
                'coherence': 0.8,
                'sync_factor': 0.9,
                'timestamp': datetime.now().isoformat()
            }
            
            # Salva estado
            success = self.core.save_state(filepath)
            self.assertTrue(success)
            
            # Limpa estado
            self.core.system_state = {}
            
            # Carrega estado
            success = self.core.load_state(filepath)
            self.assertTrue(success)
            
            # Verifica estado carregado
            self.assertIn('coherence', self.core.system_state)
            self.assertIn('sync_factor', self.core.system_state)
            self.assertIn('timestamp', self.core.system_state)
            
        finally:
            # Limpa arquivo temporário
            Path(filepath).unlink()

if __name__ == '__main__':
    unittest.main() 