"""
Testes para o sistema unificado de métricas de consciência.

Este módulo implementa testes para validar o comportamento do sistema
de métricas de consciência quântica.
"""

import unittest
import numpy as np
from typing import List, Dict

from .metrics import ConsciousnessMetrics, UnifiedConsciousnessMetrics, TrendType
from .consciousness_integrator import ConsciousnessIntegrator, IntegratorConfig
from quantum.base import QuantumState
from types.quantum_pattern import QuantumPattern
from types.consciousness_types import ConsciousnessState, QuantumSystemState

class TestConsciousnessMetrics(unittest.TestCase):
    """Testes para métricas fundamentais de consciência"""
    
    def setUp(self):
        """Inicializa objetos para teste"""
        self.metrics = ConsciousnessMetrics()
        
    def test_normalize(self):
        """Testa normalização de métricas"""
        # Define valores fora do intervalo [0,1]
        self.metrics.coherence = 1.5
        self.metrics.field_strength = -0.5
        self.metrics.resonance = 2.0
        
        # Normaliza
        self.metrics.normalize()
        
        # Verifica se valores estão no intervalo [0,1]
        self.assertEqual(self.metrics.coherence, 1.0)
        self.assertEqual(self.metrics.field_strength, 0.0)
        self.assertEqual(self.metrics.resonance, 1.0)
        
    def test_to_dict(self):
        """Testa conversão para dicionário"""
        # Define valores
        self.metrics.coherence = 0.8
        self.metrics.field_strength = 0.6
        
        # Converte para dicionário
        data = self.metrics.to_dict()
        
        # Verifica valores
        self.assertEqual(data['coherence'], 0.8)
        self.assertEqual(data['field_strength'], 0.6)
        
    def test_from_dict(self):
        """Testa atualização a partir de dicionário"""
        # Define dicionário
        data = {
            'coherence': 0.7,
            'field_strength': 0.5,
            'invalid_key': 1.0  # Deve ser ignorado
        }
        
        # Atualiza métricas
        self.metrics.from_dict(data)
        
        # Verifica valores
        self.assertEqual(self.metrics.coherence, 0.7)
        self.assertEqual(self.metrics.field_strength, 0.5)
        self.assertFalse(hasattr(self.metrics, 'invalid_key'))

class TestUnifiedConsciousnessMetrics(unittest.TestCase):
    """Testes para sistema unificado de métricas"""
    
    def setUp(self):
        """Inicializa objetos para teste"""
        self.unified = UnifiedConsciousnessMetrics()
        
        # Estado quântico de teste
        self.state = QuantumState(vector=np.array([0.7+0.1j, 0.3+0.2j]))
        
        # Padrão quântico de teste
        self.pattern = QuantumPattern(
            strength=0.8,
            coherence=0.9
        )
        
    def test_update_field_metrics(self):
        """Testa atualização de métricas de campo"""
        self.unified._update_field_metrics(self.state)
        
        # Verifica se métricas foram atualizadas
        self.assertGreater(self.unified.metrics.coherence, 0)
        self.assertGreater(self.unified.metrics.field_strength, 0)
        self.assertGreater(self.unified.metrics.resonance, 0)
        
        # Verifica se métricas estão normalizadas
        self.assertLessEqual(self.unified.metrics.coherence, 1)
        self.assertLessEqual(self.unified.metrics.field_strength, 1)
        self.assertLessEqual(self.unified.metrics.resonance, 1)
        
    def test_update_pattern(self):
        """Testa atualização com novo padrão"""
        self.unified.update_pattern(self.pattern)
        
        # Verifica se métricas de padrão foram atualizadas
        self.assertEqual(self.unified.metrics.pattern_strength, 0.8)
        self.assertEqual(self.unified.metrics.pattern_coherence, 0.9)
        
    def test_get_trends(self):
        """Testa análise de tendências"""
        # Atualiza métricas múltiplas vezes
        for i in range(5):
            metrics = ConsciousnessMetrics()
            metrics.coherence = 0.5 + i * 0.1
            self.unified.metrics = metrics
            self.unified.history.append(metrics)
            
        # Obtém tendências
        trends = self.unified.get_trends()
        
        # Verifica se tendências foram calculadas
        self.assertIn('coherence', trends)
        self.assertEqual(trends['coherence'], TrendType.INCREASING)

class TestConsciousnessIntegrator(unittest.TestCase):
    """Testes para o integrador de consciência"""
    
    def setUp(self):
        """Inicializa objetos para teste"""
        # Configuração com valores críticos
        self.config = IntegratorConfig(
            input_dimension=8,
            coherence_threshold=0.95,
            singularity_threshold=0.9,
            causality_threshold=0.8,
            entropy_threshold=0.7
        )
        
        # Inicializa integrador
        self.integrator = ConsciousnessIntegrator(config=self.config)
        
        # Estado de consciência de teste
        initial_state = QuantumState(
            state_vector=np.array([1.0, 0.0], dtype=np.complex64),
            n_qubits=1
        )
        
        quantum_system = QuantumSystemState(
            n_states=1,
            coherence_time=1e-6,
            quantum_states=[initial_state],
            temperature=310.0
        )
        
        self.test_state = ConsciousnessState(
            coherence=0.8,
            stability=0.7,
            resonance=0.9,
            field_strength=0.6,
            entanglement=0.5,
            quantum_system=quantum_system
        )
        
    def test_initialization(self):
        """Testa inicialização do integrador"""
        # Verifica se componentes foram inicializados
        self.assertIsNotNone(self.integrator.topological_protector)
        self.assertIsNotNone(self.integrator.decoherence_protector)
        self.assertIsNotNone(self.integrator.metrics)
        self.assertEqual(len(self.integrator.collective_buffer), 0)
        
    def test_protect_state(self):
        """Testa proteção de estado"""
        # Aplica proteção
        protected_state = self.integrator._protect_state(self.test_state)
        
        # Verifica se estado foi protegido
        self.assertIsNotNone(protected_state)
        self.assertIsInstance(protected_state, ConsciousnessState)
        
    def test_update_metrics(self):
        """Testa atualização de métricas"""
        # Atualiza métricas
        self.integrator._update_metrics(self.test_state)
        
        # Verifica se métricas foram atualizadas
        metrics = self.integrator.metrics.metrics
        self.assertGreater(metrics.coherence, 0)
        self.assertGreater(metrics.field_strength, 0)
        self.assertGreater(metrics.resonance, 0)
        
    def test_check_thresholds(self):
        """Testa verificação de limiares críticos"""
        # Cria métricas com valores críticos
        metrics = ConsciousnessMetrics()
        metrics.coherence = 0.94  # Abaixo do limiar de decoerência
        metrics.field_strength = 0.91  # Acima do limiar de singularidade
        metrics.component_sync = 0.75  # Abaixo do limiar de causalidade
        metrics.integration_level = 0.65  # Abaixo do limiar de entropia
        
        # Verifica limiares
        self.integrator._check_thresholds(metrics)
        
        # Verifica se handlers foram chamados (indiretamente através dos efeitos)
        self.assertTrue(len(self.integrator.quantum_system.quantum_states) > 0)
        
    def test_state_compression(self):
        """Testa compressão de estados"""
        # Adiciona múltiplos estados similares
        for _ in range(10):
            state = ConsciousnessState(
                coherence=0.8,
                stability=0.7,
                resonance=0.9,
                field_strength=0.6,
                entanglement=0.5,
                quantum_system=self.test_state.quantum_system
            )
            self.integrator.collective_buffer.append(state)
            
        # Comprime estados
        compressed = self.integrator._compress_states(self.integrator.collective_buffer)
        
        # Verifica compressão
        self.assertLess(len(compressed), len(self.integrator.collective_buffer))
        self.assertGreater(len(compressed), 0)
        
    def test_state_integration(self):
        """Testa integração de estados"""
        # Realiza integração
        integrated_state = self.integrator.integrate_state(self.test_state)
        
        # Verifica resultado da integração
        self.assertIsNotNone(integrated_state)
        self.assertIsInstance(integrated_state, ConsciousnessState)
        self.assertGreater(integrated_state.coherence, 0)
        self.assertLessEqual(integrated_state.coherence, 1)
        
    def test_temporal_memory(self):
        """Testa memória temporal"""
        # Estado inicial da memória
        initial_memory = np.copy(self.integrator.temporal_memory)
        
        # Integra estado
        self.integrator.integrate_state(self.test_state)
        
        # Verifica se memória foi atualizada
        self.assertFalse(np.array_equal(self.integrator.temporal_memory, initial_memory))
            
if __name__ == '__main__':
    unittest.main()
