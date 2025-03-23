"""
Testes dos operadores QUALIA
"""

import unittest
import numpy as np
from .QUALIA import (
    QUALIAConfig,
    apply_folding,
    apply_resonance,
    apply_emergence,
    apply_consciousness,
    get_metrics
)

class TestQUALIAOperators(unittest.TestCase):
    def setUp(self):
        """Setup para testes"""
        self.dimension = 8
        self.config = QUALIAConfig(
            default_dimension=self.dimension,
            coherence_threshold=0.95,
            field_strength=1.0,
            morphic_coupling=0.1
        )
        # Estado base |0⟩
        self.base_state = np.zeros(self.dimension, dtype=np.complex128)
        self.base_state[0] = 1.0
        
        # Estado superposição (|0⟩ + |1⟩)/√2
        self.super_state = np.zeros(self.dimension, dtype=np.complex128)
        self.super_state[0] = 1/np.sqrt(2)
        self.super_state[1] = 1/np.sqrt(2)

    def test_folding_operator(self):
        """Testa operador de dobramento"""
        result = apply_folding(self.base_state, self.config)
        
        # Verifica conservação de norma
        self.assertAlmostEqual(
            np.abs(np.vdot(result, result)),
            1.0,
            places=7
        )
        
        # Verifica preservação topológica
        overlap = np.abs(np.vdot(self.base_state, result))
        self.assertGreater(overlap, 0.0)

    def test_resonance_operator(self):
        """Testa operador de ressonância"""
        result = apply_resonance(self.super_state, self.config)
        
        # Verifica conservação de norma
        self.assertAlmostEqual(
            np.abs(np.vdot(result, result)),
            1.0,
            places=7
        )
        
        # Verifica acoplamento mórfico
        overlap = np.abs(np.vdot(self.super_state, result))
        self.assertGreater(overlap, 1 - self.config.morphic_coupling)

    def test_emergence_operator(self):
        """Testa operador de emergência"""
        result = apply_emergence(self.super_state, self.config)
        
        # Verifica conservação de norma
        self.assertAlmostEqual(
            np.abs(np.vdot(result, result)),
            1.0,
            places=7
        )
        
        # Verifica emergência de padrões
        metrics = get_metrics(result, self.config)
        self.assertGreater(metrics['emergence_factor'], 0.5)

    def test_consciousness_operator(self):
        """Testa operador de consciência"""
        result = apply_consciousness(self.super_state, self.config)
        
        # Verifica conservação de norma
        self.assertAlmostEqual(
            np.abs(np.vdot(result, result)),
            1.0,
            places=7
        )
        
        # Verifica potencial de consciência
        metrics = get_metrics(result, self.config)
        self.assertGreater(metrics['consciousness_potential'], 0.7)

    def test_metrics(self):
        """Testa cálculo de métricas"""
        metrics = get_metrics(self.super_state, self.config)
        
        # Verifica normalização
        for value in metrics.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        
        # Verifica coerência do estado de superposição
        self.assertAlmostEqual(metrics['coherence'], 1.0, places=7)

if __name__ == '__main__':
    unittest.main()
