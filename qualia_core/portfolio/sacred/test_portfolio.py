"""
Testes para o sistema de portfólio quântico
"""

import pytest
import numpy as np
from datetime import datetime

from quantum.core.qtypes.quantum_state import QuantumState
from quantum.core.qtypes.quantum_types import ConsciousnessObservation, QualiaState, SystemBehavior
from quantum.core.portfolio.sacred.geometry import SacredGeometry, SacredPattern
from quantum.core.portfolio.sacred.optimizer import QuantumPortfolioOptimizer
from quantum.core.portfolio.sacred.dark_integration import DarkPortfolioIntegrator

class TestSacredGeometry:
    """Testes para geometria sagrada"""
    
    def setup_method(self):
        """Setup para testes"""
        self.geometry = SacredGeometry(dimensions=4)
        
    def test_sacred_matrix(self):
        """Testa geração de matriz sagrada"""
        matrix = self.geometry.generate_sacred_matrix(3)
        
        # Verifica dimensões
        assert matrix.shape == (3, 3)
        
        # Verifica normalização
        assert np.abs(np.linalg.norm(matrix) - 1.0) < 1e-6
        
    def test_harmony_calculation(self):
        """Testa cálculo de harmonia"""
        weights = np.array([0.4, 0.3, 0.3])
        harmony = self.geometry.calculate_harmony(weights)
        
        # Verifica range
        assert 0 <= harmony <= 1
        
    def test_pattern_identification(self):
        """Testa identificação de padrões"""
        # Proporções φ mais precisas
        phi = (1 + np.sqrt(5)) / 2
        weights = np.array([1/phi, 1/phi**2, 1/phi**3, 1/phi**4])
        weights = weights / np.sum(weights)  # Normaliza
        
        returns = np.random.randn(100, 4)
        
        patterns = self.geometry.identify_patterns(weights, returns)
        
        # Verifica se encontrou padrão φ
        assert any(p.name == "golden_ratio" for p in patterns)
        
class TestQuantumOptimizer:
    """Testes para otimizador quântico"""
    
    def setup_method(self):
        """Setup para testes"""
        self.assets = ['BTC', 'ETH', 'ADA']
        self.optimizer = QuantumPortfolioOptimizer(
            assets=self.assets,
            dimensions=4
        )
        
    def test_portfolio_optimization(self):
        """Testa otimização de portfólio"""
        # Dados simulados
        returns = np.random.randn(100, 3)
        consciousness = ConsciousnessObservation(
            qualia=QualiaState(
                intensity=0.8,
                complexity=0.6,
                coherence=0.7
            ),
            behavior=SystemBehavior(
                pattern_type="resonant",
                frequency=0.5,
                stability=0.8
            ),
            quantum_state=None
        )
        
        # Otimiza portfólio
        state = self.optimizer.optimize(returns, consciousness)
        
        # Verifica pesos
        assert len(state.weights) == 3
        assert np.abs(np.sum(state.weights) - 1.0) < 1e-6
        assert all(0 <= w <= 1 for w in state.weights)
        
    def test_rebalance_calculation(self):
        """Testa cálculo de rebalanceamento"""
        # Otimiza primeiro
        returns = np.random.randn(100, 3)
        consciousness = ConsciousnessObservation(
            qualia=QualiaState(
                intensity=0.8,
                complexity=0.6,
                coherence=0.7
            ),
            behavior=SystemBehavior(
                pattern_type="resonant",
                frequency=0.5,
                stability=0.8
            ),
            quantum_state=None
        )
        self.optimizer.optimize(returns, consciousness)
        
        # Calcula rebalanceamento
        current_weights = np.array([0.4, 0.3, 0.3])
        needs = self.optimizer.calculate_rebalance_needs(
            current_weights,
            tolerance=0.1
        )
        
        # Verifica formato
        assert isinstance(needs, dict)
        assert all(asset in self.assets for asset in needs.keys())
        
class TestDarkIntegration:
    """Testes para integração dark"""
    
    def setup_method(self):
        """Setup para testes"""
        self.integrator = DarkPortfolioIntegrator(dimensions=4)
        
    def test_dark_metrics_calculation(self):
        """Testa cálculo de métricas ocultas"""
        # Dados simulados
        returns = np.random.randn(100, 3)
        volumes = np.random.exponential(1, size=(100, 3))
        consciousness = ConsciousnessObservation(
            qualia=QualiaState(
                intensity=0.8,
                complexity=0.6,
                coherence=0.7
            ),
            behavior=SystemBehavior(
                pattern_type="resonant",
                frequency=0.5,
                stability=0.8
            ),
            quantum_state=None
        )
        
        # Calcula métricas
        metrics = self.integrator.calculate_dark_metrics(
            returns,
            volumes,
            consciousness
        )
        
        # Verifica métricas
        assert 0 <= metrics.dark_risk <= 1
        assert 0 <= metrics.growth_potential <= 1
        assert 0 <= metrics.field_strength <= 1
        assert 0 <= metrics.coherence <= 1
        assert 0 <= metrics.resonance <= 1
        
    def test_weight_adjustment(self):
        """Testa ajuste de pesos"""
        # Dados simulados
        weights = np.array([0.4, 0.3, 0.3])
        metrics = self.integrator.calculate_dark_metrics(
            np.random.randn(100, 3),
            np.random.exponential(1, size=(100, 3)),
            ConsciousnessObservation(
                qualia=QualiaState(
                    intensity=0.8,
                    complexity=0.6,
                    coherence=0.7
                ),
                behavior=SystemBehavior(
                    pattern_type="resonant",
                    frequency=0.5,
                    stability=0.8
                ),
                quantum_state=None
            )
        )
        
        # Ajusta pesos
        adjusted = self.integrator.adjust_weights(weights, metrics)
        
        # Verifica ajustes
        assert len(adjusted) == 3
        assert np.abs(np.sum(adjusted) - 1.0) < 1e-6
        assert all(0 <= w <= 1 for w in adjusted)