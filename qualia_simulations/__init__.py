"""
QUALIA - Sistema de Simulação Quântico-Cosmológica Adaptativa
"""

from .quantum_cosmological_simulator import QuantumFieldSimulator
from .cosmological_evolution import CosmologicalEvolution
from .cosmic_dance import main, CellularAutomaton, AdaptiveQuantumCosmoIntegrator

__all__ = [
    'QuantumFieldSimulator',
    'CosmologicalEvolution',
    'main',
    'CellularAutomaton',
    'AdaptiveQuantumCosmoIntegrator'
] 