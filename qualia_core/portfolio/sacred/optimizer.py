"""
Quantum Portfolio Optimizer with Sacred Geometry
---------------------------------------------
Otimizador de portfólio usando princípios quânticos e geometria sagrada.
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

from quantum.core.state.quantum_state import QuantumState
from quantum.core.qtypes.quantum_types import ConsciousnessObservation
from quantum.core.portfolio.sacred.geometry import SacredGeometry, SacredPattern

@dataclass
class PortfolioState:
    """Estado do portfólio"""
    weights: np.ndarray
    returns: float
    volatility: float
    harmony: float
    consciousness: complex
    patterns: List[SacredPattern]
    timestamp: datetime = datetime.now()

class QuantumPortfolioOptimizer:
    """
    Otimizador de portfólio com consciência quântica e geometria sagrada.
    Integra:
    - Geometria sagrada
    - Consciência quântica
    - Campos mórficos
    """

    def __init__(self, 
                 assets: List[str],
                 dimensions: int = 8,
                 risk_tolerance: float = 0.1):
        """
        Inicializa otimizador

        Args:
            assets: Lista de ativos
            dimensions: Dimensões do espaço quântico
            risk_tolerance: Tolerância a risco (0-1)
        """
        self.assets = assets
        self.dimensions = dimensions
        self.risk_tolerance = risk_tolerance

        # Componentes
        self.sacred_geometry = SacredGeometry(dimensions)
        self.state_history: List[PortfolioState] = []

    def optimize(self,
                returns: np.ndarray,
                consciousness_obs: ConsciousnessObservation,
                constraints: Optional[Dict] = None) -> PortfolioState:
        """
        Otimiza portfólio com consciência quântica

        Args:
            returns: Matriz de retornos históricos
            consciousness_obs: Observação de consciência
            constraints: Restrições opcionais

        Returns:
            Estado otimizado do portfólio
        """
        # Prepara estado quântico
        quantum_state = self._prepare_quantum_state(returns, consciousness_obs)

        # Gera matriz sagrada
        sacred_matrix = self.sacred_geometry.generate_sacred_matrix(len(self.assets))

        # Aplica transformação quântica
        weights = self._quantum_transform(sacred_matrix, quantum_state)

        # Aplica restrições
        if constraints:
            weights = self._apply_constraints(weights, constraints)

        # Calcula métricas
        portfolio_return = np.sum(returns.mean(axis=0) * weights)
        portfolio_vol = np.sqrt(weights.T @ np.cov(returns.T) @ weights)

        # Identifica padrões
        patterns = self.sacred_geometry.identify_patterns(weights, returns)

        # Calcula harmonia
        harmony = self.sacred_geometry.calculate_harmony(weights)

        # Cria estado do portfólio
        state = PortfolioState(
            weights=weights,
            returns=portfolio_return,
            volatility=portfolio_vol,
            harmony=harmony,
            consciousness=consciousness_obs,
            patterns=patterns
        )

        # Armazena histórico
        self.state_history.append(state)

        return state

    def _prepare_quantum_state(self,
                             returns: np.ndarray,
                             consciousness_obs: ConsciousnessObservation) -> QuantumState:
        """Prepara estado quântico inicial"""
        # Calcula componentes principais
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(returns.T))

        # Normaliza autovalores
        probabilities = eigenvalues / np.sum(eigenvalues)

        # Cria estado quântico
        state_vector = np.sqrt(probabilities) * np.exp(
            1j * 2 * np.pi * self.sacred_geometry.phi * np.arange(len(probabilities))
        )

        # Integra consciência
        qualia_intensity = consciousness_obs.qualia.intensity
        qualia_coherence = consciousness_obs.qualia.coherence
        qualia_complexity = consciousness_obs.qualia.complexity

        # Aplica modulação quântica
        state_vector *= (qualia_intensity * qualia_coherence * 
                        np.exp(1j * 2 * np.pi * qualia_complexity))

        return QuantumState(
            dimensions=self.dimensions,
            state_vector=state_vector
        )

    def _quantum_transform(self,
                         sacred_matrix: np.ndarray,
                         quantum_state: QuantumState) -> np.ndarray:
        """Aplica transformação quântica nos pesos"""
        # Aplica matriz sagrada
        transformed = sacred_matrix @ quantum_state.state_vector

        # Extrai pesos das amplitudes
        weights = np.abs(transformed) ** 2

        # Normaliza
        weights = weights / np.sum(weights)

        return weights

    def _apply_constraints(self,
                         weights: np.ndarray,
                         constraints: Dict) -> np.ndarray:
        """Aplica restrições nos pesos"""
        # Restrição de soma unitária
        weights = weights / np.sum(weights)

        # Limites por ativo
        if 'limits' in constraints:
            limits = constraints['limits']
            weights = np.clip(weights, limits['min'], limits['max'])
            weights = weights / np.sum(weights)

        return weights

    def get_optimization_history(self) -> List[PortfolioState]:
        """Retorna histórico de otimizações"""
        return self.state_history

    def calculate_rebalance_needs(self,
                                current_weights: np.ndarray,
                                tolerance: float = 0.1) -> Dict[str, float]:
        """
        Calcula necessidades de rebalanceamento

        Args:
            current_weights: Pesos atuais
            tolerance: Tolerância para rebalanceamento

        Returns:
            Dicionário com ajustes necessários
        """
        if not self.state_history:
            return {}

        # Obtém última otimização
        target_weights = self.state_history[-1].weights

        # Calcula desvios
        deviations = target_weights - current_weights

        # Filtra ajustes necessários
        rebalance_needs = {}
        for asset, dev in zip(self.assets, deviations):
            if abs(dev) > tolerance:
                rebalance_needs[asset] = float(dev)

        return rebalance_needs