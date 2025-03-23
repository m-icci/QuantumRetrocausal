"""
Dark Portfolio Integration
------------------------
Integra fatores ocultos e campos mórficos na otimização de portfólio.
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

from quantum.core.state.quantum_state import QuantumState
from quantum.core.qtypes.quantum_types import ConsciousnessObservation
from quantum.core.portfolio.sacred.geometry import SacredGeometry, SacredPattern

@dataclass
class DarkMetrics:
    """Métricas ocultas do portfólio"""
    dark_risk: float
    growth_potential: float
    field_strength: float
    coherence: float
    resonance: float
    timestamp: datetime = datetime.now()

class DarkPortfolioIntegrator:
    """
    Integrador de fatores ocultos para portfólio.
    Implementa:
    - Análise de fatores ocultos
    - Campos mórficos
    - Ressonância quântica
    """

    def __init__(self, 
                 dimensions: int = 8,
                 field_coupling: float = 0.1):
        """
        Inicializa integrador

        Args:
            dimensions: Dimensões do espaço quântico
            field_coupling: Força de acoplamento com campo mórfico
        """
        self.dimensions = dimensions
        self.field_coupling = field_coupling
        self.sacred_geometry = SacredGeometry(dimensions)
        self.metrics_history: List[DarkMetrics] = []

    def calculate_dark_metrics(self,
                            returns: np.ndarray,
                            volumes: np.ndarray,
                            consciousness_obs: ConsciousnessObservation) -> DarkMetrics:
        """
        Calcula métricas ocultas do mercado

        Args:
            returns: Matriz de retornos
            volumes: Matriz de volumes
            consciousness_obs: Observação de consciência

        Returns:
            Métricas ocultas calculadas
        """
        # Calcula fatores ocultos
        dark_risk = self._calculate_dark_risk(returns, consciousness_obs)
        growth_potential = self._calculate_growth_potential(returns, volumes)

        # Calcula métricas de campo
        field_strength = self._calculate_field_strength(returns, consciousness_obs)
        coherence = consciousness_obs.get_coherence()
        resonance = self._calculate_resonance(returns, field_strength)

        # Cria métricas
        metrics = DarkMetrics(
            dark_risk=dark_risk,
            growth_potential=growth_potential,
            field_strength=field_strength,
            coherence=coherence,
            resonance=resonance
        )

        # Armazena histórico
        self.metrics_history.append(metrics)

        return metrics

    def _calculate_dark_risk(self,
                           returns: np.ndarray,
                           consciousness_obs: ConsciousnessObservation) -> float:
        """Calcula risco oculto"""
        # Calcula volatilidade tradicional
        vol = np.std(returns, axis=0)

        # Ajusta por consciência quântica
        qualia_intensity = consciousness_obs.qualia.intensity
        qualia_coherence = consciousness_obs.qualia.coherence
        qualia_complexity = consciousness_obs.qualia.complexity

        # Calcula ajuste quântico
        dark_adjustment = (qualia_intensity * qualia_coherence * 
                         np.exp(-qualia_complexity)) * self.sacred_geometry.phi

        # Calcula risco oculto
        dark_risk = np.mean(vol * dark_adjustment)

        return float(dark_risk)

    def _calculate_growth_potential(self,
                                  returns: np.ndarray,
                                  volumes: np.ndarray) -> float:
        """Calcula potencial de crescimento oculto"""
        # Calcula retorno ajustado por volume
        volume_weighted_returns = returns * np.log1p(volumes)

        # Aplica transformação φ
        transformed_returns = np.exp(
            1j * 2 * np.pi * self.sacred_geometry.phi * volume_weighted_returns
        )

        # Calcula potencial como coerência da transformação
        potential = np.abs(np.mean(transformed_returns))

        return float(potential)

    def _calculate_field_strength(self,
                                returns: np.ndarray,
                                consciousness_obs: ConsciousnessObservation) -> float:
        """
        Calcula a força do campo mórfico

        Args:
            returns: Dados de retorno
            consciousness_obs: Observação de consciência

        Returns:
            Força do campo mórfico
        """
        # Verifica se há dados suficientes
        if returns is None or len(returns) <= 1:
            return 0.0

        # Prepara estado quântico
        quantum_state = self._prepare_quantum_state(returns, consciousness_obs)

        # Gera matriz sagrada
        field_matrix = self.sacred_geometry.generate_sacred_matrix(
            max(2, len(returns[0]) if returns.ndim > 1 else 2)
        )

        # Calcula força do campo
        field_strength = np.trace(field_matrix) / np.linalg.norm(field_matrix)

        # Ajusta por consciência
        qualia_intensity = consciousness_obs.qualia.intensity
        qualia_coherence = consciousness_obs.qualia.coherence

        return field_strength * qualia_intensity * qualia_coherence

    def _calculate_resonance(self,
                           returns: np.ndarray,
                           field_strength: float) -> float:
        """Calcula ressonância com campo mórfico"""
        # Calcula correlações
        corr_matrix = np.corrcoef(returns.T)

        # Aplica transformação φ
        transformed_corr = np.exp(
            1j * 2 * np.pi * self.sacred_geometry.phi * corr_matrix
        )

        # Calcula ressonância
        resonance = np.abs(np.mean(transformed_corr)) * field_strength

        return float(resonance)

    def _prepare_quantum_state(self,
                             returns: np.ndarray,
                             consciousness_obs: ConsciousnessObservation) -> QuantumState:
        """Prepara estado quântico para análise"""
        # Garante que returns seja 2D
        if returns.ndim == 0:
            return QuantumState(
                dimensions=self.dimensions,
                state_vector=np.zeros(self.dimensions, dtype=complex)
            )
        elif returns.ndim == 1:
            returns = returns[:, np.newaxis]

        try:
            # Calcula componentes principais
            covariance_matrix = np.cov(returns.T)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

            # Normaliza autovalores
            probabilities = eigenvalues / np.sum(eigenvalues)

            # Cria vetor de estado quântico
            state_vector = np.zeros(self.dimensions, dtype=complex)

            # Mapeia os primeiros autovalores/autovetores para o estado quântico
            for i in range(min(len(probabilities), self.dimensions)):
                state_vector[i] = np.sqrt(probabilities[i]) * eigenvectors[i, 0]

            # Normaliza o vetor de estado
            state_vector /= np.linalg.norm(state_vector)

            return QuantumState(
                dimensions=self.dimensions,
                state_vector=state_vector
            )

        except (np.linalg.LinAlgError, ValueError):
            # Em caso de erro, retorna estado base
            return QuantumState(
                dimensions=self.dimensions,
                state_vector=np.zeros(self.dimensions, dtype=complex)
            )

    def adjust_weights(self,
                      weights: np.ndarray,
                      dark_metrics: DarkMetrics) -> np.ndarray:
        """
        Ajusta pesos por fatores ocultos

        Args:
            weights: Pesos originais
            dark_metrics: Métricas ocultas

        Returns:
            Pesos ajustados
        """
        # Calcula ajuste de risco
        risk_adjustment = 1 - dark_metrics.dark_risk

        # Calcula ajuste de crescimento
        growth_adjustment = dark_metrics.growth_potential

        # Calcula ajuste de campo
        field_adjustment = dark_metrics.field_strength * dark_metrics.coherence

        # Aplica ajustes
        adjusted_weights = weights * (
            risk_adjustment *
            (1 + growth_adjustment) *
            (1 + field_adjustment * self.sacred_geometry.phi)
        )

        # Normaliza
        adjusted_weights = adjusted_weights / np.sum(adjusted_weights)

        return adjusted_weights

    def get_metrics_history(self) -> List[DarkMetrics]:
        """Retorna histórico de métricas"""
        return self.metrics_history