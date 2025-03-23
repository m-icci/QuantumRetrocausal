"""
Risk analysis implementation for QUALIA Trading System
Integrates quantum risk assessment with classical financial metrics
"""
from typing import Dict, Any, Optional, Tuple, Iterator
import numpy as np
import pandas as pd
from scipy import stats
import logging
from dataclasses import dataclass, asdict
import logging
from .quantum_analysis import QuantumState

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Enhanced risk metrics data class with dict-like behavior"""
    risk_level: float
    volatility: float
    var_95: float
    var_99: float
    es_95: float
    es_99: float
    quantum_adjusted_risk: float
    market_regime: str
    stability_index: float

    def __iter__(self) -> Iterator[str]:
        """Make the class iterable over its field names"""
        return iter(self.__dataclass_fields__)

    def items(self):
        """Provide dict-like items() method"""
        return asdict(self).items()

    def values(self):
        """Provide dict-like values() method"""
        return asdict(self).values()

    def get(self, key: str, default: Any = None) -> Any:
        """Provide dict-like get method"""
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        """Support dictionary-style access"""
        return getattr(self, key)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return asdict(self)

class RiskAnalyzer:
    """Enhanced risk analysis with integrated quantum metrics"""

    def __init__(
        self,
        market_data_provider = None,
        config: Dict[str, Any] = None,
        memory_length: int = 100,
        confidence_level: float = 0.95
    ):
        # Merge config com defaults
        default_config = {
            'volatility_window': 20,
            'max_risk_score': 1.0,
            'risk_threshold': 0.7,
            'volume_threshold': 1000,
            'price_change_threshold': 0.05
        }
        self.config = {**default_config, **(config or {})}
        
        # Configurações básicas
        self.memory_length = memory_length
        self.confidence_level = confidence_level
        self.risk_memory = []
        self.epsilon = 1e-10  # Numerical stability threshold
        
        # Configurações de risco
        self.volatility_window = self.config['volatility_window']
        self.max_risk_score = self.config['max_risk_score']
        self.risk_threshold = self.config['risk_threshold']
        self.volume_threshold = self.config['volume_threshold']
        self.price_change_threshold = self.config['price_change_threshold']
        
        # Thresholds de regime
        self.regime_thresholds = {
            'high_risk': 0.7,
            'medium_risk': 0.4,
            'low_risk': 0.2
        }
        
        # Market Data Provider
        self.market_data_provider = market_data_provider
        if market_data_provider:
            logger.info(f"Market Data Provider configurado: {type(market_data_provider).__name__}")

    def calculate_risk_metrics(
        self,
        market_data: Optional[np.ndarray] = None,
        quantum_metrics: Optional[Dict[str, float]] = None
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics with quantum adjustment
        """
        try:
            # Initialize with default values
            metrics = {
                'volatility': 0.5,
                'var_95': -0.1,
                'var_99': -0.15,
                'es_95': -0.12,
                'es_99': -0.18,
                'risk_level': 0.5,
                'stability_index': 0.5
            }

            # Calculate metrics if market data is available
            if market_data is not None and len(market_data) > 1:
                returns = np.diff(np.log(np.abs(market_data) + self.epsilon))
                if len(returns) > 0:
                    metrics['volatility'] = float(np.std(returns) + self.epsilon)
                    metrics['var_95'] = float(np.percentile(returns, 5))
                    metrics['var_99'] = float(np.percentile(returns, 1))

                    # Safe calculation of expected shortfall
                    var_95_mask = returns <= metrics['var_95']
                    var_99_mask = returns <= metrics['var_99']

                    if np.any(var_95_mask):
                        metrics['es_95'] = float(np.mean(returns[var_95_mask]))
                    if np.any(var_99_mask):
                        metrics['es_99'] = float(np.mean(returns[var_99_mask]))

            # Adjust risk based on quantum metrics
            quantum_factor = 0.0
            if quantum_metrics:
                coherence = quantum_metrics.get('coherence', 0.5)
                market_stability = quantum_metrics.get('market_stability', 0.5)
                quantum_factor = (1 - coherence) * (1 - market_stability)

            # Calculate final risk level
            risk_level = min(1.0, max(0.0, metrics['volatility'] * (1 + quantum_factor)))
            stability_index = min(1.0, max(0.0, 1.0 - risk_level))

            metrics = RiskMetrics(
                risk_level=risk_level,
                volatility=metrics['volatility'],
                var_95=metrics['var_95'],
                var_99=metrics['var_99'],
                es_95=metrics['es_95'],
                es_99=metrics['es_99'],
                quantum_adjusted_risk=risk_level * (1 + quantum_factor),
                market_regime=self._determine_market_regime(risk_level, stability_index),
                stability_index=stability_index
            )

            # Ensure all metrics are finite
            metrics_dict = asdict(metrics)
            for key, value in metrics_dict.items():
                if isinstance(value, (int, float)) and not np.isfinite(value):
                    metrics_dict[key] = 0.5

            return RiskMetrics(**metrics_dict)

        except Exception as e:
            logger.error(f"Risk metrics calculation failed: {str(e)}")
            return RiskMetrics(
                risk_level=0.5,
                volatility=0.5,
                var_95=-0.1,
                var_99=-0.15,
                es_95=-0.12,
                es_99=-0.18,
                quantum_adjusted_risk=0.5,
                market_regime='medium_risk',
                stability_index=0.5
            )

    def _determine_market_regime(
        self,
        risk_score: float,
        stability_index: float
    ) -> str:
        """
        Determine current market regime using risk score and stability
        """
        try:
            # Adjust thresholds based on stability
            adjusted_high = self.regime_thresholds['high_risk'] * (1 - stability_index * 0.3)
            adjusted_medium = self.regime_thresholds['medium_risk'] * (1 - stability_index * 0.2)

            if risk_score > adjusted_high:
                return 'high_risk'
            elif risk_score > adjusted_medium:
                return 'medium_risk'
            return 'low_risk'

        except Exception as e:
            logger.error(f"Market regime determination failed: {str(e)}")
            return 'medium_risk'

    def assess_risk_regime(
        self,
        risk_metrics: RiskMetrics,
        quantum_state: Optional[QuantumState] = None
    ) -> Dict[str, Any]:
        """
        Assess current market risk regime with quantum enhancement
        """
        try:
            description = {
                'high_risk': 'High risk environment - exercise extreme caution',
                'medium_risk': 'Elevated risk levels - maintain defensive position',
                'low_risk': 'Normal market conditions - standard risk management'
            }

            # Generate quantum-enhanced recommendations
            recommendations = []

            # Risk trend analysis
            if len(self.risk_memory) >= 2:
                risk_trend = self.risk_memory[-1] - self.risk_memory[-2]
                if risk_trend > 0:
                    recommendations.append('Risk is increasing - consider reducing exposure')
                elif risk_trend < 0:
                    recommendations.append('Risk is decreasing - monitor for opportunities')

            # Volatility recommendations
            if risk_metrics.volatility > 0.05:
                recommendations.append('High volatility - adjust position sizes')

            # Quantum-enhanced recommendations
            if quantum_state and quantum_state.entropy > 0.7:
                recommendations.append('High quantum entropy - market uncertainty elevated')
            if quantum_state and quantum_state.coherence < 0.3:
                recommendations.append('Low quantum coherence - potential regime shift')

            return {
                'regime': risk_metrics.market_regime,
                'description': description[risk_metrics.market_regime],
                'risk_level': float(risk_metrics.risk_level),
                'stability_index': float(risk_metrics.stability_index),
                'recommendations': recommendations
            }

        except Exception as e:
            logger.error(f"Risk regime assessment failed: {str(e)}")
            return {
                'regime': 'high_risk',
                'description': 'Unable to assess risk regime',
                'risk_level': 1.0,
                'stability_index': 0.0,
                'recommendations': ['Exercise caution due to risk assessment failure']
            }

    def calculate_position_risk(self, position: Dict[str, Any]) -> float:
        """
        Calcula o risco de uma posição aberta
        
        Args:
            position (Dict[str, Any]): Dados da posição
            
        Returns:
            float: Nível de risco da posição (0-1)
        """
        try:
            # Calcula volatilidade do preço
            current_price = position['current_price']
            entry_price = position['entry_price']
            price_change = abs(current_price - entry_price) / entry_price
            
            # Ajusta risco baseado no PnL
            pnl = position['pnl']
            pnl_risk = 1.0 if pnl < 0 else 0.5
            
            # Combina fatores
            position_risk = (price_change * 0.6 + pnl_risk * 0.4)
            
            return min(1.0, max(0.0, position_risk))
            
        except Exception as e:
            logger.error(f"Erro ao calcular risco da posição: {e}")
            return 0.5  # Risco neutro em caso de erro