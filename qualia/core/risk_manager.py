"""
Enhanced Risk Manager with quantum-aware dynamic adjustments
Implements M-ICCI principles for risk management
"""
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import logging
from dataclasses import dataclass
from datetime import datetime
from dataclasses import asdict

from ..optimization.bayesian_optimizer import BayesianOptimizer, OptimizationResult
from ..utils.quantum_field import safe_complex_to_real, serialize_complex_value
from ..quantum_state_manager import QuantumStateManager

@dataclass
class RiskParameters:
    """Risk management parameters with quantum-awareness"""
    # Position sizing parameters
    max_position_size: float  # Maximum position size as fraction of capital
    position_scaling: float   # Dynamic position size scaling factor
    risk_adjustment_factor: float  # Risk-based position adjustment

    # Stop loss parameters with quantum integration
    base_stop_loss: float    # Base stop loss percentage
    stop_loss_scaling: float # Dynamic stop loss scaling
    trailing_stop_ratio: float # Trailing stop adjustment

    # Take profit parameters
    take_profit_ratio: float  # Base take profit ratio
    profit_scaling: float     # Dynamic profit scaling

    # Quantum integration parameters
    quantum_weight: float     # Weight for quantum metrics
    decoherence_threshold: float  # Threshold for financial decoherence
    coherence_threshold: float    # Minimum quantum coherence

    @classmethod
    def get_bounds(cls) -> Dict[str, Tuple[float, float]]:
        """Parameter bounds for optimization"""
        return {
            'max_position_size': (0.01, 0.5),
            'position_scaling': (0.5, 2.0),
            'risk_adjustment_factor': (0.1, 2.0),
            'base_stop_loss': (0.01, 0.05),
            'stop_loss_scaling': (0.5, 2.0),
            'trailing_stop_ratio': (0.5, 2.0),
            'take_profit_ratio': (1.5, 4.0),
            'profit_scaling': (0.5, 2.0),
            'quantum_weight': (0.1, 1.0),
            'decoherence_threshold': (0.3, 0.8),
            'coherence_threshold': (0.3, 0.8)
        }

class RiskManager:
    """
    Quantum-enhanced risk manager implementing M-ICCI principles
    Provides dynamic risk adjustment based on market conditions and quantum metrics
    """
    def __init__(
        self,
        initial_parameters: Optional[RiskParameters] = None,
        optimization_interval: int = 100  # trades
    ):
        self.parameters = initial_parameters or RiskParameters(
            max_position_size=0.1,
            position_scaling=1.0,
            risk_adjustment_factor=1.0,
            base_stop_loss=0.02,
            stop_loss_scaling=1.0,
            trailing_stop_ratio=1.0,
            take_profit_ratio=2.0,
            profit_scaling=1.0,
            quantum_weight=0.5,
            decoherence_threshold=0.6,
            coherence_threshold=0.6
        )

        self.optimization_interval = optimization_interval
        self.trade_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)

        # Initialize quantum components
        self.quantum_manager = QuantumStateManager()

        # Initialize optimizer with quantum enhancement
        self.optimizer = BayesianOptimizer(
            parameter_bounds=RiskParameters.get_bounds(),
            n_iterations=50
        )

        # Performance tracking
        self.last_optimization: Optional[datetime] = None
        self.optimization_history: List[OptimizationResult] = []
        self.metrics_history: List[Dict[str, float]] = []

    def calculate_position_size(
        self,
        quantum_metrics: Dict[str, float],
        account_balance: float,
        market_volatility: float,
        market_trend: float,
        decoherence: float
    ) -> float:
        """Calculate adaptive position size with quantum risk metrics"""
        try:
            # Extract quantum metrics with safe conversion
            coherence = safe_complex_to_real(quantum_metrics.get('coherence', 0.5))
            consciousness = safe_complex_to_real(quantum_metrics.get('consciousness', 0.5))

            if not np.isfinite(account_balance) or account_balance <= 0:
                return 0.0

            if not all(np.isfinite(x) for x in [market_volatility, market_trend, decoherence]):
                return 0.0

            # Quantum risk adjustment
            quantum_risk = (coherence + consciousness) / 2
            quantum_adjustment = self.parameters.quantum_weight * (quantum_risk - 0.5)

            # Decoherence-based risk scaling
            decoherence_factor = np.exp(-max(0, decoherence) / self.parameters.decoherence_threshold)

            # Volatility scaling with market trend
            volatility_factor = np.exp(
                -self.parameters.position_scaling * 
                abs(market_volatility) * 
                (1 + abs(market_trend))
            )

            # Calculate adaptive position scaling
            trend_factor = 1.0 + self.parameters.position_scaling * abs(market_trend)

            # Combine factors for final position sizing
            risk_factor = (
                (1.0 + quantum_adjustment) * 
                self.parameters.risk_adjustment_factor * 
                volatility_factor *
                trend_factor *
                decoherence_factor
            )

            # Calculate position size with bounds
            position_size = (
                self.parameters.max_position_size * 
                risk_factor * 
                account_balance
            )

            # Ensure result is finite and within bounds
            if not np.isfinite(position_size):
                return 0.0

            return float(np.clip(position_size, 0, account_balance))

        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0

    def calculate_stop_levels(
        self,
        entry_price: float,
        market_volatility: float,
        quantum_metrics: Dict[str, float],
        market_trend: float,
        decoherence: float
    ) -> Tuple[float, float]:
        """Calculate adaptive stop loss and take profit levels with quantum enhancement"""
        try:
            # Input validation
            if not np.isfinite(entry_price) or entry_price <= 0:
                # For invalid entry price, use fixed 2% stop loss and 4% take profit
                return entry_price * 0.98, entry_price * 0.96

            if not all(np.isfinite(x) for x in [market_volatility, market_trend, decoherence]):
                # For invalid metrics, use fixed 2% stop loss and 4% take profit
                return entry_price * 0.98, entry_price * 0.96

            # Extract quantum metrics
            coherence = safe_complex_to_real(quantum_metrics.get('coherence', 0.5))
            consciousness = safe_complex_to_real(quantum_metrics.get('consciousness', 0.5))

            # Base stop loss calculation with decoherence adjustment
            base_stop = entry_price * self.parameters.base_stop_loss * (1 + decoherence)

            # Adjust for volatility and market trend
            volatility_adjustment = np.clip(
                market_volatility / 0.02 * 
                (1 + self.parameters.stop_loss_scaling * abs(market_trend)),
                0.5, 2.0
            )
            adjusted_stop = base_stop * volatility_adjustment

            # Apply quantum adjustment
            quantum_factor = 1.0 + (1 - coherence) * self.parameters.quantum_weight
            coherence_adjustment = np.exp(-decoherence / self.parameters.coherence_threshold)

            # Calculate trailing stop adjustment
            trailing_adjustment = self.parameters.trailing_stop_ratio * market_trend

            # Calculate take profit with dynamic scaling
            take_profit_scaling = (
                self.parameters.take_profit_ratio * 
                (1 + self.parameters.profit_scaling * consciousness) *
                coherence_adjustment
            )
            take_profit_distance = adjusted_stop * take_profit_scaling

            # Apply final adjustments
            final_stop_loss = entry_price - (adjusted_stop * quantum_factor * (1 + trailing_adjustment))
            final_take_profit = entry_price + (take_profit_distance * quantum_factor)

            # Ensure results are finite
            if not (np.isfinite(final_stop_loss) and np.isfinite(final_take_profit)):
                # For invalid calculation results, use fixed 2% stop loss and 4% take profit
                return entry_price * 0.98, entry_price * 0.96

            return float(final_stop_loss), float(final_take_profit)

        except Exception as e:
            self.logger.error(f"Error calculating stop levels: {e}")
            # For any errors, use fixed 2% stop loss and 4% take profit
            return entry_price * 0.98, entry_price * 0.96

    def update_parameters(self, trade_result: Dict[str, Any]) -> None:
        """Update trade history and optimize parameters with quantum enhancement"""
        try:
            # Add trade to history with quantum metrics
            trade_result['quantum_metrics'] = self.quantum_manager.calculate_consciousness_metrics()
            self.trade_history.append(trade_result)

            # Check if optimization is needed
            if (len(self.trade_history) % self.optimization_interval == 0 and
                len(self.trade_history) >= self.optimization_interval):

                # Run quantum-enhanced optimization
                optimization_result = self.optimizer.optimize(
                    self._calculate_optimization_objective
                )

                # Update parameters if improvement is significant
                if optimization_result.improvement_probability > 0.6:
                    self.parameters = RiskParameters(**optimization_result.parameters)
                    self.logger.info("Risk parameters updated through quantum-enhanced optimization")

                # Record optimization result
                self.optimization_history.append(optimization_result)
                self.last_optimization = datetime.now()

                # Update metrics history
                self.metrics_history.append({
                    'optimization_score': optimization_result.expected_value,
                    'uncertainty': optimization_result.uncertainty,
                    'improvement_probability': optimization_result.improvement_probability,
                    **optimization_result.quantum_metrics
                })

        except Exception as e:
            self.logger.error(f"Error updating parameters: {e}")

    def calculate_financial_decoherence(
        self, 
        returns: np.ndarray,
        window_size: int = 20
    ) -> float:
        """
        Calculate financial decoherence metric D(ρ) = -∑pi log(pi)
        using rolling returns distribution with normalization
        """
        try:
            if len(returns) < window_size:
                return 0.0

            # Use rolling window for recent data
            recent_returns = returns[-window_size:]

            # Calculate probability distribution
            hist, bins = np.histogram(recent_returns, bins='auto', density=True)
            probabilities = hist * np.diff(bins)
            probabilities = probabilities[probabilities > 0]

            if len(probabilities) == 0:
                return 0.0

            # Normalize probabilities to sum to 1
            probabilities = probabilities / np.sum(probabilities)

            # Calculate decoherence using the formula D(ρ) = -∑pi log(pi)
            decoherence = -np.sum(probabilities * np.log(probabilities))

            # Normalize decoherence to [0,1] using a sigmoid function
            normalized_decoherence = 1 / (1 + np.exp(-decoherence))

            return float(normalized_decoherence)

        except Exception as e:
            self.logger.error(f"Error calculating decoherence: {e}")
            return 0.0

    def _calculate_optimization_objective(self, params: Dict[str, float]) -> float:
        """Calculate objective function for parameter optimization with quantum metrics"""
        try:
            if not self.trade_history:
                return 0.0

            # Create test parameters
            test_params = RiskParameters(**params)

            # Calculate performance metrics
            returns = []
            max_drawdown = 0
            current_drawdown = 0
            peak = float('-inf')
            quantum_alignment = 0.0

            # Get quantum metrics
            quantum_metrics = self.quantum_manager.calculate_consciousness_metrics()
            coherence = safe_complex_to_real(quantum_metrics.get('coherence', 0.5))

            for trade in self.trade_history[-self.optimization_interval:]:
                returns.append(trade.get('return', 0))
                equity = trade.get('equity', 0)

                if equity > peak:
                    peak = equity
                    current_drawdown = 0
                else:
                    current_drawdown = (peak - equity) / peak
                    max_drawdown = max(max_drawdown, current_drawdown)

                # Calculate quantum alignment for each trade
                trade_quantum = safe_complex_to_real(trade.get('quantum_metrics', {}).get('coherence', 0.5))
                quantum_alignment += abs(trade_quantum - test_params.coherence_threshold)

            # Calculate risk-adjusted returns
            if not returns:
                return 0.0

            returns = np.array(returns)
            sharpe = np.mean(returns) / (np.std(returns) + 1e-6)
            sortino = np.mean(returns) / (np.std(returns[returns < 0]) + 1e-6)

            # Calculate drawdown penalty
            drawdown_penalty = np.exp(-5 * max_drawdown)

            # Calculate quantum alignment score
            quantum_score = np.exp(-2 * quantum_alignment / len(returns))

            # Combine metrics with quantum enhancement
            score = (
                sharpe * 0.4 +
                sortino * 0.3 +
                drawdown_penalty * 0.2 +
                quantum_score * 0.1
            ) * (1 + 0.2 * coherence)

            return float(score)

        except Exception as e:
            self.logger.error(f"Error calculating optimization objective: {e}")
            return 0.0

    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk management metrics with quantum enhancement"""
        try:
            # Get quantum metrics
            quantum_metrics = self.quantum_manager.calculate_consciousness_metrics()

            metrics = {
                'current_parameters': asdict(self.parameters),
                'trades_since_optimization': len(self.trade_history) % self.optimization_interval,
                'total_trades': len(self.trade_history),
                'quantum_metrics': {k: safe_complex_to_real(v) for k, v in quantum_metrics.items()}
            }

            if self.optimization_history:
                latest_opt = self.optimization_history[-1]
                metrics.update({
                    'last_optimization_score': latest_opt.expected_value,
                    'parameter_uncertainty': latest_opt.uncertainty,
                    'improvement_probability': latest_opt.improvement_probability
                })

            if self.metrics_history:
                latest_metrics = self.metrics_history[-1]
                metrics['optimization_trends'] = {
                    'score_trend': self._calculate_metric_trend('optimization_score'),
                    'uncertainty_trend': self._calculate_metric_trend('uncertainty'),
                    'quantum_coherence_trend': self._calculate_metric_trend('coherence')
                }

            return metrics

        except Exception as e:
            self.logger.error(f"Error getting risk metrics: {e}")
            return {
                'error': str(e),
                'current_parameters': asdict(self.parameters)
            }

    def _calculate_metric_trend(self, metric_name: str, window: int = 10) -> float:
        """Calculate trend for a specific metric with improved precision"""
        try:
            if len(self.metrics_history) < 2:
                return 0.0

            # Extract and validate values
            values = []
            for m in self.metrics_history[-window:]:
                try:
                    v = float(m.get(metric_name, 0))
                    if np.isfinite(v):
                        values.append(v)
                except (TypeError, ValueError):
                    continue

            if len(values) < 2:
                return 0.0

            # Calculate trend with numpy and proper trend thresholding
            x = np.arange(len(values))
            try:
                coeffs = np.polyfit(x, values, 1)
                trend = coeffs[0]
                # Consider trends below 0.15 as effectively zero
                return 0.0 if abs(trend) < 0.15 else float(trend)
            except:
                return 0.0

        except Exception as e:
            self.logger.error(f"Error calculating metric trend: {e}")
            return 0.0