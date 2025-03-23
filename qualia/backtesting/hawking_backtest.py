"""
Hawking Backtest Module for QUALIA Trading System
Implements historical simulations with quantum parameter optimization
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..market_data import MarketDataProvider
from ..quantum_trader import QuantumTrader, TradeSignal
from ..analysis.quantum_analysis import QuantumAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class BacktestParameters:
    """Quantum-aware backtest parameters"""
    alpha: float  # Quantum coherence weight
    beta: float  # Morphic resonance factor
    gamma: float  # Consciousness integration weight
    decoherence_threshold: float
    consciousness_threshold: float
    resonance_threshold: float

    @classmethod
    def get_bounds(cls) -> Dict[str, Tuple[float, float]]:
        """Parameter bounds for optimization"""
        return {
            'alpha': (0.1, 1.0),
            'beta': (0.1, 1.0),
            'gamma': (0.1, 1.0),
            'decoherence_threshold': (0.3, 0.8),
            'consciousness_threshold': (0.5, 0.9),
            'resonance_threshold': (0.4, 0.8)
        }

@dataclass
class BacktestResult:
    """Enhanced backtest results with quantum metrics"""
    start_date: datetime
    end_date: datetime
    total_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    decoherence_metric: float
    ks_test_pvalue: float
    predicted_returns: List[float]
    actual_returns: List[float]
    cumulative_returns: List[float]
    equity_curve: List[float]
    quantum_metrics: Dict[str, List[float]]
    parameters: BacktestParameters
    optimization_history: List[Dict[str, float]]

class HawkingBacktest:
    """Quantum-aware backtesting engine with parameter optimization"""

    def __init__(
        self,
        trader: QuantumTrader,
        market_data: MarketDataProvider,
        quantum_analyzer: Optional[QuantumAnalyzer] = None,
        max_execution_time: int = 60,  # Maximum execution time in seconds
        optimization_iterations: int = 50
    ):
        self.trader = trader
        self.market_data = market_data
        self.quantum_analyzer = quantum_analyzer or QuantumAnalyzer()
        self.max_execution_time = max_execution_time
        self.optimization_iterations = optimization_iterations

        # Initialize optimizer
        self.optimizer = BayesianOptimizer(
            parameter_bounds=BacktestParameters.get_bounds(),
            n_iterations=optimization_iterations
        )

        # Performance tracking
        self.optimization_history: List[Dict[str, float]] = []

    def calculate_financial_decoherence(self, returns: np.ndarray) -> float:
        """Calculate financial decoherence metric D(ρ) = -∑pi log(pi)"""
        try:
            # Calculate probability distribution
            hist, bins = np.histogram(returns, bins='auto', density=True)
            probabilities = hist * np.diff(bins)
            probabilities = probabilities[probabilities > 0]

            # Calculate decoherence
            decoherence = -np.sum(probabilities * np.log(probabilities))
            return float(decoherence)

        except Exception as e:
            logger.error(f"Error calculating decoherence: {e}")
            return 0.0

    def perform_ks_test(
        self,
        predicted_returns: np.ndarray,
        actual_returns: np.ndarray
    ) -> Tuple[float, float]:
        """Perform KS test to validate prediction consistency"""
        try:
            statistic, pvalue = stats.ks_2samp(predicted_returns, actual_returns)
            return float(statistic), float(pvalue)
        except Exception as e:
            logger.error(f"Error performing KS test: {e}")
            return 0.0, 0.0

    def optimize_parameters(
        self, 
        symbol: str,
        initial_capital: float,
        start_date: datetime,
        end_date: datetime,
        window_size: int = 50
    ) -> BacktestParameters:
        """
        Optimize backtest parameters using Bayesian optimization
        """
        def objective_function(params: Dict[str, float]) -> float:
            """Objective function for parameter optimization"""
            try:
                # Run backtest with current parameters
                result = self.run_backtest(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital,
                    parameters=BacktestParameters(**params)
                )

                # Calculate comprehensive score
                sharpe_weight = 0.4
                decoherence_weight = 0.2
                ks_weight = 0.2
                drawdown_weight = 0.2

                score = (
                    sharpe_weight * result.sharpe_ratio +
                    decoherence_weight * (1 - result.decoherence_metric) +
                    ks_weight * result.ks_test_pvalue +
                    drawdown_weight * (1 + result.max_drawdown)
                )

                return float(score)

            except Exception as e:
                logger.error(f"Error in objective function: {e}")
                return float('-inf')

        # Run Bayesian optimization
        try:
            optimization_result = self.optimizer.optimize(objective_function)
            return BacktestParameters(**optimization_result.parameters)

        except Exception as e:
            logger.error(f"Parameter optimization failed: {e}")
            # Return default parameters
            return BacktestParameters(
                alpha=0.5,
                beta=0.5,
                gamma=0.5,
                decoherence_threshold=0.6,
                consciousness_threshold=0.7,
                resonance_threshold=0.6
            )

    def run_backtest(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 10000.0,
        parameters: Optional[BacktestParameters] = None
    ) -> BacktestResult:
        """Run historical backtest with quantum analysis and parameter optimization"""
        start_time = datetime.now()

        try:
            # Get historical data
            historical_data = self.market_data.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )

            if historical_data is None or len(historical_data) == 0:
                raise ValueError("No historical data available")

            # Use default parameters if none provided
            if parameters is None:
                parameters = BacktestParameters(
                    alpha=0.5,
                    beta=0.5,
                    gamma=0.5,
                    decoherence_threshold=0.6,
                    consciousness_threshold=0.7,
                    resonance_threshold=0.6
                )

            # Initialize tracking variables
            positions = []
            predicted_returns = []
            actual_returns = []
            equity_curve = [initial_capital]
            current_capital = initial_capital
            quantum_metrics = {
                'coherence': [],
                'decoherence': [],
                'entropy': []
            }

            # Pre-calculate indices for faster iteration
            indices = list(range(len(historical_data.index) - 1))
            logger.info(f"Starting backtest simulation with {len(indices)} data points")

            # Run simulation with timeout check
            for i in indices:
                # Check execution time
                if (datetime.now() - start_time).total_seconds() > self.max_execution_time:
                    logger.warning("Backtest execution timeout reached")
                    break

                # Get current and next price data
                current_data = historical_data.iloc[i]
                next_data = historical_data.iloc[i + 1]

                # Configure trader with current parameters
                self.trader.configure_quantum_parameters(
                    alpha=parameters.alpha,
                    beta=parameters.beta,
                    gamma=parameters.gamma,
                    decoherence_threshold=parameters.decoherence_threshold,
                    consciousness_threshold=parameters.consciousness_threshold,
                    resonance_threshold=parameters.resonance_threshold
                )

                # Generate trading signal
                signal = self.trader.analyze_market(symbol)

                if signal and signal.validate():
                    # Track quantum metrics efficiently
                    metrics = signal.quantum_metrics
                    quantum_metrics['coherence'].append(metrics.get('coherence', 0))
                    quantum_metrics['decoherence'].append(metrics.get('decoherence_rate', 0))
                    quantum_metrics['entropy'].append(metrics.get('entropy', 0))

                    # Calculate predicted return
                    pred_return = 0.0
                    if signal.action == 'BUY':
                        pred_return = (signal.take_profit - current_data['close']) / current_data['close']
                    elif signal.action == 'SELL':
                        pred_return = (current_data['close'] - signal.take_profit) / current_data['close']

                    predicted_returns.append(pred_return)

                    # Calculate actual return
                    actual_return = (next_data['close'] - current_data['close']) / current_data['close']
                    actual_returns.append(actual_return)

                    # Update portfolio value
                    if signal.action != 'HOLD':
                        position_size = self.trader._calculate_position_size(signal)
                        trade_return = position_size * actual_return
                        current_capital *= (1 + trade_return)

                    equity_curve.append(float(current_capital))

            # Convert to numpy arrays for efficient calculations
            predicted_returns = np.array(predicted_returns)
            actual_returns = np.array(actual_returns)
            equity_curve = np.array(equity_curve)

            # Calculate performance metrics
            total_trades = len(predicted_returns)
            win_rate = np.mean(np.sign(predicted_returns) == np.sign(actual_returns)) if total_trades > 0 else 0.0
            returns = np.diff(equity_curve) / equity_curve[:-1]
            sharpe_ratio = np.sqrt(252) * np.mean(returns) / (np.std(returns) + 1e-10)
            max_drawdown = np.min(equity_curve / np.maximum.accumulate(equity_curve) - 1)

            # Calculate advanced metrics
            decoherence = self.calculate_financial_decoherence(actual_returns)
            _, ks_pvalue = self.perform_ks_test(predicted_returns, actual_returns)
            cumulative_returns = np.cumprod(1 + returns) - 1

            logger.info(f"Backtest completed in {(datetime.now() - start_time).total_seconds():.2f}s")

            return BacktestResult(
                start_date=start_date,
                end_date=end_date,
                total_trades=total_trades,
                win_rate=float(win_rate),
                sharpe_ratio=float(sharpe_ratio),
                max_drawdown=float(max_drawdown),
                decoherence_metric=float(decoherence),
                ks_test_pvalue=float(ks_pvalue),
                predicted_returns=predicted_returns.tolist(),
                actual_returns=actual_returns.tolist(),
                cumulative_returns=cumulative_returns.tolist(),
                equity_curve=equity_curve.tolist(),
                quantum_metrics=quantum_metrics,
                parameters=parameters,
                optimization_history=self.optimization_history
            )

        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise