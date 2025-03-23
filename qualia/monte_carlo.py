"""
Monte Carlo Simulation Module for QUALIA Trading System
Implements quantum-aware path generation and advanced stress testing
"""
import numpy as np
from scipy import stats
from typing import List, Dict, Tuple, Optional, Any, Callable, Union
import logging
from datetime import datetime
import json

from .quantum_state_manager import QuantumStateManager
from .validation_layer import ValidationLayer
from .utils.quantum_field import safe_complex_to_real

logger = logging.getLogger(__name__)

class MarketScenario:
    """Defines market scenario parameters for stress testing"""
    def __init__(self,
                 name: str,
                 volatility_range: Tuple[float, float],
                 drift_range: Tuple[float, float],
                 quantum_coherence: float = 0.5,
                 description: str = ""):
        self.name = name
        self.volatility_range = volatility_range
        self.drift_range = drift_range
        self.quantum_coherence = quantum_coherence
        self.description = description

    def validate(self) -> bool:
        """Validate scenario parameters"""
        try:
            if not isinstance(self.volatility_range, tuple) or len(self.volatility_range) != 2:
                return False
            if not isinstance(self.drift_range, tuple) or len(self.drift_range) != 2:
                return False
            if not all(isinstance(x, (int, float)) for x in [*self.volatility_range, *self.drift_range]):
                return False
            if not 0 <= self.quantum_coherence <= 1:
                return False
            return True
        except Exception as e:
            logger.error(f"Scenario validation failed: {e}")
            return False

class QuantumMonteCarlo:
    """
    Enhanced Quantum-aware Monte Carlo simulation engine with stress testing.
    """
    def __init__(self, 
                 n_simulations: int = 1000,
                 time_steps: int = 100,
                 confidence_level: float = 0.95):
        """Initialize with parameter validation"""
        try:
            self.n_simulations = max(100, min(10000, n_simulations))  # Bound simulations
            self.time_steps = max(10, min(1000, time_steps))  # Bound time steps
            self.confidence_level = max(0.8, min(0.99, confidence_level))  # Bound confidence
            self.epsilon = 1e-10  # Numerical stability threshold

            # Initialize quantum components
            self.state_manager = QuantumStateManager()
            self.validator = ValidationLayer()

            # Enhanced metrics tracking with proper initialization
            self.simulation_metrics = {
                'paths': [],
                'var': [],
                'expected_shortfall': [],
                'quantum_correlation': [],
                'scenario_results': {}
            }

            # Define standard market scenarios
            self.standard_scenarios = {
                'bull_market': MarketScenario(
                    name='bull_market',
                    volatility_range=(0.1, 0.2),
                    drift_range=(0.1, 0.3),
                    quantum_coherence=0.8,
                    description="Strong upward trend with moderate volatility"
                ),
                'bear_market': MarketScenario(
                    name='bear_market',
                    volatility_range=(0.3, 0.5),
                    drift_range=(-0.3, -0.1),
                    quantum_coherence=0.4,
                    description="Downward trend with high volatility"
                ),
                'sideways': MarketScenario(
                    name='sideways',
                    volatility_range=(0.05, 0.15),
                    drift_range=(-0.05, 0.05),
                    quantum_coherence=0.6,
                    description="Range-bound market with low directional bias"
                ),
                'crisis': MarketScenario(
                    name='crisis',
                    volatility_range=(0.5, 0.8),
                    drift_range=(-0.5, -0.2),
                    quantum_coherence=0.2,
                    description="Extreme volatility with strong downward bias"
                )
            }

            # Validate all scenarios
            for scenario in self.standard_scenarios.values():
                if not scenario.validate():
                    raise ValueError(f"Invalid scenario configuration: {scenario.name}")

            logger.info(f"Enhanced Quantum Monte Carlo initialized with {n_simulations} simulations")

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise

    def generate_quantum_paths(self, 
                          initial_price: float,
                          volatility: float,
                          drift: float,
                          quantum_state: Optional[np.ndarray] = None,
                          scenario: Optional[MarketScenario] = None) -> np.ndarray:
        """
        Generate price paths with quantum influence and scenario-based adjustments
        """
        try:
            # Validate inputs
            if not all(isinstance(x, (int, float)) for x in [initial_price, volatility, drift]):
                raise ValueError("Invalid numeric inputs")

            if initial_price <= 0 or volatility <= 0:
                raise ValueError("Initial price and volatility must be positive")

            dt = 1.0 / self.time_steps
            paths = np.zeros((self.n_simulations, self.time_steps))
            paths[:, 0] = initial_price

            # Apply scenario adjustments if provided
            if scenario:
                if not scenario.validate():
                    logger.warning(f"Invalid scenario: {scenario.name}, using default parameters")
                else:
                    volatility = np.random.uniform(*scenario.volatility_range)
                    drift = np.random.uniform(*scenario.drift_range)
                    quantum_coherence = scenario.quantum_coherence
            else:
                # Extract quantum influence factors with validation
                try:
                    metrics = self.state_manager.calculate_consciousness_metrics()
                    quantum_coherence = safe_complex_to_real(metrics.get('coherence', 0.5))
                except Exception as e:
                    logger.warning(f"Error calculating consciousness metrics: {e}")
                    quantum_coherence = 0.5

            # Adjust volatility based on quantum coherence
            adjusted_volatility = volatility * (1 + 0.2 * (1 - quantum_coherence))

            # Generate correlated random walks with quantum noise
            try:
                random_factors = np.random.normal(0, 1, (self.n_simulations, self.time_steps-1))
                quantum_noise = (1 - quantum_coherence) * np.random.normal(0, 1, (self.n_simulations, self.time_steps-1))
            except Exception as e:
                logger.error(f"Error generating random factors: {e}")
                return self._generate_fallback_paths(initial_price, volatility, drift)

            # Combine classical and quantum randomness
            total_noise = 0.8 * random_factors + 0.2 * quantum_noise

            # Vectorized path calculation with validation
            try:
                for t in range(1, self.time_steps):
                    paths[:, t] = paths[:, t-1] * np.exp(
                        (drift - 0.5 * adjusted_volatility**2) * dt + 
                        adjusted_volatility * np.sqrt(dt) * total_noise[:, t-1]
                    )

                    # Validate generated values
                    if not np.all(np.isfinite(paths[:, t])):
                        logger.error("Non-finite values detected in path generation")
                        return self._generate_fallback_paths(initial_price, volatility, drift)

            except Exception as e:
                logger.error(f"Error in path calculation: {e}")
                return self._generate_fallback_paths(initial_price, volatility, drift)

            return paths

        except Exception as e:
            logger.error(f"Error generating quantum paths: {str(e)}")
            return self._generate_fallback_paths(initial_price, volatility, drift)

    def _generate_fallback_paths(self, initial_price: float, volatility: float, drift: float) -> np.ndarray:
        """Fallback path generation using simple geometric Brownian motion"""
        try:
            paths = np.zeros((self.n_simulations, self.time_steps))
            paths[:, 0] = initial_price
            dt = 1.0 / self.time_steps

            for t in range(1, self.time_steps):
                z = np.random.normal(0, 1, self.n_simulations)
                paths[:, t] = paths[:, t-1] * np.exp(
                    (drift - 0.5 * volatility**2) * dt + 
                    volatility * np.sqrt(dt) * z
                )

            return paths
        except Exception as e:
            logger.error(f"Fallback path generation failed: {e}")
            # Return constant paths as ultimate fallback
            paths = np.full((self.n_simulations, self.time_steps), initial_price)
            return paths

    def run_stress_test(self,
                       initial_price: float,
                       initial_investment: float,
                       market_data: np.ndarray,
                       custom_scenarios: Optional[Dict[str, MarketScenario]] = None) -> Dict[str, Any]:
        """
        Run comprehensive stress tests across multiple market scenarios
        """
        try:
            # Validate inputs
            if not all(isinstance(x, (int, float)) for x in [initial_price, initial_investment]):
                raise ValueError("Invalid price or investment values")

            if initial_price <= 0 or initial_investment <= 0:
                raise ValueError("Initial price and investment must be positive")

            # Combine scenarios
            scenarios = {**self.standard_scenarios}
            if custom_scenarios:
                # Validate custom scenarios
                for name, scenario in custom_scenarios.items():
                    if not isinstance(scenario, MarketScenario) or not scenario.validate():
                        logger.warning(f"Skipping invalid custom scenario: {name}")
                        continue
                    scenarios[name] = scenario

            stress_test_results = {}
            for scenario_name, scenario in scenarios.items():
                try:
                    # Run simulation with scenario parameters
                    paths = self.generate_quantum_paths(
                        initial_price=initial_price,
                        volatility=np.mean(scenario.volatility_range),
                        drift=np.mean(scenario.drift_range),
                        scenario=scenario
                    )

                    # Calculate risk metrics for scenario
                    risk_metrics = self.calculate_risk_metrics(paths, initial_investment)

                    # Store results with proper type conversion
                    stress_test_results[scenario_name] = {
                        'description': str(scenario.description),
                        'risk_metrics': {k: float(v) for k, v in risk_metrics.items()},
                        'quantum_coherence': float(scenario.quantum_coherence),
                        'worst_case_return': float(np.min(paths[:, -1]) / initial_price - 1),
                        'best_case_return': float(np.max(paths[:, -1]) / initial_price - 1),
                        'probability_of_loss': float(np.mean(paths[:, -1] < initial_price))
                    }

                except Exception as e:
                    logger.error(f"Error in scenario {scenario_name}: {e}")
                    continue

            return stress_test_results

        except Exception as e:
            logger.error(f"Error in stress testing: {e}")
            return {}

    def calculate_risk_metrics(self, paths: np.ndarray, initial_investment: float) -> Dict[str, float]:
        """Calculate comprehensive risk metrics from simulated paths"""
        try:
            # Calculate returns for each path
            final_prices = paths[:, -1]
            returns = (final_prices - paths[:, 0]) / paths[:, 0]

            # Sort returns for percentile calculations
            sorted_returns = np.sort(returns)

            # Calculate VaR with proper indexing
            var_index = int((1 - self.confidence_level) * self.n_simulations)
            value_at_risk = -sorted_returns[var_index] * initial_investment

            # Calculate Expected Shortfall (CVaR)
            expected_shortfall = -np.mean(sorted_returns[:var_index]) * initial_investment

            # Calculate additional statistics
            metrics = {
                'value_at_risk': float(value_at_risk),
                'expected_shortfall': float(expected_shortfall),
                'mean_return': float(np.mean(returns)),
                'volatility': float(np.std(returns)),
                'skewness': float(stats.skew(returns)),
                'kurtosis': float(stats.kurtosis(returns)),
                'sharpe_ratio': float(np.mean(returns) / (np.std(returns) + self.epsilon)),
                'sortino_ratio': float(np.mean(returns) / (np.std(returns[returns < 0]) + self.epsilon))
            }

            return metrics

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return {
                'value_at_risk': 0.0,
                'expected_shortfall': 0.0,
                'mean_return': 0.0,
                'volatility': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0
            }

    def reset_simulation(self) -> None:
        """Reset simulation metrics and state"""
        for key in self.simulation_metrics:
            if isinstance(self.simulation_metrics[key], list):
                self.simulation_metrics[key] = []
            elif isinstance(self.simulation_metrics[key], dict):
                self.simulation_metrics[key] = {}

        self.state_manager = QuantumStateManager()
        logger.info("Monte Carlo simulation reset")

    def run_simulation(self,
                      initial_price: float,
                      initial_investment: float,
                      market_data: np.ndarray) -> Dict[str, Any]:
        """
        Run complete Monte Carlo simulation with quantum awareness
        and proper type handling for serialization
        """
        try:
            # Get current quantum state
            quantum_state = self.state_manager.get_state()
            if quantum_state and 'density_matrix' in quantum_state:
                quantum_state = quantum_state['density_matrix']
            else:
                quantum_state = None

            # Calculate market parameters
            returns = np.diff(market_data) / market_data[:-1]
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            drift = np.mean(returns) * 252  # Annualized

            # Generate paths
            paths = self.generate_quantum_paths(
                initial_price=initial_price,
                volatility=volatility,
                drift=drift,
                quantum_state=quantum_state
            )

            # Calculate risk metrics
            risk_metrics = self.calculate_risk_metrics(
                paths=paths,
                initial_investment=initial_investment
            )

            # Get quantum metrics with safe conversion to real numbers
            quantum_metrics = self.state_manager.calculate_consciousness_metrics()
            quantum_metrics = {k: safe_complex_to_real(v) 
                             for k, v in quantum_metrics.items()}

            # Store results with proper type conversion
            simulation_result = {
                'paths': paths.tolist(),  # Convert numpy array to list
                'risk_metrics': risk_metrics,
                'quantum_metrics': quantum_metrics,
                'parameters': {
                    'volatility': float(volatility),
                    'drift': float(drift),
                    'initial_price': float(initial_price),
                    'initial_investment': float(initial_investment)
                },
                'timestamp': datetime.now().timestamp()
            }

            # Update metrics history with proper type conversion
            self.simulation_metrics['paths'].append(paths[-1].tolist())
            self.simulation_metrics['var'].append(float(risk_metrics['value_at_risk']))
            self.simulation_metrics['expected_shortfall'].append(float(risk_metrics['expected_shortfall']))
            self.simulation_metrics['quantum_correlation'].append(float(quantum_metrics.get('coherence', 0)))

            # Maintain history length
            max_history = 1000
            for key in self.simulation_metrics:
                if len(self.simulation_metrics[key]) > max_history:
                    self.simulation_metrics[key] = self.simulation_metrics[key][-max_history:]

            return simulation_result

        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {str(e)}")
            raise