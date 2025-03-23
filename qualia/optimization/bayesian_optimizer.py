"""
Bayesian Parameter Optimizer for QUALIA Trading System
Implements Gaussian Process-based parameter optimization with quantum enhancement
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Callable, Any, Optional
import logging
from dataclasses import dataclass
from datetime import datetime

from ..quantum_state_manager import QuantumStateManager
from ..utils.quantum_field import safe_complex_to_real, serialize_complex_value

@dataclass
class OptimizationResult:
    """Optimization result with uncertainty estimates"""
    parameters: Dict[str, float]
    expected_value: float
    uncertainty: float
    improvement_probability: float
    quantum_metrics: Dict[str, float]
    optimization_history: List[Dict[str, Any]]

class BayesianOptimizer:
    """
    Quantum-enhanced Bayesian optimizer using Gaussian Processes
    for trading parameter optimization.
    """
    def __init__(
        self,
        parameter_bounds: Dict[str, Tuple[float, float]],
        n_iterations: int = 100,
        exploration_weight: float = 0.1,
        noise_level: float = 0.1
    ):
        # Validate parameter bounds
        for param_name, (lower, upper) in parameter_bounds.items():
            if lower >= upper:
                raise ValueError(f"Invalid bounds for parameter {param_name}: lower bound {lower} >= upper bound {upper}")

        self.parameter_bounds = parameter_bounds
        self.n_iterations = n_iterations
        self.exploration_weight = exploration_weight
        self.noise_level = noise_level

        # Initialize optimization history
        self.X_samples: List[np.ndarray] = []
        self.y_samples: List[float] = []
        self.best_params: Optional[Dict[str, float]] = None
        self.best_value: float = float('-inf')

        # Parameter names for reference
        self.param_names = list(parameter_bounds.keys())

        # Initialize quantum components
        self.quantum_manager = QuantumStateManager()

        logging.info(f"Initialized quantum-enhanced Bayesian optimizer with {len(parameter_bounds)} parameters")

    def _objective(self, X: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> Tuple[float, float]:
        """
        Calculate objective function using Gaussian Process prediction with quantum enhancement
        Returns mean and standard deviation of the prediction
        """
        if len(self.X_samples) < 2:
            return 0.0, 1.0

        try:
            # Calculate quantum-enhanced kernel matrix
            K = self._quantum_rbf_kernel(X, X)
            K_new = self._quantum_rbf_kernel(X, x_new.reshape(1, -1))
            K_new_new = self._quantum_rbf_kernel(x_new.reshape(1, -1), x_new.reshape(1, -1))

            # Add noise to diagonal for numerical stability
            K += self.noise_level * np.eye(len(X))

            # Calculate posterior with quantum enhancement
            K_inv = np.linalg.inv(K + 1e-8 * np.eye(len(K)))
            mu = K_new.T @ K_inv @ y
            sigma = K_new_new - K_new.T @ K_inv @ K_new

            return float(mu), float(np.sqrt(abs(sigma)))

        except Exception as e:
            logging.error(f"Error in GP prediction: {e}")
            return 0.0, 1.0

    def _quantum_rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Quantum-enhanced RBF kernel with coherence modulation"""
        try:
            # Get quantum metrics
            quantum_metrics = self.quantum_manager.calculate_consciousness_metrics()
            coherence = safe_complex_to_real(quantum_metrics.get('coherence', 0.5))

            # Calculate base RBF kernel
            sq_dist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)

            # Modulate kernel with quantum coherence
            quantum_factor = 1.0 + 0.5 * coherence
            return quantum_factor * np.exp(-0.5 * sq_dist)

        except Exception as e:
            logging.error(f"Error in quantum kernel calculation: {e}")
            return np.ones((X1.shape[0], X2.shape[0]))

    def _acquisition_function(self, x: np.ndarray) -> float:
        """
        Calculate acquisition function (Expected Improvement)
        with quantum-enhanced exploration-exploitation balance
        """
        try:
            if not self.X_samples or len(self.X_samples) == 0:
                return 0.0  # Return finite default value for empty samples

            X = np.array(self.X_samples)
            y = np.array(self.y_samples)

            # Get quantum metrics for exploration balance
            quantum_metrics = self.quantum_manager.calculate_consciousness_metrics()
            coherence = safe_complex_to_real(quantum_metrics.get('coherence', 0.5))

            # Calculate mean and std with quantum enhancement
            mu, sigma = self._objective(X, y, x)

            # Adjust exploration weight based on quantum coherence
            adjusted_weight = self.exploration_weight * (1 + 0.5 * (1 - coherence))

            # Calculate improvement probability with quantum adjustment
            gamma = (mu - self.best_value - adjusted_weight) / (sigma + 1e-9)
            ei = sigma * (gamma * norm.cdf(gamma) + norm.pdf(gamma))

            return -ei  # Minimize negative EI

        except Exception as e:
            logging.error(f"Error in acquisition function: {e}")
            return 0.0

    def _suggest_next_point(self) -> np.ndarray:
        """Suggest next point to evaluate using quantum-enhanced acquisition"""
        try:
            bounds = [(b[0], b[1]) for b in self.parameter_bounds.values()]

            # Multiple random starts with quantum randomization
            n_starts = 5
            best_x = None
            best_acq = float('inf')

            # Get quantum randomization factor
            quantum_metrics = self.quantum_manager.calculate_consciousness_metrics()
            quantum_factor = safe_complex_to_real(quantum_metrics.get('coherence', 0.5))

            for _ in range(n_starts):
                # Add quantum noise to starting point
                x0 = np.random.uniform(low=[b[0] for b in bounds],
                                   high=[b[1] for b in bounds])
                x0 += quantum_factor * np.random.normal(0, 0.1, size=len(x0))
                x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])

                res = minimize(self._acquisition_function, x0, bounds=bounds,
                              method='L-BFGS-B')

                if res.fun < best_acq:
                    best_acq = res.fun
                    best_x = res.x

            return best_x if best_x is not None else np.random.uniform(
                low=[b[0] for b in bounds],
                high=[b[1] for b in bounds]
            )

        except Exception as e:
            logging.error(f"Error suggesting next point: {e}")
            # Return random point as fallback
            return np.random.uniform(
                low=[b[0] for b in self.parameter_bounds.values()],
                high=[b[1] for b in self.parameter_bounds.values()]
            )

    def optimize(self, objective_function: Callable[[Dict[str, float]], float]) -> OptimizationResult:
        """
        Run Bayesian optimization loop with quantum enhancement
        """
        try:
            optimization_history = []

            # Initial random points with quantum randomization
            n_initial = min(5, self.n_iterations)
            quantum_metrics = self.quantum_manager.calculate_consciousness_metrics()
            quantum_factor = safe_complex_to_real(quantum_metrics.get('coherence', 0.5))

            for _ in range(n_initial):
                # Generate initial points with quantum noise
                x = np.random.uniform(
                    low=[b[0] for b in self.parameter_bounds.values()],
                    high=[b[1] for b in self.parameter_bounds.values()]
                )
                x += quantum_factor * np.random.normal(0, 0.1, size=len(x))
                x = np.clip(x, 
                          [b[0] for b in self.parameter_bounds.values()],
                          [b[1] for b in self.parameter_bounds.values()])

                params = {name: float(val) for name, val in zip(self.param_names, x)}
                y = objective_function(params)

                self.X_samples.append(x)
                self.y_samples.append(float(y))

                if y > self.best_value:
                    self.best_value = float(y)
                    self.best_params = params.copy()

                optimization_history.append({
                    'iteration': len(self.X_samples),
                    'parameters': params,
                    'value': float(y),
                    'uncertainty': 1.0,
                    'quantum_metrics': self.quantum_manager.calculate_consciousness_metrics()
                })

            # Main optimization loop
            for i in range(n_initial, self.n_iterations):
                x_next = self._suggest_next_point()
                params = {name: float(val) for name, val in zip(self.param_names, x_next)}
                y = objective_function(params)

                self.X_samples.append(x_next)
                self.y_samples.append(float(y))

                if y > self.best_value:
                    self.best_value = float(y)
                    self.best_params = params.copy()

                # Calculate uncertainty
                X = np.array(self.X_samples)
                y_array = np.array(self.y_samples)
                _, uncertainty = self._objective(X, y_array, x_next)

                optimization_history.append({
                    'iteration': len(self.X_samples),
                    'parameters': params,
                    'value': float(y),
                    'uncertainty': float(uncertainty),
                    'quantum_metrics': self.quantum_manager.calculate_consciousness_metrics()
                })

            # Get final quantum metrics
            final_quantum_metrics = self.quantum_manager.calculate_consciousness_metrics()

            # Calculate improvement probability with quantum enhancement
            better_than_mean = len([v for v in self.y_samples if v > np.mean(self.y_samples)])
            improvement_prob = better_than_mean / len(self.y_samples)
            improvement_prob *= (1 + 0.2 * safe_complex_to_real(final_quantum_metrics.get('coherence', 0.5)))

            return OptimizationResult(
                parameters=self.best_params,
                expected_value=self.best_value,
                uncertainty=optimization_history[-1]['uncertainty'],
                improvement_probability=min(1.0, improvement_prob),
                quantum_metrics={k: safe_complex_to_real(v) for k, v in final_quantum_metrics.items()},
                optimization_history=optimization_history
            )

        except Exception as e:
            logging.error(f"Optimization failed: {e}")
            # Return default optimization result with empty history
            default_params = {name: (bounds[0] + bounds[1])/2 
                          for name, bounds in self.parameter_bounds.items()}
            return OptimizationResult(
                parameters=default_params,
                expected_value=float('-inf'),
                uncertainty=1.0,
                improvement_probability=0.0,
                quantum_metrics=self.quantum_manager.calculate_consciousness_metrics(),
                optimization_history=[]  # Empty history on error
            )