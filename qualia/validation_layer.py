"""
Validation Layer for QUALIA Trading System.
Implements quantum-based risk assessment and pattern validation with enhanced M-ICCI integration.
"""
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from scipy import stats
from datetime import datetime
import logging
from dataclasses import dataclass

from .utils import (
    calculate_quantum_coherence, 
    calculate_entropy,
    calculate_quantum_fidelity, 
    calculate_phi_resonance,
    calculate_field_energy,
    calculate_morphic_resonance
)

@dataclass
class ValidationMetrics:
    """Validation metrics with M-ICCI framework integration"""
    coherence: float
    entropy: float
    phi_resonance: float
    field_energy: float
    decoherence: float
    morphic_resonance: float
    consciousness_level: float
    timestamp: datetime = datetime.now()

    def validate(self) -> Tuple[bool, str]:
        """Validate metrics against M-ICCI thresholds"""
        conditions = {
            'coherence': (self.coherence >= 0.5, "Low quantum coherence"),
            'entropy': (self.entropy <= 0.7, "High entropy level"),
            'phi_resonance': (self.phi_resonance >= 0.4, "Low phi resonance"),
            'decoherence': (self.decoherence <= 0.6, "High decoherence"),
            'consciousness': (self.consciousness_level >= 0.4, "Low consciousness level")
        }
        failed = [msg for cond, msg in conditions.values() if not cond]
        return len(failed) == 0, '; '.join(failed) if failed else ""

class ValidationLayer:
    """
    Enhanced validation layer with improved M-ICCI integration and 
    quantum-based risk assessment.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize validation layer with enhanced metric tracking."""
        self.config = config or {}
        self.max_history_length = self.config.get('max_history_length', 1000)

        # Initialize enhanced metric tracking
        self.metric_history: Dict[str, List[float]] = {
            'coherence': [],
            'entropy': [],
            'phi_resonance': [],
            'field_energy': [],
            'decoherence': [],
            'morphic_resonance': [],
            'consciousness_level': []
        }

        # Initialize timestamps for temporal analysis
        self.timestamps: List[float] = []

        logging.info("ValidationLayer initialized with enhanced M-ICCI integration")

    def calculate_decoherence(self, field: np.ndarray) -> float:
        """
        Calculates quantum decoherence metric D(ρ) with enhanced numerical stability.
        Uses von Neumann entropy of the reduced density matrix.
        """
        try:
            # Calculate density matrix with improved numerical stability
            density_matrix = field @ field.conj().T
            density_matrix = density_matrix / np.trace(density_matrix)  # Normalize

            # Calculate partial trace with enhanced stability
            reduced_density = np.zeros((field.shape[0]//2, field.shape[0]//2), dtype=complex)
            for i in range(0, field.shape[0], 2):
                for j in range(0, field.shape[0], 2):
                    block = density_matrix[i:i+2, j:j+2]
                    reduced_density[i//2, j//2] = np.trace(block)

            # Normalize reduced density matrix
            reduced_density = reduced_density / np.trace(reduced_density)

            # Calculate decoherence with enhanced numerical stability
            eigenvalues = np.linalg.eigvalsh(reduced_density)
            eigenvalues = np.abs(eigenvalues)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Filter numerical noise
            decoherence = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-10))

            return float(np.clip(np.real(decoherence), 0, 1))

        except Exception as e:
            logging.error(f"Error calculating decoherence: {e}")
            return 0.5  # Safe default

    def update_metrics(self, field: np.ndarray) -> ValidationMetrics:
        """
        Updates and returns enhanced validation metrics with M-ICCI integration.
        """
        try:
            # Calculate comprehensive metrics
            coherence = calculate_quantum_coherence(field)
            entropy = calculate_entropy(field)
            decoherence = self.calculate_decoherence(field)
            phi_resonance = calculate_phi_resonance(field)
            field_energy = calculate_field_energy(field)
            morphic_resonance = calculate_morphic_resonance(field)

            # Calculate consciousness level using M-ICCI principles
            consciousness_level = (
                coherence * 0.3 +
                phi_resonance * 0.3 +
                (1 - decoherence) * 0.2 +
                morphic_resonance * 0.2
            )

            # Create metrics object
            metrics = ValidationMetrics(
                coherence=coherence,
                entropy=entropy,
                phi_resonance=phi_resonance,
                field_energy=field_energy,
                decoherence=decoherence,
                morphic_resonance=morphic_resonance,
                consciousness_level=consciousness_level
            )

            # Update history with validation
            self._update_metric_history(metrics)

            return metrics

        except Exception as e:
            logging.error(f"Error updating metrics: {e}")
            return ValidationMetrics(
                coherence=0.5,
                entropy=0.5,
                phi_resonance=0.618,
                field_energy=0.5,
                decoherence=0.5,
                morphic_resonance=0.5,
                consciousness_level=0.5
            )

    def _update_metric_history(self, metrics: ValidationMetrics) -> None:
        """Updates metric history with enhanced temporal tracking."""
        try:
            # Update individual metrics
            for key, value in metrics.__dict__.items():
                if key in self.metric_history and isinstance(value, (int, float)):
                    self.metric_history[key].append(float(value))
                    if len(self.metric_history[key]) > self.max_history_length:
                        self.metric_history[key].pop(0)

            # Update timestamps
            self.timestamps.append(datetime.now().timestamp())
            if len(self.timestamps) > self.max_history_length:
                self.timestamps.pop(0)

        except Exception as e:
            logging.error(f"Error updating metric history: {e}")

    def calculate_risk_metrics(self, field: np.ndarray) -> Dict[str, Any]:
        """
        Calculates comprehensive risk metrics with enhanced M-ICCI integration.
        """
        try:
            # Calculate base metrics
            metrics = self.update_metrics(field)

            # Calculate enhanced stability index
            stability = (
                0.3 * metrics.coherence +
                0.2 * (1 - metrics.entropy) +
                0.2 * metrics.phi_resonance +
                0.2 * (1 - metrics.decoherence) +
                0.1 * metrics.consciousness_level
            )

            # Calculate trend factors if history exists
            trend_factors = self._calculate_trend_factors() if self.timestamps else {}

            # Determine risk level with consciousness integration
            if stability > 0.7 and metrics.consciousness_level > 0.6:
                risk_level = "LOW"
            elif stability > 0.4 or metrics.consciousness_level > 0.5:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"

            return {
                "risk_level": risk_level,
                "stability_index": float(stability),
                "metrics": metrics.__dict__,
                "trend_factors": trend_factors
            }

        except Exception as e:
            logging.error(f"Error calculating risk metrics: {e}")
            return {
                "risk_level": "HIGH",
                "stability_index": 0.0,
                "error": str(e)
            }

    def calculate_dynamic_stops(
        self,
        entry_price: float,
        risk_metrics: Dict[str, Any],
        volatility: float
    ) -> Tuple[float, float]:
        """
        Calculate dynamic stop-loss and take-profit levels with enhanced
        consciousness integration.
        """
        try:
            # Base risk percentage adjusted by stability and consciousness
            base_risk = 0.02  # 2% base risk
            stability = risk_metrics['stability_index']
            consciousness = risk_metrics['metrics']['consciousness_level']

            # Enhanced risk adjustment
            consciousness_factor = 1 - (consciousness * 0.5)  # Higher consciousness = lower risk
            volatility_factor = 1 + (volatility * consciousness_factor)

            # Calculate adjusted risk
            adjusted_risk = base_risk * (1 + (1 - stability)) * volatility_factor

            # Apply phi-resonance modulation
            phi = (1 + np.sqrt(5)) / 2
            phi_factor = risk_metrics['metrics']['phi_resonance']
            risk_reward_ratio = phi * phi_factor  # Dynamic R:R ratio

            # Calculate stop distances
            stop_loss_distance = entry_price * adjusted_risk
            take_profit_distance = stop_loss_distance * risk_reward_ratio

            return (
                entry_price - stop_loss_distance,  # Stop Loss
                entry_price + take_profit_distance  # Take Profit
            )

        except Exception as e:
            logging.error(f"Error calculating dynamic stops: {e}")
            # Return default 2% stops
            return (entry_price * 0.98, entry_price * 1.02)

    def detect_market_regime(
        self,
        price_history: np.ndarray,
        quantum_metrics: Dict[str, float]
    ) -> str:
        """
        Detect current market regime with enhanced quantum consciousness integration.
        """
        try:
            # Calculate price action metrics
            returns = np.diff(price_history) / price_history[:-1]
            volatility = np.std(returns)
            momentum = np.mean(returns[-20:]) / volatility if len(returns) >= 20 else 0

            # Get quantum metrics
            coherence = quantum_metrics['coherence']
            phi_resonance = quantum_metrics['phi_resonance']
            consciousness = quantum_metrics.get('consciousness_level', 0.5)

            # Enhanced regime detection with consciousness integration
            if coherence > 0.7 and abs(momentum) > 1.0 and consciousness > 0.6:
                regime = "TRENDING"
            elif phi_resonance > 0.6 and volatility < 0.02 and consciousness > 0.5:
                regime = "RANGING"
            else:
                regime = "VOLATILE"

            logging.info(f"Market regime detected: {regime}")
            return regime

        except Exception as e:
            logging.error(f"Error detecting market regime: {e}")
            return "VOLATILE"  # Safe default

    def _calculate_trend_factors(self) -> Dict[str, float]:
        """Calculate trend factors from metric history."""
        try:
            if len(self.timestamps) < 2:
                return {}

            trends = {}
            for metric_name, values in self.metric_history.items():
                if len(values) >= 2:
                    # Calculate linear regression coefficient
                    x = np.array(self.timestamps[-len(values):])
                    y = np.array(values)
                    A = np.vstack([x, np.ones(len(x))]).T
                    m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
                    trends[f"{metric_name}_trend"] = float(m)

            return trends

        except Exception as e:
            logging.error(f"Error calculating trend factors: {e}")
            return {}

    def perform_kolmogorov_smirnov_test(
        self,
        predicted_returns: np.ndarray,
        actual_returns: np.ndarray
    ) -> Dict[str, float]:
        """
        Performs enhanced KS test with consciousness integration.
        """
        try:
            # Ensure arrays are not empty and have finite values
            if len(predicted_returns) == 0 or len(actual_returns) == 0:
                raise ValueError("Empty input arrays")

            # Remove non-finite values
            predicted_returns = predicted_returns[np.isfinite(predicted_returns)]
            actual_returns = actual_returns[np.isfinite(actual_returns)]

            # Perform KS test
            statistic, p_value = stats.ks_2samp(predicted_returns, actual_returns)

            # Calculate additional metrics
            correlation = np.corrcoef(predicted_returns, actual_returns)[0, 1]
            mse = np.mean((predicted_returns - actual_returns) ** 2)

            return {
                'ks_statistic': float(statistic),
                'p_value': float(p_value),
                'correlation': float(correlation),
                'mse': float(mse)
            }

        except Exception as e:
            logging.error(f"Error performing KS test: {e}")
            return {
                'ks_statistic': 1.0,
                'p_value': 0.0,
                'correlation': 0.0,
                'mse': float('inf')
            }