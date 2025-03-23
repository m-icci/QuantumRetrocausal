"""
Dark Finance quantum operator implementation
Handles market phase detection and quantum field interactions using hybrid classical-quantum approach
"""
import numpy as np
from typing import Dict, Optional, List
import logging
from ..validation_report import ValidationReportGenerator
from .meta_validation import MetaOperatorValidator

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class DarkFinanceOperator:
    def __init__(self, dimension: int = 8):
        self.dimension = dimension
        self.validator = MetaOperatorValidator()
        logger.debug(f"Initialized Dark Finance Operator with dimension {dimension}")

    def _ensure_valid_density_matrix(self, state: np.ndarray) -> np.ndarray:
        """Ensure the state is a valid density matrix with proper numerical handling"""
        logger.debug(f"Validating density matrix - Initial trace: {np.trace(state)}")

        # Make Hermitian with numerical stability
        state = 0.5 * (state + state.conj().T)

        # Ensure positive eigenvalues with minimum threshold
        eigenvals, eigenvecs = np.linalg.eigh(state)
        logger.debug(f"Minimum eigenvalue: {np.min(eigenvals.real)}")

        eigenvals = np.maximum(eigenvals.real, 1e-8)

        # Reconstruct with valid eigenvalues
        state = eigenvecs @ np.diag(eigenvals) @ eigenvecs.conj().T

        # Normalize trace
        trace = np.trace(state).real
        logger.debug(f"Trace before normalization: {trace}")

        if abs(trace) > 1e-8:
            state = state / trace
        else:
            logger.warning("Trace too small, using identity fallback")
            state = np.eye(state.shape[0]) / state.shape[0]

        return state

    def _calculate_classical_indicators(self, prices: np.ndarray) -> Dict[str, float]:
        """Calculate classical technical indicators"""
        if len(prices) < 2:
            logger.warning("Insufficient price data points")
            return {
                'trend': 0.0,
                'volatility': 0.0,
                'momentum': 0.0,
                'strength': 0.0
            }

        # Calculate linear trend
        x = np.arange(len(prices))
        slope, intercept = np.polyfit(x, prices, 1)
        trend_direction = np.sign(slope)

        # Calculate changes and returns
        changes = np.diff(prices)
        returns = changes / prices[:-1]

        logger.debug("Market Analysis:")
        logger.debug(f"Price range: {np.min(prices):.2f} - {np.max(prices):.2f}")
        logger.debug(f"Trend slope: {slope:.4f}")
        logger.debug(f"Average return: {np.mean(returns)*100:.2f}%")

        # Calculate trend strength from rate of change
        avg_change = np.mean(abs(changes))
        price_range = np.max(prices) - np.min(prices)
        normalized_slope = abs(slope) * len(prices) / price_range

        # Initial strength calculation
        strength = np.clip(normalized_slope * 2.0, 0, 1)
        logger.debug(f"Initial strength: {strength:.4f}")

        # Consistency boost
        consistent_moves = np.sum(np.sign(changes) == trend_direction)
        consistency = consistent_moves / len(changes)

        if consistency > 0.8:  # Strong trend consistency
            strength *= 1.5  # 50% boost
            logger.debug(f"After consistency boost: {strength:.4f}")

        # Momentum calculation
        total_return = (prices[-1] - prices[0]) / prices[0]
        momentum = np.clip(abs(total_return), 0, 1)
        logger.debug(f"Momentum: {momentum:.4f}")

        if momentum > 0.1:  # Significant movement
            strength *= (1.0 + momentum)
            logger.debug(f"After momentum boost: {strength:.4f}")

        # Volatility adjustment
        volatility = np.std(returns)
        norm_volatility = np.clip(volatility * 5, 0, 1)
        logger.debug(f"Normalized volatility: {norm_volatility:.4f}")

        if norm_volatility > 0.2:
            damping = 1.0 - (norm_volatility - 0.2) * 0.3  # 30% max reduction
            strength *= damping
            logger.debug(f"After volatility damping: {strength:.4f}")

        # Minimum strength guarantee for clear trends
        if consistency > 0.8 and momentum > 0.2:
            strength = max(strength, 0.7)
            logger.debug(f"Applied minimum strength: {strength:.4f}")

        # Final normalization
        final_strength = float(np.clip(strength, 0, 1.0))

        return {
            'trend': float(trend_direction * final_strength),
            'volatility': float(norm_volatility),
            'momentum': float(momentum),
            'strength': final_strength,
            'consistency': float(consistency)
        }

    def compute_market_field(self, prices: np.ndarray) -> np.ndarray:
        """Create quantum field with direct strength encoding"""
        indicators = self._calculate_classical_indicators(prices)
        strength = indicators['strength']

        logger.debug(f"\nComputing quantum field:")
        logger.debug(f"Classical strength: {strength:.4f}")

        # Create basis states for direct strength encoding
        trend_state = np.zeros(self.dimension, dtype=np.complex128)

        # Direct strength encoding in first basis state
        trend_state[0] = np.sqrt(strength)  # Main strength component
        trend_state[1] = np.sqrt(1 - strength)  # Orthogonal component

        logger.debug(f"Basis state weights: [{trend_state[0]:.4f}, {trend_state[1]:.4f}]")

        # Create density matrix
        field = np.outer(trend_state, trend_state.conj())

        # Add minimal noise for quantum effects
        vol_factor = min(indicators['volatility'] * 0.05, 0.02)  # Very low noise
        if vol_factor > 0:
            noise = vol_factor * (np.random.randn(self.dimension, self.dimension) + 
                                1j * np.random.randn(self.dimension, self.dimension))
            noise = noise @ noise.conj().T
            noise = noise / np.trace(noise).real
            field = (1 - vol_factor) * field + vol_factor * noise
            logger.debug(f"Added noise factor: {vol_factor:.4f}")

        return self._ensure_valid_density_matrix(field)

    def predict_market_movement(self, current_field: np.ndarray,
                              historical_fields: np.ndarray) -> Dict:
        """Predict market movement with direct strength extraction"""
        logger.debug("\nPredicting market movement")

        # Get quantum metrics
        current = self._ensure_valid_density_matrix(current_field)
        quantum_metrics = self.calculate_market_potential(current)

        # Extract strength directly from first basis state
        basis_weights = np.abs(np.diagonal(current))
        strength = float(basis_weights[0])  # Direct strength from first basis

        logger.debug(f"Extracted quantum strength: {strength:.4f}")

        # Get classical indicators
        prices = []
        if len(historical_fields) > 0:
            prices = [np.real(np.trace(f)) for f in historical_fields]
        prices.append(np.real(np.trace(current)))
        classical = self._calculate_classical_indicators(np.array(prices))

        logger.debug(f"Classical strength: {classical['strength']:.4f}")

        # Apply quantum coherence boost
        if quantum_metrics['coherence'] > 0.5:
            boost = quantum_metrics['coherence'] * 0.5  # Up to 50% boost
            strength = min(strength * (1.0 + boost), 1.0)
            logger.debug(f"After coherence boost: {strength:.4f}")

        # Ensure minimum strength for strong classical signals
        if classical['consistency'] > 0.8 and classical['momentum'] > 0.2:
            strength = max(strength, 0.7)
            logger.debug(f"Final strength: {strength:.4f}")

        # Calculate confidence
        classical_conf = 0.5 * (1 + classical['consistency'])
        quantum_conf = 0.5 * (quantum_metrics['coherence'] + quantum_metrics['entropy'])
        confidence = float(np.clip(0.5 * (classical_conf + quantum_conf), 0, 0.8))

        return {
            'direction': float(np.sign(classical['trend'])),
            'strength': strength,
            'confidence': confidence,
            'metrics': quantum_metrics,
            'classical_indicators': classical
        }

    def calculate_market_potential(self, field: np.ndarray) -> Dict:
        """Calculate quantum metrics"""
        state = self._ensure_valid_density_matrix(field)

        coherence = self.validator.calculate_coherence(state)
        entropy = self.validator.calculate_entropy(state)

        max_coherence = self.dimension * (self.dimension - 1) / 2
        max_entropy = np.log(self.dimension)

        norm_coherence = np.clip(coherence / max_coherence, 0, 1)
        norm_entropy = 1 - np.clip(entropy / max_entropy, 0, 1)

        logger.debug(f"Quantum metrics - Coherence: {norm_coherence:.4f}, Entropy: {norm_entropy:.4f}")

        return {
            'coherence': float(norm_coherence),
            'entropy': float(norm_entropy),
            'trace': float(np.real(np.trace(state)))
        }

    def apply_morphic_resonance(self, field: np.ndarray) -> Dict:
        """Apply morphic resonance and measure quantum properties"""
        state = self._ensure_valid_density_matrix(field)
        metrics = self.calculate_market_potential(state)

        return {
            'resonance_score': float(metrics['coherence']),
            'temporal_coherence': float(metrics['entropy']),
            'final_state': state
        }