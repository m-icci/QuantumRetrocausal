"""
Wavelet Operator Module
Implements wavelet analysis and prediction for quantum states
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import pywt
from numpy.typing import NDArray
import time
from .base import BaseOperator
from functools import lru_cache

class WaveletOperator(BaseOperator):
    """
    Implements wavelet analysis and prediction for quantum states

    Key features:
    1. Multi-level wavelet decomposition
    2. Feature extraction
    3. Pattern analysis
    4. Quantum state prediction
    5. Dynamic state evolution
    6. Cached computations
    """

    def __init__(self, wavelet: str = 'db4', level: int = 3, cache_size: int = 1024):
        """
        Initialize wavelet operator

        Args:
            wavelet: Wavelet type (default: 'db4')
            level: Decomposition level (default: 3)
            cache_size: Size of LRU cache for computations
        """
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        self.coeffs = None
        self._last_features = {}
        self._metrics = {}
        self._state_history = []
        self._prediction_cache = {}

    @lru_cache(maxsize=128)
    def predict_next_state(self, state_vector: Tuple[complex, ...], timesteps: int = 1) -> NDArray:
        """
        Predict future quantum states using wavelet analysis

        Args:
            state_vector: Current quantum state vector (converted to tuple for caching)
            timesteps: Number of steps to predict ahead

        Returns:
            Predicted quantum state vector
        """
        # Convert tuple back to array for computation
        state = np.array(state_vector)

        # Extract features and patterns
        coeffs = self.transform(state)
        features = self.extract_features(coeffs)
        patterns = self.analyze_patterns([features])

        # Predict evolution using wavelet coefficients
        predicted_coeffs = self._evolve_coefficients(coeffs, patterns, timesteps)

        # Reconstruct predicted state
        predicted_state = self.reconstruct(predicted_coeffs)

        # Normalize prediction
        return predicted_state / np.linalg.norm(predicted_state)

    def _evolve_coefficients(self, coeffs: List[NDArray], 
                           patterns: Dict[str, float], 
                           timesteps: int) -> List[NDArray]:
        """Evolve wavelet coefficients forward in time"""
        evolved_coeffs = []

        for c in coeffs:
            # Apply pattern-based evolution
            evolution_factor = np.exp(1j * patterns['pattern_complexity'])
            evolved = c * evolution_factor

            # Add dynamic phase based on energy correlation
            phase = patterns['energy_correlation'] * np.angle(c)
            evolved *= np.exp(1j * phase * timesteps)

            evolved_coeffs.append(evolved)

        return evolved_coeffs

    def get_visualization_data(self, state: NDArray) -> Dict[str, NDArray]:
        """
        Generate visualization data for the quantum state

        Returns:
            Dictionary containing different visualization representations
        """
        # Transform state
        coeffs = self.transform(state)

        # Generate multi-resolution representation
        scales = [np.abs(c) for c in coeffs]
        phases = [np.angle(c) for c in coeffs]

        # Create density representation
        density = np.abs(state) ** 2

        return {
            'scales': scales,
            'phases': phases,
            'density': density,
            'state': state
        }

    def calculate_efficiency_metrics(self) -> Dict[str, float]:
        """Calculate system efficiency metrics"""
        return {
            'computation_time': self._metrics.get('transform_time', 0),
            'cache_hits': self._get_cache_stats(),
            'prediction_accuracy': self._calculate_prediction_accuracy(),
            'compression_ratio': self._calculate_compression_ratio()
        }

    def _calculate_prediction_accuracy(self) -> float:
        """Calculate accuracy of previous predictions"""
        if len(self._state_history) < 2:
            return 0.0

        predictions = []
        actuals = []

        for i in range(len(self._state_history) - 1):
            pred = self.predict_next_state(tuple(self._state_history[i]))
            actual = self._state_history[i + 1]

            predictions.append(pred)
            actuals.append(actual)

        # Calculate fidelity between predicted and actual states
        fidelities = [np.abs(np.vdot(p, a))**2 
                     for p, a in zip(predictions, actuals)]

        return np.mean(fidelities)

    def _calculate_compression_ratio(self) -> float:
        """Calculate wavelet compression efficiency"""
        if not self.coeffs:
            return 0.0

        total_coeffs = sum(len(c) for c in self.coeffs)
        significant_coeffs = sum(np.count_nonzero(np.abs(c) > 1e-10) 
                               for c in self.coeffs)

        return 1 - (significant_coeffs / total_coeffs)

    def _get_cache_stats(self) -> float:
        """Get cache hit rate statistics"""
        hits = self.predict_next_state.cache_info().hits
        misses = self.predict_next_state.cache_info().misses
        total = hits + misses

        return hits / total if total > 0 else 0.0

    def apply(self, state: 'QuantumState') -> 'QuantumState':
        """
        Apply wavelet transform and reconstruction to quantum state

        Args:
            state: Input quantum state

        Returns:
            Transformed quantum state
        """
        if state is None:
            raise ValueError("Input state cannot be None")

        # Transform
        start_time = time.time()
        coeffs = self.transform(state.vector)
        self.coeffs = coeffs #Store coefficients

        # Extract features and store
        features = self.extract_features(coeffs)
        self._last_features = features

        # Update metrics
        self._update_metrics(start_time)

        # Reconstruct
        reconstructed = self.reconstruct(coeffs)

        # Normalize and return new state
        self._state_history.append(state.vector) #add to history
        return type(state)(reconstructed / np.linalg.norm(reconstructed))

    def transform(self, state_vector: NDArray) -> List[NDArray]:
        """
        Apply wavelet transform to quantum state

        Args:
            state_vector: Quantum state vector

        Returns:
            List of wavelet coefficients
        """
        # Handle complex state vector
        real_coeffs = self._wavelet_transform(np.real(state_vector))
        imag_coeffs = self._wavelet_transform(np.imag(state_vector))

        # Combine real and imaginary coefficients
        return [real_coeffs[i] + 1j * imag_coeffs[i] for i in range(len(real_coeffs))]

    def _wavelet_transform(self, vector: NDArray) -> List[NDArray]:
        """Apply wavelet transform to real vector"""
        # Pad vector to nearest power of 2 if needed
        n = len(vector)
        pad_len = int(2**np.ceil(np.log2(n)))
        if pad_len > n:
            vector = np.pad(vector, (0, pad_len - n))

        # Apply wavelet transform
        coeffs = pywt.wavedec(vector, self.wavelet, level=self.level)
        return [np.array(c, dtype=np.float64) for c in coeffs]

    def extract_features(self, coeffs: Optional[List[NDArray]] = None) -> Dict[str, float]:
        """
        Extract features from wavelet coefficients

        Args:
            coeffs: Optional list of wavelet coefficients. If None, use stored coefficients

        Returns:
            Dictionary of extracted features
        """
        if coeffs is None:
            coeffs = self.coeffs

        if not coeffs:
            raise ValueError("No coefficients available")

        # Calculate energy
        energy = np.sum([np.sum(np.abs(c)**2) for c in coeffs])

        # Calculate entropy (use real part to avoid complex warnings)
        all_coeffs = np.concatenate([np.real(c).flatten() for c in coeffs])
        hist, _ = np.histogram(all_coeffs, bins=20, density=True)
        entropy = -np.sum(p * np.log2(p + 1e-10) for p in hist if p > 0)

        # Calculate statistics
        skewness = self._calculate_skewness(coeffs)
        kurtosis = self._calculate_kurtosis(coeffs)

        return {
            'energy': float(energy),
            'entropy': float(entropy),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'statistics': {
                'mean': float(np.mean([np.mean(np.abs(c)) for c in coeffs])),
                'std': float(np.mean([np.std(np.abs(c)) for c in coeffs]))
            }
        }

    def analyze_patterns(self, patterns: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Analyze patterns in wavelet features

        Args:
            patterns: List of feature dictionaries from extract_features

        Returns:
            Dictionary of pattern analysis metrics
        """
        if not patterns:
            return {}

        # Extract feature sequences
        energies = [p['energy'] for p in patterns]
        entropies = [p['entropy'] for p in patterns]

        # Calculate temporal correlations
        energy_corr = np.corrcoef(energies[:-1], energies[1:])[0,1]
        entropy_corr = np.corrcoef(entropies[:-1], entropies[1:])[0,1]

        # Calculate pattern metrics
        return {
            'energy_correlation': float(energy_corr if not np.isnan(energy_corr) else 0.0),
            'entropy_correlation': float(entropy_corr if not np.isnan(entropy_corr) else 0.0),
            'pattern_stability': float(np.mean([np.std(energies), np.std(entropies)])),
            'pattern_complexity': float(np.mean(entropies))
        }

    def reconstruct(self, coeffs: Optional[List[NDArray]] = None) -> NDArray:
        """
        Reconstruct signal from wavelet coefficients

        Args:
            coeffs: Wavelet coefficients (optional)

        Returns:
            Reconstructed signal
        """
        if coeffs is None:
            coeffs = self.coeffs

        if coeffs is None:
            raise ValueError("No coefficients available")

        # Reconstruct signal
        reconstructed = self.inverse_transform(coeffs)

        return reconstructed

    def inverse_transform(self, coeffs: List[NDArray]) -> NDArray:
        """
        Apply inverse wavelet transform

        Args:
            coeffs: List of wavelet coefficients

        Returns:
            Reconstructed state vector
        """
        # Handle complex coefficients
        real_coeffs = [np.real(c) for c in coeffs]
        imag_coeffs = [np.imag(c) for c in coeffs]

        # Reconstruct real and imaginary parts
        real_part = pywt.waverec(real_coeffs, self.wavelet)
        imag_part = pywt.waverec(imag_coeffs, self.wavelet)

        return real_part + 1j * imag_part

    def _calculate_skewness(self, coeffs: List[NDArray]) -> float:
        """Calculate skewness of coefficients"""
        all_coeffs = np.concatenate([c.flatten() for c in coeffs])
        mean = np.mean(all_coeffs)
        std = np.std(all_coeffs)
        skew = np.mean(((all_coeffs - mean) / std) ** 3)
        return float(skew)

    def _calculate_kurtosis(self, coeffs: List[NDArray]) -> float:
        """Calculate kurtosis of coefficients"""
        all_coeffs = np.concatenate([c.flatten() for c in coeffs])
        mean = np.mean(all_coeffs)
        std = np.std(all_coeffs)
        kurt = np.mean(((all_coeffs - mean) / std) ** 4) - 3  # Excess kurtosis
        return float(kurt)

    def get_features(self) -> Dict[str, float]:
        """Get last extracted features"""
        return self._last_features.copy()

    def get_metrics(self) -> Dict[str, float]:
        """Get wavelet metrics"""
        return self._metrics.copy()

    def _update_metrics(self, start_time: float) -> None:
        """Update operator metrics"""
        self._metrics = {
            'transform_time': time.time() - start_time,
            'num_coefficients': sum(len(c) for c in self.coeffs),
            'max_level': self.level
        }

    def reset(self) -> None:
        """Reset operator state"""
        self.coeffs = None
        self._last_features = {}
        self._metrics = {}