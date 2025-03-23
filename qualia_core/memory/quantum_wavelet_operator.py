"""
Quantum Wavelet Operator implementation following QUALIA framework principles.
Implements pattern detection and coherence analysis in quantum state spaces.
Based on QUALIA Section 2.2 - Advanced Layer: QuantumMorphicField.

Reference: QUALIA's theoretical foundation - Quantum Understanding and Adaptive 
Learning Integration Architecture, focusing on emergent pattern detection and 
quantum state preservation.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import pywt
from dataclasses import dataclass

@dataclass
class WaveletConfig:
    """Configuration for quantum wavelet operations."""
    wavelet_type: str = 'db4'  # Daubechies-4 wavelet for quantum state analysis
    decomposition_level: int = 2  # Multi-scale quantum state decomposition
    threshold: float = 0.05  # Threshold for quantum pattern detection
    coherence_threshold: float = 0.4  # Quantum coherence threshold
    min_pattern_size: int = 4  # Minimum size for coherent patterns
    min_signal_length: int = 16  # Minimum length for stable wavelet transform
    boundary_mode: str = 'symmetric'  # Boundary extension mode
    quantum_tolerance: float = 1e-10  # Tolerance for quantum state verification
    phi: float = (1 + np.sqrt(5)) / 2  # Golden ratio for sacred geometry

    def __post_init__(self):
        """Validate quantum configuration parameters."""
        if self.decomposition_level < 1:
            raise ValueError("Decomposition level must be positive")
        if not 0 < self.threshold < 1:
            raise ValueError("Threshold must be between 0 and 1")
        if not 0 < self.coherence_threshold < 1:
            raise ValueError("Coherence threshold must be between 0 and 1")
        if self.min_signal_length < 16:  # Ensure stable wavelet transform
            self.min_signal_length = 16
        if self.quantum_tolerance <= 0:
            raise ValueError("Quantum tolerance must be positive")
        valid_modes = ['zero', 'constant', 'symmetric']
        if self.boundary_mode not in valid_modes:
            self.boundary_mode = 'symmetric'

class QuantumWaveletOperator:
    """
    Quantum Wavelet Operator for analyzing quantum states and detecting coherent patterns.
    Implements QUALIA framework principles for quantum state analysis and morphic field integration.
    """

    def __init__(self, config: Optional[WaveletConfig] = None):
        """Initialize quantum wavelet operator with configuration."""
        self.config = config or WaveletConfig()
        self._validate_config()
        self.wavelet = pywt.Wavelet(self.config.wavelet_type)

    def _validate_config(self):
        """Validate quantum operator configuration."""
        if self.config.decomposition_level < 1:
            raise ValueError("Decomposition level must be positive")
        if not 0 < self.config.threshold < 1:
            raise ValueError("Threshold must be between 0 and 1")

    def _verify_quantum_state(self, state: np.ndarray) -> bool:
        """Verify quantum state properties including normalization and numerical stability."""
        if not isinstance(state, np.ndarray):
            return False
        if not np.isfinite(state).all():
            return False
        norm = np.linalg.norm(state)
        return abs(norm - 1.0) < self.config.quantum_tolerance

    def _get_padded_length(self, length: int) -> int:
        """Calculate required padded length based on sacred geometry principles."""
        min_length = max(self.config.min_signal_length, 
                        self.wavelet.dec_len * (2 ** self.config.decomposition_level))
        # Use golden ratio for optimal padding
        phi_adjusted_length = int(min_length * self.config.phi)
        return 2 ** int(np.ceil(np.log2(max(length, phi_adjusted_length))))

    def _pad_quantum_state(self, state: np.ndarray) -> Tuple[np.ndarray, int]:
        """Pad quantum state while preserving quantum properties and morphic resonance."""
        if not self._verify_quantum_state(state):
            raise ValueError("Input does not satisfy quantum state requirements")

        original_length = len(state)
        padded_length = self._get_padded_length(original_length)

        # Handle padding modes with morphic field preservation
        if self.config.boundary_mode == 'zero':
            padded_state = np.pad(state, (0, padded_length - original_length),
                               mode='constant', constant_values=0)
        else:
            padded_state = np.pad(state, (0, padded_length - original_length),
                               mode=self.config.boundary_mode)

        # Preserve quantum normalization and morphic coherence
        padded_state = padded_state / np.linalg.norm(padded_state)
        return padded_state, original_length

    def decompose(self, state: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Decompose quantum state using wavelet transform.
        Implements multi-scale quantum state analysis from QUALIA framework.
        """
        padded_state, original_length = self._pad_quantum_state(state)

        # Multi-scale quantum decomposition with morphic field preservation
        coeffs = pywt.wavedec(
            padded_state,
            self.config.wavelet_type,
            mode='symmetric',
            level=self.config.decomposition_level
        )

        return {
            'approximation': coeffs[0],
            'details': coeffs[1:],
            'levels': list(range(self.config.decomposition_level)),
            'original_length': original_length,
            'morphic_coherence': self._calculate_morphic_coherence(coeffs)
        }

    def reconstruct(self, coeffs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Reconstruct quantum state from wavelet coefficients.
        Preserves quantum state properties and morphic field coherence.
        """
        coeff_list = [coeffs['approximation']] + list(coeffs['details'])
        reconstructed = pywt.waverec(
            coeff_list,
            self.config.wavelet_type,
            mode='symmetric'
        )

        # Ensure quantum properties preservation
        original_length = coeffs['original_length']
        reconstructed = reconstructed[:original_length]
        reconstructed = reconstructed / np.linalg.norm(reconstructed)
        return reconstructed

    def detectPatterns(self, coeffs: Dict[str, np.ndarray]) -> List[Dict[str, float]]:
        """
        Detect quantum patterns using sacred geometry principles.
        Implements QUALIA's pattern recognition with φ-based thresholding.
        """
        patterns = []
        if not coeffs['details']:
            return patterns

        # Calculate global energy for relative thresholding
        global_energy = sum(np.sum(np.abs(d) ** 2) for d in coeffs['details'])
        if global_energy < 1e-10:
            return patterns

        for level, detail in enumerate(coeffs['details']):
            if len(detail) < self.config.min_pattern_size:
                continue

            # Calculate quantum pattern metrics with φ-based scaling
            energy = float(np.sum(np.abs(detail) ** 2))
            relative_energy = energy / global_energy
            phi_energy = energy * self.config.phi

            mean_magnitude = np.mean(np.abs(detail))
            std = max(np.std(detail), 1e-10)

            # Calculate coherence using sacred geometry principles
            base_coherence = float(np.clip(mean_magnitude / std, 0, 1))
            phi_coherence = float(np.clip(base_coherence * self.config.phi, 0, 1))
            morphic_resonance = self._calculate_morphic_resonance(detail)

            # Adaptive thresholding based on level and φ
            level_threshold = self.config.threshold / (self.config.phi ** level)

            # Pattern detection with φ-weighted criteria
            if (relative_energy > level_threshold and 
                (phi_coherence > self.config.coherence_threshold or
                 morphic_resonance > self.config.coherence_threshold)):

                pattern = {
                    'level': level,
                    'energy': phi_energy,
                    'coherence': phi_coherence,
                    'morphic_resonance': morphic_resonance,
                    'significance': float(phi_energy * phi_coherence * morphic_resonance)
                }
                patterns.append(pattern)

        return sorted(patterns, key=lambda x: x['significance'], reverse=True)

    def _calculate_morphic_resonance(self, detail: np.ndarray) -> float:
        """Calculate morphic field resonance using sacred geometry principles."""
        if len(detail) < 2:
            return 0.0

        # Calculate φ-weighted correlation
        correlations = np.correlate(detail, detail, mode='full')
        center_idx = len(correlations) // 2
        phi_weights = np.exp(-np.abs(np.arange(len(correlations)) - center_idx) / self.config.phi)
        weighted_correlation = np.sum(correlations * phi_weights) / np.sum(phi_weights)

        return float(np.clip(weighted_correlation / (np.var(detail) + 1e-10), 0, 1))

    def _calculate_morphic_coherence(self, coeffs: List[np.ndarray]) -> float:
        """Calculate morphic field coherence across wavelet scales."""
        if not coeffs or len(coeffs) < 2:
            return 0.0

        # Calculate cross-scale coherence with φ-based weighting
        coherence_values = []
        for i in range(len(coeffs) - 1):
            coeff1, coeff2 = coeffs[i], coeffs[i + 1]
            if len(coeff1) > 0 and len(coeff2) > 0:
                correlation = self._calculate_correlation(coeff1, coeff2)
                scale_weight = self.config.phi ** (-i)  # φ-based scale weighting
                coherence_values.append(correlation * scale_weight)

        if not coherence_values:
            return 0.0

        return float(np.clip(np.mean(coherence_values), 0, 1))

    def _calculate_correlation(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Calculate correlation coefficient with robust error handling."""
        if len(data1) < 2 or len(data2) < 2:
            return 0.0

        min_len = min(len(data1), len(data2))
        data1 = data1[:min_len]
        data2 = data2[:min_len]

        if np.all(data1 == data1[0]) or np.all(data2 == data2[0]):
            return 0.0

        try:
            corr = np.corrcoef(data1, data2)[0, 1]
            return float(0.0 if np.isnan(corr) else corr)
        except (ValueError, RuntimeWarning):
            return 0.0
    
    def applyMorphology(self, coeffs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply morphological operations on wavelet coefficients.
        Implements QUALIA's morphological enhancement principles.
        """
        morphed_coeffs = {
            'approximation': coeffs['approximation'].copy(),
            'details': [],
            'levels': coeffs['levels']
        }

        if not coeffs['details']:
            return morphed_coeffs

        # Apply enhanced morphological operations on each detail level
        for detail in coeffs['details']:
            if len(detail) < self.config.min_pattern_size:
                morphed_coeffs['details'].append(np.zeros_like(detail))
                continue

            # Dynamic thresholding based on level statistics
            level_std = max(np.std(detail), 1e-10)
            threshold = self.config.threshold * level_std

            # Non-linear enhancement of significant coefficients
            morphed_detail = np.zeros_like(detail)
            significant = np.abs(detail) > threshold

            if np.any(significant):
                # Apply non-linear enhancement with numerical stability
                enhancement_factor = 1 + np.tanh(np.abs(detail[significant]) / level_std)
                morphed_detail[significant] = detail[significant] * enhancement_factor

            morphed_coeffs['details'].append(morphed_detail)

        return morphed_coeffs

    def extractStructures(self, features: Dict[str, np.ndarray]) -> List[Dict[str, float]]:
        """
        Extract coherent quantum structures from morphological features.
        Implements QUALIA's quantum structure detection principles.
        """
        structures = []

        if not features['details']:
            return structures

        for level, detail in enumerate(features['details']):
            if len(detail) < self.config.min_pattern_size:
                continue

            # Adaptive quantum thresholding with robust statistics
            level_std = max(np.std(detail), 1e-10)
            local_threshold = self.config.threshold * level_std

            # Find significant quantum regions
            significant_indices = np.where(np.abs(detail) > local_threshold)[0]

            if len(significant_indices) > 1:
                # Split into connected quantum regions
                regions = np.split(significant_indices,
                                 np.where(np.diff(significant_indices) > 1)[0] + 1)

                for region in regions:
                    if len(region) >= self.config.min_pattern_size:
                        # Extract quantum region properties
                        region_data = detail[region]

                        # Calculate quantum structure properties with robust error handling
                        amplitude = float(np.mean(np.abs(region_data)))
                        std = max(np.std(region_data), 1e-10)

                        # Calculate quantum coherence metrics
                        amplitude_coherence = np.clip(amplitude / std, 0, 1)

                        if len(region_data) > 1:
                            local_correlation = self._calculate_correlation(
                                region_data[:-1], region_data[1:]
                            )
                        else:
                            local_correlation = 0.0

                        coherence = float(np.clip(
                            0.5 * (amplitude_coherence + np.abs(local_correlation)),
                            0, 1
                        ))

                        # Calculate quantum energy density
                        energy_density = float(np.sum(np.abs(region_data)**2) / len(region))

                        if coherence > self.config.coherence_threshold:
                            structure = {
                                'level': level,
                                'size': len(region),
                                'amplitude': amplitude,
                                'coherence': coherence,
                                'energy_density': energy_density,
                                'location': float(np.mean(region)),
                                'span': (float(region[0]), float(region[-1]))
                            }
                            structures.append(structure)

        return structures

    def _calculate_correlation(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """
        Calculate correlation coefficient with robust error handling.
        """
        if len(data1) < 2 or len(data2) < 2:
            return 0.0

        # Ensure arrays have same length
        min_len = min(len(data1), len(data2))
        data1 = data1[:min_len]
        data2 = data2[:min_len]

        # Check for zero variance
        if np.all(data1 == data1[0]) or np.all(data2 == data2[0]):
            return 0.0

        try:
            corr = np.corrcoef(data1, data2)[0, 1]
            return float(0.0 if np.isnan(corr) else corr)
        except (ValueError, RuntimeWarning):
            return 0.0
    
    def _adjust_decomposition_level(self, signal_length: int) -> int:
        """
        Adjust decomposition level for quantum state analysis.
        Follows QUALIA principles of multi-scale analysis.
        """
        # Calculate maximum possible decomposition level based on signal length
        max_level = pywt.dwt_max_level(signal_length, self.wavelet.dec_len)
        target_level = min(self.config.decomposition_level, max_level)

        # Ensure we have enough samples for the requested decomposition level
        min_samples = self.wavelet.dec_len * (2 ** target_level)
        if signal_length >= min_samples:
            return target_level

        # Reduce level if signal is too short
        while target_level > 1 and signal_length < self.wavelet.dec_len * (2 ** target_level):
            target_level -= 1
        return target_level

    def analyzeCoherence(self, coeffs: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Analyze quantum coherence in wavelet decomposition.
        Implements QUALIA's quantum coherence metrics.
        """
        metrics = {
            'global_coherence': 0.0,
            'level_coherence': [],
            'pattern_stability': 0.0
        }

        if not coeffs['details']:
            return metrics

        # Calculate global quantum coherence with robust error handling
        all_coeffs = np.concatenate([coeffs['approximation']] + list(coeffs['details']))
        if len(all_coeffs) > 0:
            mean_magnitude = np.mean(np.abs(all_coeffs))
            std = max(np.std(all_coeffs), 1e-10)
            metrics['global_coherence'] = float(np.clip(mean_magnitude / std, 0, 1))

        # Calculate level-wise quantum coherence
        for detail in coeffs['details']:
            if len(detail) > 0:
                mean_magnitude = np.mean(np.abs(detail))
                std = max(np.std(detail), 1e-10)
                level_coherence = float(np.clip(mean_magnitude / std, 0, 1))
                metrics['level_coherence'].append(level_coherence)

        # Calculate quantum pattern stability with robust correlation
        if len(coeffs['details']) > 1:
            stability_values = []
            for i in range(len(coeffs['details']) - 1):
                detail1, detail2 = coeffs['details'][i], coeffs['details'][i+1]
                correlation = self._calculate_correlation(detail1, detail2)
                if not np.isnan(correlation):
                    stability_values.append(np.abs(correlation))

            if stability_values:
                metrics['pattern_stability'] = float(np.clip(np.mean(stability_values), 0, 1))

        return metrics