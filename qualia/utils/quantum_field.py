"""
Quantum Field Operations Module for QUALIA Trading System.
Implements quantum field manipulations with enhanced performance monitoring.
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
from scipy import stats
from scipy.linalg import expm
from .logging import setup_logger

logger = setup_logger(__name__)

# Performance monitoring constants
OPTIMIZATION_THRESHOLD = 1e-10
PERFORMANCE_METRICS = {
    'dimension_changes': 0,
    'energy_corrections': 0,
    'evolution_steps': 0
}

def reset_performance_metrics():
    """Reset performance monitoring counters"""
    global PERFORMANCE_METRICS
    PERFORMANCE_METRICS = {
        'dimension_changes': 0,
        'energy_corrections': 0,
        'evolution_steps': 0
    }

def get_performance_metrics() -> Dict[str, int]:
    """Return current performance metrics"""
    return PERFORMANCE_METRICS.copy()

def safe_complex_to_real(value: Any) -> float:
    """Enhanced safe conversion of complex values to float with proper magnitude handling"""
    try:
        if value is None:
            return 0.0

        if isinstance(value, (np.ndarray, list)):
            values = np.asarray([complex(v) for v in value])
            magnitudes = np.abs(values)
            return float(np.sqrt(np.mean(magnitudes**2)))

        complex_val = complex(value)
        return float(np.abs(complex_val))

    except Exception as e:
        logger.error(f"Error converting to real: {e}")
        return 0.0

def serialize_complex_value(value: Union[complex, np.complex128, float, np.ndarray]) -> Dict[str, float]:
    """Serialize complex numbers for JSON compatibility with enhanced error handling"""
    try:
        if isinstance(value, (np.ndarray, list)):
            complex_val = np.mean([complex(v) for v in np.asarray(value).flatten()])
        else:
            complex_val = complex(value)

        return {
            'magnitude': float(np.abs(complex_val)),
            'phase': float(np.angle(complex_val)),
            'real': float(complex_val.real),
            'imag': float(complex_val.imag)
        }
    except Exception as e:
        logger.error(f"Error serializing complex value: {e}")
        return {
            'magnitude': 0.0,
            'phase': 0.0,
            'real': 0.0,
            'imag': 0.0
        }

def normalize_quantum_state(state: np.ndarray, target_norm: float = 1.0) -> np.ndarray:
    """Normalize quantum state while preserving relative phase information."""
    try:
        norm = np.linalg.norm(state)
        if norm > 0:
            return state * (target_norm / norm)
        return state
    except Exception as e:
        logger.error(f"Error normalizing quantum state: {e}")
        return state

def create_quantum_field(size: int = 64) -> np.ndarray:
    """Creates a normalized quantum field with enhanced performance."""
    try:
        # Always create 64x64 field regardless of input size
        target_size = 64

        # Use vectorized operations for better performance
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio for morphic modulation
        field = np.random.randn(target_size, target_size) + 1j * np.random.randn(target_size, target_size)

        # Apply morphic resonance modulation using broadcasting
        x = np.arange(target_size).reshape(-1, 1)
        modulation = np.exp(-x / (target_size * phi))
        field *= modulation

        # Normalize while preserving phase information
        norm = np.linalg.norm(field)
        if norm > 0:
            field /= norm
            logger.debug(f"Created quantum field with size {target_size}x{target_size}")
            return field

        logger.warning("Zero norm encountered in quantum field creation")
        return np.eye(target_size, dtype=np.complex128) / np.sqrt(target_size)

    except Exception as e:
        logger.error(f"Error creating quantum field: {e}")
        return np.eye(64, dtype=np.complex128) / 8.0

def standardize_dimensions(state: np.ndarray, target_size: int = 64) -> np.ndarray:
    """Standardize state dimensions with performance monitoring."""
    try:
        global PERFORMANCE_METRICS
        PERFORMANCE_METRICS['dimension_changes'] += 1
        start_time = logger.debug(f"Starting dimension standardization: {state.size} -> {target_size}")

        if state.ndim > 1:
            state = state.flatten()
        initial_size = state.size

        # Calculate quantum properties with vectorized operations
        initial_energy = np.sum(np.abs(state)**2)
        initial_norm = np.linalg.norm(state)

        # Early normalization
        state = state / initial_norm

        if initial_size > target_size:
            magnitudes = np.abs(state)
            sorted_indices = np.argsort(magnitudes)[::-1]
            state = state[sorted_indices[:target_size]]
            scale = np.sqrt(target_size / initial_size)
            state = state * scale * initial_norm
        else:
            scale = np.sqrt(target_size / initial_size)
            padding = np.zeros(target_size - initial_size, dtype=state.dtype)
            state = np.concatenate([state * scale * initial_norm, padding])

        # Energy conservation check
        final_energy = np.sum(np.abs(state)**2)
        if not np.isclose(final_energy, initial_energy, rtol=OPTIMIZATION_THRESHOLD):
            PERFORMANCE_METRICS['energy_corrections'] += 1
            state *= np.sqrt(initial_energy / final_energy)
            logger.debug(f"Energy correction applied: {final_energy:.6f} -> {initial_energy:.6f}")

        return state

    except Exception as e:
        logger.error(f"Error in dimension standardization: {e}")
        return np.zeros(target_size, dtype=state.dtype)

def calculate_financial_decoherence(
    market_state: np.ndarray,
    reference_state: np.ndarray
) -> float:
    """Calculates financial decoherence between two quantum states."""
    try:
        # Input validation
        if not isinstance(market_state, np.ndarray) or not isinstance(reference_state, np.ndarray):
            raise TypeError("Both states must be numpy arrays")

        # Convert inputs to proper type and shape
        market_state = np.asarray(market_state, dtype=np.complex128)
        reference_state = np.asarray(reference_state, dtype=np.complex128)

        # Standardize dimensions while preserving energy density
        target_size = 64
        market_state = standardize_dimensions(market_state, target_size)
        reference_state = standardize_dimensions(reference_state, target_size)

        # Reshape to density matrices
        market_state = market_state.reshape(-1, 1)
        reference_state = reference_state.reshape(-1, 1)

        # Calculate density matrices
        rho_market = market_state @ market_state.conj().T
        rho_ref = reference_state @ reference_state.conj().T

        # Calculate trace distance with proper scaling
        diff = rho_market - rho_ref
        eigenvals = np.linalg.eigvals(diff @ diff.conj().T)
        decoherence = np.sqrt(np.abs(np.sum(eigenvals))) / 2.0  # Scale to [0,1]

        return float(np.clip(decoherence, 0, 1))

    except Exception as e:
        logger.error(f"Error calculating financial decoherence: {e}")
        return 0.5

def predict_market_state(
    current_state: Optional[np.ndarray],
    hamiltonian: Optional[np.ndarray],
    dt: float = 0.01,
    steps: int = 10
) -> np.ndarray:
    """Predicts future market state with performance monitoring."""
    try:
        if current_state is None or hamiltonian is None:
            raise TypeError("Both current_state and hamiltonian must be numpy arrays")

        if not isinstance(current_state, np.ndarray) or not isinstance(hamiltonian, np.ndarray):
            raise TypeError("Both current_state and hamiltonian must be numpy arrays")

        try:
            current_state = np.asarray(current_state, dtype=np.complex128)
            hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)
        except Exception as e:
            logger.error(f"Type conversion error: {e}")
            raise TypeError(f"Failed to convert inputs to complex arrays: {e}")

        # Dimension standardization
        target_size = 64
        if current_state.size != target_size:
            logger.debug(f"Standardizing state dimensions: {current_state.size} -> {target_size}")
            current_state = standardize_dimensions(current_state, target_size)

        if hamiltonian.shape != (target_size, target_size):
            logger.debug(f"Adjusting Hamiltonian dimensions: {hamiltonian.shape} -> ({target_size}, {target_size})")
            new_hamiltonian = np.eye(target_size, dtype=np.complex128)
            if hamiltonian.shape[0] < target_size:
                new_hamiltonian[:hamiltonian.shape[0], :hamiltonian.shape[1]] = hamiltonian
            else:
                start = (hamiltonian.shape[0] - target_size) // 2
                new_hamiltonian = hamiltonian[start:start+target_size, start:start+target_size]
            hamiltonian = new_hamiltonian

        current_state = current_state.reshape(-1, 1)

        try:
            U = expm(-1j * hamiltonian * dt)
        except Exception as e:
            logger.error(f"Evolution operator calculation failed: {e}")
            return current_state

        state = current_state.copy()
        initial_energy = np.sum(np.abs(state)**2)

        for step in range(steps):
            try:
                PERFORMANCE_METRICS['evolution_steps'] += 1
                state = U @ state
                current_energy = np.sum(np.abs(state)**2)
                if current_energy > 0:
                    state *= np.sqrt(initial_energy / current_energy)
            except Exception as e:
                logger.error(f"Evolution step {step} failed: {e}")
                return current_state

        final_energy = np.sum(np.abs(state)**2)
        logger.debug(f"Market state prediction completed: {steps} steps, energy delta: {abs(final_energy - initial_energy):.2e}")
        return state

    except TypeError as e:
        logger.error(f"Error predicting market state: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error predicting market state: {str(e)}")
        return np.zeros((64, 1), dtype=np.complex128)

def calculate_phi_resonance(field: np.ndarray, phi: float = 1.618033988749895) -> float:
    """Calculates resonance with golden ratio (phi) with enhanced error handling"""
    try:
        # Ensure field is 2D and normalized
        field = field.reshape(field.shape[0], -1)
        field = normalize_quantum_state(field)

        # Calculate coherence factor
        coherence = np.abs(np.trace(field @ field.conj().T))

        # Calculate resonance with phi
        resonance = np.exp(-np.abs(coherence - phi))

        return float(np.clip(resonance, 0, 1))

    except Exception as e:
        logger.error(f"Error calculating phi resonance: {e}")
        return 0.5

def validate_market_distribution(empirical_data: np.ndarray,
                              theoretical_data: np.ndarray) -> Dict[str, float]:
    """Validates market distribution using enhanced statistical tests.
    Incorporates quantum consciousness effects in validation.
    """
    try:
        # Ensure 1D arrays for statistical tests
        empirical_data = np.ravel(empirical_data)
        theoretical_data = np.ravel(theoretical_data)

        # Normalize data with consciousness weighting
        phi = (1 + np.sqrt(5)) / 2
        empirical_data = (empirical_data - np.mean(empirical_data)) / (np.std(empirical_data) * phi + 1e-10)
        theoretical_data = (theoretical_data - np.mean(theoretical_data)) / (np.std(theoretical_data) * phi + 1e-10)

        # Perform enhanced KS test
        ks_stat, p_value = stats.ks_2samp(empirical_data, theoretical_data)

        # Calculate quantum-consciousness correlation
        correlation = float(np.corrcoef(empirical_data, theoretical_data)[0, 1])

        # Ensure all values are real and JSON serializable using new serialization function
        return {
            'ks_statistic': safe_complex_to_real(ks_stat),
            'p_value': safe_complex_to_real(p_value),
            'quantum_correlation': safe_complex_to_real(correlation),
            'consciousness_alignment': safe_complex_to_real(np.exp(-ks_stat * phi))
        }

    except Exception as e:
        logger.error(f"Error validating market distribution: {e}")
        return {
            'ks_statistic': 0.0,
            'p_value': 1.0,
            'quantum_correlation': 0.0,
            'consciousness_alignment': 0.5
        }

def create_observer_matrix(size: int = 64) -> np.ndarray:
    """Creates a unitary observer matrix using QR decomposition with enhanced dimension handling."""
    try:
        # Ensure size is valid
        size = max(2, size)
        
        # Create random matrix with proper shape management
        random_matrix = (np.random.randn(size, size) + 
                        1j * np.random.randn(size, size))
        
        # Calculate QR decomposition
        q, _ = np.linalg.qr(random_matrix)
        
        # Ensure unitarity
        q = q @ q.conj().T
        
        return q
        
    except Exception as e:
        logger.error(f"Error creating observer matrix: {e}")
        return np.eye(size, dtype=np.complex128)