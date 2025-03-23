"""
Quantum utility functions for QUALIA Trading System.
Implements core quantum calculations and metrics.
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

def calculate_quantum_coherence(state: np.ndarray) -> float:
    """
    Calculate quantum coherence from state vector or density matrix.
    Uses l1-norm of coherence measure.

    Args:
        state: Quantum state as vector or density matrix

    Returns:
        float: Coherence measure in [0,1] range
    """
    try:
        # Convert to density matrix if needed
        if state.ndim == 1:  # State vector
            state = state.reshape(-1, 1)
            density_matrix = state @ state.conj().T
        else:  # Already a density matrix
            density_matrix = state

        # Ensure proper normalization using vectorized operations
        trace = np.abs(np.trace(density_matrix))
        if trace > 0:
            density_matrix = density_matrix / trace

        # Calculate l1-norm coherence using vectorized operations
        diag_part = np.diag(np.diag(density_matrix))
        coherence = np.sum(np.abs(density_matrix - diag_part))

        # Normalize to [0,1] range
        dimension = density_matrix.shape[0]
        max_coherence = dimension * (dimension - 1) / 2
        normalized_coherence = np.clip(coherence / max_coherence if max_coherence > 0 else 0, 0, 1)

        return float(normalized_coherence)

    except Exception as e:
        logging.error(f"Error calculating quantum coherence: {e}")
        return 0.5  # Safe default

def create_quantum_field(size: int = 64) -> np.ndarray:
    """
    Creates an initial quantum field with proper normalization.

    Args:
        size: Dimension of the quantum field

    Returns:
        Normalized complex quantum field array
    """
    try:
        # Use float32 for large matrices to improve performance
        dtype = np.complex64 if size > 128 else np.complex128

        # Create complex field with random phase factors using vectorized operations
        field = (np.random.randn(size, size).astype(dtype) + 
                1j * np.random.randn(size, size).astype(dtype))

        # Normalize the field using optimized operations
        norm = np.sqrt(np.sum(np.abs(field)**2))
        if norm > 0:
            field = field / norm

        return field

    except Exception as e:
        logging.error(f"Error creating quantum field: {e}")
        # Return identity matrix as safe fallback
        return np.eye(size, dtype=dtype) / np.sqrt(size)

def calculate_dark_finance_metrics(state: np.ndarray, phi: float = 1.618033988749895) -> Dict[str, float]:
    """
    Calculate dark finance metrics based on M-ICCI principles.

    Args:
        state: Quantum state vector or density matrix
        phi: Golden ratio (default φ ≈ 1.618033988749895)

    Returns:
        Dictionary with dark finance metrics
    """
    try:
        # Use appropriate precision based on matrix size
        dtype = np.complex64 if state.size > 128 else np.complex128
        state = state.astype(dtype)

        if state.ndim == 1:
            state = state.reshape(-1, 1)
            density = state @ state.conj().T
        else:
            density = state

        # Calculate dark ratio using phi resonance with vectorized operations
        dimension = density.shape[0]
        i, j = np.meshgrid(np.arange(dimension), np.arange(dimension), sparse=True)
        dark_phases = np.exp(1j * phi * np.pi * (i * j) / dimension, dtype=dtype)

        # Calculate dark energy density using optimized matrix operations
        dark_energy = np.abs(np.trace(dark_phases @ density))
        dark_ratio = float(np.clip(dark_energy / dimension, 0, 1))

        # Calculate dark coupling strength using vectorized operations
        coupling = phi**(-np.abs(i-j)/dimension)
        phase = np.pi * (i + j) / (phi * dimension)
        coupling_matrix = coupling * np.exp(1j * phase)

        dark_coupling = np.abs(np.trace(coupling_matrix @ density))
        dark_coupling = float(np.clip(dark_coupling / dimension, 0, 1))

        # Calculate dark coherence using optimized method
        dark_coherence = calculate_quantum_coherence(dark_phases @ density @ dark_phases.conj().T)

        return {
            'dark_ratio': dark_ratio,
            'dark_coupling': dark_coupling,
            'dark_coherence': dark_coherence,
            'dark_phi_resonance': abs(1 - (dark_ratio - 1/phi)**2)
        }

    except Exception as e:
        logging.error(f"Error calculating dark finance metrics: {e}")
        return {
            'dark_ratio': 0.5,
            'dark_coupling': 0.5,
            'dark_coherence': 0.5,
            'dark_phi_resonance': 0.618
        }

def calculate_field_energy(field: np.ndarray) -> float:
    """Calculate normalized energy of quantum field."""
    try:
        # Calculate field energy using vectorized operations and appropriate precision
        dtype = np.float32 if field.size > 128 else np.float64
        field = field.astype(dtype)

        energy = np.sum(np.abs(field)**2)

        # Normalize by dimension
        dimension = np.prod(field.shape)
        normalized_energy = np.clip(energy / dimension if dimension > 0 else 0, 0, 1)

        return float(normalized_energy)

    except Exception as e:
        logging.error(f"Error calculating field energy: {e}")
        return 0.5  # Safe default

def calculate_morphic_resonance(field: np.ndarray, phi: float = 1.618033988749895) -> float:
    """Calculate morphic resonance metric based on golden ratio."""
    try:
        # Calculate base resonance using optimized method
        base_resonance = calculate_field_energy(field)

        # Modulate with golden ratio using optimized computation
        resonance = np.abs(1 - (base_resonance - 1/phi)**2)

        return float(np.clip(resonance, 0, 1))

    except Exception as e:
        logging.error(f"Error calculating morphic resonance: {e}")
        return 0.5  # Safe default

def calculate_entropy(state: np.ndarray, epsilon: float = 1e-10) -> float:
    """Calculate von Neumann entropy for quantum state."""
    try:
        if state.ndim == 1:  # State vector
            state = state.reshape(-1, 1)
            density_matrix = state @ state.conj().T
        else:
            density_matrix = state

        # Ensure proper normalization
        trace = np.abs(np.trace(density_matrix))
        if trace > 0:
            density_matrix = density_matrix / trace

        # Calculate eigenvalues using optimized method
        eigenvalues = np.linalg.eigvalsh(density_matrix)
        eigenvalues = eigenvalues[eigenvalues > epsilon]

        if len(eigenvalues) == 0:
            return 0.5

        # Calculate von Neumann entropy using vectorized operations
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues + epsilon))

        # Normalize by maximum possible entropy
        max_entropy = np.log2(density_matrix.shape[0])
        normalized_entropy = np.clip(entropy / max_entropy if max_entropy > 0 else 0, 0, 1)

        return float(normalized_entropy)

    except Exception:
        return 0.5  # Safe default

def calculate_quantum_fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    """Calculate quantum fidelity between two quantum states."""
    try:
        # Convert to density matrices if needed
        if state1.ndim == 1:
            state1 = state1.reshape(-1, 1)
            rho1 = state1 @ state1.conj().T
        else:
            rho1 = state1

        if state2.ndim == 1:
            state2 = state2.reshape(-1, 1)
            rho2 = state2 @ state2.conj().T
        else:
            rho2 = state2

        # Ensure proper normalization
        trace1 = np.abs(np.trace(rho1))
        trace2 = np.abs(np.trace(rho2))
        if trace1 > 0:
            rho1 = rho1 / trace1
        if trace2 > 0:
            rho2 = rho2 / trace2

        # Calculate using optimized matrix operations
        sqrt_rho1 = np.linalg.sqrtm(rho1)

        # Calculate fidelity using optimized computation
        product = sqrt_rho1 @ rho2 @ sqrt_rho1
        fidelity = np.abs(np.trace(np.linalg.sqrtm(product)))

        return float(np.clip(fidelity, 0, 1))

    except Exception:
        return 0.5  # Safe default