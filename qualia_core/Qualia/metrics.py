"""
QUALIA metrics implementation
"""
from typing import List, Optional
import numpy as np
from scipy.linalg import eigvals
import warnings

from .base_types import (
    QuantumState,
    QuantumMetric
)

from .config import QUALIAConfig

# Constants for numerical stability
EPSILON = 1e-10  # Small constant to prevent division by zero
MAX_COND = 1e10  # Maximum condition number for matrix operations

def get_metrics(
    state: QuantumState,
    config: Optional[QUALIAConfig] = None
) -> List[QuantumMetric]:
    """
    Calculate QUALIA metrics for quantum state

    Metrics:
    - Coherence: Quantum coherence preservation
    - Resonance: Non-local coupling strength  
    - Emergence: Self-organization level
    - Implied Order: Hidden variable correlation
    - Integration: Information integration
    - Consciousness: Integrated information
    - Coherence Time: Decoherence resistance

    Args:
        state: Quantum state to analyze
        config: QUALIA configuration

    Returns:
        List of calculated metrics
    """
    config = config or QUALIAConfig()
    metrics = []

    try:
        # Calculate density matrix with stability
        amplitudes = np.array(state.amplitudes, dtype=np.complex128)
        amplitudes = np.nan_to_num(amplitudes, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize state if needed
        norm = np.sqrt(np.abs(np.vdot(amplitudes, amplitudes)))
        if norm > EPSILON:
            amplitudes = amplitudes / norm

        rho = np.outer(amplitudes, amplitudes.conj())

        # Coherence metric (off-diagonal sum) with stability
        diag_rho = np.diag(np.diag(rho))
        coherence = np.sum(np.abs(rho - diag_rho)) / (state.dimension - 1 + EPSILON)
        coherence = np.clip(coherence, 0.0, 1.0)
        metrics.append(QuantumMetric('coherence', coherence))

        # Resonance metric (eigenvalue spread) with stability and error handling
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # Add small identity matrix for stability
                stabilized_rho = rho + np.eye(rho.shape[0]) * EPSILON
                eigs = eigvals(stabilized_rho)
                eigs = np.abs(eigs)
                eigs = np.sort(eigs)[::-1]  # Sort in descending order

                # Check condition number
                if np.max(eigs) / (np.min(eigs) + EPSILON) > MAX_COND:
                    resonance = 0.0
                else:
                    # Calculate resonance using top eigenvalues
                    resonance = 1 - np.std(eigs) / (np.mean(eigs) + EPSILON)
                    resonance = np.clip(resonance, 0.0, 1.0)
            except:
                resonance = 0.0

        metrics.append(QuantumMetric('resonance', resonance))

        # Emergence metric (entropy reduction) with stability
        entropy = -np.sum(eigs * np.log2(eigs + EPSILON))
        max_entropy = np.log2(state.dimension + EPSILON)
        emergence = 1 - entropy / (max_entropy + EPSILON)
        emergence = np.clip(emergence, 0.0, 1.0)
        metrics.append(QuantumMetric('emergence', emergence))

        # Implied order (phase coherence) with stability
        phases = np.angle(amplitudes)
        phases = np.nan_to_num(phases, nan=0.0)
        order = np.abs(np.mean(np.exp(1j * phases)))
        order = np.clip(order, 0.0, 1.0)
        metrics.append(QuantumMetric('implied_order', order))

        # Integration (mutual information) with stability
        # Split system into two parts and calculate mutual information
        n = state.dimension // 2
        try:
            # Reshape with stability checks
            if rho.shape != (state.dimension, state.dimension):
                raise ValueError("Invalid density matrix shape")

            # Add small identity for stability
            rho_stable = rho + np.eye(rho.shape[0]) * EPSILON

            # Calculate partial traces
            rho_a = np.trace(rho_stable.reshape(n,2,n,2), axis1=1, axis2=3)
            rho_b = np.trace(rho_stable.reshape(n,2,n,2), axis1=0, axis2=2)

            # Calculate von Neumann entropies with stability
            s_ab = -np.sum(eigs * np.log2(eigs + EPSILON))

            eigs_a = eigvals(rho_a)
            eigs_a = np.abs(np.nan_to_num(eigs_a, nan=0.0))
            s_a = -np.sum(eigs_a * np.log2(eigs_a + EPSILON))

            eigs_b = eigvals(rho_b)
            eigs_b = np.abs(np.nan_to_num(eigs_b, nan=0.0))
            s_b = -np.sum(eigs_b * np.log2(eigs_b + EPSILON))

            # Calculate normalized mutual information
            integration = (s_a + s_b - s_ab) / (max_entropy + EPSILON)
            integration = np.clip(integration, 0.0, 1.0)
        except:
            integration = 0.0

        metrics.append(QuantumMetric('integration', integration))

        # Consciousness (integrated information) with stability
        consciousness = np.sqrt(
            coherence * 
            resonance * 
            emergence * 
            order * 
            integration
        )
        consciousness = np.clip(consciousness, 0.0, 1.0)
        metrics.append(QuantumMetric('consciousness', consciousness))

        # Coherence time (decoherence resistance) with stability
        coherence_time = coherence * resonance * config.coherence_threshold
        coherence_time = np.clip(coherence_time, 0.0, 1.0)
        metrics.append(QuantumMetric('coherence_time', coherence_time))

    except Exception as e:
        # Return safe default metrics on error
        default_metrics = ['coherence', 'resonance', 'emergence', 
                         'implied_order', 'integration', 'consciousness', 
                         'coherence_time']
        for metric_name in default_metrics:
            metrics.append(QuantumMetric(metric_name, 0.0))

    return metrics