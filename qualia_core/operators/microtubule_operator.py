"""
Microtubule Quantum Operator Implementation
Simulates quantum dynamics in neuronal microtubules based on Penrose-Hameroff theory.
"""
from typing import Dict, Optional, Tuple
import numpy as np

class MicrotubuleOperator:
    """
    Implements quantum operations for microtubular processing in consciousness simulation.
    Based on the Orchestrated Objective Reduction (Orch OR) model.
    """

    def __init__(self, dimension: int = 8):
        self.dimension = dimension
        self.decoherence_rate = 0.1
        self.collapse_threshold = 0.5  # Threshold for total probability
        self.coherence_history = []

    def initialize_state(self) -> np.ndarray:
        """Initialize microtubule quantum state"""
        state = np.zeros(self.dimension, dtype=np.complex128)
        state[0] = 1.0  # Start in ground state
        return state

    def apply_decoherence(self, state: np.ndarray, time_step: float) -> np.ndarray:
        """Apply environmental decoherence effects"""
        decoherence_factor = np.exp(-self.decoherence_rate * time_step)
        return state * decoherence_factor

    def check_collapse(self, state: np.ndarray) -> Tuple[bool, np.ndarray]:
        """Check for quantum state collapse based on Orch OR threshold"""
        # Calculate total probability (norm squared)
        total_probability = np.sum(np.abs(state) ** 2)

        # If total probability is below threshold, collapse is needed
        if total_probability < self.collapse_threshold:
            # Find the most probable basis state
            probabilities = np.abs(state) ** 2
            max_index = np.argmax(probabilities)

            # Create collapsed state
            collapsed_state = np.zeros_like(state)
            collapsed_state[max_index] = 1.0

            return True, collapsed_state

        return False, state

    def evolve_state(self, state: np.ndarray, time_step: float) -> Dict[str, any]:
        """
        Evolve the microtubule quantum state according to Orch OR dynamics
        """
        # Apply decoherence
        evolved_state = self.apply_decoherence(state, time_step)

        # Check for collapse
        did_collapse, final_state = self.check_collapse(evolved_state)

        # Calculate coherence
        coherence = float(np.sum(np.abs(final_state) ** 2))
        self.coherence_history.append(coherence)

        return {
            "state": final_state,
            "coherence": coherence,
            "collapse_occurred": did_collapse
        }

    def get_metrics(self) -> Dict[str, float]:
        """Get operator metrics"""
        return {
            "average_coherence": float(np.mean(self.coherence_history)) if self.coherence_history else 0.0,
            "collapse_probability": float(len([c for c in self.coherence_history if c < self.collapse_threshold]) / max(len(self.coherence_history), 1)),
            "current_coherence": float(self.coherence_history[-1]) if self.coherence_history else 0.0
        }