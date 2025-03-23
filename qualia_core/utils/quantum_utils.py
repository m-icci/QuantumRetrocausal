"""
Quantum Utils Module
------------------
Utilities for quantum operations and morphic fields.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class QuantumDevice:
    """Represents a quantum device."""
    dimensions: int
    coherence_time: float  # Coherence time in seconds
    phi: float = (1 + np.sqrt(5)) / 2  # Golden ratio
    retrocausal_window: int = 20  # Window for retrocausal analysis

    def __post_init__(self):
        """Initialize internal state."""
        self.state = np.zeros((self.dimensions,), dtype=np.complex128)
        self.last_update = datetime.now()
        self.state_history: List[np.ndarray] = []

    def update_state(self, new_state: np.ndarray):
        """Update state and maintain history."""
        self.state = new_state
        self.state_history.append(new_state.copy())
        if len(self.state_history) > self.retrocausal_window:
            self.state_history.pop(0)
        self.last_update = datetime.now()

def create_quantum_device(config: Optional[Dict] = None) -> QuantumDevice:
    """
    Create an optimized quantum device.

    Args:
        config: Optional configuration

    Returns:
        Configured quantum device
    """
    config = config or {}
    return QuantumDevice(
        dimensions=config.get('dimensions', 8),
        coherence_time=config.get('coherence_time', 1.0),
        retrocausal_window=config.get('retrocausal_window', 20)
    )

def apply_phi_scaling(data: np.ndarray, phi: float = (1 + np.sqrt(5)) / 2) -> np.ndarray:
    """
    Apply φ-adaptive scaling.

    Args:
        data: Data to scale
        phi: Golden ratio (default)

    Returns:
        Scaled data
    """
    epsilon = 1e-10  # Numerical stability threshold

    # Get maximum absolute value with stability check
    max_abs = np.max(np.abs(data))
    if max_abs > epsilon:
        # Normalize
        norm_data = data / max_abs

        # Apply φ scaling
        scaled = norm_data * np.exp(1j * phi * np.angle(norm_data))

        return scaled
    else:
        return np.zeros_like(data)

def compute_field_metrics(field: np.ndarray) -> Dict[str, float]:
    """
    Calculate quantum field metrics.

    Args:
        field: Field for analysis

    Returns:
        Dictionary with metrics
    """
    epsilon = 1e-10  # Numerical stability threshold

    # Calculate basic metrics with stability checks
    field_abs = np.abs(field)
    field_sum = np.sum(field_abs)
    if field_sum > epsilon:
        coherence = float(np.abs(np.mean(field)))
    else:
        coherence = 0.0

    # Calculate energy with stability
    energy = float(np.sum(field_abs**2))

    # Calculate entropy with stability
    prob = field_abs**2
    prob_sum = np.sum(prob)
    if prob_sum > epsilon:
        prob = prob / prob_sum
        entropy = float(-np.sum(prob * np.log(prob + epsilon)))
    else:
        entropy = 0.0

    # Calculate φ-adaptive metrics
    phi = (1 + np.sqrt(5)) / 2
    phi_phase = np.exp(-1j * phi * np.angle(field))
    phi_resonance = float(np.abs(np.mean(field * phi_phase)))

    # Calculate morphic field metrics with stability
    field_strength = np.sqrt(np.mean(field_abs**2))
    if field_strength > epsilon:
        morphic_coherence = float(np.abs(np.sum(field * np.conj(field)))) / (len(field) * field_strength**2)
    else:
        morphic_coherence = 0.0

    return {
        'coherence': coherence,
        'energy': energy,
        'entropy': entropy,
        'phi_resonance': phi_resonance,
        'field_strength': float(field_strength),
        'morphic_coherence': morphic_coherence
    }

def validate_quantum_state(state: np.ndarray, tolerance: float = 1e-6) -> bool:
    """
    Validate quantum state.

    Args:
        state: State to validate
        tolerance: Tolerance for unit norm

    Returns:
        True if state is valid
    """
    # Check unit norm
    norm = np.linalg.norm(state)
    if not np.isclose(norm, 1.0, atol=tolerance):
        return False

    # Check finite values
    if not np.all(np.isfinite(state)):
        return False

    return True

def create_phi_mask(size: int, phi: float = (1 + np.sqrt(5)) / 2) -> np.ndarray:
    """
    Create φ-adaptive mask.

    Args:
        size: Mask size
        phi: Golden ratio (default)

    Returns:
        φ-adaptive mask
    """
    angles = np.linspace(0, 2*np.pi, size)
    mask = np.exp(1j * phi * angles)
    return mask

def compute_retrocausal_factor(
    past_states: List[np.ndarray],
    future_state: np.ndarray,
    phi: float = (1 + np.sqrt(5)) / 2
) -> Tuple[float, List[float]]:
    """
    Calcula fator retrocausal e correlações φ-adaptativas.

    Args:
        past_states: Estados passados
        future_state: Estado futuro
        phi: Razão áurea

    Returns:
        Fator retrocausal (0-1) e lista de correlações
    """
    if not past_states:
        return 0.0, []

    # Calcula correlações com pesos φ-adaptativos
    correlations = []
    weights = []
    for i, past in enumerate(past_states):
        # Correlação base
        corr = np.abs(np.corrcoef(
            np.abs(past).flatten(),
            np.abs(future_state).flatten()
        )[0,1])

        # Peso φ-adaptativo
        weight = phi**(-i)  # Decaimento exponencial baseado em φ

        correlations.append(corr)
        weights.append(weight)

    # Normaliza pesos
    weights = np.array(weights)
    weights /= np.sum(weights)

    # Calcula fator retrocausal ponderado
    retrocausal_factor = float(np.sum(np.array(correlations) * weights))

    return retrocausal_factor, correlations

def compute_fibonacci_levels(
    high: float,
    low: float,
    retrocausal_factor: float
) -> Dict[str, float]:
    """
    Calcula níveis de Fibonacci com ajuste retrocausal.

    Args:
        high: Preço mais alto
        low: Preço mais baixo
        retrocausal_factor: Fator de retrocausalidade (0-1)

    Returns:
        Dicionário com níveis ajustados
    """
    # Níveis de Fibonacci clássicos
    levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]

    # Ajusta níveis com fator retrocausal
    range_price = high - low
    adjusted_levels = {}

    for level in levels:
        # Ajuste retrocausal baseado em φ
        phi = (1 + np.sqrt(5)) / 2
        adjusted = level + (retrocausal_factor * (1/phi) * (1 - level))

        # Calcula preço
        price = low + (range_price * adjusted)
        adjusted_levels[f'level_{level:.3f}'] = float(price)

    return adjusted_levels