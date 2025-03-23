"""
Quantum state implementation with proper type definitions and validation
"""
import numpy as np
from typing import Dict, Optional, Union, List, Tuple
from dataclasses import dataclass, field

@dataclass
class QuantumPattern:
    """
    Represents a quantum pattern with associated metadata
    Used for pattern recognition and matching in quantum states
    """
    data: np.ndarray
    label: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate pattern data"""
        if not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data, dtype=np.complex128)
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")

    def normalize(self) -> 'QuantumPattern':
        """Normalize pattern data"""
        norm = np.linalg.norm(self.data)
        if norm > 0:
            self.data = self.data / norm
        return self

@dataclass
class QuantumState:
    """
    Represents a quantum state with proper dimension handling and validation
    """
    vector: np.ndarray  # Estado quântico como vetor
    dimension: Optional[int] = None
    validate: bool = True
    auto_normalize: bool = False
    _normalized: bool = field(default=False, init=False, repr=False)
    metadata: Dict[str, any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize and validate quantum state"""
        if not isinstance(self.vector, np.ndarray):
            self.vector = np.array(self.vector, dtype=np.complex128)

        if self.dimension is None:
            self.dimension = self.vector.size
        elif self.dimension != self.vector.size:
            raise ValueError(f"Dimension mismatch: {self.dimension} vs {self.vector.size}")

        if self.validate:
            self._validate_state()

    def _validate_state(self):
        """Validate quantum state properties"""
        if not isinstance(self.vector, np.ndarray):
            raise TypeError("State vector must be a numpy.ndarray")

        if self.vector.dtype not in [np.complex64, np.complex128]:
            raise TypeError("State vector must be complex")

        if not np.all(np.isfinite(self.vector)):
            raise ValueError("State vector contains non-finite values")

        norm = np.linalg.norm(self.vector)
        if abs(norm) < 1e-10:
            raise ValueError("State vector cannot have zero norm")

        if not np.isclose(norm, 1.0, atol=1e-6):
            if self.auto_normalize:
                self.vector = self.vector / norm
                self._normalized = True
            else:
                raise ValueError(f"State vector not normalized (norm: {norm})")

    def get_probabilities(self) -> Dict[str, float]:
        """Calculate measurement probabilities in computational basis"""
        probs = np.abs(self.vector) ** 2
        return {f"|{i}⟩": float(p) for i, p in enumerate(probs)}

    def evolve(self, operator: np.ndarray) -> 'QuantumState':
        """Apply quantum operator to state"""
        if operator.shape != (self.dimension, self.dimension):
            raise ValueError(f"Operator dimension {operator.shape} doesn't match state dimension {self.dimension}")
        new_vector = operator @ self.vector
        return QuantumState(new_vector, validate=self.validate, auto_normalize=self.auto_normalize)

    def calculate_overlap(self, other: 'QuantumState') -> complex:
        """Calculate quantum state overlap"""
        if self.dimension != other.dimension:
            raise ValueError("States must have same dimension")
        return np.vdot(self.vector, other.vector)

    def normalize(self) -> None:
        """Normalize state vector"""
        norm = np.linalg.norm(self.vector)
        if norm > 0:
            self.vector = self.vector / norm
            self._normalized = True

    def __str__(self):
        """String representation"""
        return f"QuantumState(dim={self.dimension}, normalized={self._normalized})"