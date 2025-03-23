"""
Base types for quantum system with QUALIA alignment
Provides core type definitions with enhanced consciousness integration
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

@dataclass
class QuantumState:
    """QUALIA-enhanced quantum state representation"""
    amplitudes: np.ndarray
    dimension: int
    device: str = 'cpu'
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        # Initialize quantum state
        if not isinstance(self.amplitudes, np.ndarray):
            self.amplitudes = np.array(self.amplitudes, dtype=np.complex128)

        # Validate dimension
        if len(self.amplitudes) != self.dimension:
            raise ValueError(f"Amplitude vector length {len(self.amplitudes)} != dimension {self.dimension}")

        # Initialize metadata
        if self.metadata is None:
            self.metadata = {}

        # Normalize state vector
        norm = np.linalg.norm(self.amplitudes)
        if norm > 0:
            self.amplitudes = self.amplitudes / norm

    @property
    def vector(self):
        """Get the state vector"""
        return self.amplitudes

    def evolve(self) -> 'QuantumState':
        """Evolve quantum state one time step"""
        # Simple time evolution - apply phase rotation
        phase = np.exp(1j * np.pi / 4)  # 45-degree rotation
        evolved_amplitudes = phase * self.amplitudes

        return QuantumState(
            amplitudes=evolved_amplitudes,
            dimension=self.dimension,
            device=self.device,
            metadata=self.metadata
        )

    @property 
    def coherence(self) -> float:
        """Calculate quantum coherence"""
        return float(np.abs(np.vdot(self.amplitudes, self.amplitudes)))

    @property
    def phi_resonance(self) -> float:
        """Calculate φ (golden ratio) resonance"""
        phi = (1 + np.sqrt(5)) / 2
        phases = np.angle(self.amplitudes)
        return float(np.abs(np.mean(np.exp(1j * phi * phases))))

    @property 
    def is_valid(self) -> bool:
        """Check if state is valid"""
        if not isinstance(self.amplitudes, np.ndarray):
            return False
        if len(self.amplitudes) != self.dimension:
            return False
        if np.any(np.isnan(self.amplitudes)) or np.any(np.isinf(self.amplitudes)):
            return False
        norm = np.linalg.norm(self.amplitudes)
        return abs(norm - 1.0) < 1e-10

@dataclass 
class QuantumPattern:
    """QUALIA-enhanced pattern representation"""
    pattern_id: str
    state: QuantumState
    metadata: Optional[Dict[str, Any]] = None

    @property
    def dimension(self) -> int:
        return self.state.dimension

    @property
    def coherence(self) -> float:
        """Calculate pattern coherence"""
        return self.state.coherence

    @property
    def resonance(self) -> float:
        """Calculate pattern resonance"""
        return self.state.phi_resonance

@dataclass
class QuantumMetric:
    """QUALIA-enhanced metric representation"""
    name: str
    value: float
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not 0 <= self.value <= 1:
            raise ValueError(f"{self.name} must be between 0 and 1")

@dataclass
class QuantumOperator:
    """QUALIA-enhanced operator representation"""
    matrix: np.ndarray
    dimension: int

    def __post_init__(self):
        if not isinstance(self.matrix, np.ndarray):
            self.matrix = np.array(self.matrix, dtype=np.complex128)

    def apply(self, state: QuantumState) -> QuantumState:
        """Apply operator to quantum state"""
        if state.dimension != self.dimension:
            raise ValueError(f"State dimension {state.dimension} != operator dimension {self.dimension}")

        new_amplitudes = np.dot(self.matrix, state.amplitudes)

        return QuantumState(
            amplitudes=new_amplitudes,
            dimension=self.dimension,
            device=state.device,
            metadata=state.metadata
        )