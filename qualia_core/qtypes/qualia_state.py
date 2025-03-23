"""
Qualia State definition for quantum consciousness system.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

@dataclass
class QualiaState:
    """Represents a quantum qualia state in consciousness."""
    intensity: float
    coherence: float
    resonance: float
    field_strength: float
    state_vector: np.ndarray
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate and normalize state."""
        if not 0 <= self.intensity <= 1:
            raise ValueError("Intensity must be between 0 and 1")
        if not 0 <= self.coherence <= 1:
            raise ValueError("Coherence must be between 0 and 1")
        if not 0 <= self.resonance <= 1:
            raise ValueError("Resonance must be between 0 and 1")
        if not 0 <= self.field_strength <= 1:
            raise ValueError("Field strength must be between 0 and 1")
            
        # Ensure state vector is normalized
        norm = np.linalg.norm(self.state_vector)
        if norm > 0:
            self.state_vector = self.state_vector / norm
            
    def merge(self, other: 'QualiaState', weight: float = 0.5) -> 'QualiaState':
        """Merge two qualia states with given weight."""
        if not 0 <= weight <= 1:
            raise ValueError("Weight must be between 0 and 1")
            
        # Compute weighted averages
        intensity = weight * self.intensity + (1 - weight) * other.intensity
        coherence = weight * self.coherence + (1 - weight) * other.coherence
        resonance = weight * self.resonance + (1 - weight) * other.resonance
        field_strength = weight * self.field_strength + (1 - weight) * other.field_strength
        
        # Merge state vectors with phase alignment
        phase_diff = np.angle(np.vdot(self.state_vector, other.state_vector))
        aligned_other = other.state_vector * np.exp(-1j * phase_diff)
        state_vector = weight * self.state_vector + (1 - weight) * aligned_other
        
        # Merge metadata if present
        metadata = None
        if self.metadata or other.metadata:
            metadata = {
                **(self.metadata or {}),
                **(other.metadata or {})
            }
            
        return QualiaState(
            intensity=intensity,
            coherence=coherence,
            resonance=resonance,
            field_strength=field_strength,
            state_vector=state_vector,
            metadata=metadata
        )
        
    def get_phase(self) -> float:
        """Get the global phase of the state vector."""
        return float(np.angle(np.sum(self.state_vector)))
        
    def get_amplitude(self) -> float:
        """Get the total amplitude of the state."""
        return float(np.abs(np.sum(self.state_vector)))
        
    def get_density_matrix(self) -> np.ndarray:
        """Calculate the density matrix of the state."""
        return np.outer(self.state_vector, np.conjugate(self.state_vector))
        
    def get_purity(self) -> float:
        """Calculate the purity of the state."""
        rho = self.get_density_matrix()
        return float(np.trace(np.matmul(rho, rho)).real)
        
    def get_entropy(self) -> float:
        """Calculate the von Neumann entropy of the state."""
        rho = self.get_density_matrix()
        eigenvalues = np.linalg.eigvalsh(rho)
        # Remove very small eigenvalues to avoid log(0)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        return float(-np.sum(eigenvalues * np.log2(eigenvalues)).real)
