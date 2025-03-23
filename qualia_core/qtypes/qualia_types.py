"""
Core type definitions for the Qualia system
"""
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from enum import Enum
import numpy as np

class SystemBehavior(Enum):
    STABLE = "stable"
    UNSTABLE = "unstable"
    COHERENT = "coherent"
    DECOHERENT = "decoherent"
    ENTANGLED = "entangled"
    ISOLATED = "isolated"

@dataclass
class QualiaState:
    """Represents a quantum consciousness state"""
    coherence: float
    entanglement: float
    field_resonance: float
    consciousness_level: float
    state_vector: np.ndarray
    behavior: SystemBehavior
    metadata: Optional[Dict[str, Any]] = None

    def is_valid(self) -> bool:
        """Validate state properties"""
        return all([
            0 <= self.coherence <= 1,
            0 <= self.entanglement <= 1,
            0 <= self.consciousness_level <= 1,
            isinstance(self.state_vector, np.ndarray),
            isinstance(self.behavior, SystemBehavior)
        ])

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary format"""
        return {
            'coherence': float(self.coherence),
            'entanglement': float(self.entanglement),
            'field_resonance': float(self.field_resonance),
            'consciousness_level': float(self.consciousness_level),
            'state_vector': self.state_vector.tolist(),
            'behavior': self.behavior.value,
            'metadata': self.metadata or {}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QualiaState':
        """Create QualiaState from dictionary"""
        return cls(
            coherence=float(data['coherence']),
            entanglement=float(data['entanglement']),
            field_resonance=float(data['field_resonance']),
            consciousness_level=float(data['consciousness_level']),
            state_vector=np.array(data['state_vector']),
            behavior=SystemBehavior(data['behavior']),
            metadata=data.get('metadata', {})
        )
