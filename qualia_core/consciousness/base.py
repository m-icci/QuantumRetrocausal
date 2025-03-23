"""
Base classes for consciousness framework
"""
from dataclasses import dataclass
from typing import Optional, Protocol
import numpy as np
from ..QUALIA.base_types import QuantumState

class ConsciousnessBase(Protocol):
    """Base protocol for consciousness implementations"""

    def apply_field(self, state: QuantumState) -> QuantumState:
        """Apply consciousness transformation to quantum state"""
        ...

    def calculate_resonance(self, state: QuantumState) -> float:
        """Calculate resonance between state and consciousness"""
        ...

@dataclass
class ConsciousnessState:
    """State of consciousness"""
    quantum_state: QuantumState
    field: ConsciousnessBase
    metadata: Optional[dict] = None

    def evolve(self) -> 'ConsciousnessState':
        """Evolve consciousness state"""
        new_state = self.field.apply_field(self.quantum_state)
        resonance = self.field.calculate_resonance(new_state)

        return ConsciousnessState(
            quantum_state=new_state,
            field=self.field,
            metadata={
                **(self.metadata or {}),
                'resonance': resonance
            }
        )