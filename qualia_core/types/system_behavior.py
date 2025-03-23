"""
System behavior type definitions and interfaces.
"""
from enum import Enum
from typing import Protocol, Dict, Any

class SystemBehavior(Enum):
    """System behavior types for quantum consciousness."""
    COHERENT = "coherent"
    RESONANT = "resonant"
    ENTANGLED = "entangled"
    DECOHERENT = "decoherent"
    PROTECTED = "protected"
    MEASURED = "measured"

    def to_dict(self) -> Dict[str, Any]:
        """Convert behavior to dictionary representation."""
        return {
            'type': self.name,
            'value': self.value
        }
