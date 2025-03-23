"""
Quantum pattern implementation for consciousness system.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
from .pattern_types import PatternType

@dataclass
class QuantumPattern:
    """Quantum pattern with type classification and metadata."""
    pattern_type: PatternType
    strength: float
    coherence: float
    data: Optional[np.ndarray] = None
    timestamp: datetime = field(default_factory=datetime.now)
    pattern_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate pattern after initialization."""
        if not 0 <= self.strength <= 1:
            raise ValueError(f"Pattern strength must be between 0 and 1, got {self.strength}")
        if not 0 <= self.coherence <= 1:
            raise ValueError(f"Pattern coherence must be between 0 and 1, got {self.coherence}")

        if self.data is not None and not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data, dtype=np.complex128)

    def get_metrics(self) -> Dict[str, float]:
        """Get pattern metrics."""
        return {
            'strength': float(self.strength),
            'coherence': float(self.coherence)
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary representation"""
        result = {
            'type': self.pattern_type.value,
            'strength': float(self.strength),
            'coherence': float(self.coherence),
            'pattern_id': self.pattern_id,
            'metrics': self.get_metrics(),
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }

        if self.data is not None:
            result['data'] = self.data.tolist()

        return result