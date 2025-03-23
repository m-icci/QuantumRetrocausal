"""
Unified Quantum Pattern Implementation
Combines consciousness, QUALIA, and sacred geometry features
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
from .pattern_types import PatternType

@dataclass
class QuantumPattern:
    """
    Unified quantum pattern with enhanced consciousness integration

    Attributes:
        pattern_type: Type classification of the pattern
        strength: Pattern strength (0-1)
        coherence: Quantum coherence measure (0-1)
        data: Quantum state data
        pattern_id: Unique pattern identifier
        timestamp: Pattern creation time
        metadata: Additional pattern metadata
    """
    pattern_type: PatternType
    strength: float
    coherence: float
    data: Optional[np.ndarray] = None
    pattern_id: str = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and initialize pattern"""
        # Validate core metrics
        if not 0 <= self.strength <= 1:
            raise ValueError(f"Pattern strength must be between 0 and 1, got {self.strength}")
        if not 0 <= self.coherence <= 1:
            raise ValueError(f"Pattern coherence must be between 0 and 1, got {self.coherence}")

        # Initialize quantum data
        if self.data is not None and not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data, dtype=np.complex128)

    def calculate_overlap(self, other: 'QuantumPattern') -> complex:
        """Calculate quantum overlap between patterns"""
        if self.data is None or other.data is None:
            return 0
        return np.vdot(self.data, other.data)

    def get_metrics(self) -> Dict[str, float]:
        """Get comprehensive pattern metrics"""
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

    @property
    def is_valid(self) -> bool:
        """Check if pattern is valid"""
        # Check core validity
        if not (0 <= self.strength <= 1 and 0 <= self.coherence <= 1):
            return False

        # Check data validity if present
        if self.data is not None:
            if not isinstance(self.data, np.ndarray):
                return False
            if np.any(np.isnan(self.data)) or np.any(np.isinf(self.data)):
                return False

        return True