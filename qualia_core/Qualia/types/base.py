"""
Base quantum types for QUALIA framework
"""
from dataclasses import dataclass
from typing import Any, Optional
import numpy as np

@dataclass
class QuantumState:
    """Base quantum state representation"""
    vector: Any  # Changed from np.ndarray to Any for initialization flexibility
    metadata: Optional[dict] = None

    def __post_init__(self):
        """Validate and normalize state vector"""
        # Convert to numpy array if not already
        if not isinstance(self.vector, np.ndarray):
            try:
                self.vector = np.array(self.vector, dtype=np.complex128)
            except Exception as e:
                raise ValueError(f"Could not convert vector to numpy array: {e}")

        # Ensure normalization
        norm = np.linalg.norm(self.vector)
        if norm > 0:
            self.vector = self.vector / norm

        if self.metadata is None:
            self.metadata = {}