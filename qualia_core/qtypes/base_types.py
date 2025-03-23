"""
Base type definitions for quantum system
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

@dataclass
class BaseQuantumMetric:
    """Base class for quantum metrics"""
    name: str
    value: float
    metadata: Optional[Dict[str, Any]] = None
    sacred_factor: float = 1.0
    consciousness_weight: float = 1.0

    @property
    def weighted_value(self) -> float:
        """Get consciousness-weighted metric value"""
        return self.value * self.sacred_factor * self.consciousness_weight

    def __post_init__(self):
        if not 0 <= self.value <= 1:
            raise ValueError(f"{self.name} must be between 0 and 1")