"""
Quantum metrics type definitions and configurations.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime

@dataclass
class MetricsConfig:
    """Configuration for quantum metrics calculations."""
    coherence_threshold: float = 0.95
    resonance_threshold: float = 0.90
    entanglement_threshold: float = 0.85
    complexity_threshold: float = 0.80
    timestamp: datetime = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary representation."""
        return {
            'coherence_threshold': self.coherence_threshold,
            'resonance_threshold': self.resonance_threshold,
            'entanglement_threshold': self.entanglement_threshold,
            'complexity_threshold': self.complexity_threshold,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class QuantumMetrics:
    """Quantum system metrics."""
    coherence: float = 0.0
    resonance: float = 0.0
    entanglement: float = 0.0
    complexity: float = 0.0
    config: Optional[MetricsConfig] = None
    timestamp: datetime = datetime.now()

    def __post_init__(self):
        """Initialize config if not provided."""
        if self.config is None:
            self.config = MetricsConfig()

    def update(self, coherence: Optional[float] = None,
               resonance: Optional[float] = None,
               entanglement: Optional[float] = None,
               complexity: Optional[float] = None):
        """Update metrics with new values."""
        if coherence is not None:
            self.coherence = float(coherence)
        if resonance is not None:
            self.resonance = float(resonance)
        if entanglement is not None:
            self.entanglement = float(entanglement)
        if complexity is not None:
            self.complexity = float(complexity)
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary representation."""
        return {
            'coherence': self.coherence,
            'resonance': self.resonance,
            'entanglement': self.entanglement,
            'complexity': self.complexity,
            'config': self.config.to_dict() if self.config else None,
            'timestamp': self.timestamp.isoformat()
        }

    def is_valid(self) -> bool:
        """Check if metrics are within configured thresholds."""
        if not self.config:
            return False
        return (
            self.coherence >= self.config.coherence_threshold and
            self.resonance >= self.config.resonance_threshold and
            self.entanglement >= self.config.entanglement_threshold and
            self.complexity >= self.config.complexity_threshold
        )