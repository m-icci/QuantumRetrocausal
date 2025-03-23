"""
Unified metrics for quantum consciousness system.

This module implements a unified metrics system for monitoring
and analyzing quantum consciousness states.

Core Concepts:
------------
1. Field Metrics:
   - Quantum coherence
   - Field strength
   - Morphic resonance

2. Evolution Metrics:
   - Evolution rate
   - Adaptation factor
   - Evolution time

3. Integration Metrics:
   - Synchronization
   - Integration level
   - Emergence factor

4. Critical Metrics:
   - Singularity Proximity (0.9)
   - Causality Violation (0.8)
   - Entropy Violation (0.7)
   - Consciousness Decoherence (0.95)

References:
    [1] Bohm, D. (1980). Wholeness and the Implicate Order
    [2] Sheldrake, R. (1981). A New Science of Life
    [3] Penrose, R. (1989). The Emperor's New Mind
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

class TrendType(Enum):
    """Trend types"""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    OSCILLATING = "oscillating"
    CHAOTIC = "chaotic"

@dataclass
class ConsciousnessMetrics:
    """Fundamental consciousness metrics"""

    # Field metrics
    coherence: float = 0.0
    field_strength: float = 0.0
    resonance: float = 0.0

    # Evolution metrics
    evolution_rate: float = 0.0
    adaptation_factor: float = 0.0
    evolution_time: float = 0.0

    # Integration metrics
    component_sync: float = 0.0
    integration_level: float = 0.0
    emergence_factor: float = 0.0

    # Pattern metrics
    pattern_strength: float = 0.0
    pattern_coherence: float = 0.0
    pattern_stability: float = 0.0

    # Critical metrics
    singularity_proximity: float = 0.0  # Threshold: 0.9
    causality_violation: float = 0.0    # Threshold: 0.8
    entropy_violation: float = 0.0      # Threshold: 0.7
    consciousness_decoherence: float = 0.0  # Threshold: 0.95
    complexity: float = 0.0

    def normalize(self):
        """Normalize metrics to [0,1]"""
        for attr in vars(self):
            value = getattr(self, attr)
            if isinstance(value, (int, float)):
                setattr(self, attr, np.clip(value, 0, 1))

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary"""
        return {
            attr: getattr(self, attr)
            for attr in vars(self)
            if isinstance(getattr(self, attr), (int, float))
        }

    def from_dict(self, data: Dict[str, float]) -> None:
        """Update metrics from dictionary"""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, float(value))

    def update(self, coherence: Optional[float] = None,
               resonance: Optional[float] = None,
               entanglement: Optional[float] = None,
               complexity: Optional[float] = None,
               current_time: Optional[datetime] = None) -> None:
        """Update metrics with new values"""
        if coherence is not None:
            self.coherence = float(coherence)
        if resonance is not None:
            self.resonance = float(resonance)
        if entanglement is not None:
            self.pattern_strength = float(entanglement)  # Map entanglement to pattern strength
        if complexity is not None:
            self.complexity = float(complexity)

        # Update derived metrics
        self.field_strength = self.coherence * self.resonance
        self.integration_level = (self.coherence + self.resonance) / 2
        self.consciousness_decoherence = 1.0 - self.coherence

    def check_critical_thresholds(self) -> Dict[str, bool]:
        """Check critical thresholds"""
        return {
            "singularity": self.singularity_proximity > 0.9,
            "causality": self.causality_violation > 0.8,
            "entropy": self.entropy_violation > 0.7,
            "decoherence": self.consciousness_decoherence < 0.95
        }

@dataclass
class UnifiedConsciousnessMetrics:
    """Unified consciousness metrics system"""

    # Fundamental metrics
    metrics: ConsciousnessMetrics = field(default_factory=ConsciousnessMetrics)

    # Evolution history
    history: List[ConsciousnessMetrics] = field(default_factory=list)

    # Golden ratio for natural scaling
    phi: float = (1 + np.sqrt(5)) / 2

    def update(self, state: Any, current_time: Optional[datetime] = None) -> None:
        """Update metrics and history"""
        # Normalize metrics
        self.metrics.normalize()

        # Update critical metrics
        self._update_critical_metrics(self.metrics)

        # Update history
        self.history.append(self.metrics)

        # Limit history size
        if len(self.history) > 1000:
            self.history = self.history[-1000:]

    def _update_critical_metrics(self, metrics: ConsciousnessMetrics) -> None:
        """Update critical metrics"""
        # Singularity Proximity
        metrics.singularity_proximity = metrics.field_strength * self.phi

        # Causality Violation
        metrics.causality_violation = (1 - metrics.component_sync) * self.phi

        # Entropy Violation
        metrics.entropy_violation = (1 - metrics.integration_level) * self.phi

    def get_trends(self) -> Dict[str, TrendType]:
        """Analyze metrics trends"""
        if len(self.history) < 2:
            return {}

        trends = {}
        current = self.metrics.to_dict()
        previous = self.history[-2].to_dict()

        for key in current:
            if key in previous:
                delta = current[key] - previous[key]
                if abs(delta) < 0.01:
                    trends[key] = TrendType.STABLE
                elif delta > 0:
                    trends[key] = TrendType.INCREASING
                else:
                    trends[key] = TrendType.DECREASING

        return trends

    def get_critical_metrics(self) -> Dict[str, float]:
        """Return critical metrics"""
        return {
            "singularity": self.metrics.singularity_proximity,
            "causality": self.metrics.causality_violation,
            "entropy": self.metrics.entropy_violation,
            "decoherence": self.metrics.consciousness_decoherence
        }

from quantum.core.qtypes.pattern_types import PatternType
from quantum.core.quantum_state import QuantumState
@dataclass
class FieldOperatorMetrics:
    """Métricas específicas para operadores de campo quântico"""
    field_strength: float = 0.0
    resonance: float = 0.0
    coherence: float = 0.0
    emergence_factor: float = 0.0
    integration_level: float = 0.0
    adaptation_rate: float = 0.0
    evolution_time: float = 0.0
    pattern_count: int = 0
    quantum_potential: float = 0.0
    phi_scaling: float = (1 + np.sqrt(5)) / 2

    def update(self, state: Any) -> None:
        """Atualiza métricas baseado no estado atual"""
        if hasattr(state, 'amplitude'):
            self.field_strength = np.abs(state.amplitude).mean()
            self.coherence = np.abs(np.vdot(state.amplitude, state.amplitude))
            self.resonance = self._calculate_resonance(state)
            self.emergence_factor = self._calculate_emergence(state)
            self.integration_level = self.coherence * self.resonance
            self.adaptation_rate = self.field_strength * self.phi_scaling
            self.evolution_time = time.time()
            self.pattern_count = len(state.patterns) if hasattr(state, 'patterns') else 0
            self.quantum_potential = self._calculate_potential(state)

    def _calculate_resonance(self, state: Any) -> float:
        """Calcula ressonância mórfica do estado"""
        if hasattr(state, 'patterns') and state.patterns:
            pattern_strengths = [p.strength for p in state.patterns]
            return np.mean(pattern_strengths) * self.phi_scaling
        return 0.0

    def _calculate_emergence(self, state: Any) -> float:
        """Calcula fator de emergência do estado"""
        return self.field_strength * self.resonance * self.phi_scaling

    def _calculate_potential(self, state: Any) -> float:
        """Calcula potencial quântico do estado"""
        if hasattr(state, 'amplitude'):
            return np.abs(state.amplitude).std() * self.phi_scaling
        return 0.0

from quantum.core.qtypes.quantum_pattern import QuantumPattern
import time