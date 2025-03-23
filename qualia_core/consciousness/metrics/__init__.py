"""
Consciousness metrics package initialization with restructured dependencies
"""

from quantum.core.qtypes import BaseQuantumMetric, MetricsConfig
from .metrics import (
    ConsciousnessMonitor,
    ConsciousnessMetrics,
    UnifiedConsciousnessMetrics,
    TrendType
)
from .qualia_metrics import QualiaMetrics

# Re-export the calculate_field_metrics function from qualia_metrics
from .qualia_metrics import calculate_field_metrics

__all__ = [
    'BaseQuantumMetric',
    'MetricsConfig',
    'ConsciousnessMonitor',
    'ConsciousnessMetrics',
    'UnifiedConsciousnessMetrics',
    'TrendType',
    'QualiaMetrics',
    'calculate_field_metrics'
]