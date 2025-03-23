"""
QUALIA Trading System - Utils Module
Implements quantum-aware utility functions and helpers aligned with M-ICCI model
"""
from .logging import (
    setup_logger,
    log_quantum_state,
    log_market_event,
    setup_performance_logging,
    log_performance_metric
)

from .quantum_utils import (
    calculate_quantum_coherence,
    calculate_entropy,
    calculate_quantum_fidelity,
    calculate_field_energy,
    calculate_morphic_resonance,
    calculate_dark_finance_metrics
)

from .quantum_field import (
    calculate_phi_resonance,
    predict_market_state,
    validate_market_distribution
)

from ..analysis.quantum_analysis import QuantumAnalyzer

# Importações específicas de métodos
calculate_field_entropy = QuantumAnalyzer.calculate_field_entropy

# Placeholder para calculate_integration_index
def calculate_integration_index(quantum_field):
    """
    Placeholder function for integration index calculation.
    This should be replaced with the actual implementation from ai_insights.
    """
    return 0.5  # Default value

__all__ = [
    'setup_logger',
    'log_quantum_state',
    'log_market_event',
    'setup_performance_logging',
    'log_performance_metric',
    'calculate_quantum_coherence',
    'calculate_entropy',
    'calculate_quantum_fidelity',
    'calculate_field_energy',
    'calculate_morphic_resonance',
    'calculate_phi_resonance',
    'calculate_dark_finance_metrics',
    'predict_market_state',
    'validate_market_distribution',
    'calculate_field_entropy',
    'calculate_integration_index'
]