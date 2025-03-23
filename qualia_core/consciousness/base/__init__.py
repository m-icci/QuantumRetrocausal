"""
Base classes and interfaces for quantum consciousness system.

This module provides the foundational abstractions for consciousness implementations:
- Base interfaces for quantum consciousness
- Core configuration types
- Abstract base classes
"""

from .base import (
    IQuantumConsciousness,
    ConsciousnessBase,
    ConsciousnessConfig,
    ConsciousnessField as BaseQuantumConsciousness
)

__all__ = [
    'IQuantumConsciousness',
    'ConsciousnessBase', 
    'ConsciousnessConfig',
    'BaseQuantumConsciousness'
]