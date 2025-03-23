"""
Consciousness field package initialization

This module provides core interfaces for quantum consciousness fields, including:
- Field operators for consciousness manipulation
- Resonance and morphic field implementations
- Integration with M-ICCI framework
"""

from .consciousness_field import ConsciousnessField
from ..base.base import ConsciousnessState
from typing import List, Dict, Optional, Any

# Export core quantum consciousness interfaces
__all__ = [
    'ConsciousnessField',
    'ConsciousnessState'
]

# Version tracking
VERSION = '1.0.0'

# Field configuration constants aligned with sacred geometry
FIELD_CONFIG = {
    'MIN_DIMENSION': 2,
    'DEFAULT_STRENGTH': 1.0,  # Base field strength
    'DEFAULT_RESONANCE': 1.0, # Base resonance level
    'PHI': (1 + 5**0.5) / 2,  # Golden ratio for sacred geometry
    'CONSCIOUSNESS_COUPLING': 0.1  # Default consciousness-field coupling
}