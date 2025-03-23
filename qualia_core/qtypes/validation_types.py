"""
Validation Types Module

Defines types used for quantum state validation.
"""

from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ValidationResult:
    """Result of quantum state validation."""
    is_valid: bool
    metrics: Dict[str, float]
    messages: List[str]
