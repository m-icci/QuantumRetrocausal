"""
Trading-specific quantum types.
This module defines the core types used in quantum trading operations.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

@dataclass
class MarketQuantumState:
    """Quantum state with market-specific attributes."""
    vector: np.ndarray
    symbol: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate and normalize state vector"""
        if self.metadata is None:
            self.metadata = {}
        # Ensure vector is normalized
        if not isinstance(self.vector, np.ndarray):
            self.vector = np.array(self.vector, dtype=np.complex128)
        norm = np.linalg.norm(self.vector)
        if norm > 0:
            self.vector = self.vector / norm

@dataclass
class TradingPattern:
    """Trading pattern identified in market data."""
    pattern_type: str  # e.g., 'bullish', 'bearish'
    confidence: float  # Pattern confidence score
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class MarketHolographicMemory:
    """Market-specific holographic memory."""

    def __init__(self, dimension: int = 1024):
        """Initialize market holographic memory."""
        self.dimension = dimension
        self.patterns: List[TradingPattern] = []

    def find_similar_patterns(
        self,
        state: MarketQuantumState,
        threshold: float = 0.7
    ) -> List[Tuple[int, TradingPattern, float]]:
        """
        Find trading patterns similar to current state.

        Args:
            state: Current market quantum state
            threshold: Similarity threshold

        Returns:
            List of (pattern_id, pattern, similarity) tuples
        """
        similar_patterns = []
        for i, pattern in enumerate(self.patterns):
            # Calculate pattern similarity using quantum state overlap
            similarity = np.abs(np.vdot(state.vector, self.get_pattern_state(pattern)))
            if similarity >= threshold:
                similar_patterns.append((i, pattern, similarity))

        return similar_patterns

    def get_pattern_state(self, pattern: TradingPattern) -> np.ndarray:
        """Convert trading pattern to quantum state vector."""
        # Basic implementation - should be enhanced based on pattern type
        state = np.zeros(self.dimension, dtype=complex)
        if pattern.pattern_type == 'bullish':
            # Encode bullish pattern
            state[0] = np.cos(pattern.confidence * np.pi/2)
            state[1] = np.sin(pattern.confidence * np.pi/2)
        else:
            # Encode bearish pattern
            state[0] = np.cos(pattern.confidence * np.pi/2)
            state[1] = -np.sin(pattern.confidence * np.pi/2)

        return state

    def store_pattern(self, pattern: TradingPattern):
        """Store trading pattern in memory."""
        self.patterns.append(pattern)