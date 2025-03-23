"""
Test script for the Holographic Market Memory system
"""
import os
import sys
import numpy as np
from datetime import datetime

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from quantum.core.QUALIA.trading.memory.holographic_market_memory import AdvancedHolographicMarketMemory

def generate_test_pattern(size: int = 10, trend: str = 'up') -> np.ndarray:
    """Generate test market pattern with specified trend."""
    base = np.random.random((size, size))
    if trend == 'up':
        # Add upward trend component
        trend_component = np.linspace(0, 1, size)
        base += trend_component[:, np.newaxis]
    elif trend == 'down':
        # Add downward trend component
        trend_component = np.linspace(1, 0, size)
        base += trend_component[:, np.newaxis]
    return base / np.max(base)  # Normalize

def test_holographic_memory():
    """Test the holographic memory implementation"""
    print("Starting holographic memory test...")

    # Initialize memory system
    memory = AdvancedHolographicMarketMemory(capacity=1000)
    print("Memory system initialized successfully")

    # Test pattern recognition with different trends
    up_trend = generate_test_pattern(trend='up')
    down_trend = generate_test_pattern(trend='down')
    similar_up = generate_test_pattern(trend='up')  # Should be similar to up_trend

    print("\nStoring test patterns...")
    memory.store_pattern(up_trend)
    memory.store_pattern(down_trend)

    print("\nTesting pattern recognition...")
    # Test similarity with an upward trend pattern
    similar = memory.retrieve_similar_pattern(similar_up)
    signal = memory.generate_trading_signal(similar_up)

    # Print results
    print("\nTest Results:")
    print("Similar pattern found:", similar is not None)
    print("Trading signal:", signal)
    print("Memory metrics:", memory.get_memory_metrics())

    # Test wavelet-based similarity
    if similar is not None:
        similarity = memory._compute_similarity(similar_up, up_trend)
        print(f"Similarity score with up trend: {similarity:.4f}")
        similarity = memory._compute_similarity(similar_up, down_trend)
        print(f"Similarity score with down trend: {similarity:.4f}")

if __name__ == "__main__":
    test_holographic_memory()