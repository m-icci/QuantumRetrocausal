"""
Test suite for Quantum Bayesian Forecasting operator
"""

import os
import sys
# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))

import numpy as np
from datetime import datetime
import pytest

from quantum.core.types import MarketData
from quantum.core.QUALIA.trading.operators.bayesian_forecasting import QuantumBayesianForecaster

def generate_test_market_data(trend: str = 'up') -> MarketData:
    """Generate test market data with specified trend"""
    base_price = 100.0
    if trend == 'up':
        price = base_price * (1 + 0.01 * np.random.randn() + 0.005)
    else:
        price = base_price * (1 + 0.01 * np.random.randn() - 0.005)

    return MarketData(
        symbol="BTC-USD",
        price=price,
        volume=100.0 * np.random.rand(),
        timestamp=datetime.now(),
        high=price * 1.01,
        low=price * 0.99,
        open=base_price,
        close=price
    )

def test_bayesian_forecaster():
    """Test the quantum Bayesian forecasting system"""
    print("\nStarting Quantum Bayesian Forecaster test...")

    # Initialize forecaster
    forecaster = QuantumBayesianForecaster()
    print("Forecaster initialized successfully")

    # Generate test data sequence
    up_trend_data = [generate_test_market_data('up') for _ in range(5)]
    down_trend_data = [generate_test_market_data('down') for _ in range(5)]

    print("\nTesting forecasting with upward trend...")
    # Test upward trend prediction
    states = []
    confidences = []
    for data in up_trend_data:
        next_state, confidence = forecaster.predict_next_state(data)
        states.append(next_state)
        confidences.append(confidence)

    print(f"Average confidence (up trend): {np.mean(confidences):.4f}")

    print("\nTesting forecasting with downward trend...")
    # Reset forecaster
    forecaster = QuantumBayesianForecaster()

    # Test downward trend prediction
    states = []
    confidences = []
    for data in down_trend_data:
        next_state, confidence = forecaster.predict_next_state(data)
        states.append(next_state)
        confidences.append(confidence)

    print(f"Average confidence (down trend): {np.mean(confidences):.4f}")

    # Test prediction stability
    state_differences = [np.linalg.norm(states[i] - states[i-1]) 
                        for i in range(1, len(states))]
    print(f"\nAverage state stability: {np.mean(state_differences):.4f}")

    # Verify quantum properties
    for state in states:
        # Check hermiticity
        assert np.allclose(state, state.T.conj()), "State matrix is not Hermitian"
        # Check trace preservation
        assert np.abs(np.trace(state) - 1.0) < 1e-6, "Trace is not preserved"

    print("\nQuantum properties verified successfully")

if __name__ == "__main__":
    test_bayesian_forecaster()