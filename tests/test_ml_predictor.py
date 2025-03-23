"""
Test suite for ML-based state predictor implementation with enhanced confidence validation
"""
import pytest
import numpy as np
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from qualia.ml_state_predictor import (
    MLStatePredictor,
    QuantumTransformer,
    MarketSequenceDataset,
    PredictionMetrics
)

@pytest.fixture
def ml_predictor():
    """Fixture for MLStatePredictor instance"""
    return MLStatePredictor(input_dim=4, sequence_length=10)

@pytest.fixture
def sample_market_data():
    """Fixture for sample market data"""
    return np.random.randn(100, 4)  # 100 timesteps, 4 features

def test_confidence_validation(ml_predictor, sample_market_data):
    """Test that confidence values are never zero"""
    # Train the model briefly
    dataloader = ml_predictor.prepare_data(sample_market_data)
    ml_predictor.train_epoch(dataloader)

    # Test multiple predictions
    sequence = sample_market_data[-10:]
    predictions = []
    for _ in range(10):  # Test multiple times for robustness
        metrics = ml_predictor.predict_next_state(sequence)
        predictions.append(metrics)

        # Verify confidence is never zero or extremely close to zero
        assert metrics.confidence > 0.01, "Confidence should never be near zero"
        assert metrics.quantum_alignment > 0.01, "Quantum alignment should never be near zero"

def test_quantum_metrics_validation(ml_predictor):
    """Test quantum metrics are properly calculated and validated"""
    quantum_metrics = ml_predictor.quantum_manager.calculate_consciousness_metrics()

    # Verify all metrics are reasonable values
    required_metrics = ['coherence', 'consciousness', 'morphic_resonance', 
                       'integration_index', 'entropy']

    for metric in required_metrics:
        value = quantum_metrics.get(metric, 0)
        assert 0.01 <= value <= 1.0, f"{metric} should be between 0.01 and 1.0"

def test_prediction_confidence_integration(ml_predictor, sample_market_data):
    """Test integration of ML and quantum confidence scores"""
    sequence = sample_market_data[-10:]
    metrics = ml_predictor.predict_next_state(sequence)

    # Verify confidence calculation uses both ML and quantum metrics
    assert hasattr(metrics, 'confidence'), "Metrics should include confidence"
    assert hasattr(metrics, 'quantum_alignment'), "Metrics should include quantum alignment"

    # Test confidence bounds
    assert 0.01 <= metrics.confidence <= 1.0, "Confidence should be between 0.01 and 1.0"
    assert 0.01 <= metrics.quantum_alignment <= 1.0, "Quantum alignment should be between 0.01 and 1.0"

def test_dataset_creation(sample_market_data):
    """Test market sequence dataset creation"""
    dataset = MarketSequenceDataset(sample_market_data, sequence_length=10)
    assert len(dataset) == len(sample_market_data) - 10

    # Test single item
    sequence, target = dataset[0]
    assert sequence.shape == (10, 4)
    assert target.shape == (4,)
    assert torch.is_tensor(sequence)
    assert torch.is_tensor(target)

def test_transformer_model():
    """Test quantum transformer model"""
    model = QuantumTransformer(input_dim=4)
    batch_size = 32
    seq_len = 10

    # Test forward pass with correct input shape (batch_first=True)
    x = torch.randn(batch_size, seq_len, 4)  # Changed shape order
    prediction, confidence = model(x)

    assert prediction.shape == (batch_size, 4)
    assert confidence.shape == (batch_size, 1)
    assert torch.all(confidence >= 0) and torch.all(confidence <= 1)

def test_prediction_metrics():
    """Test prediction metrics calculation"""
    metrics = PredictionMetrics(
        predicted_value=1.0,
        confidence=0.8,
        quantum_alignment=0.7,
        entropy=0.5
    )

    assert metrics.predicted_value == 1.0
    assert metrics.confidence == 0.8
    assert metrics.quantum_alignment == 0.7
    assert metrics.entropy == 0.5
    assert isinstance(metrics.timestamp, datetime)