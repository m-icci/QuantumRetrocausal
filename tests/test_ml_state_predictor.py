"""
Test suite for ML State Predictor functionality
"""
import pytest
import numpy as np
import torch
from datetime import datetime
from unittest.mock import MagicMock, patch
from qualia.ml_state_predictor import (
    MLStatePredictor,
    PredictionMetrics,
    MarketSequenceDataset,
    QuantumTransformer
)

@pytest.fixture
def predictor():
    """Create MLStatePredictor instance for testing"""
    return MLStatePredictor(
        input_dim=64,
        sequence_length=50,
        batch_size=32
    )

@pytest.fixture
def sample_data():
    """Generate sample market data for testing"""
    return np.random.randn(100, 64)  # 100 timesteps, 64 features

def test_predictor_initialization(predictor):
    """Test MLStatePredictor initialization"""
    assert predictor.input_dim == 64
    assert predictor.sequence_length == 50
    assert predictor.batch_size == 32
    assert isinstance(predictor.model, QuantumTransformer)
    assert predictor.device in ['cuda', 'cpu']

def test_prepare_data(predictor, sample_data):
    """Test data preparation functionality"""
    dataloader = predictor.prepare_data(sample_data)
    assert isinstance(dataloader, torch.utils.data.DataLoader)

    # Test batch shape
    batch_sequences, batch_targets = next(iter(dataloader))
    assert batch_sequences.shape[1] == predictor.sequence_length
    assert batch_sequences.shape[2] == predictor.input_dim

def test_train_epoch(predictor, sample_data):
    """Test training functionality"""
    dataloader = predictor.prepare_data(sample_data)
    metrics = predictor.train_epoch(dataloader)

    assert isinstance(metrics, dict)
    assert 'loss' in metrics
    assert 'quantum_alignment' in metrics
    assert 0 <= metrics['quantum_alignment'] <= 1

def test_predict_next_state(predictor, sample_data):
    """Test prediction functionality"""
    sequence = sample_data[:50]  # Use sequence_length samples
    metrics = predictor.predict_next_state(sequence)

    assert isinstance(metrics, PredictionMetrics)
    assert isinstance(metrics.predicted_value, float)
    assert isinstance(metrics.confidence, float)
    assert isinstance(metrics.quantum_alignment, float)
    assert isinstance(metrics.entropy, float)

    # Check value ranges
    assert 0 <= metrics.confidence <= 1
    assert 0 <= metrics.quantum_alignment <= 1
    assert metrics.entropy >= 0

MIN_CONFIDENCE = 0.1
MIN_QUANTUM_ALIGNMENT = 0.1

def test_error_handling(predictor):
    """Test error handling for invalid inputs"""
    # Test sequence too short
    with pytest.raises(ValueError, match="Input sequence length .* is less than required"):
        predictor.predict_next_state(np.random.randn(10, 64))

    # Test wrong feature dimension
    with pytest.raises(ValueError, match="Input features dimension .* does not match model input dimension"):
        predictor.predict_next_state(np.random.randn(50, 32))

    # Test invalid 1D input
    with pytest.raises(ValueError, match="1D input sequence length .* does not match required dimensions"):
        predictor.predict_next_state(np.random.randn(100))  # Wrong total length

    # Test valid 1D input
    sequence = np.random.randn(predictor.sequence_length * predictor.input_dim)  # Correct total length
    result = predictor.predict_next_state(sequence)
    assert isinstance(result, PredictionMetrics)
    assert result.confidence >= MIN_CONFIDENCE
    assert result.quantum_alignment >= MIN_QUANTUM_ALIGNMENT

def test_quantum_transformer():
    """Test quantum transformer model"""
    model = QuantumTransformer(input_dim=64)
    batch_size = 32
    seq_len = 50

    x = torch.randn(batch_size, seq_len, 64)
    prediction, confidence = model(x)

    assert prediction.shape == (batch_size, 64)
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

def test_market_dataset():
    """Test market sequence dataset functionality"""
    data = np.random.randn(100, 64)
    dataset = MarketSequenceDataset(data, sequence_length=50)

    assert len(dataset) == 50  # 100 - sequence_length

    sequence, target = dataset[0]
    assert sequence.shape == (50, 64)
    assert target.shape == (64,)
    assert torch.is_tensor(sequence)
    assert torch.is_tensor(target)