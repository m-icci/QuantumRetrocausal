"""
Tests for the quantum pattern analyzer component.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from quantum_trading.analysis.quantum_pattern_analyzer import QuantumPatternAnalyzer
from quantum_trading.data.data_loader import DataLoader

@pytest.fixture
def config():
    """Test configuration fixture."""
    return {
        'quantum': {
            'entanglement_threshold': 0.7,
            'coherence_window': 100,
            'decoherence_rate': 0.1,
            'quantum_parameters': {
                'n_qubits': 20,
                'circuit_depth': 5,
                'measurement_basis': 'computational'
            }
        },
        'pattern_analysis': {
            'min_pattern_length': 10,
            'max_pattern_length': 100,
            'significance_level': 0.05,
            'correlation_threshold': 0.8
        }
    }

@pytest.fixture
def mock_data_loader():
    """Mock DataLoader fixture."""
    loader = AsyncMock(spec=DataLoader)
    
    # Mock historical data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='1min')
    loader.get_historical_data.return_value = pd.DataFrame({
        'timestamp': dates,
        'price': [50000.0 + np.sin(i/1000)*1000 + np.random.normal(0, 100) for i in range(len(dates))],
        'volume': [1000.0 + np.random.random()*100 for _ in range(len(dates))]
    })
    
    return loader

@pytest.mark.asyncio
async def test_analyzer_initialization(config):
    """Test quantum pattern analyzer initialization."""
    analyzer = QuantumPatternAnalyzer(config)
    assert analyzer.config == config
    assert analyzer.entanglement_threshold == config['quantum']['entanglement_threshold']
    assert analyzer.coherence_window == config['quantum']['coherence_window']
    assert analyzer.decoherence_rate == config['quantum']['decoherence_rate']

@pytest.mark.asyncio
async def test_quantum_state_preparation(config, mock_data_loader):
    """Test quantum state preparation from market data."""
    analyzer = QuantumPatternAnalyzer(config)
    
    # Get sample data
    data = await mock_data_loader.get_historical_data()
    
    # Prepare quantum state
    quantum_state = analyzer.prepare_quantum_state(data)
    assert isinstance(quantum_state, np.ndarray)
    assert len(quantum_state.shape) == 2  # Complex amplitudes matrix
    assert quantum_state.shape[1] == 2**config['quantum']['quantum_parameters']['n_qubits']

@pytest.mark.asyncio
async def test_entanglement_detection(config):
    """Test entanglement detection between market variables."""
    analyzer = QuantumPatternAnalyzer(config)
    
    # Generate sample data
    n_samples = 1000
    prices = np.random.normal(50000, 1000, n_samples)
    volumes = prices * 0.8 + np.random.normal(0, 100, n_samples)  # Correlated with prices
    data = pd.DataFrame({
        'price': prices,
        'volume': volumes
    })
    
    # Calculate entanglement
    entanglement = analyzer.calculate_entanglement(data)
    assert isinstance(entanglement, float)
    assert 0 <= entanglement <= 1

@pytest.mark.asyncio
async def test_coherence_calculation(config):
    """Test quantum coherence calculation."""
    analyzer = QuantumPatternAnalyzer(config)
    
    # Generate sample quantum state
    n_qubits = config['quantum']['quantum_parameters']['n_qubits']
    state = np.random.complex128(np.random.rand(2**n_qubits) + 1j * np.random.rand(2**n_qubits))
    state = state / np.linalg.norm(state)
    
    # Calculate coherence
    coherence = analyzer.calculate_coherence(state)
    assert isinstance(coherence, float)
    assert 0 <= coherence <= 1

@pytest.mark.asyncio
async def test_decoherence_analysis(config):
    """Test decoherence analysis over time."""
    analyzer = QuantumPatternAnalyzer(config)
    
    # Generate time series of quantum states
    n_qubits = config['quantum']['quantum_parameters']['n_qubits']
    n_timesteps = 100
    states = []
    for _ in range(n_timesteps):
        state = np.random.complex128(np.random.rand(2**n_qubits) + 1j * np.random.rand(2**n_qubits))
        state = state / np.linalg.norm(state)
        states.append(state)
    
    # Calculate decoherence
    decoherence = analyzer.analyze_decoherence(states)
    assert isinstance(decoherence, float)
    assert 0 <= decoherence <= 1

@pytest.mark.asyncio
async def test_pattern_detection(config, mock_data_loader):
    """Test quantum pattern detection."""
    analyzer = QuantumPatternAnalyzer(config)
    
    # Get sample data
    data = await mock_data_loader.get_historical_data()
    
    # Detect patterns
    patterns = analyzer.detect_patterns(data)
    assert isinstance(patterns, list)
    for pattern in patterns:
        assert isinstance(pattern, dict)
        assert 'start_index' in pattern
        assert 'end_index' in pattern
        assert 'confidence' in pattern
        assert 'type' in pattern

@pytest.mark.asyncio
async def test_quantum_prediction(config, mock_data_loader):
    """Test quantum-based prediction."""
    analyzer = QuantumPatternAnalyzer(config)
    
    # Get sample data
    data = await mock_data_loader.get_historical_data()
    
    # Make prediction
    prediction = analyzer.predict_next_state(data)
    assert isinstance(prediction, dict)
    assert 'price_direction' in prediction
    assert 'confidence' in prediction
    assert 'timeframe' in prediction
    assert 0 <= prediction['confidence'] <= 1

@pytest.mark.asyncio
async def test_quantum_optimization(config):
    """Test quantum circuit optimization."""
    analyzer = QuantumPatternAnalyzer(config)
    
    # Generate sample parameters
    params = {
        'theta': np.random.rand(config['quantum']['quantum_parameters']['circuit_depth']),
        'phi': np.random.rand(config['quantum']['quantum_parameters']['circuit_depth'])
    }
    
    # Optimize quantum circuit
    result = analyzer.optimize_quantum_circuit(params)
    assert isinstance(result, dict)
    assert 'optimized_params' in result
    assert 'cost_value' in result
    assert result['cost_value'] >= 0

@pytest.mark.asyncio
async def test_error_handling(config):
    """Test error handling."""
    analyzer = QuantumPatternAnalyzer(config)
    
    # Test with invalid quantum state
    with pytest.raises(ValueError):
        analyzer.calculate_coherence(np.array([1, 2, 3]))  # Invalid quantum state
    
    # Test with invalid configuration
    invalid_config = config.copy()
    invalid_config['quantum']['quantum_parameters']['n_qubits'] = -1
    with pytest.raises(ValueError):
        QuantumPatternAnalyzer(invalid_config)

@pytest.mark.asyncio
async def test_quantum_metrics(config, mock_data_loader):
    """Test quantum metrics calculation."""
    analyzer = QuantumPatternAnalyzer(config)
    
    # Get sample data
    data = await mock_data_loader.get_historical_data()
    
    # Calculate quantum metrics
    metrics = analyzer.calculate_quantum_metrics(data)
    assert isinstance(metrics, dict)
    assert 'entanglement' in metrics
    assert 'coherence' in metrics
    assert 'decoherence' in metrics
    assert all(0 <= v <= 1 for v in metrics.values())

@pytest.mark.asyncio
async def test_pattern_significance(config, mock_data_loader):
    """Test pattern significance testing."""
    analyzer = QuantumPatternAnalyzer(config)
    
    # Get sample data
    data = await mock_data_loader.get_historical_data()
    
    # Test pattern significance
    patterns = analyzer.detect_patterns(data)
    for pattern in patterns:
        significance = analyzer.test_pattern_significance(pattern, data)
        assert isinstance(significance, dict)
        assert 'p_value' in significance
        assert 'is_significant' in significance
        assert isinstance(significance['is_significant'], bool)

if __name__ == '__main__':
    pytest.main(['-v', __file__]) 