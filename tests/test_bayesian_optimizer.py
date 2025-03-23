"""
Tests for the Bayesian optimization component.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from quantum_trading.optimization.bayesian_optimizer import BayesianOptimizer
from quantum_trading.data.data_loader import DataLoader

@pytest.fixture
def config():
    """Test configuration fixture."""
    return {
        'optimization': {
            'method': 'bayesian',
            'max_iterations': 100,
            'parameters': {
                'min_profit': [0.0003, 0.001],
                'max_loss': [0.0002, 0.0005],
                'position_time': [60, 600]
            }
        },
        'quantum': {
            'optimization': {
                'quantum_annealing': True,
                'qbit_count': 20,
                'annealing_cycles': 1000
            }
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

@pytest.fixture
def objective_function():
    """Test objective function fixture."""
    def objective(params):
        # Simulate a simple objective function
        x = params['min_profit']
        y = params['max_loss']
        z = params['position_time']
        return -(x**2 + y**2 + (z-300)**2/90000) + 1.0
    return objective

@pytest.mark.asyncio
async def test_optimizer_initialization(config):
    """Test Bayesian optimizer initialization."""
    optimizer = BayesianOptimizer(config)
    assert optimizer.config == config
    assert optimizer.max_iterations == config['optimization']['max_iterations']
    assert optimizer.parameter_ranges == config['optimization']['parameters']

@pytest.mark.asyncio
async def test_parameter_space_validation(config):
    """Test parameter space validation."""
    optimizer = BayesianOptimizer(config)
    
    # Test valid parameter ranges
    valid = optimizer.validate_parameter_ranges(config['optimization']['parameters'])
    assert valid is True
    
    # Test invalid parameter ranges
    invalid_ranges = {
        'param1': [1.0],  # Missing upper bound
        'param2': [2.0, 1.0]  # Lower bound > upper bound
    }
    with pytest.raises(ValueError):
        optimizer.validate_parameter_ranges(invalid_ranges)

@pytest.mark.asyncio
async def test_optimization_run(config, objective_function):
    """Test optimization run."""
    optimizer = BayesianOptimizer(config)
    
    # Run optimization
    result = await optimizer.optimize(objective_function)
    
    assert isinstance(result, dict)
    assert 'best_params' in result
    assert 'best_score' in result
    assert 'optimization_history' in result
    assert len(result['optimization_history']) <= config['optimization']['max_iterations']
    
    # Verify best parameters are within bounds
    for param, value in result['best_params'].items():
        bounds = config['optimization']['parameters'][param]
        assert bounds[0] <= value <= bounds[1]

@pytest.mark.asyncio
async def test_quantum_enhanced_optimization(config, objective_function):
    """Test quantum-enhanced optimization."""
    optimizer = BayesianOptimizer(config)
    
    # Run quantum-enhanced optimization
    result = await optimizer.optimize_quantum(objective_function)
    
    assert isinstance(result, dict)
    assert 'best_params' in result
    assert 'best_score' in result
    assert 'quantum_metrics' in result
    assert 'annealing_schedule' in result['quantum_metrics']

@pytest.mark.asyncio
async def test_acquisition_function(config):
    """Test acquisition function calculation."""
    optimizer = BayesianOptimizer(config)
    
    # Generate some sample data
    X = np.random.uniform(0, 1, (10, 3))
    y = np.random.uniform(0, 1, 10)
    
    # Calculate acquisition values
    values = optimizer.calculate_acquisition(X, y)
    assert isinstance(values, np.ndarray)
    assert len(values) == len(X)
    assert np.all(np.isfinite(values))

@pytest.mark.asyncio
async def test_surrogate_model(config):
    """Test surrogate model fitting and prediction."""
    optimizer = BayesianOptimizer(config)
    
    # Generate some sample data
    X = np.random.uniform(0, 1, (10, 3))
    y = np.random.uniform(0, 1, 10)
    
    # Fit model
    optimizer.fit_surrogate_model(X, y)
    
    # Make predictions
    X_new = np.random.uniform(0, 1, (5, 3))
    predictions = optimizer.predict_surrogate_model(X_new)
    assert isinstance(predictions, tuple)
    assert len(predictions) == 2  # mean and std
    assert len(predictions[0]) == len(X_new)

@pytest.mark.asyncio
async def test_convergence_checking(config, objective_function):
    """Test convergence checking."""
    optimizer = BayesianOptimizer(config)
    
    # Run optimization with convergence checking
    result = await optimizer.optimize(
        objective_function,
        convergence_threshold=1e-4,
        patience=5
    )
    
    assert 'convergence_achieved' in result
    assert isinstance(result['convergence_achieved'], bool)
    if result['convergence_achieved']:
        assert len(result['optimization_history']) < config['optimization']['max_iterations']

@pytest.mark.asyncio
async def test_parallel_optimization(config, objective_function):
    """Test parallel optimization."""
    optimizer = BayesianOptimizer(config)
    
    # Run parallel optimization
    result = await optimizer.optimize_parallel(
        objective_function,
        n_parallel=4
    )
    
    assert isinstance(result, dict)
    assert 'best_params' in result
    assert 'parallel_results' in result
    assert len(result['parallel_results']) == 4

@pytest.mark.asyncio
async def test_parameter_importance(config, objective_function):
    """Test parameter importance analysis."""
    optimizer = BayesianOptimizer(config)
    
    # Run optimization
    result = await optimizer.optimize(objective_function)
    
    # Calculate parameter importance
    importance = optimizer.calculate_parameter_importance()
    assert isinstance(importance, dict)
    assert all(param in importance for param in config['optimization']['parameters'])
    assert all(0 <= v <= 1 for v in importance.values())

@pytest.mark.asyncio
async def test_error_handling(config):
    """Test error handling."""
    optimizer = BayesianOptimizer(config)
    
    # Test with invalid objective function
    async def invalid_objective(params):
        raise ValueError("Test error")
    
    with pytest.raises(ValueError):
        await optimizer.optimize(invalid_objective)
    
    # Test with invalid parameter types
    invalid_config = config.copy()
    invalid_config['optimization']['parameters']['invalid'] = 'not_a_list'
    with pytest.raises(ValueError):
        BayesianOptimizer(invalid_config)

if __name__ == '__main__':
    pytest.main(['-v', __file__])
