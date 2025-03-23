"""
Test suite for enhanced Monte Carlo simulation with stress testing
"""
import pytest
import numpy as np
from qualia.monte_carlo import QuantumMonteCarlo, MarketScenario

@pytest.fixture
def monte_carlo():
    """Fixture for QuantumMonteCarlo instance"""
    return QuantumMonteCarlo(n_simulations=100, time_steps=50)

def test_market_scenario_initialization():
    """Test market scenario initialization"""
    scenario = MarketScenario(
        name='test_scenario',
        volatility_range=(0.1, 0.2),
        drift_range=(0.05, 0.15),
        quantum_coherence=0.7,
        description="Test scenario"
    )
    
    assert scenario.name == 'test_scenario'
    assert scenario.volatility_range == (0.1, 0.2)
    assert scenario.drift_range == (0.05, 0.15)
    assert scenario.quantum_coherence == 0.7
    assert scenario.description == "Test scenario"

def test_path_generation(monte_carlo):
    """Test quantum path generation"""
    initial_price = 100.0
    volatility = 0.2
    drift = 0.1
    
    paths = monte_carlo.generate_quantum_paths(
        initial_price=initial_price,
        volatility=volatility,
        drift=drift
    )
    
    assert paths.shape == (monte_carlo.n_simulations, monte_carlo.time_steps)
    assert np.all(paths[:, 0] == initial_price)
    assert np.all(np.isfinite(paths))

def test_stress_testing(monte_carlo):
    """Test stress testing across scenarios"""
    initial_price = 100.0
    initial_investment = 10000.0
    market_data = np.array([100.0] * 100)  # Dummy market data
    
    # Run stress tests
    results = monte_carlo.run_stress_test(
        initial_price=initial_price,
        initial_investment=initial_investment,
        market_data=market_data
    )
    
    # Verify results
    assert isinstance(results, dict)
    assert len(results) > 0
    
    # Check standard scenarios
    standard_scenarios = ['bull_market', 'bear_market', 'sideways', 'crisis']
    for scenario in standard_scenarios:
        assert scenario in results
        scenario_result = results[scenario]
        assert 'risk_metrics' in scenario_result
        assert 'description' in scenario_result
        assert 'quantum_coherence' in scenario_result
        assert 'worst_case_return' in scenario_result
        assert 'best_case_return' in scenario_result
        
def test_risk_metrics_calculation(monte_carlo):
    """Test risk metrics calculation"""
    # Generate sample paths
    paths = np.random.randn(monte_carlo.n_simulations, monte_carlo.time_steps)
    paths = np.exp(paths)  # Convert to price paths
    initial_investment = 10000.0
    
    metrics = monte_carlo.calculate_risk_metrics(paths, initial_investment)
    
    # Verify metrics
    required_metrics = [
        'value_at_risk',
        'expected_shortfall',
        'mean_return',
        'volatility',
        'skewness',
        'kurtosis',
        'sharpe_ratio',
        'sortino_ratio'
    ]
    
    for metric in required_metrics:
        assert metric in metrics
        assert isinstance(metrics[metric], float)
        assert np.isfinite(metrics[metric])

def test_custom_scenario_integration(monte_carlo):
    """Test integration of custom scenarios"""
    custom_scenarios = {
        'custom_scenario': MarketScenario(
            name='custom_scenario',
            volatility_range=(0.3, 0.4),
            drift_range=(-0.1, 0.1),
            quantum_coherence=0.5,
            description="Custom test scenario"
        )
    }
    
    results = monte_carlo.run_stress_test(
        initial_price=100.0,
        initial_investment=10000.0,
        market_data=np.array([100.0] * 100),
        custom_scenarios=custom_scenarios
    )
    
    assert 'custom_scenario' in results
    custom_result = results['custom_scenario']
    assert custom_result['description'] == "Custom test scenario"
    assert isinstance(custom_result['risk_metrics'], dict)
