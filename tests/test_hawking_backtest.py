"""
Test suite for Hawking Backtest module
Validates quantum parameter optimization and performance analysis
"""
import pytest
import numpy as np
from datetime import datetime, timedelta

from qualia.backtesting.hawking_backtest import (
    HawkingBacktest, BacktestResult, BacktestParameters
)
from qualia.quantum_trader import QuantumTrader, TradeSignal
from qualia.market_data import MarketDataProvider
from qualia.analysis.quantum_analysis import QuantumAnalyzer

@pytest.fixture
def backtest():
    """Fixture for HawkingBacktest instance"""
    trader = QuantumTrader()
    market_data = MarketDataProvider()
    quantum_analyzer = QuantumAnalyzer()
    return HawkingBacktest(trader, market_data, quantum_analyzer)

def test_backtest_parameters():
    """Test backtest parameters initialization and bounds"""
    params = BacktestParameters(
        alpha=0.5,
        beta=0.5,
        gamma=0.5,
        decoherence_threshold=0.6,
        consciousness_threshold=0.7,
        resonance_threshold=0.6
    )

    bounds = BacktestParameters.get_bounds()
    assert all(0 < getattr(params, key) < 1 for key in bounds.keys())
    assert all(len(bound) == 2 for bound in bounds.values())

def test_financial_decoherence():
    """Test financial decoherence metric calculation"""
    backtest = HawkingBacktest(QuantumTrader(), MarketDataProvider())

    # Generate sample returns
    returns = np.random.normal(0, 0.02, 1000)

    # Calculate decoherence
    decoherence = backtest.calculate_financial_decoherence(returns)

    # Verify metric properties
    assert isinstance(decoherence, float)
    assert decoherence >= 0  # Entropy is non-negative
    assert np.isfinite(decoherence)

def test_parameter_optimization():
    """Test quantum parameter optimization"""
    backtest = HawkingBacktest(
        QuantumTrader(),
        MarketDataProvider(),
        optimization_iterations=5  # Reduced for testing
    )

    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()

    # Run optimization
    params = backtest.optimize_parameters(
        symbol='BTC/USDT',
        initial_capital=10000.0,
        start_date=start_date,
        end_date=end_date
    )

    # Verify parameters
    assert isinstance(params, BacktestParameters)
    bounds = BacktestParameters.get_bounds()

    for param_name, bound in bounds.items():
        value = getattr(params, param_name)
        assert bound[0] <= value <= bound[1], f"{param_name} out of bounds"

def test_ks_test():
    """Test Kolmogorov-Smirnov test implementation"""
    backtest = HawkingBacktest(QuantumTrader(), MarketDataProvider())

    # Generate two similar but slightly different distributions
    pred_returns = np.random.normal(0.001, 0.02, 1000)
    actual_returns = np.random.normal(0, 0.02, 1000)

    # Perform KS test
    statistic, pvalue = backtest.perform_ks_test(pred_returns, actual_returns)

    # Verify test properties
    assert isinstance(statistic, float)
    assert isinstance(pvalue, float)
    assert 0 <= statistic <= 1
    assert 0 <= pvalue <= 1

def test_backtest_execution():
    """Test full backtest execution with parameter optimization"""
    backtest = HawkingBacktest(
        QuantumTrader(),
        MarketDataProvider(),
        optimization_iterations=5  # Reduced for testing
    )

    # Run backtest
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()

    result = backtest.run_backtest(
        symbol='BTC/USDT',
        start_date=start_date,
        end_date=end_date,
        initial_capital=10000.0
    )

    # Verify result properties
    assert isinstance(result, BacktestResult)
    assert result.start_date == start_date
    assert result.end_date == end_date
    assert result.total_trades >= 0
    assert 0 <= result.win_rate <= 1
    assert np.isfinite(result.sharpe_ratio)
    assert -1 <= result.max_drawdown <= 0
    assert result.decoherence_metric >= 0
    assert 0 <= result.ks_test_pvalue <= 1

    # Verify quantum metrics and parameters
    assert result.parameters is not None
    assert isinstance(result.parameters, BacktestParameters)
    assert len(result.optimization_history) >= 0

    # Verify return series
    assert len(result.predicted_returns) == len(result.actual_returns)
    assert len(result.equity_curve) > 0
    assert all(isinstance(x, float) for x in result.equity_curve)