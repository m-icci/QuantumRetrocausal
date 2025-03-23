"""
Testes para módulo de backtesting
============================

Testes unitários para o sistema de backtesting do QUALIA.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from quantum_trading.backtesting import (
    BacktestEngine,
    StrategyTester,
    PortfolioSimulator,
    RiskAnalyzer
)
from quantum_trading.exceptions import BacktestError

@pytest.fixture
def sample_data():
    """Fixture com dados de exemplo para backtesting."""
    np.random.seed(42)
    dates = pd.date_range(
        start="2023-01-01",
        end="2023-12-31",
        freq="1min"
    )
    
    data = pd.DataFrame({
        "timestamp": dates,
        "open": np.cumsum(np.random.normal(0, 0.1, len(dates))) + 100,
        "high": np.cumsum(np.random.normal(0, 0.1, len(dates))) + 101,
        "low": np.cumsum(np.random.normal(0, 0.1, len(dates))) + 99,
        "close": np.cumsum(np.random.normal(0, 0.1, len(dates))) + 100,
        "volume": np.random.exponential(1, len(dates)) * 10
    })
    
    return data

@pytest.fixture
def strategy_config():
    """Fixture com configuração de estratégia."""
    return {
        "entry": {
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "volume_threshold": 1.5
        },
        "exit": {
            "take_profit": 0.02,
            "stop_loss": 0.01,
            "trailing_stop": 0.005
        },
        "risk": {
            "position_size": 0.1,
            "max_positions": 3,
            "max_drawdown": 0.2
        }
    }

def test_backtest_engine(sample_data, strategy_config):
    """Testa motor de backtesting."""
    engine = BacktestEngine(
        initial_capital=10000,
        commission=0.001
    )
    
    # Executa backtest
    results = engine.run(
        data=sample_data,
        strategy_config=strategy_config
    )
    
    # Verifica resultados
    assert "trades" in results
    assert "metrics" in results
    assert "equity_curve" in results
    assert len(results["trades"]) > 0
    assert results["metrics"]["total_trades"] > 0
    assert len(results["equity_curve"]) == len(sample_data)

def test_strategy_testing(sample_data, strategy_config):
    """Testa avaliação de estratégia."""
    tester = StrategyTester(
        strategy_config=strategy_config,
        initial_capital=10000
    )
    
    # Executa testes
    results = tester.evaluate(
        data=sample_data,
        metrics=["sharpe_ratio", "max_drawdown", "win_rate"]
    )
    
    # Verifica resultados
    assert "sharpe_ratio" in results
    assert "max_drawdown" in results
    assert "win_rate" in results
    assert -1 <= results["sharpe_ratio"] <= 5
    assert 0 <= results["max_drawdown"] <= 1
    assert 0 <= results["win_rate"] <= 1

def test_portfolio_simulation(sample_data):
    """Testa simulação de portfólio."""
    simulator = PortfolioSimulator(
        initial_capital=100000,
        risk_free_rate=0.03
    )
    
    # Define alocações
    allocations = {
        "BTC/USDT": 0.4,
        "ETH/USDT": 0.3,
        "SOL/USDT": 0.3
    }
    
    # Executa simulação
    results = simulator.simulate(
        data=sample_data,
        allocations=allocations,
        rebalance_freq="1d"
    )
    
    # Verifica resultados
    assert "portfolio_value" in results
    assert "weights_history" in results
    assert "rebalance_events" in results
    assert len(results["portfolio_value"]) == len(sample_data)
    assert len(results["weights_history"]) > 0

def test_risk_analysis(sample_data, strategy_config):
    """Testa análise de risco."""
    analyzer = RiskAnalyzer()
    
    # Executa backtest primeiro
    engine = BacktestEngine(initial_capital=10000)
    backtest_results = engine.run(
        data=sample_data,
        strategy_config=strategy_config
    )
    
    # Analisa riscos
    risk_metrics = analyzer.analyze(
        trades=backtest_results["trades"],
        equity_curve=backtest_results["equity_curve"]
    )
    
    # Verifica métricas
    assert "var_95" in risk_metrics
    assert "cvar_95" in risk_metrics
    assert "beta" in risk_metrics
    assert "correlation" in risk_metrics
    assert risk_metrics["var_95"] > 0
    assert risk_metrics["cvar_95"] >= risk_metrics["var_95"]

def test_backtest_validation(sample_data, strategy_config):
    """Testa validação de parâmetros de backtest."""
    engine = BacktestEngine(initial_capital=10000)
    
    # Testa dados inválidos
    with pytest.raises(BacktestError):
        engine.run(
            data=None,
            strategy_config=strategy_config
        )
    
    # Testa configuração inválida
    invalid_config = strategy_config.copy()
    del invalid_config["risk"]
    with pytest.raises(BacktestError):
        engine.run(
            data=sample_data,
            strategy_config=invalid_config
        )

def test_walk_forward_analysis(sample_data, strategy_config):
    """Testa análise walk-forward."""
    tester = StrategyTester(
        strategy_config=strategy_config,
        initial_capital=10000
    )
    
    # Executa análise walk-forward
    results = tester.walk_forward_analysis(
        data=sample_data,
        train_size=0.6,
        test_size=0.4,
        window_size="30D"
    )
    
    # Verifica resultados
    assert "windows" in results
    assert "performance" in results
    assert "optimization_results" in results
    assert len(results["windows"]) > 0
    assert all(w["train_score"] > 0 for w in results["windows"])
    assert all(w["test_score"] > 0 for w in results["windows"])

def test_monte_carlo_simulation(sample_data, strategy_config):
    """Testa simulação Monte Carlo."""
    simulator = PortfolioSimulator(
        initial_capital=100000,
        risk_free_rate=0.03
    )
    
    # Executa simulação
    results = simulator.monte_carlo(
        data=sample_data,
        strategy_config=strategy_config,
        num_simulations=100
    )
    
    # Verifica resultados
    assert "simulations" in results
    assert "confidence_intervals" in results
    assert "risk_metrics" in results
    assert len(results["simulations"]) == 100
    assert "95%" in results["confidence_intervals"]

def test_backtest_reporting(sample_data, strategy_config, tmp_path):
    """Testa geração de relatórios de backtest."""
    engine = BacktestEngine(initial_capital=10000)
    
    # Executa backtest
    results = engine.run(
        data=sample_data,
        strategy_config=strategy_config
    )
    
    # Gera relatório
    report_path = tmp_path / "backtest_report.html"
    engine.generate_report(
        results=results,
        output_file=report_path
    )
    
    # Verifica relatório
    assert report_path.exists()
    with open(report_path) as f:
        content = f.read()
        assert "Trade Analysis" in content
        assert "Risk Metrics" in content
        assert "Performance Summary" in content

def test_optimization_integration(sample_data, strategy_config):
    """Testa integração com otimização."""
    tester = StrategyTester(
        strategy_config=strategy_config,
        initial_capital=10000
    )
    
    # Define espaço de parâmetros
    param_space = {
        "entry.rsi_period": (10, 30),
        "entry.rsi_overbought": (60, 80),
        "exit.take_profit": (0.01, 0.05)
    }
    
    # Executa otimização com backtesting
    results = tester.optimize(
        data=sample_data,
        param_space=param_space,
        num_trials=10,
        optimization_target="sharpe_ratio"
    )
    
    # Verifica resultados
    assert "best_params" in results
    assert "best_score" in results
    assert "optimization_path" in results
    assert results["best_score"] > 0 