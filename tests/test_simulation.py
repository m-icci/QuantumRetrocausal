"""
Testes para módulo de simulação
=====================

Testes unitários para o sistema de simulação do QUALIA.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from quantum_trading.simulation import (
    MarketSimulator,
    QuantumSimulator,
    OrderSimulator,
    RiskSimulator,
    SimulationEngine
)
from quantum_trading.exceptions import SimulationError

@pytest.fixture
def simulation_config():
    """Fixture com configuração de simulação."""
    return {
        "market": {
            "initial_price": 100.0,
            "volatility": 0.2,
            "drift": 0.1,
            "tick_size": 0.01
        },
        "quantum": {
            "num_qubits": 4,
            "shots": 1000,
            "noise_model": "basic"
        },
        "order": {
            "latency": 50,  # ms
            "slippage": 0.001,
            "fees": 0.001
        },
        "risk": {
            "initial_balance": 100000,
            "max_position": 0.1,
            "stop_loss": 0.02
        }
    }

@pytest.fixture
def sample_market_data():
    """Fixture com dados de mercado de exemplo."""
    np.random.seed(42)
    dates = pd.date_range(
        start="2023-01-01",
        end="2023-12-31",
        freq="1min"
    )
    
    return pd.DataFrame({
        "timestamp": dates,
        "open": np.cumsum(np.random.normal(0, 0.1, len(dates))) + 100,
        "high": np.cumsum(np.random.normal(0, 0.1, len(dates))) + 101,
        "low": np.cumsum(np.random.normal(0, 0.1, len(dates))) + 99,
        "close": np.cumsum(np.random.normal(0, 0.1, len(dates))) + 100,
        "volume": np.random.exponential(1, len(dates)) * 1000
    })

def test_market_simulator(simulation_config):
    """Testa simulador de mercado."""
    simulator = MarketSimulator(config=simulation_config)
    
    # Simula mercado
    simulation = simulator.simulate(
        duration=timedelta(days=30),
        interval="1min"
    )
    
    # Verifica resultados
    assert isinstance(simulation, pd.DataFrame)
    assert all(col in simulation.columns for col in ["open", "high", "low", "close", "volume"])
    assert len(simulation) > 0
    assert simulation["close"].std() > 0
    
    # Testa continuidade
    assert all(simulation["high"] >= simulation["low"])
    assert all(simulation["open"] >= simulation["low"])
    assert all(simulation["close"] >= simulation["low"])

def test_quantum_simulator(simulation_config):
    """Testa simulador quântico."""
    simulator = QuantumSimulator(config=simulation_config)
    
    # Prepara estado quântico
    state = simulator.prepare_state(
        num_qubits=4,
        initial_state=[0.5, 0.5, 0.5, 0.5]
    )
    
    # Aplica operações
    result = simulator.apply_operations(
        state=state,
        operations=["H", "CNOT", "X"]
    )
    
    # Verifica resultados
    assert "final_state" in result
    assert "measurements" in result
    assert len(result["measurements"]) == simulation_config["quantum"]["shots"]
    
    # Verifica coerência
    assert abs(np.sum(np.abs(result["final_state"])**2) - 1.0) < 1e-10

def test_order_simulator(simulation_config, sample_market_data):
    """Testa simulador de ordens."""
    simulator = OrderSimulator(config=simulation_config)
    
    # Cria ordem
    order = {
        "type": "market",
        "side": "buy",
        "amount": 1.0,
        "timestamp": datetime.now()
    }
    
    # Simula execução
    execution = simulator.simulate_execution(
        order=order,
        market_data=sample_market_data
    )
    
    # Verifica resultados
    assert "filled_amount" in execution
    assert "average_price" in execution
    assert "fees" in execution
    assert "latency" in execution
    assert execution["filled_amount"] > 0
    assert execution["latency"] >= 0

def test_risk_simulator(simulation_config):
    """Testa simulador de risco."""
    simulator = RiskSimulator(config=simulation_config)
    
    # Define posição
    position = {
        "asset": "BTC/USDT",
        "amount": 1.0,
        "entry_price": 50000,
        "current_price": 49000
    }
    
    # Simula risco
    risk = simulator.simulate_risk(position)
    
    # Verifica resultados
    assert "value_at_risk" in risk
    assert "max_loss" in risk
    assert "margin_call_price" in risk
    assert risk["value_at_risk"] > 0
    assert risk["max_loss"] <= position["amount"] * position["entry_price"]

def test_simulation_engine(simulation_config, sample_market_data):
    """Testa motor de simulação."""
    engine = SimulationEngine(config=simulation_config)
    
    # Define estratégia
    def strategy(data):
        close = data["close"]
        sma = close.rolling(window=20).mean()
        return pd.Series(np.where(close > sma, 1, -1), index=data.index)
    
    # Executa simulação
    results = engine.run_simulation(
        strategy=strategy,
        market_data=sample_market_data
    )
    
    # Verifica resultados
    assert "trades" in results
    assert "performance" in results
    assert "portfolio" in results
    assert len(results["trades"]) > 0
    assert results["performance"]["total_return"] != 0

def test_monte_carlo_simulation(simulation_config):
    """Testa simulação Monte Carlo."""
    engine = SimulationEngine(config=simulation_config)
    
    # Executa simulações
    simulations = engine.run_monte_carlo(
        num_simulations=100,
        duration=timedelta(days=30)
    )
    
    # Verifica resultados
    assert len(simulations) == 100
    assert all(isinstance(sim, pd.DataFrame) for sim in simulations)
    
    # Calcula estatísticas
    stats = engine.calculate_monte_carlo_stats(simulations)
    assert "mean_return" in stats
    assert "var_95" in stats
    assert "max_drawdown" in stats

def test_stress_testing(simulation_config, sample_market_data):
    """Testa testes de estresse."""
    engine = SimulationEngine(config=simulation_config)
    
    # Define cenários de estresse
    scenarios = [
        {"volatility": 0.5, "drift": -0.2},  # Alta volatilidade, tendência negativa
        {"volatility": 0.8, "drift": 0.0},   # Volatilidade extrema
        {"volatility": 0.3, "drift": -0.5}   # Crash de mercado
    ]
    
    # Executa testes de estresse
    results = engine.run_stress_tests(
        scenarios=scenarios,
        market_data=sample_market_data
    )
    
    # Verifica resultados
    assert len(results) == len(scenarios)
    assert all("portfolio_value" in r for r in results)
    assert all("max_drawdown" in r for r in results)
    assert all("var_95" in r for r in results)

def test_scenario_analysis(simulation_config):
    """Testa análise de cenários."""
    engine = SimulationEngine(config=simulation_config)
    
    # Define cenários
    scenarios = {
        "bull": {"drift": 0.2, "volatility": 0.15},
        "bear": {"drift": -0.2, "volatility": 0.25},
        "sideways": {"drift": 0.0, "volatility": 0.1}
    }
    
    # Executa análise
    analysis = engine.analyze_scenarios(scenarios)
    
    # Verifica resultados
    assert all(scenario in analysis for scenario in scenarios)
    assert all("expected_return" in analysis[s] for s in scenarios)
    assert all("risk_metrics" in analysis[s] for s in scenarios)

def test_backtest_simulation(simulation_config, sample_market_data):
    """Testa simulação de backtest."""
    engine = SimulationEngine(config=simulation_config)
    
    # Define estratégia
    def strategy(data):
        close = data["close"]
        sma_fast = close.rolling(window=10).mean()
        sma_slow = close.rolling(window=30).mean()
        return pd.Series(np.where(sma_fast > sma_slow, 1, -1), index=data.index)
    
    # Executa backtest
    results = engine.run_backtest(
        strategy=strategy,
        market_data=sample_market_data,
        initial_balance=100000
    )
    
    # Verifica resultados
    assert "equity_curve" in results
    assert "trades" in results
    assert "metrics" in results
    assert results["metrics"]["sharpe_ratio"] != 0
    assert len(results["trades"]) > 0

def test_simulation_persistence(simulation_config, tmp_path):
    """Testa persistência de simulação."""
    engine = SimulationEngine(config=simulation_config)
    
    # Executa simulação
    simulation = engine.run_simulation(
        duration=timedelta(days=30),
        interval="1min"
    )
    
    # Salva resultados
    save_path = tmp_path / "simulation_results.pkl"
    engine.save_results(simulation, save_path)
    
    # Carrega resultados
    loaded_simulation = engine.load_results(save_path)
    
    # Verifica dados
    assert all(key in loaded_simulation for key in simulation.keys())
    assert all(
        loaded_simulation[key].equals(simulation[key])
        if isinstance(simulation[key], pd.DataFrame)
        else loaded_simulation[key] == simulation[key]
        for key in simulation.keys()
    ) 