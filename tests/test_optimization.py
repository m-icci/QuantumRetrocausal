"""
Testes para módulo de otimização
=====================

Testes unitários para o sistema de otimização do QUALIA.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from quantum_trading.optimization import (
    QuantumOptimizer,
    StrategyOptimizer,
    HyperparameterOptimizer,
    ResourceOptimizer,
    OptimizationPipeline
)
from quantum_trading.exceptions import OptimizationError

@pytest.fixture
def optimization_data():
    """Fixture com dados para otimização."""
    np.random.seed(42)
    
    return {
        "parameters": {
            "window_size": np.arange(10, 100, 10),
            "threshold": np.linspace(0.1, 0.9, 9),
            "leverage": np.array([1, 2, 3, 4, 5])
        },
        "constraints": {
            "max_window": 100,
            "min_threshold": 0.1,
            "max_leverage": 5
        },
        "objective": lambda x: -(x["window_size"] * 0.1 + x["threshold"] * 0.5 + x["leverage"] * 0.2)
    }

@pytest.fixture
def market_data():
    """Fixture com dados de mercado."""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="1H")
    np.random.seed(42)
    
    return pd.DataFrame({
        "timestamp": dates,
        "open": np.random.normal(100, 10, len(dates)),
        "high": np.random.normal(105, 10, len(dates)),
        "low": np.random.normal(95, 10, len(dates)),
        "close": np.random.normal(100, 10, len(dates)),
        "volume": np.random.normal(1000, 100, len(dates))
    }).set_index("timestamp")

def test_quantum_optimizer(optimization_data):
    """Testa otimizador quântico."""
    optimizer = QuantumOptimizer()
    
    # Configura otimização
    optimizer.configure(
        n_qubits=4,
        n_layers=2,
        learning_rate=0.01
    )
    
    # Executa otimização
    result = optimizer.optimize(
        parameters=optimization_data["parameters"],
        constraints=optimization_data["constraints"],
        objective=optimization_data["objective"]
    )
    
    # Verifica resultados
    assert "optimal_params" in result
    assert "optimal_value" in result
    assert "convergence" in result
    
    # Valida restrições
    params = result["optimal_params"]
    assert params["window_size"] <= optimization_data["constraints"]["max_window"]
    assert params["threshold"] >= optimization_data["constraints"]["min_threshold"]
    assert params["leverage"] <= optimization_data["constraints"]["max_leverage"]

def test_strategy_optimizer(optimization_data, market_data):
    """Testa otimizador de estratégia."""
    optimizer = StrategyOptimizer()
    
    # Define estratégia
    def strategy(data, params):
        sma = data["close"].rolling(window=int(params["window_size"])).mean()
        signals = np.where(data["close"] > sma * (1 + params["threshold"]), 1, 0)
        return pd.Series(signals, index=data.index)
    
    # Executa otimização
    result = optimizer.optimize(
        strategy=strategy,
        data=market_data,
        parameters=optimization_data["parameters"],
        constraints=optimization_data["constraints"]
    )
    
    # Verifica resultados
    assert "optimal_params" in result
    assert "performance" in result
    assert "trades" in result
    
    # Valida performance
    assert result["performance"]["sharpe_ratio"] > 0
    assert result["performance"]["max_drawdown"] < 1

def test_hyperparameter_optimizer(optimization_data):
    """Testa otimizador de hiperparâmetros."""
    optimizer = HyperparameterOptimizer()
    
    # Define modelo
    model = Mock()
    model.fit = Mock(return_value=None)
    model.score = Mock(return_value=0.8)
    
    # Executa otimização
    result = optimizer.optimize(
        model=model,
        parameters=optimization_data["parameters"],
        cv_folds=5
    )
    
    # Verifica resultados
    assert "best_params" in result
    assert "best_score" in result
    assert "cv_results" in result
    
    # Valida scores
    assert 0 <= result["best_score"] <= 1
    assert len(result["cv_results"]) > 0

def test_resource_optimizer(optimization_data):
    """Testa otimizador de recursos."""
    optimizer = ResourceOptimizer()
    
    # Define recursos
    resources = {
        "cpu": np.linspace(0.1, 1.0, 10),
        "memory": np.linspace(1, 16, 16),
        "bandwidth": np.linspace(0.1, 1.0, 10)
    }
    
    # Executa otimização
    result = optimizer.optimize(
        resources=resources,
        workload=lambda x: -(x["cpu"] * 0.4 + x["memory"] * 0.4 + x["bandwidth"] * 0.2),
        constraints={"max_total": 10}
    )
    
    # Verifica resultados
    assert "allocation" in result
    assert "efficiency" in result
    assert "utilization" in result
    
    # Valida alocação
    allocation = result["allocation"]
    assert sum(allocation.values()) <= 10

def test_optimization_pipeline(optimization_data, market_data):
    """Testa pipeline de otimização."""
    pipeline = OptimizationPipeline()
    
    # Adiciona otimizadores
    pipeline.add_optimizer("quantum", QuantumOptimizer())
    pipeline.add_optimizer("strategy", StrategyOptimizer())
    
    # Configura pipeline
    pipeline.configure(
        sequence=["quantum", "strategy"],
        data=market_data,
        parameters=optimization_data["parameters"],
        constraints=optimization_data["constraints"]
    )
    
    # Executa pipeline
    result = pipeline.run()
    
    # Verifica resultados
    assert "quantum" in result
    assert "strategy" in result
    assert "final_params" in result

def test_optimization_constraints(optimization_data):
    """Testa restrições de otimização."""
    optimizer = QuantumOptimizer()
    
    # Adiciona restrições
    constraints = {
        "max_window": 50,  # Mais restritivo
        "min_threshold": 0.2,  # Mais restritivo
        "max_leverage": 3  # Mais restritivo
    }
    
    # Executa otimização
    result = optimizer.optimize(
        parameters=optimization_data["parameters"],
        constraints=constraints,
        objective=optimization_data["objective"]
    )
    
    # Verifica restrições
    params = result["optimal_params"]
    assert params["window_size"] <= 50
    assert params["threshold"] >= 0.2
    assert params["leverage"] <= 3

def test_optimization_convergence(optimization_data):
    """Testa convergência de otimização."""
    optimizer = QuantumOptimizer()
    
    # Configura critérios
    optimizer.configure(
        max_iterations=1000,
        tolerance=1e-6,
        early_stopping=True
    )
    
    # Executa otimização
    result = optimizer.optimize(
        parameters=optimization_data["parameters"],
        constraints=optimization_data["constraints"],
        objective=optimization_data["objective"]
    )
    
    # Verifica convergência
    assert "iterations" in result
    assert "convergence_curve" in result
    assert result["iterations"] <= 1000

def test_optimization_stability(optimization_data):
    """Testa estabilidade de otimização."""
    optimizer = QuantumOptimizer()
    results = []
    
    # Executa múltiplas otimizações
    for _ in range(5):
        result = optimizer.optimize(
            parameters=optimization_data["parameters"],
            constraints=optimization_data["constraints"],
            objective=optimization_data["objective"]
        )
        results.append(result["optimal_value"])
    
    # Verifica estabilidade
    std = np.std(results)
    assert std < 0.1  # Baixa variância

def test_optimization_parallelization(optimization_data):
    """Testa paralelização de otimização."""
    optimizer = QuantumOptimizer()
    
    # Configura paralelização
    optimizer.configure(
        n_jobs=4,
        backend="multiprocessing"
    )
    
    # Executa otimização
    result = optimizer.optimize(
        parameters=optimization_data["parameters"],
        constraints=optimization_data["constraints"],
        objective=optimization_data["objective"]
    )
    
    # Verifica resultados
    assert "parallel_evaluations" in result
    assert result["parallel_evaluations"] > 0 