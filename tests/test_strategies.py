"""
Testes para módulo de estratégias
===========================

Testes unitários para o sistema de estratégias do QUALIA.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from quantum_trading.strategies import (
    QuantumStrategy,
    TrendStrategy,
    MeanReversionStrategy,
    VolatilityStrategy,
    StrategyFactory
)
from quantum_trading.exceptions import StrategyError

@pytest.fixture
def sample_data():
    """Fixture com dados de exemplo para estratégias."""
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
def quantum_config():
    """Fixture com configuração quântica."""
    return {
        "num_qubits": 4,
        "num_layers": 2,
        "entanglement": "circular",
        "measurement_basis": "computational",
        "optimization": {
            "method": "COBYLA",
            "maxiter": 100
        }
    }

def test_quantum_strategy(sample_data, quantum_config):
    """Testa estratégia quântica."""
    strategy = QuantumStrategy(
        config=quantum_config,
        window_size=20
    )
    
    # Treina estratégia
    strategy.train(sample_data)
    
    # Gera sinais
    signals = strategy.generate_signals(sample_data)
    
    # Verifica sinais
    assert len(signals) == len(sample_data)
    assert all(s in [-1, 0, 1] for s in signals)
    assert "quantum_state" in strategy.get_state()
    assert "circuit_params" in strategy.get_state()

def test_trend_strategy(sample_data):
    """Testa estratégia de tendência."""
    strategy = TrendStrategy(
        short_window=20,
        long_window=50,
        momentum_period=14
    )
    
    # Gera sinais
    signals = strategy.generate_signals(sample_data)
    
    # Verifica sinais
    assert len(signals) == len(sample_data)
    assert all(s in [-1, 0, 1] for s in signals)
    
    # Verifica indicadores
    indicators = strategy.get_indicators()
    assert "sma_short" in indicators
    assert "sma_long" in indicators
    assert "momentum" in indicators

def test_mean_reversion_strategy(sample_data):
    """Testa estratégia de reversão à média."""
    strategy = MeanReversionStrategy(
        window=20,
        std_dev=2.0,
        rsi_period=14
    )
    
    # Gera sinais
    signals = strategy.generate_signals(sample_data)
    
    # Verifica sinais
    assert len(signals) == len(sample_data)
    assert all(s in [-1, 0, 1] for s in signals)
    
    # Verifica bandas
    bands = strategy.get_bands()
    assert "upper" in bands
    assert "lower" in bands
    assert "middle" in bands

def test_volatility_strategy(sample_data):
    """Testa estratégia de volatilidade."""
    strategy = VolatilityStrategy(
        atr_period=14,
        volatility_window=20,
        threshold=1.5
    )
    
    # Gera sinais
    signals = strategy.generate_signals(sample_data)
    
    # Verifica sinais
    assert len(signals) == len(sample_data)
    assert all(s in [-1, 0, 1] for s in signals)
    
    # Verifica métricas
    metrics = strategy.get_metrics()
    assert "atr" in metrics
    assert "historical_volatility" in metrics
    assert "volatility_ratio" in metrics

def test_strategy_factory():
    """Testa fábrica de estratégias."""
    factory = StrategyFactory()
    
    # Cria diferentes estratégias
    quantum = factory.create("quantum", num_qubits=4)
    trend = factory.create("trend", short_window=20, long_window=50)
    mean_reversion = factory.create("mean_reversion", window=20)
    volatility = factory.create("volatility", atr_period=14)
    
    # Verifica tipos
    assert isinstance(quantum, QuantumStrategy)
    assert isinstance(trend, TrendStrategy)
    assert isinstance(mean_reversion, MeanReversionStrategy)
    assert isinstance(volatility, VolatilityStrategy)

def test_strategy_validation(sample_data):
    """Testa validação de estratégias."""
    strategy = TrendStrategy(
        short_window=20,
        long_window=50
    )
    
    # Testa dados inválidos
    with pytest.raises(StrategyError):
        strategy.generate_signals(None)
    
    # Testa parâmetros inválidos
    with pytest.raises(StrategyError):
        TrendStrategy(
            short_window=50,  # maior que long_window
            long_window=20
        )

def test_strategy_persistence(sample_data, tmp_path):
    """Testa persistência de estratégias."""
    strategy = QuantumStrategy(
        config=quantum_config,
        window_size=20
    )
    
    # Treina estratégia
    strategy.train(sample_data)
    
    # Salva estado
    save_path = tmp_path / "strategy_state.json"
    strategy.save_state(save_path)
    
    # Carrega estado em nova instância
    new_strategy = QuantumStrategy(
        config=quantum_config,
        window_size=20
    )
    new_strategy.load_state(save_path)
    
    # Verifica estados
    assert strategy.get_state() == new_strategy.get_state()

def test_strategy_combination(sample_data):
    """Testa combinação de estratégias."""
    trend = TrendStrategy(short_window=20, long_window=50)
    volatility = VolatilityStrategy(atr_period=14)
    
    # Gera sinais individuais
    trend_signals = trend.generate_signals(sample_data)
    volatility_signals = volatility.generate_signals(sample_data)
    
    # Combina sinais
    combined_signals = np.where(
        (trend_signals == volatility_signals) & (trend_signals != 0),
        trend_signals,
        0
    )
    
    # Verifica sinais combinados
    assert len(combined_signals) == len(sample_data)
    assert all(s in [-1, 0, 1] for s in combined_signals)
    assert sum(abs(combined_signals)) <= sum(abs(trend_signals))

def test_strategy_optimization(sample_data):
    """Testa otimização de estratégia."""
    strategy = TrendStrategy(
        short_window=20,
        long_window=50,
        momentum_period=14
    )
    
    # Define espaço de parâmetros
    param_space = {
        "short_window": (10, 30),
        "long_window": (40, 100),
        "momentum_period": (10, 20)
    }
    
    # Otimiza estratégia
    best_params = strategy.optimize(
        data=sample_data,
        param_space=param_space,
        metric="sharpe_ratio",
        num_trials=10
    )
    
    # Verifica resultados
    assert "short_window" in best_params
    assert "long_window" in best_params
    assert "momentum_period" in best_params
    assert best_params["short_window"] < best_params["long_window"]

def test_strategy_analysis(sample_data):
    """Testa análise de estratégia."""
    strategy = MeanReversionStrategy(
        window=20,
        std_dev=2.0,
        rsi_period=14
    )
    
    # Gera sinais
    signals = strategy.generate_signals(sample_data)
    
    # Analisa estratégia
    analysis = strategy.analyze(
        data=sample_data,
        signals=signals
    )
    
    # Verifica análise
    assert "win_rate" in analysis
    assert "profit_factor" in analysis
    assert "max_drawdown" in analysis
    assert "trade_analysis" in analysis
    assert 0 <= analysis["win_rate"] <= 1
    assert analysis["profit_factor"] > 0 