"""
Testes para módulo de análise
=====================

Testes unitários para o sistema de análise do QUALIA.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from quantum_trading.analysis import (
    QuantumAnalyzer,
    TechnicalAnalyzer,
    StatisticalAnalyzer,
    MachineLearningAnalyzer,
    PortfolioAnalyzer
)
from quantum_trading.exceptions import AnalysisError

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

@pytest.fixture
def portfolio_data():
    """Fixture com dados de portfólio."""
    return {
        "BTC/USDT": {
            "position": 1.5,
            "entry_price": 45000,
            "current_price": 47000,
            "pnl": 3000,
            "risk": 0.1
        },
        "ETH/USDT": {
            "position": 10,
            "entry_price": 3000,
            "current_price": 3200,
            "pnl": 2000,
            "risk": 0.15
        }
    }

def test_quantum_analyzer(market_data):
    """Testa analisador quântico."""
    analyzer = QuantumAnalyzer()
    
    # Configura análise
    analyzer.configure(
        quantum_bits=4,
        entanglement_depth=2,
        measurement_basis="computational"
    )
    
    # Executa análise quântica
    results = analyzer.analyze(market_data)
    
    # Verifica resultados
    assert "quantum_state" in results
    assert "entanglement_measure" in results
    assert "quantum_indicators" in results
    
    # Valida métricas
    assert 0 <= results["entanglement_measure"] <= 1
    assert len(results["quantum_indicators"]) > 0

def test_technical_analyzer(market_data):
    """Testa analisador técnico."""
    analyzer = TechnicalAnalyzer()
    
    # Configura indicadores
    analyzer.configure_indicators([
        {"name": "SMA", "params": {"window": 20}},
        {"name": "RSI", "params": {"window": 14}},
        {"name": "MACD", "params": {"fast": 12, "slow": 26, "signal": 9}}
    ])
    
    # Executa análise
    results = analyzer.analyze(market_data)
    
    # Verifica indicadores
    assert "SMA" in results
    assert "RSI" in results
    assert "MACD" in results
    
    # Valida sinais
    signals = analyzer.generate_signals(results)
    assert "buy_signals" in signals
    assert "sell_signals" in signals

def test_statistical_analyzer(market_data):
    """Testa analisador estatístico."""
    analyzer = StatisticalAnalyzer()
    
    # Executa análise
    stats = analyzer.analyze(market_data)
    
    # Verifica estatísticas
    assert "mean" in stats
    assert "std" in stats
    assert "skew" in stats
    assert "kurtosis" in stats
    
    # Testa normalidade
    normality = analyzer.test_normality(market_data["close"])
    assert "statistic" in normality
    assert "p_value" in normality
    
    # Testa estacionariedade
    stationarity = analyzer.test_stationarity(market_data["close"])
    assert "adf_stat" in stationarity
    assert "p_value" in stationarity

def test_machine_learning_analyzer(market_data):
    """Testa analisador de machine learning."""
    analyzer = MachineLearningAnalyzer()
    
    # Prepara dados
    X, y = analyzer.prepare_data(market_data)
    
    # Treina modelo
    model = analyzer.train_model(X, y)
    
    # Faz previsões
    predictions = analyzer.predict(model, X)
    assert len(predictions) > 0
    
    # Avalia modelo
    metrics = analyzer.evaluate_model(model, X, y)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    
    # Valida features
    importance = analyzer.feature_importance(model)
    assert len(importance) > 0

def test_portfolio_analyzer(portfolio_data):
    """Testa analisador de portfólio."""
    analyzer = PortfolioAnalyzer()
    
    # Analisa portfólio
    analysis = analyzer.analyze_portfolio(portfolio_data)
    
    # Verifica métricas
    assert "total_value" in analysis
    assert "total_pnl" in analysis
    assert "portfolio_risk" in analysis
    
    # Calcula retornos
    returns = analyzer.calculate_returns(portfolio_data)
    assert "absolute_return" in returns
    assert "percentage_return" in returns
    
    # Analisa risco
    risk = analyzer.analyze_risk(portfolio_data)
    assert "var" in risk
    assert "sharpe_ratio" in risk
    assert "max_drawdown" in risk

def test_quantum_technical_integration(market_data):
    """Testa integração quântica-técnica."""
    quantum = QuantumAnalyzer()
    technical = TechnicalAnalyzer()
    
    # Executa análises
    quantum_results = quantum.analyze(market_data)
    technical_results = technical.analyze(market_data)
    
    # Combina resultados
    combined = {
        "quantum": quantum_results,
        "technical": technical_results
    }
    
    # Verifica integração
    assert "quantum_state" in combined["quantum"]
    assert "SMA" in combined["technical"]

def test_statistical_ml_integration(market_data):
    """Testa integração estatística-ML."""
    statistical = StatisticalAnalyzer()
    ml = MachineLearningAnalyzer()
    
    # Executa análises
    stats = statistical.analyze(market_data)
    
    # Usa estatísticas como features
    X = pd.DataFrame(stats).T
    y = market_data["close"].pct_change().shift(-1)[:-1]
    
    # Treina modelo
    model = ml.train_model(X, y)
    predictions = ml.predict(model, X)
    
    # Verifica resultados
    assert len(predictions) > 0

def test_portfolio_optimization(portfolio_data, market_data):
    """Testa otimização de portfólio."""
    analyzer = PortfolioAnalyzer()
    
    # Otimiza alocação
    optimal = analyzer.optimize_allocation(
        portfolio_data,
        market_data,
        risk_free_rate=0.02
    )
    
    # Verifica resultados
    assert "weights" in optimal
    assert "expected_return" in optimal
    assert "expected_risk" in optimal
    assert "sharpe_ratio" in optimal

def test_analysis_persistence():
    """Testa persistência de análise."""
    analyzer = TechnicalAnalyzer()
    
    # Salva configuração
    analyzer.save_configuration("test_config")
    
    # Carrega configuração
    loaded = analyzer.load_configuration("test_config")
    
    # Verifica configuração
    assert loaded == analyzer.get_configuration()
    
    # Remove arquivo
    analyzer.remove_configuration("test_config")

def test_analysis_validation(market_data):
    """Testa validação de análise."""
    analyzer = StatisticalAnalyzer()
    
    # Testa dados válidos
    assert analyzer.validate_data(market_data)
    
    # Testa dados inválidos
    invalid_data = market_data.copy()
    invalid_data.loc[invalid_data.index[0], "close"] = np.nan
    
    with pytest.raises(AnalysisError):
        analyzer.validate_data(invalid_data) 