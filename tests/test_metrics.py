"""
Testes para módulo de métricas
=========================

Testes unitários para o sistema de métricas do QUALIA.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from quantum_trading.metrics import (
    QuantumMetrics,
    MarketMetrics,
    PerformanceMetrics,
    MetricsCollector
)
from quantum_trading.exceptions import MetricsError

@pytest.fixture
def sample_data():
    """Fixture com dados de exemplo para métricas."""
    np.random.seed(42)
    timestamps = [
        datetime.now() + timedelta(minutes=i)
        for i in range(100)
    ]
    prices = np.cumsum(np.random.normal(0, 0.1, 100)) + 100
    volumes = np.random.exponential(1, 100) * 10
    return {
        "timestamps": timestamps,
        "prices": prices,
        "volumes": volumes
    }

@pytest.fixture
def metrics_collector():
    """Fixture com coletor de métricas configurado."""
    return MetricsCollector(
        window_size=20,
        metrics_interval="1m"
    )

def test_quantum_metrics_calculation():
    """Testa cálculo de métricas quânticas."""
    data = np.random.normal(0, 1, 100)
    qm = QuantumMetrics()
    
    # Calcula métricas
    metrics = qm.calculate(data)
    
    # Verifica resultados
    assert "entropy" in metrics
    assert "coherence" in metrics
    assert "complexity" in metrics
    assert 0 <= metrics["entropy"] <= 1
    assert 0 <= metrics["coherence"] <= 1
    assert metrics["complexity"] >= 0

def test_market_metrics_calculation(sample_data):
    """Testa cálculo de métricas de mercado."""
    mm = MarketMetrics()
    
    # Calcula métricas
    metrics = mm.calculate(
        prices=sample_data["prices"],
        volumes=sample_data["volumes"],
        window=20
    )
    
    # Verifica resultados
    assert "volatility" in metrics
    assert "momentum" in metrics
    assert "volume_profile" in metrics
    assert "price_trend" in metrics
    assert metrics["volatility"] >= 0
    assert isinstance(metrics["momentum"], float)
    assert len(metrics["volume_profile"]) > 0

def test_performance_metrics_calculation():
    """Testa cálculo de métricas de performance."""
    trades = [
        {"price": 100, "amount": 1, "side": "buy"},
        {"price": 105, "amount": 1, "side": "sell"},
        {"price": 98, "amount": 1.5, "side": "buy"},
        {"price": 102, "amount": 1.5, "side": "sell"}
    ]
    
    pm = PerformanceMetrics()
    
    # Calcula métricas
    metrics = pm.calculate(trades)
    
    # Verifica resultados
    assert "total_profit" in metrics
    assert "win_rate" in metrics
    assert "profit_factor" in metrics
    assert "sharpe_ratio" in metrics
    assert metrics["total_profit"] > 0
    assert 0 <= metrics["win_rate"] <= 1
    assert metrics["profit_factor"] > 0

def test_metrics_collector_integration(metrics_collector, sample_data):
    """Testa integração do coletor de métricas."""
    # Adiciona dados
    for i in range(len(sample_data["timestamps"])):
        metrics_collector.add_data(
            timestamp=sample_data["timestamps"][i],
            price=sample_data["prices"][i],
            volume=sample_data["volumes"][i]
        )
    
    # Coleta métricas
    metrics = metrics_collector.collect()
    
    # Verifica resultados
    assert "quantum" in metrics
    assert "market" in metrics
    assert "performance" in metrics
    assert len(metrics["quantum"]) > 0
    assert len(metrics["market"]) > 0
    assert len(metrics["performance"]) > 0

def test_metrics_validation():
    """Testa validação de métricas."""
    qm = QuantumMetrics()
    
    # Testa dados inválidos
    with pytest.raises(MetricsError):
        qm.calculate(None)
    
    with pytest.raises(MetricsError):
        qm.calculate([])
    
    with pytest.raises(MetricsError):
        qm.calculate(np.array([np.nan, 1, 2]))

def test_metrics_window_size(metrics_collector):
    """Testa janela deslizante de métricas."""
    # Adiciona dados
    for i in range(50):
        metrics_collector.add_data(
            timestamp=datetime.now() + timedelta(minutes=i),
            price=100 + i,
            volume=10
        )
    
    # Verifica tamanho da janela
    assert len(metrics_collector.price_window) == 20
    assert len(metrics_collector.volume_window) == 20
    
    # Verifica que dados mais antigos foram removidos
    assert metrics_collector.price_window[0] == 130  # 50-20=30, 100+30=130

def test_metrics_persistence(metrics_collector, tmp_path):
    """Testa persistência de métricas."""
    # Adiciona dados
    for i in range(10):
        metrics_collector.add_data(
            timestamp=datetime.now() + timedelta(minutes=i),
            price=100 + i,
            volume=10
        )
    
    # Salva métricas
    metrics_file = tmp_path / "metrics.json"
    metrics_collector.save(metrics_file)
    
    # Carrega métricas
    new_collector = MetricsCollector.load(metrics_file)
    
    # Verifica dados
    assert len(new_collector.price_window) == len(metrics_collector.price_window)
    assert len(new_collector.volume_window) == len(metrics_collector.volume_window)
    np.testing.assert_array_almost_equal(
        new_collector.price_window,
        metrics_collector.price_window
    )

def test_metrics_aggregation(metrics_collector, sample_data):
    """Testa agregação de métricas."""
    # Adiciona dados
    for i in range(len(sample_data["timestamps"])):
        metrics_collector.add_data(
            timestamp=sample_data["timestamps"][i],
            price=sample_data["prices"][i],
            volume=sample_data["volumes"][i]
        )
    
    # Agrega métricas por hora
    hourly_metrics = metrics_collector.aggregate("1h")
    
    # Verifica resultados
    assert len(hourly_metrics) > 0
    for timestamp, metrics in hourly_metrics.items():
        assert isinstance(timestamp, datetime)
        assert "quantum" in metrics
        assert "market" in metrics
        assert "performance" in metrics

def test_metrics_correlation(metrics_collector, sample_data):
    """Testa correlação entre métricas."""
    # Adiciona dados
    for i in range(len(sample_data["timestamps"])):
        metrics_collector.add_data(
            timestamp=sample_data["timestamps"][i],
            price=sample_data["prices"][i],
            volume=sample_data["volumes"][i]
        )
    
    # Calcula correlações
    correlations = metrics_collector.calculate_correlations()
    
    # Verifica resultados
    assert "price_volume" in correlations
    assert "entropy_volatility" in correlations
    assert "coherence_momentum" in correlations
    assert -1 <= correlations["price_volume"] <= 1
    assert -1 <= correlations["entropy_volatility"] <= 1
    assert -1 <= correlations["coherence_momentum"] <= 1 