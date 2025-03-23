"""
Testes para módulo de monitoramento
=====================

Testes unitários para o sistema de monitoramento do QUALIA.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from quantum_trading.monitoring import (
    SystemMonitor,
    PerformanceMonitor,
    ResourceMonitor,
    AlertManager,
    MonitoringManager
)
from quantum_trading.exceptions import MonitoringError

@pytest.fixture
def monitoring_config():
    """Fixture com configuração de monitoramento."""
    return {
        "system": {
            "check_interval": 60,
            "metrics": ["cpu", "memory", "disk", "network"],
            "thresholds": {
                "cpu_percent": 80,
                "memory_percent": 85,
                "disk_percent": 90
            }
        },
        "performance": {
            "window_size": "1h",
            "metrics": ["latency", "throughput", "success_rate"],
            "thresholds": {
                "max_latency": 1000,  # ms
                "min_throughput": 100,  # ops/s
                "min_success_rate": 0.95
            }
        },
        "resource": {
            "check_interval": 300,
            "resources": ["cpu", "memory", "disk", "gpu"],
            "limits": {
                "max_memory": "8G",
                "max_cpu_cores": 4,
                "max_gpu_memory": "4G"
            }
        },
        "alerts": {
            "channels": ["email", "slack", "telegram"],
            "levels": ["info", "warning", "error", "critical"],
            "throttle_interval": 300
        }
    }

@pytest.fixture
def sample_metrics():
    """Fixture com métricas de exemplo."""
    np.random.seed(42)
    dates = pd.date_range(
        start=datetime.now() - timedelta(hours=1),
        end=datetime.now(),
        freq="1min"
    )
    
    return pd.DataFrame({
        "timestamp": dates,
        "cpu_percent": np.random.uniform(20, 90, len(dates)),
        "memory_percent": np.random.uniform(40, 95, len(dates)),
        "disk_percent": np.random.uniform(50, 85, len(dates)),
        "latency_ms": np.random.exponential(100, len(dates)),
        "throughput": np.random.normal(150, 20, len(dates)),
        "success_rate": np.random.beta(9, 1, len(dates))
    })

def test_system_monitor(monitoring_config):
    """Testa monitor do sistema."""
    monitor = SystemMonitor(config=monitoring_config)
    
    # Coleta métricas
    metrics = monitor.collect_metrics()
    
    # Verifica métricas
    assert all(m in metrics for m in monitoring_config["system"]["metrics"])
    assert all(isinstance(metrics[m], (int, float)) for m in metrics)
    
    # Verifica thresholds
    status = monitor.check_thresholds(metrics)
    assert isinstance(status, dict)
    assert all(s in ["ok", "warning", "critical"] for s in status.values())

def test_performance_monitor(monitoring_config, sample_metrics):
    """Testa monitor de performance."""
    monitor = PerformanceMonitor(config=monitoring_config)
    
    # Analisa performance
    analysis = monitor.analyze_performance(sample_metrics)
    
    # Verifica análise
    assert "latency" in analysis
    assert "throughput" in analysis
    assert "success_rate" in analysis
    
    # Verifica estatísticas
    stats = monitor.calculate_statistics(sample_metrics)
    assert "mean" in stats
    assert "std" in stats
    assert "percentiles" in stats

def test_resource_monitor(monitoring_config):
    """Testa monitor de recursos."""
    monitor = ResourceMonitor(config=monitoring_config)
    
    # Monitora recursos
    usage = monitor.monitor_resources()
    
    # Verifica uso
    assert all(r in usage for r in monitoring_config["resource"]["resources"])
    
    # Verifica limites
    status = monitor.check_resource_limits(usage)
    assert isinstance(status, dict)
    assert all(s in ["ok", "warning", "critical"] for s in status.values())

def test_alert_manager(monitoring_config):
    """Testa gerenciador de alertas."""
    manager = AlertManager(config=monitoring_config)
    
    # Cria alerta
    alert = {
        "level": "warning",
        "message": "High CPU usage detected",
        "timestamp": datetime.now(),
        "metrics": {
            "cpu_percent": 85
        }
    }
    
    # Mock para canais de notificação
    mock_channels = {
        "email": Mock(),
        "slack": Mock(),
        "telegram": Mock()
    }
    
    for name, channel in mock_channels.items():
        manager.register_channel(name, channel)
    
    # Envia alerta
    manager.send_alert(alert)
    
    # Verifica envios
    assert all(c.send.called for c in mock_channels.values())

def test_monitoring_manager(monitoring_config):
    """Testa gerenciador de monitoramento."""
    manager = MonitoringManager(config=monitoring_config)
    
    # Registra monitores
    monitors = {
        "system": Mock(),
        "performance": Mock(),
        "resource": Mock(),
        "alerts": Mock()
    }
    
    for name, monitor in monitors.items():
        manager.register_monitor(name, monitor)
    
    # Inicia monitoramento
    manager.start_monitoring()
    
    # Verifica monitores
    assert all(m.start.called for m in monitors.values())
    
    # Para monitoramento
    manager.stop_monitoring()
    assert all(m.stop.called for m in monitors.values())

def test_metric_collection(monitoring_config, sample_metrics):
    """Testa coleta de métricas."""
    manager = MonitoringManager(config=monitoring_config)
    
    # Registra coletor
    def metric_collector():
        return sample_metrics.iloc[-1].to_dict()
    
    manager.register_collector("test", metric_collector)
    
    # Coleta métricas
    metrics = manager.collect_all_metrics()
    
    # Verifica métricas
    assert "test" in metrics
    assert all(m in metrics["test"] for m in sample_metrics.columns)

def test_monitoring_persistence(monitoring_config, sample_metrics):
    """Testa persistência de monitoramento."""
    manager = MonitoringManager(config=monitoring_config)
    
    # Salva métricas
    manager.save_metrics(sample_metrics, "test_metrics")
    
    # Carrega métricas
    loaded = manager.load_metrics("test_metrics")
    
    # Verifica métricas
    assert isinstance(loaded, pd.DataFrame)
    assert all(loaded.equals(sample_metrics))

def test_alert_rules(monitoring_config):
    """Testa regras de alerta."""
    manager = AlertManager(config=monitoring_config)
    
    # Define regra
    def cpu_rule(metrics):
        return metrics["cpu_percent"] > 80
    
    manager.add_rule("high_cpu", cpu_rule)
    
    # Testa regra
    metrics = {
        "cpu_percent": 85,
        "memory_percent": 70
    }
    
    alerts = manager.check_rules(metrics)
    
    # Verifica alertas
    assert len(alerts) > 0
    assert alerts[0]["rule"] == "high_cpu"
    assert alerts[0]["level"] == "warning"

def test_monitoring_dashboard(monitoring_config, sample_metrics):
    """Testa dashboard de monitoramento."""
    manager = MonitoringManager(config=monitoring_config)
    
    # Gera dashboard
    dashboard = manager.generate_dashboard(sample_metrics)
    
    # Verifica componentes
    assert "system_status" in dashboard
    assert "performance_metrics" in dashboard
    assert "resource_usage" in dashboard
    assert "alerts" in dashboard

def test_monitoring_validation(monitoring_config):
    """Testa validação de monitoramento."""
    manager = MonitoringManager(config=monitoring_config)
    
    # Configuração válida
    assert manager.validate_config(monitoring_config)
    
    # Configuração inválida
    invalid_config = monitoring_config.copy()
    del invalid_config["system"]["metrics"]
    
    with pytest.raises(MonitoringError):
        manager.validate_config(invalid_config) 