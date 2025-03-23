"""
Módulo de Monitoramento
=====================

Módulo responsável pelo sistema de monitoramento do QUALIA.
"""

import psutil
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Callable

class SystemMonitor:
    def __init__(self):
        self.metrics = {}

    def collect_metrics(self) -> Dict:
        """Coleta métricas do sistema."""
        return {
            'cpu': psutil.cpu_percent(),
            'memory': psutil.virtual_memory().percent,
            'disk': psutil.disk_usage('/').percent
        }

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}

    def collect_metrics(self, data: np.ndarray) -> Dict:
        """Coleta métricas de performance."""
        return {
            'latency': np.mean(data),
            'throughput': len(data),
            'error_rate': np.sum(data < 0) / len(data)
        }

class ResourceMonitor:
    def __init__(self):
        self.metrics = {}

    def collect_metrics(self) -> Dict:
        """Coleta métricas de recursos."""
        return {
            'cpu_cores': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'disk_total': psutil.disk_usage('/').total
        }

class AlertManager:
    def __init__(self):
        self.alerts = []
        self.rules = {}

    def add_rule(self, name: str, rule: Callable, threshold: float):
        """Adiciona uma regra de alerta."""
        self.rules[name] = (rule, threshold)

    def check_alerts(self, metrics: Dict) -> List[str]:
        """Verifica alertas com base nas métricas."""
        alerts = []
        for name, (rule, threshold) in self.rules.items():
            if rule(metrics) > threshold:
                alerts.append(f"Alert: {name} exceeded threshold {threshold}")
        return alerts

class MonitoringManager:
    def __init__(self):
        self.system_monitor = SystemMonitor()
        self.performance_monitor = PerformanceMonitor()
        self.resource_monitor = ResourceMonitor()
        self.alert_manager = AlertManager()
        self.metrics_history = []

    def collect_all_metrics(self, performance_data: Optional[np.ndarray] = None) -> Dict:
        """Coleta todas as métricas."""
        metrics = {
            'timestamp': datetime.now(),
            'system': self.system_monitor.collect_metrics(),
            'resources': self.resource_monitor.collect_metrics()
        }
        
        if performance_data is not None:
            metrics['performance'] = self.performance_monitor.collect_metrics(performance_data)
        
        self.metrics_history.append(metrics)
        return metrics

    def check_alerts(self) -> List[str]:
        """Verifica todos os alertas."""
        if not self.metrics_history:
            return []
        return self.alert_manager.check_alerts(self.metrics_history[-1])

    def get_metrics_history(self) -> List[Dict]:
        """Retorna o histórico de métricas."""
        return self.metrics_history

__all__ = [
    'SystemMonitor',
    'PerformanceMonitor',
    'ResourceMonitor',
    'AlertManager',
    'MonitoringManager'
] 