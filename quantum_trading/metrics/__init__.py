"""
Módulo de Métricas
=================

Módulo responsável pelo sistema de métricas do QUALIA.
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

class QuantumMetrics:
    def __init__(self):
        self.metrics = {}

    def calculate_metrics(self, data: np.ndarray) -> Dict:
        """Calcula métricas quânticas."""
        return {
            'coherence': np.mean(data),
            'entanglement': np.std(data),
            'quantum_state': np.sum(data)
        }

class MarketMetrics:
    def __init__(self):
        self.metrics = {}

    def calculate_metrics(self, data: np.ndarray) -> Dict:
        """Calcula métricas de mercado."""
        return {
            'volatility': np.std(data),
            'trend': np.mean(data),
            'momentum': np.sum(np.diff(data))
        }

class PerformanceMetrics:
    def __init__(self):
        self.metrics = {}

    def calculate_metrics(self, data: np.ndarray) -> Dict:
        """Calcula métricas de performance."""
        return {
            'sharpe_ratio': self._calculate_sharpe(data),
            'max_drawdown': self._calculate_drawdown(data),
            'win_rate': self._calculate_win_rate(data)
        }

    def _calculate_sharpe(self, data: np.ndarray) -> float:
        """Calcula o Sharpe Ratio."""
        returns = np.diff(data) / data[:-1]
        return np.mean(returns) / np.std(returns) if len(returns) > 0 else 0.0

    def _calculate_drawdown(self, data: np.ndarray) -> float:
        """Calcula o Maximum Drawdown."""
        peak = np.maximum.accumulate(data)
        drawdown = (peak - data) / peak
        return np.max(drawdown) if len(drawdown) > 0 else 0.0

    def _calculate_win_rate(self, data: np.ndarray) -> float:
        """Calcula o Win Rate."""
        returns = np.diff(data)
        wins = np.sum(returns > 0)
        return wins / len(returns) if len(returns) > 0 else 0.0

class MetricsCollector:
    def __init__(self):
        self.quantum_metrics = QuantumMetrics()
        self.market_metrics = MarketMetrics()
        self.performance_metrics = PerformanceMetrics()
        self.metrics_history = []

    def collect_metrics(self, data: np.ndarray) -> Dict:
        """Coleta todas as métricas."""
        metrics = {
            'timestamp': datetime.now(),
            'quantum': self.quantum_metrics.calculate_metrics(data),
            'market': self.market_metrics.calculate_metrics(data),
            'performance': self.performance_metrics.calculate_metrics(data)
        }
        self.metrics_history.append(metrics)
        return metrics

    def get_metrics_history(self) -> List[Dict]:
        """Retorna o histórico de métricas."""
        return self.metrics_history

__all__ = [
    'QuantumMetrics',
    'MarketMetrics',
    'PerformanceMetrics',
    'MetricsCollector'
] 