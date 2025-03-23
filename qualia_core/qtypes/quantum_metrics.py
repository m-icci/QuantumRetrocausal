"""
Base types for quantum metrics system
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
import numpy as np

@dataclass
class BaseQuantumMetric:
    """Base class for quantum metrics"""
    name: str
    value: float
    metadata: Optional[Dict] = None

    def __post_init__(self):
        if not 0 <= self.value <= 1:
            raise ValueError(f"{self.name} must be between 0 and 1")

@dataclass 
class MetricsConfig:
    """Configuration for quantum metrics calculation"""
    phi: float = (1 + np.sqrt(5)) / 2  # Golden ratio
    coherence_threshold: float = 0.8
    resonance_threshold: float = 0.7
    field_strength_threshold: float = 0.6

@dataclass
class QuantumMetrics:
    """
    Comprehensive quantum metrics collection
    """
    metrics: List[BaseQuantumMetric] = field(default_factory=list)

    def add_metric(self, metric: BaseQuantumMetric):
        """
        Add a new quantum metric to the collection

        Args:
            metric (BaseQuantumMetric): Metric to add
        """
        self.metrics.append(metric)

    def get_metric_by_name(self, name: str) -> Optional[BaseQuantumMetric]:
        """
        Retrieve a metric by its name

        Args:
            name (str): Name of the metric

        Returns:
            Optional[BaseQuantumMetric]: Metric if found, None otherwise
        """
        for metric in self.metrics:
            if metric.name == name:
                return metric
        return None

    def calculate_aggregate_metric(self) -> float:
        """
        Calculate an aggregate metric from all collected metrics

        Returns:
            float: Aggregate metric value between 0 and 1
        """
        if not self.metrics:
            return 0.0

        total_value = sum(metric.value for metric in self.metrics)
        return total_value / len(self.metrics)

    def analyze_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data and generate quantum metrics

        Args:
            market_data (Dict[str, Any]): Raw market data

        Returns:
            Dict[str, Any]: Analysis and quantum metrics
        """
        metrics = {
            'should_trade': False,
            'confidence': 0.0,
            'volatility': 0.0,
            'market_sentiment': 0.0,
            'quantum_coherence': 0.0
        }

        if not market_data:
            return metrics

        # Calculate volatility
        prices = [float(data['price']) for data in market_data.values()]
        metrics['volatility'] = np.std(prices) / np.mean(prices)

        # Calculate market sentiment
        price_changes = np.diff(prices)
        positive_changes = np.sum(price_changes > 0)
        total_changes = len(price_changes)
        metrics['market_sentiment'] = positive_changes / total_changes if total_changes > 0 else 0.5

        # Calculate quantum coherence
        metrics['quantum_coherence'] = 1 - metrics['volatility']

        # Determine if we should trade
        metrics['should_trade'] = (
            metrics['quantum_coherence'] > 0.7 and
            metrics['market_sentiment'] > 0.6 and
            metrics['volatility'] < 0.2
        )

        # Calculate confidence
        metrics['confidence'] = (
            metrics['quantum_coherence'] * 0.5 +
            metrics['market_sentiment'] * 0.3 +
            (1 - metrics['volatility']) * 0.2
        )

        return metrics