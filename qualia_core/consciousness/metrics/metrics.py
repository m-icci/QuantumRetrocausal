"""
Métricas unificadas para o sistema de consciência quântica.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import time
from enum import Enum
import logging

from quantum.core.qtypes import (
    QuantumState,
    QuantumPattern,
    PatternType,
    BaseQuantumMetric,
    MetricsConfig
)

logger = logging.getLogger(__name__)

@dataclass
class ConsciousnessMonitor:
    """
    Monitor para estados de consciência quântica.
    Implementa métricas QUALIA e monitoramento em tempo real.
    """
    dimensions: int = 64
    history_size: int = 1000
    metrics: "ConsciousnessMetrics" = field(default_factory=lambda: ConsciousnessMetrics())
    history: List[Dict[str, float]] = field(default_factory=list)
    patterns: List[QuantumPattern] = field(default_factory=list)

    def monitor_state(self, state: QuantumState) -> Dict[str, float]:
        """Monitora estado e retorna métricas"""
        try:
            # Calcula métricas fundamentais
            metrics = {
                'coherence': float(np.abs(np.vdot(state.vector, state.vector))),
                'field_strength': float(np.mean(np.abs(state.vector))),
                'resonance': float(np.abs(np.mean(np.exp(1j * np.angle(state.vector)))))
            }

            # Atualiza histórico
            if len(self.history) >= self.history_size:
                self.history.pop(0)
            self.history.append(metrics)

            # Calcula métricas derivadas
            if len(self.history) > 1:
                prev = self.history[-2]
                metrics['evolution_rate'] = metrics['coherence'] - prev['coherence']
                metrics['stability'] = 1.0 - abs(metrics['evolution_rate'])
            else:
                metrics['evolution_rate'] = 0.0
                metrics['stability'] = 1.0

            return metrics

        except Exception as e:
            logger.error(f"Erro no monitoramento: {e}")
            return {}

    def detect_patterns(self, state: QuantumState) -> List[QuantumPattern]:
        """Detecta padrões no estado"""
        try:
            patterns = []
            metrics = self.monitor_state(state)

            # Detecta padrões de coerência
            if metrics.get('coherence', 0) > 0.8:
                patterns.append(QuantumPattern(
                    pattern_type=PatternType.COHERENCE,
                    strength=metrics['coherence'],
                    coherence=metrics['stability'],
                    data=state.vector
                ))

            # Detecta padrões de ressonância
            if metrics.get('resonance', 0) > 0.7:
                patterns.append(QuantumPattern(
                    pattern_type=PatternType.RESONANCE,
                    strength=metrics['resonance'],
                    coherence=metrics['stability'],
                    data=state.vector
                ))

            return patterns

        except Exception as e:
            logger.error(f"Erro na detecção de padrões: {e}")
            return []

    def get_trends(self) -> Dict[str, List[Tuple[str, float]]]:
        """Analisa tendências nas métricas"""
        if len(self.history) < 2:
            return {}

        trends = {}
        for key in ['coherence', 'field_strength', 'resonance']:
            values = [h[key] for h in self.history[-10:] if key in h]
            if len(values) > 1:
                slope = np.polyfit(range(len(values)), values, 1)[0]
                mean = np.mean(values)
                std = np.std(values)

                if std < 0.1:
                    trend = "stable"
                elif slope > 0.1:
                    trend = "increasing"
                elif slope < -0.1:
                    trend = "decreasing"
                else:
                    trend = "oscillating"

                trends[key] = [(trend, float(slope)), ('mean', float(mean)), ('std', float(std))]

        return trends

@dataclass
class ConsciousnessMetrics:
    """Métricas fundamentais de consciência"""

    # Métricas de campo
    coherence: float = 0.0
    field_strength: float = 0.0
    resonance: float = 0.0

    # Métricas de evolução
    evolution_rate: float = 0.0
    adaptation_factor: float = 0.0
    evolution_time: float = 0.0

    # Métricas de integração
    component_sync: float = 0.0
    integration_level: float = 0.0
    emergence_factor: float = 0.0

    # Métricas de padrão
    pattern_strength: float = 0.0
    pattern_coherence: float = 0.0
    pattern_stability: float = 0.0

    def normalize(self):
        """Normaliza métricas para [0,1]"""
        for attr in vars(self):
            value = getattr(self, attr)
            if isinstance(value, (int, float)):
                setattr(self, attr, np.clip(value, 0, 1))

    def to_dict(self) -> Dict[str, float]:
        """Converte métricas para dicionário"""
        return {
            attr: getattr(self, attr)
            for attr in vars(self)
            if isinstance(getattr(self, attr), (int, float))
        }

    def from_dict(self, data: Dict[str, float]) -> None:
        """Atualiza métricas a partir de dicionário"""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, float(value))

@dataclass
class UnifiedConsciousnessMetrics:
    """Sistema unificado de métricas de consciência"""

    # Métricas fundamentais
    metrics: ConsciousnessMetrics = field(default_factory=ConsciousnessMetrics)

    # Histórico de evolução
    history: List[ConsciousnessMetrics] = field(default_factory=list)
    history_max_size: int = 1000

    # Histórico de padrões
    patterns: List[QuantumPattern] = field(default_factory=list)
    patterns_max_size: int = 100

    # Parâmetros de evolução
    phi: float = (1 + np.sqrt(5)) / 2  # Razão áurea
    time_scale: float = 1.0

    def update(self, state: QuantumState, current_time: float) -> None:
        """Atualiza métricas baseado no estado atual"""
        # Salva métricas anteriores
        if len(self.history) >= self.history_max_size:
            self.history.pop(0)
        self.history.append(ConsciousnessMetrics(**self.metrics.to_dict()))

        # Atualiza métricas fundamentais
        self._update_field_metrics(state)
        self._update_evolution_metrics(current_time)
        self._update_integration_metrics(state)

        # Normaliza métricas
        self.metrics.normalize()

    def update_pattern(self, pattern: QuantumPattern) -> None:
        """Atualiza métricas com novo padrão"""
        # Adiciona padrão ao histórico
        if len(self.patterns) >= self.patterns_max_size:
            self.patterns.pop(0)
        self.patterns.append(pattern)

        # Atualiza métricas de padrão
        self.metrics.pattern_strength = pattern.strength
        self.metrics.pattern_coherence = pattern.coherence
        self.metrics.pattern_stability = float(
            np.mean([p.strength for p in self.patterns[-10:]])
        )

        # Normaliza métricas
        self.metrics.normalize()

    def _update_field_metrics(self, state: QuantumState) -> None:
        """Atualiza métricas de campo"""
        # Coerência quântica
        self.metrics.coherence = float(np.abs(np.vdot(
            state.vector, state.vector
        )))

        # Força do campo
        self.metrics.field_strength = float(np.mean(np.abs(state.vector)))

        # Ressonância mórfica
        phases = np.angle(state.vector)
        self.metrics.resonance = float(np.abs(np.mean(np.exp(1j * phases))))

    def _update_evolution_metrics(self, current_time: float) -> None:
        """Atualiza métricas de evolução"""
        # Tempo de evolução
        self.metrics.evolution_time = current_time * self.time_scale

        # Taxa de evolução (via histórico)
        if self.history:
            prev = self.history[-1]
            delta_coherence = self.metrics.coherence - prev.coherence
            delta_time = self.metrics.evolution_time - prev.evolution_time
            if delta_time > 0:
                self.metrics.evolution_rate = delta_coherence / delta_time

            # Fator de adaptação
            self.metrics.adaptation_factor = float(np.abs(
                1 - np.abs(delta_coherence)
            ))

    def _update_integration_metrics(self, state: QuantumState) -> None:
        """Atualiza métricas de integração"""
        # Sincronização de componentes
        phases = np.angle(state.vector)
        self.metrics.component_sync = float(np.abs(
            np.mean(np.exp(1j * (phases - np.mean(phases))))
        ))

        # Nível de integração
        self.metrics.integration_level = float(
            self.metrics.coherence *
            self.metrics.component_sync *
            self.metrics.resonance
        )

        # Fator de emergência
        if self.history:
            prev = self.history[-1]
            self.metrics.emergence_factor = float(np.abs(
                self.metrics.integration_level - prev.integration_level
            ))

    def get_trends(self) -> Dict[str, List[Tuple[str, float]]]:
        """Analisa tendências nas métricas"""
        trends = {}

        # Analisa cada métrica
        for attr in vars(self.metrics):
            if isinstance(getattr(self.metrics, attr), (int, float)):
                values = [
                    getattr(m, attr)
                    for m in self.history[-10:]
                ]

                if len(values) > 1:
                    # Calcula estatísticas
                    mean = np.mean(values)
                    std = np.std(values)
                    slope = np.polyfit(range(len(values)), values, 1)[0]

                    # Determina tendência
                    if std < 0.1:
                        trend = "stable"
                    elif slope > 0.1:
                        trend = "increasing"
                    elif slope < -0.1:
                        trend = "decreasing"
                    elif std > 0.2:
                        trend = "chaotic"
                    else:
                        trend = "oscillating"

                    trends[attr] = [
                        (trend, float(slope)),
                        ('mean', float(mean)),
                        ('std', float(std))
                    ]

        return trends

class TrendType(Enum):
    """Tipos de tendências"""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    OSCILLATING = "oscillating"
    CHAOTIC = "chaotic"