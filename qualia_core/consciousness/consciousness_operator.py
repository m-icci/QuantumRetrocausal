"""
Quantum Consciousness Operator - Operador de Consciência Quântica

Este módulo implementa o operador de consciência quântica.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

from qualia_core.qtypes import (
    QuantumState,
    QuantumPattern,
    QuantumMetric,
    QuantumOperator
)

@dataclass
class ConsciousnessConfig:
    """Configuração do operador de consciência."""
    dimensions: int = 4
    num_qubits: int = 8
    temperature: float = 0.1
    entropy: float = 0.5
    coherence: float = 0.8
    entanglement: float = 0.6
    decoherence: float = 0.2
    emergence: float = 0.7
    retrocausality: float = 0.4
    holography: float = 0.5
    oscillation: float = 0.3
    adaptation: float = 0.6
    evolution: float = 0.4
    consciousness: float = 0.5

class QuantumConsciousnessOperator(QuantumOperator):
    """Operador de consciência quântica."""
    
    def __init__(
        self,
        config: Optional[ConsciousnessConfig] = None,
        metrics: Optional[List[QuantumMetric]] = None
    ):
        """Inicializa o operador de consciência.
        
        Args:
            config: Configuração do operador
            metrics: Métricas quânticas
        """
        # Inicializa com matriz identidade para dimensão padrão
        identity_matrix = np.eye(config.dimensions if config else 4, dtype=np.complex128)
        super().__init__(
            matrix=identity_matrix,
            dimension=config.dimensions if config else 4
        )
        
        self.config = config or ConsciousnessConfig()
        self.metrics = metrics or []
        self.logger = logging.getLogger(__name__)
        
        # Inicializa estado quântico
        self.state = QuantumState(
            amplitudes=np.ones(self.config.dimensions, dtype=np.complex128) / np.sqrt(self.config.dimensions),
            dimension=self.config.dimensions
        )
        
        # Inicializa padrões
        self.patterns: List[QuantumPattern] = []
        
        # Inicializa histórico
        self.history: List[Dict[str, Any]] = []
        self.timestamps: List[datetime] = []
        
        self.logger.info("QuantumConsciousnessOperator initialized")
    
    def update_state(self, new_state: QuantumState) -> None:
        """Atualiza o estado quântico.
        
        Args:
            new_state: Novo estado quântico
        """
        self.state = new_state
        self._update_metrics()
        self._update_patterns()
        self._update_history()
    
    def add_pattern(self, pattern: QuantumPattern) -> None:
        """Adiciona um padrão quântico.
        
        Args:
            pattern: Padrão quântico
        """
        self.patterns.append(pattern)
        self._update_metrics()
    
    def remove_pattern(self, pattern_id: str) -> None:
        """Remove um padrão quântico.
        
        Args:
            pattern_id: ID do padrão
        """
        self.patterns = [p for p in self.patterns if p.id != pattern_id]
        self._update_metrics()
    
    def get_pattern(self, pattern_id: str) -> Optional[QuantumPattern]:
        """Obtém um padrão quântico.
        
        Args:
            pattern_id: ID do padrão
            
        Returns:
            Padrão quântico ou None
        """
        for pattern in self.patterns:
            if pattern.id == pattern_id:
                return pattern
        return None
    
    def _update_metrics(self) -> None:
        """Atualiza as métricas quânticas."""
        # Atualiza métricas existentes ou adiciona novas
        coherence_metric = QuantumMetric(
            name="coherence",
            value=float(np.abs(np.vdot(self.state.amplitudes, self.state.amplitudes)))
        )
        self.metrics = [coherence_metric]  # Simplificado para exemplo
    
    def _update_patterns(self) -> None:
        """Atualiza os padrões quânticos."""
        for pattern in self.patterns:
            # Atualiza o padrão com o estado atual
            pattern.state = self.apply(pattern.state)
    
    def _update_history(self) -> None:
        """Atualiza o histórico."""
        self.history.append({
            'state': self.state,
            'patterns': self.patterns,
            'metrics': self.metrics
        })
        self.timestamps.append(datetime.now())
        
        # Mantém histórico limitado
        if len(self.history) > 1000:
            self.history.pop(0)
            self.timestamps.pop(0)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Obtém resumo das métricas.
        
        Returns:
            Resumo das métricas
        """
        return {
            metric.name: metric.value for metric in self.metrics
        }
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """Obtém resumo dos padrões.
        
        Returns:
            Resumo dos padrões
        """
        return {
            'num_patterns': len(self.patterns),
            'pattern_ids': [p.id for p in self.patterns],
            'pattern_metrics': [self.apply(p.state).coherence for p in self.patterns]
        }
    
    def get_history_summary(self) -> Dict[str, Any]:
        """Obtém resumo do histórico.
        
        Returns:
            Resumo do histórico
        """
        return {
            'num_entries': len(self.history),
            'time_range': (
                self.timestamps[0] if self.timestamps else None,
                self.timestamps[-1] if self.timestamps else None
            ),
            'metrics_trend': [
                {m.name: m.value for m in entry.get('metrics', [])}
                for entry in self.history[-10:] if 'metrics' in entry
            ]
        }
