"""
Rastreador de Evolução Quântica YAA-ICCI.
Segue o mantra: INVESTIGAR → INTEGRAR → INOVAR

Este módulo preserva e analisa a evolução temporal dos estados quânticos,
mantendo um registro histórico das transformações e métricas do sistema.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
from collections import deque
import json
from pathlib import Path
from dataclasses import dataclass, field

from ..qtypes.quantum_state import QuantumState
from ..operators.base.quantum_operators import QuantumOperator
from ..measurement.quantum_measurement import QuantumMeasurement

logger = logging.getLogger(__name__)

@dataclass
class EvolutionRecord:
    """
    Registro detalhado de evolução quântica.
    Mantém informações sobre estados inicial e final, operador aplicado e medições.
    """
    initial_state: QuantumState
    final_state: QuantumState
    duration: float
    operator: Optional[QuantumOperator] = None
    measurements: Dict[str, QuantumMeasurement] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte registro para dicionário"""
        return {
            'initial_state': self.initial_state.state_vector.tolist(),
            'final_state': self.final_state.state_vector.tolist(),
            'duration': self.duration,
            'operator_name': self.operator.name if self.operator else None,
            'measurements': {k: m.to_dict() for k, m in self.measurements.items()},
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }

class QuantumEvolutionTracker:
    """
    Rastreia e analisa a evolução temporal de estados quânticos.
    Preserva o histórico e identifica padrões emergentes.
    """
    
    def __init__(self, history_size: int = 1000):
        """
        Inicializa o rastreador.
        
        Args:
            history_size: Tamanho máximo do histórico
        """
        # Configuração
        self.history_size = history_size
        self.metrics_dir = Path.home() / '.quantum' / 'metrics'
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Estado interno
        self.evolution_history: deque[EvolutionRecord] = deque(maxlen=history_size)
        self.metric_history = deque(maxlen=history_size)
        self.is_tracking = False
        self.start_time = None
        
        # Métricas
        self.current_metrics = {
            'coherence': 0.0,
            'entropy': 0.0,
            'complexity': 0.0,
            'stability': 0.0
        }
        
    def start_tracking(self):
        """Inicia rastreamento de evolução."""
        if self.is_tracking:
            logger.warning("Rastreamento já estava ativo")
            return
            
        self.is_tracking = True
        self.start_time = datetime.now()
        logger.info("Iniciando rastreamento de evolução quântica")
        
    def stop_tracking(self):
        """Finaliza rastreamento e salva métricas."""
        if not self.is_tracking:
            logger.warning("Rastreamento não estava ativo")
            return
            
        self.is_tracking = False
        self._save_metrics()
        logger.info("Rastreamento finalizado")
        
    def record_state(self, state: QuantumState):
        """
        Registra estado quântico no histórico.
        
        Args:
            state: Estado quântico atual
        """
        if not self.is_tracking:
            return
            
        # Registra estado
        self.evolution_history.append(EvolutionRecord(
            initial_state=state,
            final_state=state,
            duration=0.0
        ))
        
        # Atualiza métricas
        self._update_metrics()
        
    def get_metrics(self) -> Dict[str, float]:
        """
        Retorna métricas atuais do sistema.
        
        Returns:
            Dicionário com métricas
        """
        return self.current_metrics.copy()
        
    def _calculate_metrics(self, state: QuantumState) -> Dict[str, float]:
        """
        Calcula métricas para um estado.
        
        Args:
            state: Estado quântico
            
        Returns:
            Métricas calculadas
        """
        # Matriz densidade
        rho = np.outer(state.state_vector, state.state_vector.conj())
        
        # Coerência (pureza)
        coherence = np.abs(np.trace(np.matmul(rho, rho)))
        
        # Entropia de von Neumann
        eigenvalues = np.linalg.eigvalsh(rho)
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-10))
        
        # Complexidade (gradiente de amplitude)
        gradients = np.gradient(np.abs(state.state_vector))
        complexity = np.mean(np.abs(gradients))
        
        # Estabilidade (variação temporal)
        if len(self.evolution_history) > 1:
            prev_state = self.evolution_history[-1].initial_state
            stability = 1.0 - np.mean(np.abs(state.state_vector - prev_state.state_vector))
        else:
            stability = 1.0
            
        return {
            'coherence': float(coherence),
            'entropy': float(entropy),
            'complexity': float(complexity),
            'stability': float(stability)
        }
        
    def _update_metrics(self):
        """Atualiza métricas globais baseado no histórico."""
        if not self.evolution_history:
            return
            
        # Calcula médias móveis
        alpha = 0.1  # Taxa de atualização
        metrics = self._calculate_metrics(self.evolution_history[-1].final_state)
        
        for key in self.current_metrics:
            self.current_metrics[key] = (1 - alpha) * self.current_metrics[key] + \
                                      alpha * metrics[key]
                                      
        # Registra métricas
        self.metric_history.append({
            'timestamp': datetime.now(),
            'metrics': self.current_metrics.copy()
        })
        
    def _save_metrics(self):
        """Salva métricas em arquivo."""
        if not self.evolution_history:
            return
            
        # Prepara dados
        data = {
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'final_metrics': self.current_metrics,
            'history': [
                {
                    'timestamp': entry.timestamp.isoformat(),
                    'metrics': self._calculate_metrics(entry.final_state)
                }
                for entry in self.evolution_history
            ]
        }
        
        # Salva arquivo
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        metrics_file = self.metrics_dir / f'evolution_metrics_{timestamp}.json'
        
        try:
            metrics_file.write_text(json.dumps(data, indent=2))
            logger.info(f"Métricas salvas em: {metrics_file}")
        except Exception as e:
            logger.error(f"Erro ao salvar métricas: {e}")
            
    def analyze_trends(self) -> Dict[str, Any]:
        """
        Analisa tendências na evolução do sistema.
        
        Returns:
            Análise de tendências
        """
        if len(self.evolution_history) < 2:
            return {}
            
        # Extrai séries temporais
        times = []
        metrics = {k: [] for k in self.current_metrics}
        
        for entry in self.evolution_history:
            times.append((entry.timestamp - self.start_time).total_seconds())
            for k, v in self._calculate_metrics(entry.final_state).items():
                metrics[k].append(v)
                
        # Calcula tendências
        trends = {}
        for key, values in metrics.items():
            values = np.array(values)
            trends[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'trend': float(np.polyfit(times, values, deg=1)[0])
            }
            
        return trends
