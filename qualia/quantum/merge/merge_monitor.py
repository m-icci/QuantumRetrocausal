"""
QUALIA Quantum Merge System - Monitor de Merge
--------------------------------------------

Este módulo implementa o sistema de monitoramento para operações de
merge quântico, incluindo métricas e detecção de anomalias.
"""

import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
from prometheus_client import start_http_server, Gauge, Counter, Histogram
import threading
import time

from ...utils.logging import setup_logger
from ...core.metrics import MetricsCollector
from ...core.anomaly_detection import AnomalyDetector

class QuantumMergeMonitor:
    """
    Monitor avançado para o QuantumMergeSimulator integrado com QUALIA
    """
    def __init__(self, port: int = 8000):
        """
        Inicializa o monitor com métricas Prometheus e integração QUALIA

        Args:
            port: Porta para o servidor Prometheus
        """
        # Configurar logger
        self.logger = setup_logger('quantum_merge_monitor')
        
        # Inicializar coletor de métricas QUALIA
        self.metrics_collector = MetricsCollector()
        
        # Métricas Prometheus
        self._setup_prometheus_metrics(port)
        
        # Histórico de métricas
        self.metrics_history: List[Dict[str, float]] = []
        
        # Detector de anomalias
        self.anomaly_detector = AnomalyDetector()
    
    def _setup_prometheus_metrics(self, port: int):
        """Configura métricas Prometheus"""
        # Métricas de coerência
        self.quantum_coherence = Gauge('quantum_coherence', 
            'Coerência quântica atual do sistema')
        self.phase_coherence = Gauge('phase_coherence',
            'Coerência de fase entre estados')
        
        # Métricas de entropia
        self.system_entropy = Gauge('system_entropy',
            'Entropia total do sistema')
        self.merge_entropy = Gauge('merge_entropy',
            'Entropia durante operações de merge')
        
        # Contadores de eventos
        self.merge_attempts = Counter('merge_attempts',
            'Número total de tentativas de merge')
        self.successful_merges = Counter('successful_merges',
            'Número de merges bem-sucedidos')
        self.failed_merges = Counter('failed_merges',
            'Número de merges falhos')
        
        # Histogramas
        self.merge_duration = Histogram('merge_duration_seconds',
            'Duração das operações de merge',
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0))
        
        # Iniciar servidor em thread separada
        self.monitoring_thread = threading.Thread(
            target=self._start_monitoring_server,
            args=(port,)
        )
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
    
    def _start_monitoring_server(self, port: int):
        """Inicia servidor Prometheus"""
        start_http_server(port)
        self.logger.info(f"Servidor de monitoramento iniciado na porta {port}")
    
    def update_metrics(self, metrics: Dict[str, float]):
        """
        Atualiza métricas do sistema e QUALIA

        Args:
            metrics: Dicionário com métricas atualizadas
        """
        # Atualizar métricas Prometheus
        self.quantum_coherence.set(metrics.get('coherence', 0))
        self.phase_coherence.set(metrics.get('phase_coherence', 0))
        self.system_entropy.set(metrics.get('entropy', 0))
        self.merge_entropy.set(metrics.get('merge_entropy', 0))
        
        # Atualizar métricas QUALIA
        self.metrics_collector.update(metrics)
        
        # Registrar no histórico
        metrics['timestamp'] = datetime.now().timestamp()
        self.metrics_history.append(metrics)
        
        # Log de métricas importantes
        self.logger.info(
            f"Métricas atualizadas - Coerência: {metrics.get('coherence', 0):.3f}, "
            f"Entropia: {metrics.get('entropy', 0):.3f}"
        )
    
    def record_merge_attempt(self, success: bool, duration: float):
        """
        Registra tentativa de merge

        Args:
            success: Se o merge foi bem-sucedido
            duration: Duração da operação em segundos
        """
        self.merge_attempts.inc()
        if success:
            self.successful_merges.inc()
        else:
            self.failed_merges.inc()
        self.merge_duration.observe(duration)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Gera resumo estatístico das métricas

        Returns:
            Dicionário com estatísticas das métricas
        """
        if not self.metrics_history:
            return {}
        
        df = pd.DataFrame(self.metrics_history)
        summary = {
            'coherence_mean': df['coherence'].mean(),
            'coherence_std': df['coherence'].std(),
            'entropy_mean': df['entropy'].mean(),
            'entropy_std': df['entropy'].std(),
            'success_rate': (
                self.successful_merges._value /
                max(1, self.merge_attempts._value)
            )
        }
        
        # Adicionar métricas QUALIA
        summary.update(self.metrics_collector.get_summary())
        
        return summary
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """
        Detecta anomalias nas métricas usando sistema QUALIA

        Returns:
            Lista de anomalias detectadas
        """
        if len(self.metrics_history) < 10:
            return []
        
        df = pd.DataFrame(self.metrics_history)
        return self.anomaly_detector.detect(df)
