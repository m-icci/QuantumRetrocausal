"""
Coletor de Métricas QUALIA
--------------------------
Implementa um sistema de coleta e análise de métricas para o ambiente QUALIA
com suporte para múltiplas dimensões quânticas e retrocausalidade.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import json
import os

# Configuração do logger
logger = logging.getLogger(__name__)

@dataclass
class QuantumMetric:
    """Representação de uma métrica quântica"""
    name: str
    value: float
    timestamp: str
    dimension: int
    coherence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário"""
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp,
            'dimension': self.dimension,
            'coherence': self.coherence,
            'metadata': self.metadata or {}
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuantumMetric':
        """Cria a partir de um dicionário"""
        return cls(
            name=data['name'],
            value=data['value'],
            timestamp=data['timestamp'],
            dimension=data['dimension'],
            coherence=data.get('coherence', 1.0),
            metadata=data.get('metadata', {})
        )


class MetricsCollector:
    """
    Coletor de métricas para sistema QUALIA
    
    Responsável por:
    1. Coletar métricas de diferentes componentes
    2. Armazenar histórico de métricas
    3. Calcular estatísticas e tendências
    4. Prever comportamento futuro (retrocausalidade)
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Inicializa o coletor de métricas
        
        Args:
            storage_path: Caminho para armazenamento de métricas (opcional)
        """
        self.metrics_history: Dict[str, List[QuantumMetric]] = {}
        self.last_update = datetime.now()
        self.coherence_threshold = 0.85
        self.storage_path = storage_path
        
        # Cache dimensional para análise retrocausal
        self.dimensional_cache: Dict[int, Dict[str, List[float]]] = {}
        
        logger.info("Coletor de Métricas QUALIA inicializado")
        
    def add_metric(self, 
                  name: str, 
                  value: float, 
                  dimension: int, 
                  coherence: float = 1.0,
                  metadata: Optional[Dict[str, Any]] = None) -> QuantumMetric:
        """
        Adiciona uma nova métrica ao histórico
        
        Args:
            name: Nome da métrica
            value: Valor numérico
            dimension: Dimensão quântica associada
            coherence: Nível de coerência (0-1)
            metadata: Metadados adicionais
            
        Returns:
            A métrica registrada
        """
        timestamp = datetime.now().isoformat()
        
        metric = QuantumMetric(
            name=name,
            value=value,
            timestamp=timestamp,
            dimension=dimension,
            coherence=coherence,
            metadata=metadata or {}
        )
        
        if name not in self.metrics_history:
            self.metrics_history[name] = []
            
        self.metrics_history[name].append(metric)
        
        # Atualiza o cache dimensional
        if dimension not in self.dimensional_cache:
            self.dimensional_cache[dimension] = {}
        
        if name not in self.dimensional_cache[dimension]:
            self.dimensional_cache[dimension][name] = []
            
        self.dimensional_cache[dimension][name].append(value)
        
        # Atualiza timestamp
        self.last_update = datetime.now()
        
        return metric
    
    def get_metrics(self, name: str, limit: int = 100) -> List[QuantumMetric]:
        """
        Obtém métricas pelo nome
        
        Args:
            name: Nome da métrica
            limit: Número máximo de registros
            
        Returns:
            Lista de métricas
        """
        if name not in self.metrics_history:
            return []
            
        return self.metrics_history[name][-limit:]
    
    def get_average(self, name: str, dimension: Optional[int] = None) -> float:
        """
        Calcula a média de uma métrica
        
        Args:
            name: Nome da métrica
            dimension: Filtrar por dimensão (opcional)
            
        Returns:
            Média do valor
        """
        if name not in self.metrics_history:
            return 0.0
            
        metrics = self.metrics_history[name]
        
        if dimension is not None:
            metrics = [m for m in metrics if m.dimension == dimension]
            
        if not metrics:
            return 0.0
            
        return sum(m.value for m in metrics) / len(metrics)
    
    def calculate_trend(self, name: str, window: int = 10) -> float:
        """
        Calcula a tendência (taxa de variação) para uma métrica
        
        Args:
            name: Nome da métrica
            window: Tamanho da janela para cálculo
            
        Returns:
            Taxa percentual de variação (-1 a 1)
        """
        if name not in self.metrics_history or len(self.metrics_history[name]) < window:
            return 0.0
            
        recent = self.metrics_history[name][-window:]
        
        if len(recent) < 2:
            return 0.0
            
        first_value = recent[0].value
        last_value = recent[-1].value
        
        if first_value == 0:
            return 0.0
            
        return (last_value - first_value) / abs(first_value)
    
    def predict_future_value(self, name: str, steps_ahead: int = 1) -> float:
        """
        Implementa predição retrocausal para estimar valor futuro
        
        Args:
            name: Nome da métrica
            steps_ahead: Passos à frente para prever
            
        Returns:
            Valor previsto
        """
        if name not in self.metrics_history or len(self.metrics_history[name]) < 3:
            return 0.0
            
        # Implementação simples com média móvel ponderada
        values = [m.value for m in self.metrics_history[name][-5:]]
        weights = np.array([0.1, 0.15, 0.2, 0.25, 0.3])  # Pesos crescentes para valores mais recentes
        
        # Ajusta tamanhos se necessário
        if len(values) < len(weights):
            weights = weights[-len(values):]
            weights = weights / weights.sum()  # Renormaliza
            
        # Calcula média ponderada e adiciona tendência
        prediction = np.average(values, weights=weights)
        trend = self.calculate_trend(name)
        
        # Aplica tendência para cada passo futuro com atenuação
        prediction = prediction * (1 + trend * steps_ahead * 0.5)
        
        return prediction
    
    def get_dimensional_metrics(self, dimension: int) -> Dict[str, List[float]]:
        """
        Obtém todas as métricas para uma dimensão específica
        
        Args:
            dimension: Dimensão quântica
            
        Returns:
            Dicionário com métricas por nome
        """
        return self.dimensional_cache.get(dimension, {})
    
    def analyze_dimensional_coherence(self, dimension: int) -> float:
        """
        Analisa a coerência dos dados para uma dimensão
        
        Args:
            dimension: Dimensão quântica
            
        Returns:
            Valor de coerência (0-1)
        """
        if dimension not in self.dimensional_cache:
            return 0.0
            
        # Calcula coerência baseada na volatilidade dos dados
        coherence_values = []
        
        for metric_name, values in self.dimensional_cache[dimension].items():
            if len(values) < 2:
                continue
                
            # Calcula coeficiente de variação como medida de volatilidade
            std_dev = np.std(values)
            mean = np.mean(values)
            
            if mean == 0:
                continue
                
            cv = std_dev / abs(mean)
            
            # Converte em valor de coerência (menor volatilidade = maior coerência)
            coherence = max(0.0, min(1.0, 1.0 - min(cv, 1.0)))
            coherence_values.append(coherence)
            
        if not coherence_values:
            return self.coherence_threshold
            
        return sum(coherence_values) / len(coherence_values)
    
    def save_metrics(self, filepath: Optional[str] = None) -> bool:
        """
        Salva as métricas em arquivo JSON
        
        Args:
            filepath: Caminho do arquivo (opcional)
            
        Returns:
            Sucesso da operação
        """
        if not filepath and not self.storage_path:
            logger.warning("Caminho para armazenamento não fornecido")
            return False
            
        target_path = filepath or os.path.join(
            self.storage_path, 
            f"qualia_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        try:
            # Converte para formato serializável
            serialized = {}
            for name, metrics in self.metrics_history.items():
                serialized[name] = [m.to_dict() for m in metrics]
                
            with open(target_path, 'w') as f:
                json.dump(serialized, f, indent=2)
                
            logger.info(f"Métricas salvas em: {target_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao salvar métricas: {str(e)}")
            return False
    
    def load_metrics(self, filepath: str) -> bool:
        """
        Carrega métricas de um arquivo JSON
        
        Args:
            filepath: Caminho do arquivo
            
        Returns:
            Sucesso da operação
        """
        if not os.path.exists(filepath):
            logger.error(f"Arquivo não encontrado: {filepath}")
            return False
            
        try:
            with open(filepath, 'r') as f:
                serialized = json.load(f)
                
            # Deserializa métricas
            for name, metrics_data in serialized.items():
                self.metrics_history[name] = [
                    QuantumMetric.from_dict(data) for data in metrics_data
                ]
                
                # Atualiza cache dimensional
                for metric in self.metrics_history[name]:
                    dim = metric.dimension
                    if dim not in self.dimensional_cache:
                        self.dimensional_cache[dim] = {}
                    
                    if name not in self.dimensional_cache[dim]:
                        self.dimensional_cache[dim][name] = []
                        
                    self.dimensional_cache[dim][name].append(metric.value)
                    
            logger.info(f"Métricas carregadas de: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar métricas: {str(e)}")
            return False
