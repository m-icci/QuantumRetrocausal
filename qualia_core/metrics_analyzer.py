#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
metrics_analyzer.py - Analisador de Métricas de Mineração
--------------------------------------
Este módulo implementa ferramentas para análise de métricas de mineração
que serão utilizadas pelo adaptador neural para otimização.

Funções principais:
1. Coleta e processamento de métricas de mineração
2. Detecção de tendências e padrões
3. Cálculo de métricas derivadas para tomada de decisão
"""

import time
import math
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsAnalyzer:
    """
    Analisador de métricas de mineração para otimização neural
    
    Responsável por processar métricas brutas e extrair insights
    que serão utilizados para ajuste adaptativo de parâmetros.
    """
    
    def __init__(self, window_size: int = 10):
        """
        Inicializa o analisador de métricas
        
        Args:
            window_size: Tamanho da janela de análise
        """
        self.window_size = window_size
        self.metrics_history = []
        self.derived_metrics = {}
        self.trends = {}
        self.last_analysis_time = 0
    
    def add_metrics(self, metrics: Dict[str, Any]):
        """
        Adiciona novas métricas à história
        
        Args:
            metrics: Métricas a serem adicionadas
        """
        # Adicionar timestamp se não existir
        if "timestamp" not in metrics:
            metrics["timestamp"] = time.time()
        
        # Adicionar ao histórico
        self.metrics_history.append(metrics)
        
        # Manter tamanho da janela
        if len(self.metrics_history) > self.window_size:
            self.metrics_history = self.metrics_history[-self.window_size:]
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analisa métricas atuais e extrai tendências
        
        Returns:
            Métricas derivadas e tendências
        """
        if not self.metrics_history:
            return {"status": "sem_dados"}
        
        # Processar apenas a cada 5 segundos no máximo
        current_time = time.time()
        if current_time - self.last_analysis_time < 5 and self.derived_metrics:
            return self.derived_metrics
        
        self.last_analysis_time = current_time
        
        # Extrair métricas principais
        self._extract_performance_metrics()
        self._analyze_trends()
        self._calculate_stability()
        
        # Retornar resultados
        return self.derived_metrics
    
    def _extract_performance_metrics(self):
        """
        Extrai métricas de desempenho dos dados históricos
        """
        # Garantir que temos dados suficientes
        if len(self.metrics_history) < 2:
            return
        
        # Extrair valores para análise
        timestamps = [m.get("timestamp", 0) for m in self.metrics_history]
        hash_rates = [m.get("hash_rate", 0) for m in self.metrics_history]
        valid_shares = [m.get("valid_shares", 0) for m in self.metrics_history]
        
        # Calcular média e tendência da taxa de hash
        avg_hash_rate = sum(hash_rates) / len(hash_rates) if hash_rates else 0
        
        # Calcular taxa de shares válidos
        total_shares = sum(valid_shares)
        shares_per_second = total_shares / (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0
        
        # Calcular métricas derivadas
        self.derived_metrics.update({
            "avg_hash_rate": avg_hash_rate,
            "shares_per_second": shares_per_second,
            "total_shares": total_shares,
            "time_window": timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
        })
    
    def _analyze_trends(self):
        """
        Analisa tendências nas métricas
        """
        if len(self.metrics_history) < 3:
            return
        
        # Extrair valores para análise de tendência
        hash_rates = [m.get("hash_rate", 0) for m in self.metrics_history]
        coherence = [m.get("coherence", 0) for m in self.metrics_history]
        
        # Calcular tendência linear simples
        hash_trend = self._calculate_trend(hash_rates)
        coherence_trend = self._calculate_trend(coherence)
        
        # Armazenar tendências
        self.trends = {
            "hash_rate": hash_trend,
            "coherence": coherence_trend
        }
        
        # Adicionar às métricas derivadas
        self.derived_metrics.update({
            "hash_rate_trend": hash_trend,
            "coherence_trend": coherence_trend
        })
    
    def _calculate_trend(self, values: List[float]) -> float:
        """
        Calcula tendência linear simples
        
        Args:
            values: Lista de valores
            
        Returns:
            Valor da tendência (-1 a 1)
        """
        if len(values) < 2:
            return 0
        
        # Abordagem simples: comparar média da primeira e segunda metade
        half = len(values) // 2
        first_half = values[:half]
        second_half = values[half:]
        
        avg_first = sum(first_half) / len(first_half) if first_half else 0
        avg_second = sum(second_half) / len(second_half) if second_half else 0
        
        # Evitar divisão por zero
        if avg_first == 0:
            return 1 if avg_second > 0 else 0
        
        # Normalizar para -1 a 1
        change = (avg_second - avg_first) / avg_first
        trend = max(min(change, 1.0), -1.0)
        
        return trend
    
    def _calculate_stability(self):
        """
        Calcula índice de estabilidade das métricas
        """
        if len(self.metrics_history) < 3:
            return
        
        # Extrair valores para análise de estabilidade
        hash_rates = [m.get("hash_rate", 0) for m in self.metrics_history]
        
        # Calcular coeficiente de variação (desvio padrão / média)
        mean = sum(hash_rates) / len(hash_rates) if hash_rates else 0
        if mean == 0:
            stability = 0
        else:
            variance = sum((x - mean) ** 2 for x in hash_rates) / len(hash_rates)
            std_dev = math.sqrt(variance)
            cv = std_dev / mean if mean != 0 else float('inf')
            
            # Converter para índice de estabilidade (0-1)
            stability = max(0, min(1, 1 - cv)) if cv != float('inf') else 0
        
        # Adicionar às métricas derivadas
        self.derived_metrics["stability_index"] = stability
    
    def get_derived_metrics(self) -> Dict[str, Any]:
        """
        Obtém métricas derivadas já calculadas
        
        Returns:
            Métricas derivadas
        """
        return self.derived_metrics
    
    def suggest_improvements(self) -> Dict[str, Any]:
        """
        Sugere melhorias com base nas análises
        
        Returns:
            Sugestões de melhorias para parâmetros
        """
        # Analisar se ainda não foi feito
        if not self.derived_metrics:
            self.analyze()
        
        suggestions = {}
        
        # Lógica para sugestões de melhoria
        hash_trend = self.derived_metrics.get("hash_rate_trend", 0)
        stability = self.derived_metrics.get("stability_index", 0)
        
        # Sugestão para coerência
        if hash_trend < -0.2:
            suggestions["coherence"] = "aumentar"
        elif hash_trend > 0.2 and stability < 0.5:
            suggestions["coherence"] = "manter"
        
        # Sugestão para granularidade
        if stability < 0.3:
            suggestions["granularity"] = "reduzir"
        elif stability > 0.8:
            suggestions["granularity"] = "aumentar"
        
        return suggestions


# Função para teste
def test_analyzer():
    """Testa o analisador de métricas"""
    
    analyzer = MetricsAnalyzer(window_size=5)
    
    # Adicionar algumas métricas de teste
    for i in range(5):
        metrics = {
            "timestamp": time.time() + i * 10,
            "hash_rate": 100 + i * 10 + random.randint(-5, 5),
            "valid_shares": i,
            "coherence": 0.5 + i * 0.05
        }
        analyzer.add_metrics(metrics)
        time.sleep(0.1)  # Pequena pausa para timestamps diferentes
    
    # Analisar métricas
    results = analyzer.analyze()
    
    print("Métricas derivadas:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    
    print("\nSugestões de melhoria:")
    suggestions = analyzer.suggest_improvements()
    for param, action in suggestions.items():
        print(f"  {param}: {action}")
    
    return results

if __name__ == "__main__":
    import random
    test_analyzer()
