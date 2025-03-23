#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
neural_adapter.py - Adaptador Neural para QUALIA
--------------------------------------
Este módulo implementa um sistema neural adaptativo que otimiza 
os parâmetros de mineração com base em feedback e análise de padrões.

O adaptador neural:
1. Coleta métricas de desempenho da mineração
2. Analisa padrões temporais nos dados coletados
3. Ajusta dinamicamente os parâmetros para otimizar resultados
4. Utiliza memória de curto e longo prazo para refinamento contínuo

Versão completa: Integração com análise de métricas e ajuste adaptativo
"""

import time
import logging
import numpy as np
import math
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path

# Importação dos novos módulos de análise e ajuste
from .metrics_analyzer import MetricsAnalyzer
from .parameter_adjuster import ParameterAdjuster

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NeuralParameter:
    """
    Representa um parâmetro adaptativo neural com restrições e histórico
    """
    def __init__(
            self, 
            name: str, 
            value: float, 
            min_value: float, 
            max_value: float,
            learning_rate: float = 0.05,
            weight: float = 1.0
        ):
        """
        Inicializa um parâmetro neural adaptativo
        
        Args:
            name: Nome do parâmetro
            value: Valor inicial
            min_value: Valor mínimo permitido
            max_value: Valor máximo permitido
            learning_rate: Taxa de aprendizado para ajustes
            weight: Peso do parâmetro nas decisões de ajuste
        """
        self.name = name
        self.value = value
        self.min_value = min_value
        self.max_value = max_value
        self.learning_rate = learning_rate
        self.weight = weight
        
        # Histórico e métricas
        self.history = []  # [(timestamp, value, performance_metric)]
        self.last_update = time.time()
        self.update_count = 0
        self.best_value = value
        self.best_performance = 0.0
    
    def update(self, new_value: float, performance: float = None) -> float:
        """
        Atualiza o valor do parâmetro respeitando limites
        
        Args:
            new_value: Novo valor proposto
            performance: Métrica de desempenho associada (opcional)
            
        Returns:
            Valor ajustado (após aplicação de limites)
        """
        # Aplicar limites
        adjusted_value = max(min(new_value, self.max_value), self.min_value)
        
        # Registrar no histórico se performance foi fornecida
        now = time.time()
        if performance is not None:
            self.history.append((now, adjusted_value, performance))
            
            # Truncar histórico se ficar muito grande
            if len(self.history) > 1000:
                self.history = self.history[-1000:]
            
            # Atualizar melhor valor se desempenho melhorou
            if performance > self.best_performance:
                self.best_performance = performance
                self.best_value = adjusted_value
        
        # Atualizar estado
        self.value = adjusted_value
        self.last_update = now
        self.update_count += 1
        
        return adjusted_value
    
    def adjust_with_gradient(self, gradient: float, performance: float = None) -> float:
        """
        Ajusta o valor com base em gradiente e taxa de aprendizado
        
        Args:
            gradient: Gradiente para ajuste (-1 a 1)
            performance: Métrica de desempenho atual (opcional)
            
        Returns:
            Novo valor após ajuste
        """
        # Calcular incremento com base no gradiente e taxa de aprendizado
        range_size = self.max_value - self.min_value
        delta = gradient * self.learning_rate * range_size
        
        # Aplicar incremento ao valor atual
        new_value = self.value + delta
        
        # Atualizar com o novo valor
        return self.update(new_value, performance)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converte o parâmetro para dicionário
        
        Returns:
            Representação do parâmetro como dicionário
        """
        return {
            "name": self.name,
            "value": self.value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "learning_rate": self.learning_rate,
            "weight": self.weight,
            "update_count": self.update_count,
            "best_value": self.best_value,
            "best_performance": self.best_performance
        }


class QUALIANeuralAdapter:
    """
    Adaptador neural para otimização de parâmetros QUALIA
    
    Implementa um sistema neural adaptativo que ajusta parâmetros
    com base em observações de desempenho e padrões identificados.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa o adaptador neural
        
        Args:
            config: Configurações do adaptador
        """
        # Configuração padrão
        self.config = {
            "learning_rate": 0.05,
            "memory_window": 120,  # Janela de memória em segundos
            "adaptation_speed": 0.2,
            "exploration_rate": 0.1,
            "history_file": None,
            "enable_long_term_memory": True
        }
        
        # Atualizar com configuração fornecida
        if config:
            self.config.update(config)
        
        # Inicializar parâmetros adaptativos
        self.parameters = {}
        self._init_default_parameters()
        
        # Histórico de desempenho
        self.performance_history = []
        self.env_snapshots = []
        
        # Última atualização e contadores
        self.last_update = time.time()
        self.update_count = 0
        self.training_iterations = 0
        
        # Métricas
        self.metrics = {
            "best_performance": 0.0,
            "best_parameters": {},
            "improvement_rate": 0.0,
            "stability_index": 0.0,
            "last_gradients": {}
        }
        
        # Inicializar analisador de métricas e ajustador de parâmetros
        self.metrics_analyzer = MetricsAnalyzer(window_size=int(self.config["memory_window"] / 10))
        
        # Obter parâmetros para o ajustador
        initial_params = {name: param.value for name, param in self.parameters.items()}
        param_limits = {name: (param.min_value, param.max_value) for name, param in self.parameters.items()}
        
        self.parameter_adjuster = ParameterAdjuster(
            initial_params=initial_params,
            param_limits=param_limits,
            learning_rate=self.config["learning_rate"]
        )
        
        # Períodos de operação
        self.last_analysis_time = 0
        self.last_adaptation_time = 0
        self.analysis_interval = 10  # segundos
        self.adaptation_interval = 30  # segundos
        
        logger.info("QUALIA Neural Adapter inicializado com sistema integrado de análise e adaptação")

    def _init_default_parameters(self):
        """
        Inicializa os parâmetros adaptativos padrão
        """
        # Parâmetros fundamentais QUALIA
        self.add_parameter("coherence_target", 0.65, 0.3, 0.95)
        self.add_parameter("granularity", 21, 8, 144)
        self.add_parameter("field_energy", 0.7, 0.2, 1.0)
        self.add_parameter("mutation_rate", 0.03, 0.001, 0.2)
        
        # Parâmetros operacionais
        self.add_parameter("retrocausality_weight", 0.4, 0.0, 0.8)
        self.add_parameter("quantum_bias", 0.5, 0.0, 1.0)
        self.add_parameter("collapse_threshold", 0.7, 0.3, 0.9)
    
    def add_parameter(self, name: str, value: float, min_value: float, max_value: float, 
                     learning_rate: float = None, weight: float = 1.0):
        """
        Adiciona um parâmetro adaptativo ao sistema
        
        Args:
            name: Nome do parâmetro
            value: Valor inicial
            min_value: Valor mínimo permitido
            max_value: Valor máximo permitido
            learning_rate: Taxa de aprendizado específica (opcional)
            weight: Peso do parâmetro nas decisões
        """
        # Usar learning_rate global se não especificado
        if learning_rate is None:
            learning_rate = self.config["learning_rate"]
        
        # Criar parâmetro
        self.parameters[name] = NeuralParameter(
            name=name,
            value=value,
            min_value=min_value,
            max_value=max_value,
            learning_rate=learning_rate,
            weight=weight
        )
        
        logger.debug(f"Parâmetro adicionado: {name}={value} ({min_value}-{max_value})")
    
    def get_parameter(self, name: str) -> float:
        """
        Obtém o valor atual de um parâmetro
        
        Args:
            name: Nome do parâmetro
            
        Returns:
            Valor atual do parâmetro
        """
        if name not in self.parameters:
            logger.warning(f"Parâmetro não encontrado: {name}, usando 0.0")
            return 0.0
        
        return self.parameters[name].value
    
    def get_all_parameters(self) -> Dict[str, float]:
        """
        Obtém todos os valores de parâmetros atuais
        
        Returns:
            Dicionário com nome e valor de cada parâmetro
        """
        return {name: param.value for name, param in self.parameters.items()}
    
    def record_performance(self, metrics: Dict[str, Any], parameters: Dict[str, float] = None):
        """
        Registra dados de desempenho para análise e ajuste
        
        Args:
            metrics: Métricas de desempenho
            parameters: Parâmetros usados (se diferentes dos atuais)
        """
        timestamp = time.time()
        
        # Usar parâmetros atuais se não fornecidos
        if parameters is None:
            parameters = self.get_all_parameters()
        
        # Adicionar ao histórico
        record = {
            "timestamp": timestamp,
            "metrics": metrics,
            "parameters": parameters.copy()
        }
        self.performance_history.append(record)
        
        # Truncar histórico se ficar muito grande
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
        
        # Atualizar métricas se fornecida uma métrica de desempenho principal
        if "performance" in metrics:
            performance = metrics["performance"]
            
            # Atualizar melhor desempenho
            if performance > self.metrics["best_performance"]:
                self.metrics["best_performance"] = performance
                self.metrics["best_parameters"] = parameters.copy()
                logger.info(f"Novo melhor desempenho: {performance:.4f}")
    
    def capture_environment(self) -> Dict[str, Any]:
        """
        Captura dados do ambiente atual para análise
        
        Returns:
            Dados do ambiente de execução
        """
        # Placeholder para métricas de sistema
        # Em uma implementação completa, capturaria CPU, memória, etc.
        env_data = {
            "timestamp": time.time(),
            "cpu_load": 0.5,  # Simulado - em prod. usaria psutil
            "memory_usage": 0.4,  # Simulado - em prod. usaria psutil
            "process_count": 1,
            "parameters": self.get_all_parameters()
        }
        
        # Adicionar ao histórico
        self.env_snapshots.append(env_data)
        
        # Truncar se ficar muito grande
        if len(self.env_snapshots) > 100:
            self.env_snapshots = self.env_snapshots[-100:]
        
        return env_data
    
    def adapt(self, performance_metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Adapta parâmetros com base em métricas de desempenho
        
        Este é um método base que será expandido na Parte 3
        
        Args:
            performance_metrics: Métricas de desempenho atuais
            
        Returns:
            Novos valores de parâmetros
        """
        # Registrar desempenho
        self.record_performance(performance_metrics)
        
        # Capturar estado do ambiente
        env = self.capture_environment()
        
        # Computar gradientes simples para cada parâmetro
        # Isto será expandido na implementação completa
        gradients = self._compute_simple_gradients(performance_metrics)
        
        # Aplicar gradientes para ajustar parâmetros
        for param_name, gradient in gradients.items():
            if param_name in self.parameters:
                performance = performance_metrics.get("performance", None)
                self.parameters[param_name].adjust_with_gradient(gradient, performance)
        
        # Atualizar contadores
        self.update_count += 1
        self.last_update = time.time()
        
        # Retornar parâmetros atualizados
        return self.get_all_parameters()
    
    def _compute_simple_gradients(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Calcula gradientes simples para adaptação de parâmetros
        
        Implementação básica que será expandida na Parte 3
        
        Args:
            metrics: Métricas de desempenho atuais
            
        Returns:
            Gradientes para cada parâmetro
        """
        # Simples demonstração de gradientes para adaptação
        # Na implementação completa utilizaria análise de tendências
        
        # Extrair métricas principais
        hash_rate = metrics.get("hash_rate", 0)
        valid_shares = metrics.get("valid_shares", 0)
        hash_success_rate = metrics.get("success_rate", 0)
        coherence = metrics.get("coherence", 0.5)
        
        # Gradientes iniciais
        gradients = {
            "coherence_target": 0.0,
            "granularity": 0.0,
            "field_energy": 0.0,
            "mutation_rate": 0.0,
            "retrocausality_weight": 0.0,
            "quantum_bias": 0.0,
            "collapse_threshold": 0.0
        }
        
        # Exemplo simples: ajustar coerência com base no sucesso
        if hash_success_rate > 0.6:
            gradients["coherence_target"] = 0.05  # Aumentar coerência se taxa de sucesso alta
        else:
            gradients["coherence_target"] = -0.03  # Diminuir se baixa
        
        # Ajustar granularidade com base na taxa de hash
        # Exemplo muito simplificado - será expandido na implementação completa
        if "previous_hash_rate" in self.metrics:
            if hash_rate > self.metrics["previous_hash_rate"]:
                # Se hash rate melhorou, manter direção
                gradients["granularity"] = self.metrics.get("last_gradients", {}).get("granularity", 0.02)
            else:
                # Se piorou, inverter direção
                gradients["granularity"] = -self.metrics.get("last_gradients", {}).get("granularity", 0.02)
        
        # Armazenar para próxima iteração
        self.metrics["previous_hash_rate"] = hash_rate
        self.metrics["last_gradients"] = gradients
        
        return gradients
    
    def get_state(self) -> Dict[str, Any]:
        """
        Retorna o estado atual do adaptador neural
        
        Returns:
            Estado atual como dicionário
        """
        return {
            "parameters": {k: v.to_dict() for k, v in self.parameters.items()},
            "metrics": self.metrics,
            "update_count": self.update_count,
            "training_iterations": self.training_iterations,
            "last_update": self.last_update,
            "config": self.config
        }

    def update_metrics(self, metrics: Dict[str, Any]):
        """
        Atualiza o analisador de métricas com novos dados
        
        Args:
            metrics: Métricas de desempenho atuais
        """
        # Adicionar timestamp se não existir
        if "timestamp" not in metrics:
            metrics["timestamp"] = time.time()
        
        # Adicionar às métricas
        self.metrics_analyzer.add_metrics(metrics)
        
        # Registrar no histórico de desempenho
        self.performance_history.append((metrics.get("timestamp"), metrics))
        
        # Limitar tamanho do histórico
        max_history = int(self.config["memory_window"] * 10)  # 10 entradas por segundo no máximo
        if len(self.performance_history) > max_history:
            self.performance_history = self.performance_history[-max_history:]
        
        logger.debug(f"Métricas atualizadas: {metrics}")
    
    def analyze_and_adapt(self) -> Dict[str, Any]:
        """
        Analisa métricas e adapta parâmetros se necessário
        
        Returns:
            Parâmetros atuais após possível adaptação
        """
        current_time = time.time()
        
        # Verificar se é hora de analisar
        if current_time - self.last_analysis_time < self.analysis_interval:
            return {name: param.value for name, param in self.parameters.items()}
        
        # Analisar métricas
        self.last_analysis_time = current_time
        derived_metrics = self.metrics_analyzer.analyze()
        
        # Verificar se há dados suficientes
        if not derived_metrics or derived_metrics.get("status") == "sem_dados":
            logger.info("Dados insuficientes para análise")
            return {name: param.value for name, param in self.parameters.items()}
        
        # Atualizar métricas globais
        self.metrics.update({
            "stability_index": derived_metrics.get("stability_index", 0),
            "improvement_rate": derived_metrics.get("hash_rate_trend", 0)
        })
        
        # Verificar se é hora de adaptar
        if current_time - self.last_adaptation_time < self.adaptation_interval:
            return {name: param.value for name, param in self.parameters.items()}
        
        # Adaptar parâmetros
        self.last_adaptation_time = current_time
        suggestions = self.metrics_analyzer.suggest_improvements()
        
        if not suggestions:
            logger.info("Nenhuma sugestão de adaptação disponível")
            return {name: param.value for name, param in self.parameters.items()}
        
        # Obter métricas antes do ajuste
        metrics_before = derived_metrics.copy()
        
        # Ajustar parâmetros
        adjusted_params = self.parameter_adjuster.adjust_parameters(derived_metrics, suggestions)
        
        # Atualizar parâmetros neurais com os novos valores
        for name, value in adjusted_params.items():
            if name in self.parameters:
                performance = derived_metrics.get("avg_hash_rate", 0)
                self.parameters[name].update(value, performance)
        
        logger.info(f"Parâmetros adaptados: {adjusted_params}")
        
        # Programar avaliação de eficácia após adaptação
        self._schedule_efficacy_evaluation(metrics_before)
        
        return adjusted_params
    
    def _schedule_efficacy_evaluation(self, metrics_before: Dict[str, Any]):
        """
        Programa avaliação de eficácia para futuro próximo
        
        Args:
            metrics_before: Métricas antes do ajuste
        """
        # Implementação simples: armazenar métricas anteriores para avaliação futura
        self.env_snapshots.append(("pre_adjustment", time.time(), metrics_before))
        
        # Limitar número de snapshots
        if len(self.env_snapshots) > 20:
            self.env_snapshots = self.env_snapshots[-20:]
    
    def evaluate_efficacy(self):
        """
        Avalia a eficácia de ajustes anteriores
        """
        # Verificar se há snapshots para avaliar
        if not self.env_snapshots:
            return
        
        current_time = time.time()
        derived_metrics = self.metrics_analyzer.analyze()
        
        # Processar snapshots
        for idx, (snapshot_type, timestamp, metrics) in enumerate(self.env_snapshots[:]):
            # Verificar se o snapshot é de pré-ajuste e se já passou tempo suficiente
            if snapshot_type == "pre_adjustment" and current_time - timestamp > self.adaptation_interval:
                # Avaliar eficácia
                self.parameter_adjuster.evaluate_adjustment_efficacy(metrics, derived_metrics)
                
                # Remover snapshot processado
                self.env_snapshots.pop(idx)
                
                # Adaptar taxa de aprendizado com base na eficácia
                self.parameter_adjuster.adapt_learning_rate()
                
                logger.info("Avaliação de eficácia de ajuste realizada")
                break
    
    def get_derived_metrics(self) -> Dict[str, Any]:
        """
        Obtém métricas derivadas atuais
        
        Returns:
            Métricas derivadas
        """
        return self.metrics_analyzer.get_derived_metrics()
    
    def get_adjustment_statistics(self) -> Dict[str, Any]:
        """
        Obtém estatísticas de ajuste
        
        Returns:
            Estatísticas de ajuste
        """
        return self.parameter_adjuster.get_adjustment_statistics()
    
    def reset_adaptation(self):
        """
        Reinicia contadores de adaptação
        """
        self.parameter_adjuster.reset_counters()
        self.last_analysis_time = 0
        self.last_adaptation_time = 0

def demo_adapter():
    """Demonstração básica do adaptador neural"""
    adapter = QUALIANeuralAdapter()
    
    # Simular alguns ciclos de adaptação
    for i in range(5):
        # Métricas simuladas
        metrics = {
            "hash_rate": 100 + i * 10,
            "valid_shares": i,
            "success_rate": 0.5 + i * 0.1,
            "coherence": 0.6 + i * 0.05,
            "performance": 0.4 + i * 0.1
        }
        
        # Adaptar parâmetros
        new_params = adapter.adapt(metrics)
        
        # Exibir resultados
        print(f"\nCiclo {i+1}:")
        print(f"Métricas: {metrics}")
        print(f"Parâmetros adaptados: {new_params}")
    
    return adapter.get_state()

if __name__ == "__main__":
    # Testar adaptador neural
    state = demo_adapter()
    print("\nEstado final:")
    print(json.dumps(state["metrics"], indent=2))
