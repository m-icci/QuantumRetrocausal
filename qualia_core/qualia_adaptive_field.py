#!/usr/bin/env python3
"""
Campo de Consciência Dinâmico para Mineração QUALIA

Este módulo implementa uma camada adaptativa que monitora o comportamento da rede
e otimiza as estratégias de mineração em tempo real, baseado nos princípios do
modelo QUALIA de superposição e colapso direcionado.

O Campo de Consciência Dinâmico atua como uma entidade emergente que:
1. Observa padrões em múltiplas dimensões (tempo, espaço de rede, performance)
2. Adapta-se dinamicamente às condições observadas
3. Influencia o comportamento do sistema de mineração para otimizar resultados
4. Mantém uma "memória" de estados e transições para aprendizado contínuo

Autor: QUALIA Mining Team
Data: 2025-03-17
"""

import time
import logging
import threading
import numpy as np
import random
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from collections import deque
import scipy.stats as stats
from datetime import datetime

# Configuração de logging
logger = logging.getLogger("qualia.adaptive_field")

class AdaptiveMetric:
    """
    Classe para armazenar e analisar métricas adaptativas.
    Mantém um histórico temporal e realiza análises estatísticas.
    """
    
    def __init__(self, name: str, window_size: int = 100, 
                smoothing_factor: float = 0.3, threshold_sensitivity: float = 1.5):
        """
        Inicializa uma métrica adaptativa.
        
        Args:
            name: Nome da métrica
            window_size: Tamanho da janela de observação
            smoothing_factor: Fator de suavização para média móvel exponencial
            threshold_sensitivity: Sensibilidade para detecção de anomalias
        """
        self.name = name
        self.window_size = window_size
        self.smoothing_factor = smoothing_factor
        self.threshold_sensitivity = threshold_sensitivity
        
        # Histórico e estatísticas
        self.values = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        self.ema_value = None  # Média móvel exponencial
        self.std_dev = None    # Desvio padrão
        self.min_value = None  # Valor mínimo
        self.max_value = None  # Valor máximo
        self.trend = 0.0       # Tendência (positiva, negativa, neutra)
        
        # Limites adaptativos
        self.upper_threshold = None
        self.lower_threshold = None
        
        # Metadados
        self.last_anomaly_time = None
        self.anomaly_count = 0
        self.last_update_time = None
    
    def add_value(self, value: float, timestamp: Optional[float] = None):
        """
        Adiciona um novo valor à métrica e atualiza estatísticas.
        
        Args:
            value: Novo valor a ser adicionado
            timestamp: Timestamp do valor (se None, usa o tempo atual)
        """
        if timestamp is None:
            timestamp = time.time()
            
        # Adiciona valor ao histórico
        self.values.append(value)
        self.timestamps.append(timestamp)
        
        # Atualiza estatísticas
        self._update_statistics()
        
        # Detecta anomalias
        is_anomaly = self._detect_anomaly(value)
        if is_anomaly:
            self.anomaly_count += 1
            self.last_anomaly_time = timestamp
            
        self.last_update_time = timestamp
        return is_anomaly
    
    def _update_statistics(self):
        """Atualiza todas as estatísticas da métrica."""
        if not self.values:
            return
            
        # Calcula média móvel exponencial
        if self.ema_value is None:
            self.ema_value = self.values[-1]
        else:
            self.ema_value = (self.smoothing_factor * self.values[-1] + 
                             (1 - self.smoothing_factor) * self.ema_value)
        
        # Atualiza estatísticas básicas
        values_array = np.array(self.values)
        self.std_dev = np.std(values_array)
        self.min_value = np.min(values_array)
        self.max_value = np.max(values_array)
        
        # Calcula tendência usando regressão linear se tiver pelo menos 3 pontos
        if len(self.values) >= 3:
            x = np.array(range(len(self.values)))
            y = np.array(self.values)
            slope, _, _, _, _ = stats.linregress(x, y)
            self.trend = slope
            
        # Atualiza limites adaptativos
        self.upper_threshold = self.ema_value + (self.threshold_sensitivity * self.std_dev)
        self.lower_threshold = self.ema_value - (self.threshold_sensitivity * self.std_dev)
    
    def _detect_anomaly(self, value: float) -> bool:
        """
        Detecta se um valor é uma anomalia com base nos limites adaptativos.
        
        Args:
            value: Valor a ser verificado
            
        Returns:
            True se for uma anomalia, False caso contrário
        """
        if self.upper_threshold is None or self.lower_threshold is None:
            return False
            
        return value > self.upper_threshold or value < self.lower_threshold
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Retorna um resumo das estatísticas atuais da métrica.
        
        Returns:
            Dicionário com as estatísticas
        """
        return {
            "name": self.name,
            "current_value": self.values[-1] if self.values else None,
            "ema_value": self.ema_value,
            "std_dev": self.std_dev,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "trend": self.trend,
            "upper_threshold": self.upper_threshold,
            "lower_threshold": self.lower_threshold,
            "anomaly_count": self.anomaly_count,
            "sample_count": len(self.values)
        }


class FieldNode:
    """
    Representa um nó no Campo de Consciência Dinâmico.
    Cada nó monitora métricas específicas e pode influenciar partes do sistema.
    """
    
    def __init__(self, name: str, node_type: str, 
                metrics_config: List[Dict[str, Any]] = None,
                influence_weight: float = 1.0):
        """
        Inicializa um nó do campo.
        
        Args:
            name: Nome único do nó
            node_type: Tipo do nó (ex: 'pool', 'network', 'miner')
            metrics_config: Configuração das métricas a serem monitoradas
            influence_weight: Peso da influência deste nó no campo
        """
        self.name = name
        self.node_type = node_type
        self.influence_weight = influence_weight
        self.creation_time = time.time()
        self.last_activity = self.creation_time
        
        # Inicializa métricas
        self.metrics: Dict[str, AdaptiveMetric] = {}
        if metrics_config:
            for config in metrics_config:
                metric_name = config.get("name")
                window_size = config.get("window_size", 100)
                smoothing_factor = config.get("smoothing_factor", 0.3)
                threshold_sensitivity = config.get("threshold_sensitivity", 1.5)
                
                self.add_metric(metric_name, window_size, smoothing_factor, threshold_sensitivity)
        
        # Estado e metadados
        self.state = "active"  # active, passive, dormant
        self.connections = []  # Conexões com outros nós
        self.awareness_level = 0.5  # Nível de "consciência" (0.0 a 1.0)
        self.metadata = {}  # Metadados adicionais
    
    def add_metric(self, name: str, window_size: int = 100, 
                  smoothing_factor: float = 0.3, threshold_sensitivity: float = 1.5) -> AdaptiveMetric:
        """
        Adiciona uma nova métrica a este nó.
        
        Args:
            name: Nome da métrica
            window_size: Tamanho da janela de observação
            smoothing_factor: Fator de suavização
            threshold_sensitivity: Sensibilidade a anomalias
            
        Returns:
            Métrica criada
        """
        metric = AdaptiveMetric(
            name=name, 
            window_size=window_size,
            smoothing_factor=smoothing_factor,
            threshold_sensitivity=threshold_sensitivity
        )
        self.metrics[name] = metric
        return metric
    
    def update_metric(self, metric_name: str, value: float) -> bool:
        """
        Atualiza o valor de uma métrica e retorna se foi detectada anomalia.
        
        Args:
            metric_name: Nome da métrica
            value: Novo valor
            
        Returns:
            True se uma anomalia foi detectada
        """
        if metric_name not in self.metrics:
            raise ValueError(f"Métrica '{metric_name}' não existe neste nó")
            
        self.last_activity = time.time()
        is_anomaly = self.metrics[metric_name].add_value(value)
        
        # Aumenta o nível de consciência se detectar anomalia
        if is_anomaly:
            self.awareness_level = min(1.0, self.awareness_level + 0.1)
        else:
            # Diminui gradualmente o nível de consciência se tudo estiver normal
            self.awareness_level = max(0.1, self.awareness_level * 0.99)
            
        return is_anomaly
    
    def get_state_vector(self) -> Dict[str, Any]:
        """
        Retorna um vetor do estado atual deste nó.
        
        Returns:
            Dicionário representando o estado do nó
        """
        metrics_summary = {name: metric.get_summary() 
                          for name, metric in self.metrics.items()}
        
        return {
            "name": self.name,
            "type": self.node_type,
            "awareness_level": self.awareness_level,
            "state": self.state,
            "influence_weight": self.influence_weight,
            "age": time.time() - self.creation_time,
            "last_activity_delta": time.time() - self.last_activity,
            "metrics": metrics_summary
        }
    
    def connect_to(self, other_node: 'FieldNode', connection_strength: float = 1.0):
        """
        Estabelece uma conexão direcional com outro nó.
        
        Args:
            other_node: Nó alvo para a conexão
            connection_strength: Força da conexão
        """
        self.connections.append({
            "target": other_node,
            "strength": connection_strength,
            "established_at": time.time()
        })


class QualiaAdaptiveField:
    """
    Campo de Consciência Dinâmico para mineração QUALIA.
    
    Esta classe representa o campo adaptativo completo, composto por múltiplos
    nós que monitoram diferentes aspectos do sistema de mineração e influenciam
    seu comportamento de forma emergente.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa o Campo de Consciência Dinâmico.
        
        Args:
            config_path: Caminho opcional para arquivo de configuração JSON
        """
        self.nodes: Dict[str, FieldNode] = {}
        self.last_update_time = time.time()
        self.creation_time = self.last_update_time
        self.field_state = "initializing"  # initializing, active, learning, optimizing
        
        # Estado global do campo
        self.global_awareness = 0.5
        self.field_cycles = 0
        self.anomaly_counter = 0
        self.optimization_counter = 0
        
        # Carrega configuração se fornecida
        self.config = {}
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            except Exception as e:
                logger.error(f"Erro ao carregar configuração: {e}")
                self.config = {}
        
        # Inicializa nós padrão se não houver configuração
        if not self.config:
            self._initialize_default_nodes()
        else:
            self._initialize_from_config()
            
        logger.info(f"Campo de Consciência Dinâmico inicializado com {len(self.nodes)} nós")
        self.field_state = "active"
    
    def _initialize_default_nodes(self):
        """Inicializa os nós padrão do campo."""
        # Nó de rede para monitoramento de conexões
        network_node = self._create_node(
            name="network_monitor",
            node_type="network",
            metrics_config=[
                {"name": "latency", "window_size": 100},
                {"name": "connection_stability", "window_size": 50},
                {"name": "bandwidth", "window_size": 30}
            ],
            influence_weight=1.2
        )
        
        # Nó de pool para monitoramento de pools
        pool_node = self._create_node(
            name="pool_monitor",
            node_type="pool",
            metrics_config=[
                {"name": "response_time", "window_size": 100},
                {"name": "share_acceptance", "window_size": 200},
                {"name": "pool_stability", "window_size": 50}
            ],
            influence_weight=1.5
        )
        
        # Nó de desempenho para monitoramento de performance
        performance_node = self._create_node(
            name="performance_monitor",
            node_type="performance",
            metrics_config=[
                {"name": "hash_rate", "window_size": 50},
                {"name": "energy_efficiency", "window_size": 100},
                {"name": "optimization_gain", "window_size": 30}
            ],
            influence_weight=1.3
        )
        
        # Nó de adaptação para ajustes dinâmicos
        adaptation_node = self._create_node(
            name="adaptation_controller",
            node_type="controller",
            metrics_config=[
                {"name": "strategy_effectiveness", "window_size": 50},
                {"name": "adaptation_rate", "window_size": 30}
            ],
            influence_weight=1.8
        )
        
        # Estabelece conexões iniciais entre nós
        network_node.connect_to(pool_node, 0.8)
        pool_node.connect_to(performance_node, 0.9)
        performance_node.connect_to(adaptation_node, 1.0)
        adaptation_node.connect_to(network_node, 0.7)
    
    def _initialize_from_config(self):
        """Inicializa nós a partir da configuração carregada."""
        if "nodes" not in self.config:
            logger.warning("Configuração não contém definição de nós")
            self._initialize_default_nodes()
            return
            
        # Cria nós definidos na configuração
        for node_config in self.config["nodes"]:
            self._create_node(
                name=node_config.get("name", f"node_{len(self.nodes)}"),
                node_type=node_config.get("type", "generic"),
                metrics_config=node_config.get("metrics", []),
                influence_weight=node_config.get("influence", 1.0)
            )
            
        # Estabelece conexões definidas na configuração
        if "connections" in self.config:
            for conn in self.config["connections"]:
                source = conn.get("source")
                target = conn.get("target")
                strength = conn.get("strength", 1.0)
                
                if source in self.nodes and target in self.nodes:
                    self.nodes[source].connect_to(self.nodes[target], strength)
    
    def _create_node(self, name: str, node_type: str, 
                     metrics_config: List[Dict[str, Any]], 
                     influence_weight: float = 1.0) -> FieldNode:
        """
        Cria um novo nó e o adiciona ao campo.
        
        Args:
            name: Nome único do nó
            node_type: Tipo do nó
            metrics_config: Configuração das métricas
            influence_weight: Peso de influência
            
        Returns:
            Nó criado
        """
        if name in self.nodes:
            logger.warning(f"Nó '{name}' já existe, será substituído")
            
        node = FieldNode(
            name=name,
            node_type=node_type,
            metrics_config=metrics_config,
            influence_weight=influence_weight
        )
        
        self.nodes[name] = node
        return node
    
    def update_metric(self, node_name: str, metric_name: str, value: float) -> bool:
        """
        Atualiza o valor de uma métrica em um nó específico.
        
        Args:
            node_name: Nome do nó
            metric_name: Nome da métrica
            value: Novo valor
            
        Returns:
            True se uma anomalia foi detectada
        """
        if node_name not in self.nodes:
            logger.error(f"Nó '{node_name}' não encontrado")
            return False
            
        self.last_update_time = time.time()
        
        # Atualiza a métrica no nó
        try:
            is_anomaly = self.nodes[node_name].update_metric(metric_name, value)
            
            # Atualiza contadores globais
            if is_anomaly:
                self.anomaly_counter += 1
                self.global_awareness = min(1.0, self.global_awareness + 0.05)
            else:
                # Diminui gradualmente o nível de consciência global
                self.global_awareness = max(0.2, self.global_awareness * 0.995)
                
            # Propaga a atualização pelo campo
            self._propagate_update(node_name, metric_name, value, is_anomaly)
            
            return is_anomaly
        except Exception as e:
            logger.error(f"Erro ao atualizar métrica: {e}")
            return False
    
    def _propagate_update(self, source_node: str, metric_name: str, 
                         value: float, is_anomaly: bool):
        """
        Propaga uma atualização de métrica pelos nós conectados.
        
        Args:
            source_node: Nome do nó de origem
            metric_name: Nome da métrica atualizada
            value: Valor da métrica
            is_anomaly: Se a atualização representa uma anomalia
        """
        # Implementação básica de propagação
        # Na versão atual, apenas aumenta o nível de consciência dos nós conectados
        if source_node not in self.nodes:
            return
            
        for connection in self.nodes[source_node].connections:
            target_node = connection["target"]
            strength = connection["strength"]
            
            # Aumenta o nível de consciência do nó alvo proporcionalmente 
            # à força da conexão e se é uma anomalia
            awareness_delta = 0.02 * strength
            if is_anomaly:
                awareness_delta *= 3
                
            target_node.awareness_level = min(1.0, 
                                             target_node.awareness_level + awareness_delta)
    
    def process_cycle(self) -> Dict[str, Any]:
        """
        Processa um ciclo do campo, calculando influências e tomando decisões.
        
        Returns:
            Dicionário com resultados do ciclo
        """
        self.field_cycles += 1
        cycle_start_time = time.time()
        
        # Calcula estatísticas globais do campo
        active_nodes = sum(1 for node in self.nodes.values() if node.state == "active")
        total_awareness = sum(node.awareness_level for node in self.nodes.values())
        avg_awareness = total_awareness / len(self.nodes) if self.nodes else 0
        
        # Decide se é necessário otimizar o sistema com base no estado do campo
        should_optimize = (
            self.global_awareness > 0.7 or  # Alta consciência global
            self.anomaly_counter > 5 or     # Muitas anomalias recentes
            self.field_cycles % 10 == 0     # Periodicamente
        )
        
        # Se necessário, realiza otimização
        optimization_actions = []
        if should_optimize:
            optimization_actions = self._optimize_system()
            self.optimization_counter += 1
            self.anomaly_counter = max(0, self.anomaly_counter - 3)  # Reduz contador de anomalias
        
        # Prepara resultado do ciclo
        cycle_result = {
            "cycle_id": self.field_cycles,
            "cycle_time": time.time() - cycle_start_time,
            "field_state": self.field_state,
            "global_awareness": self.global_awareness,
            "active_nodes": active_nodes,
            "average_awareness": avg_awareness,
            "anomaly_count": self.anomaly_counter,
            "optimization_performed": bool(optimization_actions),
            "optimization_actions": optimization_actions,
            "nodes_state": {name: node.get_state_vector() 
                           for name, node in self.nodes.items()}
        }
        
        return cycle_result
    
    def _optimize_system(self) -> List[Dict[str, Any]]:
        """
        Determina e aplica otimizações com base no estado atual do campo.
        
        Returns:
            Lista de ações de otimização realizadas
        """
        self.field_state = "optimizing"
        optimization_actions = []
        
        # Identifica nós com alta consciência (indicando necessidade de atenção)
        high_awareness_nodes = [
            (name, node) for name, node in self.nodes.items() 
            if node.awareness_level > 0.7
        ]
        
        # Exemplos de ações de otimização baseadas no tipo de nó
        for name, node in high_awareness_nodes:
            if node.node_type == "network" and "latency" in node.metrics:
                # Exemplo: Sugere reduzir timeout de conexão se a latência estiver alta
                latency_metric = node.metrics["latency"]
                if latency_metric.trend > 0:  # Latência aumentando
                    optimization_actions.append({
                        "type": "network_adjustment",
                        "target": "connection_timeout",
                        "action": "increase",
                        "reason": "Latência de rede aumentando",
                        "confidence": node.awareness_level
                    })
            
            elif node.node_type == "pool" and "share_acceptance" in node.metrics:
                # Exemplo: Sugere mudar de pool se a taxa de aceitação estiver baixa
                acceptance_metric = node.metrics["share_acceptance"]
                if acceptance_metric.ema_value and acceptance_metric.ema_value < 0.8:
                    optimization_actions.append({
                        "type": "pool_adjustment",
                        "target": "pool_selection",
                        "action": "switch_pool",
                        "reason": "Taxa de aceitação de shares baixa",
                        "confidence": node.awareness_level
                    })
            
            elif node.node_type == "performance" and "hash_rate" in node.metrics:
                # Exemplo: Sugere ajustar parâmetros de mineração se o hash rate estiver baixo
                hashrate_metric = node.metrics["hash_rate"]
                if hashrate_metric.trend < 0:  # Hash rate diminuindo
                    optimization_actions.append({
                        "type": "mining_adjustment",
                        "target": "thread_count",
                        "action": "optimize",
                        "reason": "Hash rate diminuindo",
                        "confidence": node.awareness_level
                    })
        
        # Registra as ações de otimização
        if optimization_actions:
            action_summary = ", ".join([a["type"] for a in optimization_actions])
            logger.info(f"Aplicando {len(optimization_actions)} otimizações: {action_summary}")
        
        self.field_state = "active"
        return optimization_actions
    
    def get_recommended_parameters(self, system_type: str) -> Dict[str, Any]:
        """
        Retorna parâmetros recomendados para um sistema específico com base no estado do campo.
        
        Args:
            system_type: Tipo de sistema ('mining', 'pool', 'network', etc.)
            
        Returns:
            Dicionário com parâmetros recomendados
        """
        # Processa um ciclo para garantir recomendações atualizadas
        self.process_cycle()
        
        recommendations = {}
        
        if system_type == "mining":
            # Recomendações para o sistema de mineração
            performance_node = self.nodes.get("performance_monitor")
            if performance_node and "hash_rate" in performance_node.metrics:
                hash_metric = performance_node.metrics["hash_rate"]
                # Adapta threads baseado na tendência do hash rate
                if hash_metric.trend < 0:
                    # Hash rate diminuindo, ajusta threads
                    recommendations["thread_adjustment"] = "optimize"
                    recommendations["thread_confidence"] = 0.7
            
            # Recomendações baseadas no nível de consciência global
            recommendations["adaptive_optimization"] = self.global_awareness > 0.6
            recommendations["optimization_level"] = min(3, int(self.global_awareness * 5))
            
        elif system_type == "network":
            # Recomendações para configurações de rede
            network_node = self.nodes.get("network_monitor")
            if network_node:
                if "latency" in network_node.metrics:
                    latency = network_node.metrics["latency"]
                    # Ajusta timeout baseado na latência observada
                    if latency.ema_value:
                        recommendations["timeout"] = max(5, min(30, latency.ema_value * 2))
                
                if "connection_stability" in network_node.metrics:
                    stability = network_node.metrics["connection_stability"]
                    # Ajusta intervalo de reconexão baseado na estabilidade
                    if stability.ema_value:
                        stability_factor = max(0.2, min(1.0, stability.ema_value))
                        recommendations["reconnect_interval"] = int(10 / stability_factor)
        
        elif system_type == "pool":
            # Recomendações para seleção e configuração de pools
            pool_node = self.nodes.get("pool_monitor")
            if pool_node and "share_acceptance" in pool_node.metrics:
                acceptance = pool_node.metrics["share_acceptance"]
                if acceptance.ema_value and acceptance.ema_value < 0.7:
                    recommendations["switch_pool"] = True
                    recommendations["switch_confidence"] = 0.8
            
        return recommendations
    
    def integrate_external_input(self, input_type: str, data: Any):
        """
        Integra informações externas ao campo para influenciar seu comportamento.
        
        Args:
            input_type: Tipo de entrada ('user_decision', 'environment_change', etc.)
            data: Dados associados à entrada
        """
        if input_type == "user_decision":
            # Usuário tomou uma decisão que deve influenciar o campo
            decision = data.get("decision")
            confidence = data.get("confidence", 0.8)
            
            # Aumenta consciência global para refletir intervenção humana
            self.global_awareness = min(1.0, self.global_awareness + (0.1 * confidence))
            
            # Registra a decisão como um evento no sistema
            logger.info(f"Decisão do usuário integrada: {decision} (confiança: {confidence})")
            
        elif input_type == "environment_change":
            # Mudança no ambiente de mineração
            change_type = data.get("type")
            severity = data.get("severity", 0.5)
            
            # Ajusta consciência global baseado na severidade da mudança
            self.global_awareness = min(1.0, self.global_awareness + (0.05 * severity))
            
            # Registra a mudança
            logger.info(f"Mudança no ambiente registrada: {change_type} (severidade: {severity})")
    
    def save_state(self, filepath: str) -> bool:
        """
        Salva o estado atual do campo em um arquivo JSON.
        
        Args:
            filepath: Caminho para o arquivo de saída
            
        Returns:
            True se o salvamento foi bem-sucedido
        """
        try:
            # Constrói representação do estado
            field_state = {
                "timestamp": time.time(),
                "global_awareness": self.global_awareness,
                "field_cycles": self.field_cycles,
                "field_state": self.field_state,
                "nodes": {}
            }
            
            # Adiciona estado de cada nó
            for name, node in self.nodes.items():
                field_state["nodes"][name] = node.get_state_vector()
                
            # Salva em arquivo JSON
            with open(filepath, 'w') as f:
                json.dump(field_state, f, indent=2)
                
            logger.info(f"Estado do campo salvo em {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao salvar estado do campo: {e}")
            return False
    
    def load_state(self, filepath: str) -> bool:
        """
        Carrega o estado do campo a partir de um arquivo JSON.
        
        Args:
            filepath: Caminho para o arquivo de entrada
            
        Returns:
            True se o carregamento foi bem-sucedido
        """
        if not os.path.exists(filepath):
            logger.error(f"Arquivo não encontrado: {filepath}")
            return False
            
        try:
            with open(filepath, 'r') as f:
                state_data = json.load(f)
                
            # Restaura propriedades globais
            self.global_awareness = state_data.get("global_awareness", 0.5)
            self.field_cycles = state_data.get("field_cycles", 0)
            self.field_state = state_data.get("field_state", "active")
            
            # Restaura nós (simplificado - não restaura todas as propriedades)
            if "nodes" in state_data:
                for name, node_state in state_data["nodes"].items():
                    if name in self.nodes:
                        self.nodes[name].awareness_level = node_state.get("awareness_level", 0.5)
                        self.nodes[name].state = node_state.get("state", "active")
                    else:
                        logger.warning(f"Nó '{name}' não existe no campo atual, ignorando")
            
            logger.info(f"Estado do campo carregado de {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar estado do campo: {e}")
            return False


# Exemplo de como criar e usar o campo
def create_adaptive_field(config_path: Optional[str] = None) -> QualiaAdaptiveField:
    """
    Cria uma instância do Campo de Consciência Dinâmico.
    
    Args:
        config_path: Caminho opcional para arquivo de configuração
        
    Returns:
        Instância configurada do campo
    """
    return QualiaAdaptiveField(config_path)


if __name__ == "__main__":
    # Configuração de logging para teste
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Cria e testa o campo
    field = create_adaptive_field()
    
    # Exemplo: Simula algumas atualizações de métricas
    field.update_metric("network_monitor", "latency", 120)
    field.update_metric("network_monitor", "latency", 150)
    field.update_metric("network_monitor", "latency", 200)  # Deveria causar anomalia
    
    field.update_metric("pool_monitor", "share_acceptance", 0.95)
    field.update_metric("pool_monitor", "share_acceptance", 0.92)
    field.update_metric("pool_monitor", "share_acceptance", 0.60)  # Deveria causar anomalia
    
    # Processa um ciclo e imprime resultados
    cycle_result = field.process_cycle()
    logger.info(f"Resultado do ciclo: {cycle_result['optimization_actions']}")
    
    # Obtém recomendações para o sistema de mineração
    mining_params = field.get_recommended_parameters("mining")
    logger.info(f"Parâmetros recomendados para mineração: {mining_params}")
    
    # Salva estado para uso futuro
    field.save_state("adaptive_field_state.json")
