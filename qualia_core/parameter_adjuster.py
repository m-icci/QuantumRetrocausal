#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
parameter_adjuster.py - Ajustador Adaptativo de Parâmetros
--------------------------------------
Este módulo implementa o ajuste inteligente de parâmetros para o
sistema QUALIA com base nas métricas analisadas.

Funções principais:
1. Ajuste gradual de parâmetros de mineração
2. Otimização com base em tendências
3. Adaptação inteligente aos resultados observados
"""

import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParameterAdjuster:
    """
    Ajusta parâmetros de mineração de forma adaptativa
    
    Responsável por modificar parâmetros do sistema QUALIA
    com base nos insights gerados pelo analisador de métricas.
    """
    
    def __init__(self, 
                 initial_params: Dict[str, Any],
                 param_limits: Dict[str, Tuple[float, float]] = None,
                 learning_rate: float = 0.05):
        """
        Inicializa o ajustador de parâmetros
        
        Args:
            initial_params: Parâmetros iniciais
            param_limits: Limites mínimo e máximo para cada parâmetro
            learning_rate: Taxa de aprendizado para ajustes
        """
        self.current_params = initial_params.copy()
        self.param_history = [initial_params.copy()]
        self.learning_rate = learning_rate
        
        # Definir limites padrão se não fornecidos
        self.param_limits = param_limits or {
            "coherence": (0.1, 1.0),
            "granularity": (1, 128),
            "field_energy": (0.1, 10.0),
            "retrocausality_weight": (0.0, 1.0),
            "mutation_rate": (0.001, 0.2)
        }
        
        # Contagem de ajustes por parâmetro
        self.adjust_counter = {param: 0 for param in self.current_params}
        
        # Histórico de eficácia de ajustes
        self.adjustment_efficacy = {param: [] for param in self.current_params}
        
        self.last_adjustment_time = {param: 0 for param in self.current_params}
        self.cooldown_periods = {
            "coherence": 30,  # 30 segundos
            "granularity": 60,
            "field_energy": 45,
            "retrocausality_weight": 120,
            "mutation_rate": 90
        }
    
    def adjust_parameters(self, 
                          metrics: Dict[str, Any], 
                          suggestions: Dict[str, str]) -> Dict[str, Any]:
        """
        Ajusta parâmetros com base nas métricas e sugestões
        
        Args:
            metrics: Métricas derivadas do analisador
            suggestions: Sugestões de ajuste ("aumentar", "reduzir", "manter")
            
        Returns:
            Parâmetros atualizados
        """
        current_time = time.time()
        params_changed = False
        
        # Processar cada sugestão
        for param, action in suggestions.items():
            # Verificar se o parâmetro existe
            if param not in self.current_params:
                logger.warning(f"Parâmetro {param} não encontrado")
                continue
            
            # Verificar período de cooldown
            if current_time - self.last_adjustment_time.get(param, 0) < self.cooldown_periods.get(param, 30):
                logger.debug(f"Parâmetro {param} em período de cooldown")
                continue
            
            # Aplicar o ajuste
            old_value = self.current_params[param]
            
            if action == "aumentar":
                self._increase_parameter(param)
            elif action == "reduzir":
                self._decrease_parameter(param)
            
            # Registrar o ajuste
            if self.current_params[param] != old_value:
                self.last_adjustment_time[param] = current_time
                self.adjust_counter[param] += 1
                params_changed = True
                logger.info(f"Ajustado {param}: {old_value} -> {self.current_params[param]}")
        
        # Se algum parâmetro foi alterado, salvar no histórico
        if params_changed:
            self.param_history.append(self.current_params.copy())
        
        return self.current_params
    
    def _increase_parameter(self, param: str):
        """
        Aumenta o valor de um parâmetro respeitando os limites
        
        Args:
            param: Nome do parâmetro
        """
        min_val, max_val = self.param_limits.get(param, (0, float('inf')))
        current = self.current_params[param]
        
        # Calcular o delta com base no tipo de parâmetro
        if isinstance(current, int):
            # Parâmetros inteiros como granularidade
            if current < 10:
                delta = 1
            elif current < 50:
                delta = 2
            else:
                delta = 4
        else:
            # Parâmetros de ponto flutuante
            delta = current * self.learning_rate
        
        # Aplicar o ajuste e limitar ao máximo
        new_value = min(current + delta, max_val)
        
        # Arredondar para inteiro se necessário
        if isinstance(current, int):
            new_value = int(round(new_value))
        
        self.current_params[param] = new_value
    
    def _decrease_parameter(self, param: str):
        """
        Diminui o valor de um parâmetro respeitando os limites
        
        Args:
            param: Nome do parâmetro
        """
        min_val, max_val = self.param_limits.get(param, (0, float('inf')))
        current = self.current_params[param]
        
        # Calcular o delta com base no tipo de parâmetro
        if isinstance(current, int):
            # Parâmetros inteiros
            if current <= 10:
                delta = 1
            elif current <= 50:
                delta = 2
            else:
                delta = 4
        else:
            # Parâmetros de ponto flutuante
            delta = current * self.learning_rate
        
        # Aplicar o ajuste e limitar ao mínimo
        new_value = max(current - delta, min_val)
        
        # Arredondar para inteiro se necessário
        if isinstance(current, int):
            new_value = int(round(new_value))
        
        self.current_params[param] = new_value
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """
        Obtém os parâmetros atuais
        
        Returns:
            Parâmetros atuais
        """
        return self.current_params
    
    def evaluate_adjustment_efficacy(self, 
                                     metrics_before: Dict[str, Any], 
                                     metrics_after: Dict[str, Any]):
        """
        Avalia a eficácia de ajustes anteriores
        
        Args:
            metrics_before: Métricas antes do ajuste
            metrics_after: Métricas após o ajuste
        """
        if len(self.param_history) < 2:
            return
        
        # Verificar alterações de desempenho
        hash_rate_before = metrics_before.get("avg_hash_rate", 0)
        hash_rate_after = metrics_after.get("avg_hash_rate", 0)
        
        if hash_rate_before == 0:
            return
        
        # Calcular melhoria percentual
        improvement = (hash_rate_after - hash_rate_before) / hash_rate_before
        
        # Comparar parâmetros atuais com anteriores
        prev_params = self.param_history[-2]
        current_params = self.param_history[-1]
        
        # Registrar eficácia para cada parâmetro alterado
        for param in current_params:
            if param in prev_params and current_params[param] != prev_params[param]:
                # Armazenar a eficácia do ajuste
                self.adjustment_efficacy[param].append(improvement)
                
                # Manter apenas os últimos 10 ajustes
                if len(self.adjustment_efficacy[param]) > 10:
                    self.adjustment_efficacy[param] = self.adjustment_efficacy[param][-10:]
    
    def adapt_learning_rate(self):
        """
        Adapta a taxa de aprendizado com base na eficácia dos ajustes
        """
        for param, efficacy_list in self.adjustment_efficacy.items():
            if len(efficacy_list) < 3:
                continue
            
            # Calcular média de eficácia
            avg_efficacy = sum(efficacy_list) / len(efficacy_list)
            
            # Ajustar taxa de aprendizado
            if avg_efficacy > 0.05:  # Ajustes positivos significativos
                self.learning_rate = min(0.2, self.learning_rate * 1.2)
            elif avg_efficacy < -0.05:  # Ajustes negativos significativos
                self.learning_rate = max(0.01, self.learning_rate * 0.8)
    
    def reset_counters(self):
        """
        Reinicia contadores de ajuste
        """
        self.adjust_counter = {param: 0 for param in self.current_params}
    
    def get_adjustment_statistics(self) -> Dict[str, Any]:
        """
        Obtém estatísticas de ajuste
        
        Returns:
            Estatísticas de ajuste por parâmetro
        """
        stats = {
            "learning_rate": self.learning_rate,
            "adjustments": self.adjust_counter.copy(),
            "efficacy": {}
        }
        
        # Calcular eficácia média para cada parâmetro
        for param, efficacy_list in self.adjustment_efficacy.items():
            if efficacy_list:
                stats["efficacy"][param] = sum(efficacy_list) / len(efficacy_list)
            else:
                stats["efficacy"][param] = 0
        
        return stats


# Função para teste
def test_adjuster():
    """Testa o ajustador de parâmetros"""
    
    # Parâmetros iniciais
    initial_params = {
        "coherence": 0.5,
        "granularity": 16,
        "field_energy": 1.0,
        "retrocausality_weight": 0.3,
        "mutation_rate": 0.05
    }
    
    adjuster = ParameterAdjuster(initial_params)
    
    # Simular métricas e sugestões
    metrics = {
        "avg_hash_rate": 100,
        "stability_index": 0.6,
        "hash_rate_trend": -0.15
    }
    
    suggestions = {
        "coherence": "aumentar",
        "granularity": "reduzir"
    }
    
    # Ajustar parâmetros
    new_params = adjuster.adjust_parameters(metrics, suggestions)
    
    print("Parâmetros ajustados:")
    for param, value in new_params.items():
        print(f"  {param}: {value}")
    
    # Simular evolução de métricas após ajuste
    metrics_after = {
        "avg_hash_rate": 110,
        "stability_index": 0.65,
        "hash_rate_trend": 0.05
    }
    
    # Avaliar eficácia
    adjuster.evaluate_adjustment_efficacy(metrics, metrics_after)
    
    # Obter estatísticas
    stats = adjuster.get_adjustment_statistics()
    
    print("\nEstatísticas de ajuste:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return new_params

if __name__ == "__main__":
    test_adjuster()
