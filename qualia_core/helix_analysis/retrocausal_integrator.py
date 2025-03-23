#!/usr/bin/env python3
"""
RetrocausalIntegrator: Módulo responsável por integrar feedback retrocausal no campo quântico.
Implementa mecanismos para que estados futuros influenciem estados presentes,
criando um sistema de feedback temporal que otimiza o campo da hélice.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple, Deque
from collections import deque
from datetime import datetime

logger = logging.getLogger("retrocausal_integrator")

class RetrocausalIntegrator:
    """
    Integrador de feedback retrocausal para o campo da hélice.
    Permite que estados futuros influenciem estados presentes através
    de um mecanismo de feedback temporal.
    """
    
    def __init__(self, tau: int = 10, lambda_coupling: float = 0.5, memory_size: int = 100):
        """
        Inicializa o integrador retrocausal.
        
        Args:
            tau: Janela de tempo para feedback retrocausal
            lambda_coupling: Intensidade do acoplamento temporal
            memory_size: Tamanho da memória temporal
        """
        self.tau = tau
        self.lambda_coupling = lambda_coupling
        self.memory_size = memory_size
        
        # Estruturas de dados para armazenamento temporal
        self.temporal_memory = deque(maxlen=memory_size)
        self.feedback_history = deque(maxlen=memory_size)
        self.external_feedback = {}  # Mapeia step -> feedback
        
        # Métricas
        self.retrocausal_metrics = {
            'feedback_strength': 0.0,
            'temporal_coherence': 0.0,
            'stability': 0.5,
            'adaptation_rate': 0.01
        }
        
        # Estado interno
        self._last_update = 0
        
        logger.info(f"RetrocausalIntegrator inicializado com tau={tau}, lambda={lambda_coupling}")
    
    def apply_feedback(self, field: np.ndarray, current_step: int) -> np.ndarray:
        """
        Aplica feedback retrocausal ao campo atual.
        
        Args:
            field: Campo atual para aplicar o feedback
            current_step: Passo atual da simulação
            
        Returns:
            Campo modificado com influência retrocausal
        """
        if field is None or field.size == 0:
            logger.warning("Campo vazio fornecido para feedback retrocausal")
            return field
        
        # Adicionar estado atual à memória temporal
        self._add_to_memory(field, current_step)
        
        # Se não tivermos dados suficientes, retornar campo inalterado
        if len(self.temporal_memory) <= self.tau:
            return field
        
        # Calcular feedback retrocausal
        feedback_field = self._calculate_feedback(current_step)
        
        # Se não houver feedback, retornar campo inalterado
        if feedback_field is None:
            return field
        
        # Aplicar feedback com intensidade lambda
        modified_field = field + self.lambda_coupling * feedback_field
        
        # Normalizar o campo
        if np.sum(np.abs(modified_field)) > 0:
            modified_field = modified_field / np.sqrt(np.sum(np.abs(modified_field)**2))
        
        # Armazenar o feedback aplicado
        self.feedback_history.append({
            'step': current_step,
            'timestamp': datetime.now().isoformat(),
            'feedback_norm': np.linalg.norm(feedback_field),
            'field_norm': np.linalg.norm(field)
        })
        
        # Atualizar métricas
        self._update_metrics(field, feedback_field, current_step)
        
        self._last_update = current_step
        logger.debug(f"Feedback retrocausal aplicado no passo {current_step}")
        
        return modified_field
    
    def get_feedback_metrics(self, field: np.ndarray, current_step: int) -> Dict[str, float]:
        """
        Obtém métricas do feedback retrocausal atual.
        
        Args:
            field: Campo atual
            current_step: Passo atual da simulação
            
        Returns:
            Métricas do feedback retrocausal
        """
        # Se as métricas estiverem desatualizadas, atualizá-las
        if self._last_update != current_step:
            feedback_field = self._calculate_feedback(current_step)
            if feedback_field is not None:
                self._update_metrics(field, feedback_field, current_step)
                self._last_update = current_step
        
        return self.retrocausal_metrics.copy()
    
    def add_external_feedback(self, step: int, feedback_value: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Adiciona feedback externo para influenciar o campo.
        
        Args:
            step: Passo da simulação para o feedback
            feedback_value: Valor do feedback (-1 a 1)
            metadata: Dados adicionais sobre o feedback
        """
        if step < 0:
            logger.warning(f"Tentativa de adicionar feedback para passo negativo: {step}")
            return
        
        # Limitar feedback entre -1 e 1
        feedback_value = max(-1.0, min(1.0, feedback_value))
        
        # Adicionar ou atualizar feedback externo
        if step in self.external_feedback:
            # Média ponderada com feedback existente
            existing = self.external_feedback[step]
            existing_value = existing['value']
            existing_count = existing['count']
            
            new_value = (existing_value * existing_count + feedback_value) / (existing_count + 1)
            new_count = existing_count + 1
            
            if metadata:
                if 'metadata' not in existing:
                    existing['metadata'] = [metadata]
                else:
                    existing['metadata'].append(metadata)
            
            self.external_feedback[step] = {
                'value': new_value,
                'count': new_count,
                'timestamp': datetime.now().isoformat(),
                'metadata': existing.get('metadata', [])
            }
        else:
            self.external_feedback[step] = {
                'value': feedback_value,
                'count': 1,
                'timestamp': datetime.now().isoformat(),
                'metadata': [metadata] if metadata else []
            }
        
        logger.debug(f"Feedback externo adicionado para passo {step}: {feedback_value}")
        
        # Limpar feedbacks antigos (manter apenas os últimos memory_size)
        if len(self.external_feedback) > self.memory_size:
            # Remover os mais antigos
            steps_to_remove = sorted(self.external_feedback.keys())[:-self.memory_size]
            for s in steps_to_remove:
                del self.external_feedback[s]
    
    def reset(self) -> None:
        """Reseta o integrador para estado inicial"""
        self.temporal_memory.clear()
        self.feedback_history.clear()
        self.external_feedback.clear()
        
        # Resetar métricas
        self.retrocausal_metrics = {
            'feedback_strength': 0.0,
            'temporal_coherence': 0.0,
            'stability': 0.5,
            'adaptation_rate': 0.01
        }
        
        self._last_update = 0
        logger.info("RetrocausalIntegrator resetado")
    
    def _add_to_memory(self, field: np.ndarray, step: int) -> None:
        """
        Adiciona o campo atual à memória temporal.
        
        Args:
            field: Campo a ser armazenado
            step: Passo atual da simulação
        """
        # Fazer cópia para evitar modificações externas
        field_copy = field.copy()
        
        self.temporal_memory.append({
            'field': field_copy,
            'step': step,
            'timestamp': datetime.now().isoformat()
        })
    
    def _calculate_feedback(self, current_step: int) -> Optional[np.ndarray]:
        """
        Calcula o campo de feedback retrocausal.
        
        Args:
            current_step: Passo atual da simulação
            
        Returns:
            Campo de feedback ou None se não for possível calcular
        """
        if len(self.temporal_memory) <= self.tau:
            return None
        
        # Obter campo atual
        current_field = None
        for entry in self.temporal_memory:
            if entry['step'] == current_step:
                current_field = entry['field']
                break
        
        if current_field is None:
            return None
        
        # Inicializar campo de feedback
        feedback_field = np.zeros_like(current_field)
        feedback_count = 0
        
        # Calcular contribuição de estados futuros já observados
        for entry in self.temporal_memory:
            step = entry['step']
            # Considerar apenas estados dentro da janela tau
            if step > current_step and step <= current_step + self.tau:
                future_field = entry['field']
                
                # Calcular fator de decaimento temporal
                decay = np.exp(-(step - current_step) / self.tau)
                
                # Adicionar contribuição ao feedback
                feedback_field += decay * future_field
                feedback_count += 1
        
        # Adicionar contribuição de feedback externo
        for step in range(current_step - self.tau, current_step + self.tau + 1):
            if step in self.external_feedback:
                external = self.external_feedback[step]
                feedback_value = external['value']
                
                # Calcular fator de decaimento temporal (mais forte para passos próximos)
                temporal_distance = abs(step - current_step)
                decay = np.exp(-temporal_distance / self.tau)
                
                # Criar um padrão de onda baseado no feedback
                pattern = self._create_feedback_pattern(current_field.shape, feedback_value)
                
                # Adicionar contribuição ao feedback
                feedback_field += decay * pattern * abs(feedback_value)
                feedback_count += 1
        
        # Normalizar se houver contribuições
        if feedback_count > 0:
            feedback_field = feedback_field / feedback_count
            
            # Normalizar magnitude
            if np.sum(np.abs(feedback_field)) > 0:
                feedback_field = feedback_field / np.sqrt(np.sum(np.abs(feedback_field)**2))
            
            return feedback_field
        
        return None
    
    def _create_feedback_pattern(self, shape: Tuple[int, ...], value: float) -> np.ndarray:
        """
        Cria um padrão de campo baseado no valor de feedback.
        
        Args:
            shape: Forma do campo
            value: Valor do feedback (-1 a 1)
            
        Returns:
            Padrão de campo
        """
        # Criar padrão baseado no valor de feedback
        # Valores positivos criam padrões construtivos, negativos destrutivos
        
        # Inicializar com ruído aleatório de baixa amplitude
        pattern = np.random.normal(0, 0.01, shape)
        
        # Para feedback positivo, criar padrão construtivo (ondas concêntricas)
        if value > 0:
            center = (shape[0] // 2, shape[1] // 2)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    # Distância ao centro
                    dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                    # Ondas concêntricas com amplitude proporcional ao feedback
                    pattern[i, j] += value * np.sin(dist * 0.1) * np.exp(-dist * 0.05)
        
        # Para feedback negativo, criar padrão destrutivo (interferência)
        else:
            # Criar padrão de interferência
            x = np.linspace(-3, 3, shape[0])
            y = np.linspace(-3, 3, shape[1])
            X, Y = np.meshgrid(x, y)
            
            # Padrão de interferência
            interference = np.sin(X * 2) * np.sin(Y * 2) * abs(value)
            pattern += interference
        
        # Normalizar o padrão
        if np.sum(np.abs(pattern)) > 0:
            pattern = pattern / np.sqrt(np.sum(np.abs(pattern)**2))
        
        return pattern
    
    def _update_metrics(self, field: np.ndarray, feedback_field: np.ndarray, current_step: int) -> None:
        """
        Atualiza métricas do integrador retrocausal.
        
        Args:
            field: Campo atual
            feedback_field: Campo de feedback calculado
            current_step: Passo atual da simulação
        """
        # Calcular força do feedback
        feedback_norm = np.linalg.norm(feedback_field)
        field_norm = np.linalg.norm(field)
        
        if field_norm > 0:
            feedback_strength = feedback_norm / field_norm
        else:
            feedback_strength = 0
        
        # Limitar entre 0 e 1
        feedback_strength = min(1.0, max(0.0, feedback_strength))
        
        # Calcular coerência temporal (correlação entre estados passados)
        temporal_coherence = self._calculate_temporal_coherence(current_step)
        
        # Calcular estabilidade (inverso da variação entre feedbacks consecutivos)
        stability = self._calculate_stability()
        
        # Calcular taxa de adaptação baseada na estabilidade e força do feedback
        adaptation_rate = 0.01 * (1 + feedback_strength) * (1 + stability)
        
        # Atualizar métricas
        self.retrocausal_metrics = {
            'feedback_strength': float(feedback_strength),
            'temporal_coherence': float(temporal_coherence),
            'stability': float(stability),
            'adaptation_rate': float(adaptation_rate)
        }
    
    def _calculate_temporal_coherence(self, current_step: int) -> float:
        """
        Calcula a coerência temporal entre estados passados.
        
        Args:
            current_step: Passo atual da simulação
            
        Returns:
            Coerência temporal (0-1)
        """
        if len(self.temporal_memory) < 3:
            return 0.5  # Valor padrão
        
        # Obter campos relevantes (últimos 3 na janela tau)
        relevant_fields = []
        
        for entry in sorted(self.temporal_memory, key=lambda x: x['step'], reverse=True):
            if entry['step'] <= current_step and len(relevant_fields) < 3:
                relevant_fields.append(entry['field'])
        
        if len(relevant_fields) < 3:
            return 0.5
        
        # Calcular correlações entre campos consecutivos
        correlation_1 = self._calculate_field_correlation(relevant_fields[0], relevant_fields[1])
        correlation_2 = self._calculate_field_correlation(relevant_fields[1], relevant_fields[2])
        
        # Coerência temporal é a média das correlações
        coherence = (correlation_1 + correlation_2) / 2
        
        return min(1.0, max(0.0, coherence))
    
    def _calculate_field_correlation(self, field1: np.ndarray, field2: np.ndarray) -> float:
        """
        Calcula a correlação entre dois campos.
        
        Args:
            field1: Primeiro campo
            field2: Segundo campo
            
        Returns:
            Correlação normalizada (0-1)
        """
        try:
            # Produto interno normalizado
            inner_product = np.sum(field1 * field2.conj())
            norm1 = np.linalg.norm(field1)
            norm2 = np.linalg.norm(field2)
            
            if norm1 > 0 and norm2 > 0:
                correlation = abs(inner_product) / (norm1 * norm2)
            else:
                correlation = 0
            
            return float(correlation)
        except:
            return 0.5  # Valor padrão em caso de erro
    
    def _calculate_stability(self) -> float:
        """
        Calcula a estabilidade do feedback retrocausal.
        
        Returns:
            Estabilidade (0-1)
        """
        if len(self.feedback_history) < 2:
            return 0.5  # Valor padrão
        
        # Obter últimos feedbacks
        recent_feedbacks = list(self.feedback_history)[-5:]
        
        if len(recent_feedbacks) < 2:
            return 0.5
        
        # Calcular variação entre feedbacks consecutivos
        variations = []
        for i in range(1, len(recent_feedbacks)):
            prev = recent_feedbacks[i-1]['feedback_norm']
            curr = recent_feedbacks[i]['feedback_norm']
            
            if prev > 0:
                variation = abs(curr - prev) / prev
            else:
                variation = 0
            
            variations.append(variation)
        
        # Estabilidade é o inverso da variação média
        if variations:
            avg_variation = sum(variations) / len(variations)
            stability = 1.0 / (1.0 + 10.0 * avg_variation)  # Fator 10 para sensibilidade
        else:
            stability = 0.5
        
        return min(1.0, max(0.0, stability)) 