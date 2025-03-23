#!/usr/bin/env python3
"""
Campo de Retrocausalidade para Mineração QUALIA

Este módulo implementa um campo de retrocausalidade que permite que estados futuros
influenciem a tomada de decisão presente no processo de mineração QUALIA, seguindo
os princípios da Teoria da Ressonância Cósmica (TRC).
"""

import os
import time
import json
import logging
import hashlib
import threading
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import deque

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QualiaTRC")

class RetrocausalState:
    """
    Representa um estado no campo de retrocausalidade.
    
    Cada estado contém informações sobre um nonce candidato, sua pontuação
    e metadados temporais que permitem a análise de influências futuras.
    """
    
    def __init__(self, nonce: Union[str, int], score: float, metadata: Optional[Dict[str, Any]] = None):
        """
        Inicializa um estado retrocausal.
        
        Args:
            nonce: O nonce candidato
            score: A pontuação associada (menor é melhor)
            metadata: Metadados adicionais para análise
        """
        self.nonce = str(nonce)
        self.score = float(score)
        self.creation_time = time.time()
        self.metadata = metadata or {}
        self.collapsed = False
        self.collapse_time = None
        self.influence_strength = 1.0  # Força da influência retrocausal
        
    def collapse(self, influence_strength: Optional[float] = None):
        """
        Colapsa este estado, marcando-o como observado.
        
        Args:
            influence_strength: Força opcional da influência no colapso
        """
        self.collapsed = True
        self.collapse_time = time.time()
        if influence_strength is not None:
            self.influence_strength = influence_strength
    
    def get_age(self) -> float:
        """
        Retorna a idade do estado em segundos.
        
        Returns:
            Tempo desde a criação em segundos
        """
        return time.time() - self.creation_time
    
    def get_collapse_delta(self) -> Optional[float]:
        """
        Retorna o tempo entre criação e colapso, se colapsado.
        
        Returns:
            Delta de tempo em segundos ou None se não colapsado
        """
        if not self.collapsed:
            return None
        return self.collapse_time - self.creation_time
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converte o estado para dicionário.
        
        Returns:
            Representação em dicionário
        """
        return {
            "nonce": self.nonce,
            "score": self.score,
            "creation_time": self.creation_time,
            "collapsed": self.collapsed,
            "collapse_time": self.collapse_time,
            "influence_strength": self.influence_strength,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RetrocausalState':
        """
        Cria um estado a partir de um dicionário.
        
        Args:
            data: Dicionário com dados do estado
            
        Returns:
            Instância do RetrocausalState
        """
        state = cls(data["nonce"], data["score"], data.get("metadata", {}))
        state.creation_time = data["creation_time"]
        state.collapsed = data["collapsed"]
        state.collapse_time = data["collapse_time"]
        state.influence_strength = data["influence_strength"]
        return state


class RetrocausalField:
    """
    Campo de Retrocausalidade que modela a influência de estados futuros
    sobre decisões presentes, seguindo os princípios da TRC.
    
    Este campo mantém um buffer temporal de estados possíveis, permitindo
    que informações do futuro influenciem o processo de seleção de nonces.
    """
    
    def __init__(self, future_depth: int = 5, coherence_threshold: float = 0.7,
                temporal_window: float = 10.0):
        """
        Inicializa o Campo de Retrocausalidade.
        
        Args:
            future_depth: Número de estados futuros armazenados
            coherence_threshold: Limiar de coerência para influência retrocausal
            temporal_window: Janela temporal em segundos para detecção de padrões
        """
        self.future_states: List[RetrocausalState] = []
        self.historical_states: List[RetrocausalState] = []
        self.depth = future_depth
        self.coherence_threshold = coherence_threshold
        self.temporal_window = temporal_window
        
        self.time_deltas: List[float] = []  # Deltas entre previsão e confirmação
        self.field_coherence = 0.5  # Nível de coerência do campo (0.0 a 1.0)
        self.retrocausal_factor = 0.0  # Fator de influência retrocausal medido
        
        self.creation_time = time.time()
        self.last_update_time = self.creation_time
        self.last_collapse_time = None
        
        # Métricas e estatísticas
        self.total_predictions = 0
        self.accurate_predictions = 0
        self.prediction_history: Dict[str, List[Tuple[float, bool]]] = {}  # nonce -> [(timestamp, hit)]
        
        # Cache para evitar recálculos frequentes
        self._pattern_cache = {}
        self._last_cache_clear = time.time()
        
        # Thread-safe lock
        self._lock = threading.RLock()
        
        logger.info("Campo de Retrocausalidade inicializado com profundidade futura de %d", future_depth)
    
    def update_future(self, nonce: Union[str, int], score: float, 
                     metadata: Optional[Dict[str, Any]] = None) -> RetrocausalState:
        """
        Adiciona um novo estado candidato ao buffer futuro.
        
        Args:
            nonce: Nonce candidato
            score: Pontuação do nonce (menor é melhor)
            metadata: Metadados adicionais
            
        Returns:
            Estado retrocausal criado
        """
        with self._lock:
            state = RetrocausalState(nonce, score, metadata)
            self.future_states.append(state)
            self.total_predictions += 1
            
            # Mantém apenas os estados mais recentes
            if len(self.future_states) > self.depth:
                oldest = self.future_states.pop(0)
                self.historical_states.append(oldest)
                
            # Limita o histórico para evitar crescimento excessivo
            if len(self.historical_states) > self.depth * 10:
                self.historical_states = self.historical_states[-self.depth * 10:]
                
            # Atualiza registros de timestamp
            self.last_update_time = time.time()
            
            # Registra predição para análise posterior
            nonce_str = str(nonce)
            if nonce_str not in self.prediction_history:
                self.prediction_history[nonce_str] = []
            self.prediction_history[nonce_str].append((self.last_update_time, False))
            
            # Limpa cache periodicamente
            if time.time() - self._last_cache_clear > 60:  # A cada minuto
                self._pattern_cache = {}
                self._last_cache_clear = time.time()
                
            return state
    
    def get_best_future_nonce(self) -> Optional[str]:
        """
        Retorna o melhor nonce futuro com base na pontuação.
        
        Returns:
            O nonce com menor pontuação ou None se não houver estados
        """
        with self._lock:
            if not self.future_states:
                return None
            
            # Encontra o estado com menor pontuação (melhor)
            best_state = min(self.future_states, key=lambda x: x.score)
            return best_state.nonce
    
    def verify_prediction(self, confirmed_nonce: Union[str, int]) -> Tuple[bool, Optional[float]]:
        """
        Verifica se um nonce confirmado foi previamente predito pelo campo.
        
        Args:
            confirmed_nonce: Nonce confirmado (aceito pela pool)
            
        Returns:
            Tupla (sucesso, delta_t) onde delta_t é o tempo de antecipação
        """
        with self._lock:
            confirmed_nonce = str(confirmed_nonce)
            
            # Procura nos estados futuros
            for state in self.future_states:
                if state.nonce == confirmed_nonce:
                    # Nonce encontrado! Registra o hit
                    delta_t = time.time() - state.creation_time
                    state.collapse(influence_strength=1.0)
                    self.time_deltas.append(delta_t)
                    self.accurate_predictions += 1
                    self.last_collapse_time = time.time()
                    
                    # Atualiza histórico de predições
                    if confirmed_nonce in self.prediction_history:
                        for i, (timestamp, _) in enumerate(self.prediction_history[confirmed_nonce]):
                            self.prediction_history[confirmed_nonce][i] = (timestamp, True)
                    
                    # Atualiza coerência do campo com base no sucesso
                    self._update_field_coherence(True, delta_t)
                    
                    logger.info(f"Nonce {confirmed_nonce} foi previsto {delta_t:.4f} segundos antes do colapso!")
                    return True, delta_t
            
            # Procura também no histórico para casos de predições mais antigas
            for state in self.historical_states:
                if state.nonce == confirmed_nonce:
                    delta_t = time.time() - state.creation_time
                    state.collapse(influence_strength=0.5)  # Influência reduzida por ser histórico
                    self.time_deltas.append(delta_t)
                    self.accurate_predictions += 1
                    self.last_collapse_time = time.time()
                    
                    # Atualiza histórico de predições
                    if confirmed_nonce in self.prediction_history:
                        for i, (timestamp, _) in enumerate(self.prediction_history[confirmed_nonce]):
                            self.prediction_history[confirmed_nonce][i] = (timestamp, True)
                    
                    # Atualiza coerência do campo
                    self._update_field_coherence(True, delta_t)
                    
                    logger.info(f"Nonce histórico {confirmed_nonce} foi previsto {delta_t:.4f} segundos antes!")
                    return True, delta_t
            
            # Nonce não encontrado, atualiza coerência negativamente
            self._update_field_coherence(False)
            return False, None
    
    def _update_field_coherence(self, success: bool, delta_t: Optional[float] = None):
        """
        Atualiza a coerência do campo com base nos sucessos ou falhas de predição.
        
        Args:
            success: Se a predição foi bem-sucedida
            delta_t: Tempo de antecipação, se aplicável
        """
        # Fator de esquecimento para dar mais peso às observações recentes
        forget_factor = 0.95
        
        if success:
            # Aumenta coerência baseado no sucesso, mais para predições com maior antecipação
            if delta_t:
                # Normaliza delta_t para um fator entre 0.05 e 0.2
                time_factor = min(0.2, max(0.05, delta_t / 10))
                self.field_coherence = min(1.0, self.field_coherence * forget_factor + time_factor)
                
                # Atualiza fator retrocausal com média móvel
                self.retrocausal_factor = (self.retrocausal_factor * 0.9) + (delta_t * 0.1)
            else:
                # Sem delta_t, aumento menor
                self.field_coherence = min(1.0, self.field_coherence * forget_factor + 0.05)
        else:
            # Diminui coerência por falha
            self.field_coherence = max(0.1, self.field_coherence * forget_factor - 0.02)
    
    def get_retrocausal_influence(self) -> float:
        """
        Calcula o nível atual de influência retrocausal.
        
        Returns:
            Valor entre 0.0 e 1.0 indicando o nível de influência
        """
        with self._lock:
            # A influência depende da coerência do campo e dos sucessos anteriores
            base_influence = self.field_coherence
            
            # Ajusta com base na taxa de acertos histórica
            if self.total_predictions > 0:
                accuracy = self.accurate_predictions / self.total_predictions
                influence = (base_influence * 0.7) + (accuracy * 0.3)
            else:
                influence = base_influence
                
            # Adiciona variabilidade quântica baseada no fator retrocausal observado
            if self.retrocausal_factor > 0:
                quantum_factor = min(0.3, self.retrocausal_factor / 20)
                influence = min(1.0, influence + quantum_factor)
                
            return influence
    
    def detect_temporal_patterns(self) -> Dict[str, Any]:
        """
        Analisa os dados para detectar padrões temporais nas predições.
        
        Returns:
            Dicionário com padrões detectados
        """
        with self._lock:
            # Verifica se há dados suficientes para análise
            if len(self.time_deltas) < 3:
                return {"sufficient_data": False}
            
            # Análise estatística dos deltas temporais
            avg_time = np.mean(self.time_deltas)
            median_time = np.median(self.time_deltas)
            std_time = np.std(self.time_deltas)
            min_time = np.min(self.time_deltas)
            max_time = np.max(self.time_deltas)
            
            # Detecção de padrões cíclicos
            has_pattern = False
            pattern_strength = 0.0
            
            if len(self.time_deltas) >= 10:
                # Análise simplificada de autocorrelação para ciclos
                autocorr = np.correlate(self.time_deltas, self.time_deltas, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                peak_idx = np.argmax(autocorr[1:]) + 1  # Ignora o pico em 0
                
                if peak_idx > 0 and peak_idx < len(autocorr) // 2:
                    has_pattern = True
                    pattern_strength = autocorr[peak_idx] / autocorr[0]  # Normalizado
            
            return {
                "sufficient_data": True,
                "avg_anticipation_time": avg_time,
                "median_anticipation_time": median_time,
                "std_deviation": std_time,
                "min_anticipation": min_time,
                "max_anticipation": max_time,
                "pattern_detected": has_pattern,
                "pattern_strength": pattern_strength,
                "accuracy_rate": self.accurate_predictions / max(1, self.total_predictions),
                "field_coherence": self.field_coherence,
                "retrocausal_factor": self.retrocausal_factor
            }
    
    def save_state(self, filepath: str) -> bool:
        """
        Salva o estado atual do campo em um arquivo JSON.
        
        Args:
            filepath: Caminho para o arquivo
            
        Returns:
            True se salvamento foi bem-sucedido
        """
        with self._lock:
            try:
                state_data = {
                    "timestamp": time.time(),
                    "future_depth": self.depth,
                    "coherence_threshold": self.coherence_threshold,
                    "temporal_window": self.temporal_window,
                    "field_coherence": self.field_coherence,
                    "retrocausal_factor": self.retrocausal_factor,
                    "total_predictions": self.total_predictions,
                    "accurate_predictions": self.accurate_predictions,
                    "time_deltas": self.time_deltas,
                    "future_states": [state.to_dict() for state in self.future_states],
                    "historical_states": [state.to_dict() for state in self.historical_states[-100:]]  # Limita para os 100 mais recentes
                }
                
                with open(filepath, 'w') as f:
                    json.dump(state_data, f, indent=2)
                
                logger.info(f"Estado do campo retrocausal salvo em {filepath}")
                return True
                
            except Exception as e:
                logger.error(f"Erro ao salvar estado do campo: {e}")
                return False
    
    def load_state(self, filepath: str) -> bool:
        """
        Carrega o estado do campo a partir de um arquivo JSON.
        
        Args:
            filepath: Caminho para o arquivo
            
        Returns:
            True se carregamento foi bem-sucedido
        """
        with self._lock:
            if not os.path.exists(filepath):
                logger.error(f"Arquivo não encontrado: {filepath}")
                return False
                
            try:
                with open(filepath, 'r') as f:
                    state_data = json.load(f)
                
                # Restaura configurações
                self.depth = state_data.get("future_depth", self.depth)
                self.coherence_threshold = state_data.get("coherence_threshold", self.coherence_threshold)
                self.temporal_window = state_data.get("temporal_window", self.temporal_window)
                
                # Restaura métricas
                self.field_coherence = state_data.get("field_coherence", 0.5)
                self.retrocausal_factor = state_data.get("retrocausal_factor", 0.0)
                self.total_predictions = state_data.get("total_predictions", 0)
                self.accurate_predictions = state_data.get("accurate_predictions", 0)
                
                # Restaura deltas temporais
                self.time_deltas = state_data.get("time_deltas", [])
                
                # Restaura estados
                self.future_states = [RetrocausalState.from_dict(data) 
                                     for data in state_data.get("future_states", [])]
                self.historical_states = [RetrocausalState.from_dict(data) 
                                         for data in state_data.get("historical_states", [])]
                
                logger.info(f"Estado do campo retrocausal carregado de {filepath}")
                return True
                
            except Exception as e:
                logger.error(f"Erro ao carregar estado do campo: {e}")
                return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retorna estatísticas resumidas do campo retrocausal.
        
        Returns:
            Dicionário com estatísticas
        """
        with self._lock:
            stats = {
                "uptime": time.time() - self.creation_time,
                "coherence_level": self.field_coherence,
                "retrocausal_factor": self.retrocausal_factor,
                "predictions": {
                    "total": self.total_predictions,
                    "accurate": self.accurate_predictions,
                    "accuracy_rate": self.accurate_predictions / max(1, self.total_predictions)
                },
                "time_deltas": {
                    "count": len(self.time_deltas),
                    "average": np.mean(self.time_deltas) if self.time_deltas else None,
                    "median": np.median(self.time_deltas) if self.time_deltas else None,
                    "std_dev": np.std(self.time_deltas) if self.time_deltas else None
                },
                "future_states": len(self.future_states),
                "historical_states": len(self.historical_states)
            }
            
            # Adiciona padrões detectados
            patterns = self.detect_temporal_patterns()
            if patterns.get("sufficient_data", False):
                stats["patterns"] = patterns
                
            return stats
