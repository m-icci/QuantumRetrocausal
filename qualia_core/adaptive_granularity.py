"""
Sistema de Granularidade Adaptativa para QUALIA
"""
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from enum import Enum
from core.constants import QualiaConstants
from core.monitoring.metrics_collector import MetricsCollector
import math
import logging

logger = logging.getLogger(__name__)

class ProcessingMode(Enum):
    """Modos de processamento do QUALIA"""
    ATOMIC = "atomic"           # Operações atômicas (1-2 bits)
    BALANCED = "balanced"       # Processamento balanceado (5-34 bits)
    EMERGENT = "emergent"      # Padrões emergentes (144-377 bits)

@dataclass
class GranularityMetrics:
    """Métricas de desempenho para cada granularidade"""
    coherence: float           # Coerência quântica
    resonance: float          # Ressonância do sistema
    entropy: float           # Medida de complexidade do sistema
    stability: float          # Estabilidade do processamento
    execution_time: float     # Tempo de execução
    timestamp: datetime = datetime.now()

class AdaptiveGranularity:
    """Controlador de granularidade adaptativa baseado em Fibonacci"""
    
    def __init__(self):
        # Níveis de granularidade por modo
        self.atomic_granularities = QualiaConstants.ATOMIC_GRANULARITIES
        self.balanced_granularities = QualiaConstants.BALANCED_GRANULARITIES
        self.emergent_granularities = QualiaConstants.EMERGENT_GRANULARITIES
        
        # Cache de métricas por granularidade
        self.metrics_cache: Dict[int, List[GranularityMetrics]] = {}
        
        # Estado atual
        self.current_mode = ProcessingMode.BALANCED
        self.current_granularity: int = self.balanced_granularities[2]  # Granularidade inicial = 21
        
        # Coletor de métricas
        self.metrics_collector = MetricsCollector()
        
    def select_granularity(self, mode: ProcessingMode, hardware_load: float,
                          coherence_threshold: float = QualiaConstants.MIN_COHERENCE_THRESHOLD) -> int:
        """
        Seleciona granularidade ideal baseado no modo e carga do hardware
        
        Args:
            mode: Modo de processamento desejado
            hardware_load: Carga atual do hardware (0-1)
            coherence_threshold: Limiar mínimo de coerência
            
        Returns:
            Granularidade otimizada
        """
        self.current_mode = mode
        available_granularities = {
            ProcessingMode.ATOMIC: self.atomic_granularities,
            ProcessingMode.BALANCED: self.balanced_granularities,
            ProcessingMode.EMERGENT: self.emergent_granularities
        }[mode]
        
        # Se não há métricas, seleciona granularidade inicial para o modo
        if not any(g in self.metrics_cache for g in available_granularities):
            if mode == ProcessingMode.ATOMIC:
                return available_granularities[0]  # Menor granularidade
            elif mode == ProcessingMode.EMERGENT:
                return available_granularities[-1]  # Maior granularidade
            else:
                return available_granularities[len(available_granularities)//2]  # Granularidade média
        
        # Verifica entropia média do sistema
        avg_entropy = 0.0
        count = 0
        for g in available_granularities:
            if g in self.metrics_cache and self.metrics_cache[g]:
                avg_entropy += self.metrics_cache[g][-1].entropy
                count += 1
        
        if count > 0:
            avg_entropy /= count
        
        # Seleciona candidatos baseado na entropia e carga do hardware
        if hardware_load >= QualiaConstants.HARDWARE_LOAD_HIGH:
            # Com alta carga, sempre usa granularidades menores
            candidates = sorted(available_granularities)[:2]
        else:
            # Para outras situações, calcula scores para todas as granularidades
            candidates = available_granularities
            
        # Seleciona melhor granularidade baseado nas métricas
        best_granularity = None
        best_score = float('-inf')
        
        for g in candidates:
            if g in self.metrics_cache and self.metrics_cache[g]:
                self.current_granularity = g  # Temporariamente define para calcular score
                metrics = self.metrics_cache[g][-1]
                score = self._calculate_efficiency_score(metrics, hardware_load)
                
                if score > best_score and metrics.coherence >= coherence_threshold:
                    best_score = score
                    best_granularity = g
        
        # Se não encontrou granularidade adequada, usa a menor do modo atual
        if best_granularity is None:
            best_granularity = min(available_granularities)
            
        self.current_granularity = best_granularity
        return best_granularity
    
    def update_metrics(self, granularity: int, metrics: GranularityMetrics) -> None:
        """
        Atualiza métricas para uma granularidade específica e ajusta a granularidade se necessário
        
        Args:
            granularity: Granularidade a ser atualizada
            metrics: Novas métricas
        """
        if granularity not in self.metrics_cache:
            self.metrics_cache[granularity] = []
            
        # Mantém histórico limitado de métricas
        self.metrics_cache[granularity] = (
            self.metrics_cache[granularity][-9:] + [metrics]
        )
        
        # Atualiza métricas no coletor
        self.metrics_collector.collect_system_metrics(metrics, granularity)
        
        # Avalia se deve mudar a granularidade
        score = self._calculate_efficiency_score(metrics, 0.5)  # Usando carga média como base
        
        # Decide se deve aumentar ou diminuir a granularidade
        if score < 0.7:  # Score baixo, tenta diminuir granularidade
            candidates = [g for g in self.balanced_granularities if g < granularity]
            if candidates:
                self.current_granularity = max(candidates)
        elif score > 0.75:  # Score alto, tenta aumentar granularidade
            candidates = [g for g in self.balanced_granularities if g > granularity]
            if candidates:
                self.current_granularity = min(candidates)
    
    def _calculate_efficiency_score(self, metrics: GranularityMetrics, hardware_load: float) -> float:
        """Calcula score composto considerando múltiplas métricas"""
        # Pesos para cada métrica
        weights = {
            'coherence': 0.3,
            'resonance': 0.2,
            'entropy': 0.2,
            'stability': 0.2,
            'performance': 0.1
        }
        
        # Normaliza o tempo de execução (quanto menor, melhor)
        performance_score = 1.0 / (1.0 + metrics.execution_time * 10)
        
        # Calcula score ponderado
        score = (
            weights['coherence'] * metrics.coherence +
            weights['resonance'] * metrics.resonance +
            weights['entropy'] * (1 - metrics.entropy) +  # Menor entropia é melhor
            weights['stability'] * metrics.stability +
            weights['performance'] * performance_score
        )
        
        # Ajusta score com base na carga do hardware
        hardware_factor = 1.0 - (hardware_load * 0.5)  # Carga alta reduz o score
        score *= hardware_factor
        
        return score
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Gera relatório de desempenho do sistema adaptativo"""
        report = {
            "current_mode": self.current_mode.value,
            "current_granularity": self.current_granularity,
            "metrics_by_granularity": {},
            "timestamp": datetime.now()
        }
        
        # Agrega métricas por granularidade
        for g, metrics_list in self.metrics_cache.items():
            if metrics_list:
                avg_metrics = {
                    "coherence": np.mean([m.coherence for m in metrics_list]),
                    "resonance": np.mean([m.resonance for m in metrics_list]),
                    "entropy": np.mean([m.entropy for m in metrics_list]),
                    "stability": np.mean([m.stability for m in metrics_list]),
                    "execution_time": np.mean([m.execution_time for m in metrics_list])
                }
                report["metrics_by_granularity"][g] = avg_metrics
        
        return report

class AdaptiveGranularitySystem:
    """
    Sistema de Granularidade Adaptativa Quântica para Mineração
    
    Características:
    1. Otimizado para processamento de mineração
    2. Utiliza sequências Fibonacci para granularidade natural
    3. Antecipação retrocausal de métricas
    4. Auto-ajuste baseado em métricas de mineração em tempo real
    """
    
    def __init__(
        self,
        fibonacci_levels: bool = True,
        min_granularity: int = 1,
        max_granularity: int = 377,
        learning_rate: float = 0.1
    ):
        """
        Inicializa o sistema de granularidade adaptativa
        
        Args:
            fibonacci_levels: Usar níveis Fibonacci para granularidade
            min_granularity: Granularidade mínima
            max_granularity: Granularidade máxima
            learning_rate: Taxa de aprendizado para adaptação
        """
        self.fibonacci_levels = fibonacci_levels
        self.min_granularity = min_granularity
        self.max_granularity = max_granularity
        self.learning_rate = learning_rate
        
        # Níveis de granularidade disponíveis
        if fibonacci_levels:
            self.available_granularities = self._generate_fibonacci_sequence(
                min_value=min_granularity,
                max_value=max_granularity
            )
        else:
            self.available_granularities = list(range(min_granularity, max_granularity + 1))
        
        # Métricas por granularidade
        self.metrics_cache = {}
        
        # Granularidade atual
        self.current_granularity = self.available_granularities[len(self.available_granularities) // 2]
        
        # Histórico de eficiência por granularidade
        self.efficiency_history = {g: [] for g in self.available_granularities}
        
        # Métricas de adaptação retrocausal
        self.retrocausal_predictions = {}
        
        logger.info(f"Sistema de Granularidade Adaptativa inicializado com {len(self.available_granularities)} níveis")
    
    def _generate_fibonacci_sequence(self, min_value: int = 1, max_value: int = 377) -> List[int]:
        """Gera sequência de Fibonacci dentro do intervalo especificado"""
        sequence = [1, 2]
        while sequence[-1] + sequence[-2] <= max_value:
            sequence.append(sequence[-1] + sequence[-2])
        
        return [n for n in sequence if min_value <= n <= max_value]
    
    def select_optimal_granularity(
        self,
        difficulty: float,
        hardware_load: float,
        coherence_threshold: float = 0.85,
        mining_metrics: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Seleciona granularidade ótima baseado em múltiplas métricas
        
        Args:
            difficulty: Dificuldade atual da rede
            hardware_load: Carga atual do hardware (0-1)
            coherence_threshold: Limiar de coerência mínima
            mining_metrics: Métricas adicionais de mineração
            
        Returns:
            Granularidade ótima para as condições atuais
        """
        # Se não houver histórico, retorna granularidade padrão
        if not any(self.efficiency_history.values()):
            return self.current_granularity
        
        # Calcula pontuação para cada granularidade
        scores = {}
        for granularity in self.available_granularities:
            # Fatores de pontuação
            coherence_factor = self._get_coherence_factor(granularity)
            hardware_factor = self._calculate_hardware_factor(granularity, hardware_load)
            difficulty_factor = self._calculate_difficulty_factor(granularity, difficulty)
            history_factor = self._calculate_history_factor(granularity)
            
            # Pontuação final combinada
            scores[granularity] = (
                0.35 * coherence_factor +
                0.25 * hardware_factor +
                0.25 * difficulty_factor +
                0.15 * history_factor
            )
            
            # Aplica ajuste retrocausal se disponível
            if granularity in self.retrocausal_predictions:
                retrocausal_bonus = self.retrocausal_predictions[granularity] * 0.1
                scores[granularity] += retrocausal_bonus
        
        # Seleciona granularidade com maior pontuação
        optimal_granularity = max(scores.items(), key=lambda x: x[1])[0]
        
        # Atualiza granularidade atual
        self.current_granularity = optimal_granularity
        
        logger.debug(f"Granularidade ótima selecionada: {optimal_granularity} (score: {scores[optimal_granularity]:.4f})")
        
        return optimal_granularity
    
    def _get_coherence_factor(self, granularity: int) -> float:
        """Obtém fator de coerência para a granularidade"""
        # Proporção áurea para coerência
        phi = (1 + 5 ** 0.5) / 2
        return 0.65 + 0.35 * (1 / (1 + abs(math.log(granularity / (phi ** 8)))))
    
    def _calculate_hardware_factor(self, granularity: int, hardware_load: float) -> float:
        """Calcula fator de hardware para a granularidade"""
        # Granularidade pequena é melhor com carga alta
        if hardware_load > 0.8 and granularity > self.available_granularities[len(self.available_granularities) // 2]:
            return 0.5 - (hardware_load - 0.8) * 0.5
        
        # Granularidade grande é melhor com carga baixa
        if hardware_load < 0.3 and granularity < self.available_granularities[len(self.available_granularities) // 2]:
            return 0.5 - (0.3 - hardware_load) * 0.5
            
        return 1.0
    
    def _calculate_difficulty_factor(self, granularity: int, difficulty: float) -> float:
        """Calcula fator de dificuldade para a granularidade"""
        # Dificuldade alta favorece granularidades menores para busca mais intensiva
        log_difficulty = math.log10(max(1.0, difficulty))
        if log_difficulty > 7:  # Dificuldade muito alta
            if granularity in self.available_granularities[:3]:  # Granularidades menores
                return 1.0
            else:
                return 0.7
        elif log_difficulty < 5:  # Dificuldade baixa
            if granularity in self.available_granularities[-3:]:  # Granularidades maiores
                return 1.0
            else:
                return 0.8
                
        return 0.9  # Dificuldade média
    
    def _calculate_history_factor(self, granularity: int) -> float:
        """Calcula fator histórico para a granularidade"""
        if not self.efficiency_history[granularity]:
            return 0.5  # Valor neutro para granularidades sem histórico
            
        # Média ponderada com mais peso para observações recentes
        recent_efficiencies = self.efficiency_history[granularity][-5:]
        if not recent_efficiencies:
            return 0.5
            
        weights = [0.1, 0.15, 0.2, 0.25, 0.3][:len(recent_efficiencies)]
        weights = [w / sum(weights) for w in weights]
        
        weighted_avg = sum(e * w for e, w in zip(recent_efficiencies, weights))
        return weighted_avg
    
    def update_metrics(self, granularity: int, mining_performance: Dict[str, Any]) -> None:
        """
        Atualiza métricas para a granularidade baseado no desempenho de mineração
        
        Args:
            granularity: Granularidade utilizada
            mining_performance: Métricas de desempenho da mineração
        """
        if granularity not in self.available_granularities:
            return
            
        # Calcula eficiência normalizada
        hashrate = mining_performance.get('hashrate', 0)
        energy = mining_performance.get('energy_consumption', 1)
        shares = mining_performance.get('shares_accepted', 0)
        
        # Eficiência é hashrate por energia, normalizada
        efficiency = hashrate / energy if energy > 0 else 0
        efficiency = min(1.0, efficiency / 100.0)  # Normaliza para [0,1]
        
        # Adiciona ao histórico
        self.efficiency_history[granularity].append(efficiency)
        
        # Mantém histórico com tamanho limitado
        if len(self.efficiency_history[granularity]) > 20:
            self.efficiency_history[granularity].pop(0)
            
        # Atualiza cache de métricas
        if granularity not in self.metrics_cache:
            self.metrics_cache[granularity] = []
            
        self.metrics_cache[granularity].append({
            'efficiency': efficiency,
            'hashrate': hashrate,
            'energy': energy,
            'shares': shares,
            'timestamp': datetime.now().isoformat()
        })
        
        # Mantém cache com tamanho limitado
        if len(self.metrics_cache[granularity]) > 100:
            self.metrics_cache[granularity].pop(0)
    
    def predict_future_efficiency(self) -> Dict[int, float]:
        """
        Realiza previsão retrocausal de eficiência futura para cada granularidade
        
        Returns:
            Dicionário com previsões por granularidade
        """
        predictions = {}
        
        for granularity in self.available_granularities:
            history = self.efficiency_history[granularity]
            if len(history) < 5:
                continue
                
            # Análise de tendência simples
            recent_trend = sum(y - x for x, y in zip(history[:-1], history[1:])) / (len(history) - 1)
            
            # Previsão baseada na tendência recente
            last_value = history[-1]
            predicted = last_value + recent_trend * 3  # Prevê 3 passos à frente
            
            # Normaliza para intervalo válido [0, 1]
            predicted = max(0.0, min(1.0, predicted))
            
            predictions[granularity] = predicted
            
        # Atualiza previsões retrocausais
        self.retrocausal_predictions = predictions
        
        return predictions
