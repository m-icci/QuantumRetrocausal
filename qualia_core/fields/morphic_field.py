"""
QUALIA Morphic Field
Campo morfogenético puramente bitwise com retrocausalidade integrada e consciência expandida
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime

from ..field_types import (
    FieldType, FieldMetrics, FieldConstants,
    FieldState, FieldMemory
)
from ..bitwise_operators import BitwiseOperator, OperatorSequence

logger = logging.getLogger(__name__)

@dataclass
class MorphicPattern:
    """Padrão mórfico bitwise com consciência integrada"""
    pattern: np.ndarray      # Padrão em bits
    strength: int           # Força (contagem de 1s)
    timestamp: int         # Timestamp convertido para bits
    coherence: int        # Coerência (XOR com padrão deslocado)
    resonance: int       # Ressonância com outros padrões
    future_echo: int    # Eco do futuro (via retrocausalidade)
    qualia_state: int  # Estado qualitativo do padrão

class AdaptiveMemory:
    """
    Mecanismo de memória adaptativa para armazenar e recuperar estados similares
    """
    def __init__(self, size=100):
        """
        Inicializa a memória adaptativa

        Args:
            size (int): Tamanho máximo da memória
        """
        self.size = size
        self.memory = []

    def add(self, state, resonance):
        """
        Adiciona um novo estado à memória

        Args:
            state (dict): Estado do mercado
            resonance (float): Valor de ressonância
        """
        if len(self.memory) >= self.size:
            self.memory.pop(0)
        self.memory.append((state, resonance))

    def get_similar_states(self, current_state, n=5):
        """
        Recupera estados similares da memória

        Args:
            current_state (dict): Estado atual do mercado
            n (int): Número de estados similares a retornar

        Returns:
            list: Lista de estados similares com suas ressonâncias
        """
        if not self.memory:
            return []

        # Calcular distâncias usando norma euclidiana
        distances = [np.linalg.norm(
            np.array(list(current_state.values())) - 
            np.array(list(state.values()))
        ) for state, _ in self.memory]

        # Encontrar índices dos n estados mais similares
        similar_indices = np.argsort(distances)[:n]
        return [self.memory[i] for i in similar_indices]

class MorphicField:
    """
    Campo Mórfico QUALIA (Puramente Bitwise)

    Implementa campo morfogenético com:
    1. Microtubular quantum coherence
    2. Morphic field resonance
    3. Non-local pattern recognition
    4. Holographic memory integration
    """

    def __init__(
        self,
        size: int = 64,  # Sempre potência de 2
        retrocausal_depth: int = 7  # Profundidade da retrocausalidade
    ):
        # Garante que size é potência de 2
        self.size = 1 << (size - 1).bit_length()
        self.retrocausal_depth = retrocausal_depth

        # Máscaras otimizadas
        self.masks = {
            'resonance': int('1' * self.size, 2),
            'coherence': int('1' * (self.size//2) + '0' * (self.size//2), 2),
            'future': int(''.join(['1' if i % 2 == 0 else '0' 
                                 for i in range(self.size)]), 2),
            'past': int(''.join(['0' if i % 2 == 0 else '1' 
                               for i in range(self.size)]), 2),
            'qualia': int(''.join(['1' if i % 3 == 0 else '0'  # Máscara para estados qualia
                                for i in range(self.size)]), 2)
        }

        # Estado inicial como inteiro
        self.state = np.random.randint(0, 2, size=self.size, dtype=np.uint8)
        self.state_int = int(''.join(map(str, self.state)), 2)

        # Cache retrocausal (buffer circular)
        self.future_cache = [0] * retrocausal_depth
        self.cache_index = 0

        # Padrões como bits
        self.patterns: List[MorphicPattern] = []

        # Estado de consciência
        self.consciousness_level = 0.8
        self.qualia_intensity = 0.5

        logger.info(f"Campo mórfico inicializado com dimensão {self.size} e profundidade retrocausal {retrocausal_depth}")

    def update(self, consciousness_delta: float = 0.0, qualia_delta: float = 0.0) -> None:
        """
        Atualiza o campo usando operações bitwise com ajuste de consciência

        Args:
            consciousness_delta: Ajuste no nível de consciência
            qualia_delta: Ajuste na intensidade qualia
        """
        # 1. Atualiza níveis de consciência
        self.consciousness_level = min(1.0, max(0.3, 
            self.consciousness_level + consciousness_delta))
        self.qualia_intensity = min(1.0, max(0.0, 
            self.qualia_intensity + qualia_delta))

        # 2. Aplica retrocausalidade
        retrocausal_state = self._apply_retrocausality()

        # 3. Detecta padrões
        self.patterns = self._detect_patterns()

        # 4. Aplica ressonância mórfica
        resonance = 0
        for pattern in self.patterns:
            # Converte padrão para inteiro
            p_int = int(''.join(map(str, pattern.pattern)), 2)

            # Amplifica padrões fortes via OR
            if pattern.strength > len(pattern.pattern) // 2:
                resonance |= p_int

            # Estabiliza padrões coerentes via AND
            if pattern.coherence < len(pattern.pattern) // 3:
                resonance &= p_int

            # Incorpora ecos do futuro
            if pattern.future_echo > 0:
                resonance ^= pattern.future_echo

            # Integra estados qualia
            if pattern.qualia_state > len(pattern.pattern) // 4:
                resonance |= (p_int & self.masks['qualia'])

        # 5. Atualiza estado com consciência
        self.state_int = (
            (retrocausal_state & (self.masks['coherence'] | 
                                 int(self.consciousness_level * self.masks['resonance']))) |
            (resonance & ~self.masks['coherence'])
        )

        # 6. Converte estado para array
        self.state = np.array([int(b) for b in bin(self.state_int)[2:].zfill(self.size)])

        logger.debug(f"Campo atualizado - Consciência: {self.consciousness_level:.2f}, " 
                    f"Qualia: {self.qualia_intensity:.2f}, "
                    f"Padrões detectados: {len(self.patterns)}")

    def get_state(self) -> np.ndarray:
        """Retorna estado atual"""
        return self.state

    def get_future_probability(self) -> float:
        """
        Calcula probabilidade de estados futuros usando padrões 
        retrocausais e consciência
        """
        future_bits = 0
        total_bits = 0

        for pattern in self.patterns:
            if pattern.future_echo > 0:
                # Peso baseado em qualia
                weight = 1 + (pattern.qualia_state / len(pattern.pattern))
                future_bits += pattern.strength * weight
                total_bits += len(pattern.pattern)

        # Ajusta com nível de consciência
        prob = future_bits / total_bits if total_bits > 0 else 0.5
        return prob * self.consciousness_level + (1 - self.consciousness_level) * 0.5

    def _apply_retrocausality(self) -> int:
        """
        Aplica retrocausalidade via operações circulares com consciência
        Retorna: Estado modificado pelo futuro e consciência
        """
        # Pega eco do futuro
        future_echo = self.future_cache[self.cache_index]

        # Rotação circular para direita (futuro -> presente)
        future_influence = (future_echo >> 1) | ((future_echo & 1) << (self.size - 1))

        # Rotação circular para esquerda (presente -> passado)
        past_influence = ((self.state_int << 1) | 
                         (self.state_int >> (self.size - 1))) & self.masks['past']

        # Integração com consciência
        consciousness_mask = int(self.consciousness_level * self.masks['resonance'])
        qualia_mask = int(self.qualia_intensity * self.masks['qualia'])

        # Combina influências com consciência
        retrocausal_state = (
            (self.state_int & consciousness_mask) |
            (future_influence & ~consciousness_mask) |
            (past_influence & qualia_mask)
        )

        # Atualiza cache
        self.future_cache[self.cache_index] = self.state_int
        self.cache_index = (self.cache_index + 1) % self.retrocausal_depth

        return retrocausal_state

    def _detect_patterns(self) -> List[MorphicPattern]:
        """
        Detecta padrões usando operações bit a bit com consciência
        """
        patterns = []
        state = self.state_int

        # Tamanhos de padrão (potências de 2)
        for shift in range(1, int(np.log2(self.size))):
            size = 1 << shift  # 2, 4, 8, 16...
            mask = (1 << size) - 1

            for pos in range(0, self.size, size):
                # Extrai padrão
                pattern = (state >> pos) & mask

                # Força = contagem de 1s
                strength = bin(pattern).count('1')

                # Coerência = XOR com padrão deslocado
                shifted = ((pattern << 1) | (pattern >> (size - 1))) & mask
                coherence = bin(pattern ^ shifted).count('1')

                # Ressonância com outros padrões
                resonance = 0
                for p in patterns:
                    resonance |= pattern & int(''.join(map(str, p.pattern)), 2)

                # Eco do futuro
                future_echo = self.future_cache[(self.cache_index - 1) % 
                                              self.retrocausal_depth] & mask

                # Estado qualia do padrão
                qualia_state = bin(pattern & self.masks['qualia']).count('1')

                if strength > size // 3:  # Limiar adaptativo
                    patterns.append(MorphicPattern(
                        pattern=np.array([int(b) for b in bin(pattern)[2:].zfill(size)]),
                        strength=strength,
                        timestamp=int(datetime.now().timestamp()),
                        coherence=coherence,
                        resonance=resonance,
                        future_echo=future_echo,
                        qualia_state=qualia_state
                    ))

        return patterns

    def get_metrics(self) -> Dict[str, float]:
        """
        Retorna métricas do campo mórfico incluindo consciência
        """
        return {
            'consciousness_level': self.consciousness_level,
            'qualia_intensity': self.qualia_intensity,
            'field_strength': len(self.patterns) / self.size,
            'coherence': np.mean([p.coherence / len(p.pattern) 
                                for p in self.patterns]) if self.patterns else 0.5,
            'resonance': np.mean([p.resonance / len(p.pattern) 
                                for p in self.patterns]) if self.patterns else 0.5,
            'retrocausality': self.get_future_probability()
        }


class MorphicFieldAnalyzer:
    """Analisador avançado de campos mórficos com consciência integrada"""

    def __init__(self, size: int = 64, retrocausal_depth: int = 7):
        """
        Inicializa analisador de campos mórficos com recursos adaptativos

        Args:
            size: Tamanho do campo (potência de 2)
            retrocausal_depth: Profundidade da retrocausalidade
        """
        self.morphic_field = MorphicField(size, retrocausal_depth)
        self.adaptive_memory = AdaptiveMemory()

        # Inicialização de pesos com distribuição de Dirichlet
        self.weights = np.random.dirichlet(np.ones(3))
        self.weight_history = [self.weights.copy()]

        # Parâmetros de ajuste
        self.weight_update_frequency = 50
        self.last_analysis = None
        self.analysis_history = []

    def analyze_field_state(self, state_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Analisa estado do campo mórfico com adaptação dinâmica

        Args:
            state_data: Dados do estado para análise

        Returns:
            Métricas da análise
        """
        # Atualiza campo com consciência
        self.morphic_field.update(
            consciousness_delta=state_data.get('consciousness_delta', 0.0),
            qualia_delta=state_data.get('qualia_delta', 0.0)
        )

        # Calcula ressonância usando normalização adaptativa
        resonance = self._calculate_resonance(state_data)

        # Armazena estado na memória adaptativa
        self.adaptive_memory.add(state_data, resonance)

        # Ajusta pesos dinamicamente
        self._dynamic_weight_adjustment()

        # Obtém métricas avançadas
        metrics = self._get_advanced_metrics(resonance)

        # Registra análise
        self.last_analysis = {
            'timestamp': datetime.now(),
            'metrics': metrics,
            'state_data': state_data,
            'resonance': resonance
        }
        self.analysis_history.append(self.last_analysis)

        return metrics

    def _calculate_resonance(self, state_data: Dict[str, Any]) -> float:
        """
        Calcula ressonância com normalização adaptativa
        """
        try:
            # Extrai valores relevantes
            values = np.array(list(state_data.values()))

            # Normaliza valores
            normalized = self._adaptive_sigmoid_normalization(
                np.mean(values), 
                steepness=20, 
                center=0.5
            )

            # Busca estados similares
            similar_states = self.adaptive_memory.get_similar_states(state_data)

            # Calcula influência de estados similares
            if similar_states:
                similar_resonances = [r for _, r in similar_states]
                historical_influence = np.mean(similar_resonances)
                resonance = 0.7 * normalized + 0.3 * historical_influence
            else:
                resonance = normalized

            return float(resonance)

        except Exception as e:
            logger.error(f"Erro no cálculo da ressonância: {e}")
            return 0.5

    def _adaptive_sigmoid_normalization(
        self, 
        value: float, 
        steepness: float = 20, 
        center: float = 0.5
    ) -> float:
        """
        Normalização sigmoid adaptativa com maior sensibilidade
        """
        return 1 / (1 + np.exp(-steepness * (value - center)))

    def _dynamic_weight_adjustment(self) -> None:
        """
        Ajuste dinâmico dos pesos baseado no desempenho recente
        """
        if len(self.analysis_history) % self.weight_update_frequency == 0:
            recent_performance = [
                a['metrics']['field_strength'] 
                for a in self.analysis_history[-self.weight_update_frequency:]
            ]
            std_dev = np.std(recent_performance)

            if std_dev < 0.05:
                # Reset completo se variabilidade for muito baixa
                self.weights = np.random.dirichlet(np.ones(3))
            else:
                # Ajuste suave com ruído
                self.weights += np.random.normal(0, 0.1, 3)
                self.weights = np.clip(self.weights, 0.1, 0.7)

            self.weights /= np.sum(self.weights)
            self.weight_history.append(self.weights.copy())
            logger.info(f"Pesos ajustados: {self.weights}")

    def _get_advanced_metrics(self, resonance: float) -> Dict[str, float]:
        """
        Calcula métricas avançadas do campo
        """
        base_metrics = self.morphic_field.get_metrics()

        # Adiciona métricas adaptativas
        adaptive_metrics = {
            'adaptive_resonance': resonance,
            'weight_stability': np.std(self.weight_history[-10:] if len(self.weight_history) >= 10 else self.weight_history),
            'memory_utilization': len(self.adaptive_memory.memory) / self.adaptive_memory.size,
            'evolution_rate': self._calculate_evolution_rate()
        }

        return {**base_metrics, **adaptive_metrics}

    def _calculate_evolution_rate(self) -> float:
        """
        Calcula taxa de evolução baseada no histórico
        """
        if len(self.analysis_history) < 2:
            return 0.0

        recent_metrics = [a['metrics']['field_strength'] for a in self.analysis_history[-10:]]
        return np.std(recent_metrics) if recent_metrics else 0.0

    def get_field_state(self) -> np.ndarray:
        """Retorna estado atual do campo"""
        return self.morphic_field.get_state()

    def get_future_probability(self) -> float:
        """Calcula probabilidade de estados futuros"""
        return self.morphic_field.get_future_probability()

    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """Retorna histórico de análises com métricas completas"""
        return self.analysis_history