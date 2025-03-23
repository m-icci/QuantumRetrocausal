"""
Metacognitive Entropy Adaptor (MEA)
Camada adaptativa para QuantumNexus que permite aprendizado e auto-otimização
baseado em dinâmicas de entropia.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Callable
from qualia.core.metrics import QuantumMetrics
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class MetacognitiveEntropyAdaptor:
    def __init__(self, quantum_nexus, metrics: Optional[QuantumMetrics] = None):
        """
        Inicializa o Adaptador de Entropia Metacognitiva
        
        Args:
            quantum_nexus: Instância do QuantumNexus
            metrics: Instância opcional de QuantumMetrics
        """
        self.quantum_nexus = quantum_nexus
        self.metrics = metrics or QuantumMetrics()
        
        # Limiares de adaptação
        self.entropy_thresholds = {
            'exploration': {
                'min': 3.0,   # Limiar mínimo para exploração
                'max': 7.0    # Limiar máximo para exploração
            },
            'stabilization': {
                'min': 2.0,   # Limiar mínimo para estabilização
                'max': 5.0    # Limiar máximo para estabilização
            }
        }
        
        # Estratégias de adaptação
        self.adaptation_strategies = {
            'high_coherence': self._increase_quantum_coherence,
            'structural_compression': self._compress_kolmogorov_structure,
            'controlled_randomness': self._modulate_shannon_diversity
        }
        
        # Histórico de adaptações
        self.adaptation_history: List[Dict[str, Any]] = []

    def analyze_entropy_dynamics(self, quantum_state: np.ndarray) -> Dict[str, Any]:
        """
        Analisa as dinâmicas de entropia do estado quântico atual
        
        Args:
            quantum_state: Estado quântico a ser analisado
        
        Returns:
            Dicionário com métricas e estratégia de adaptação
        """
        logger.debug(f"Quantum state shape: {quantum_state.shape}")
        logger.debug(f"Quantum state min: {quantum_state.min()}, max: {quantum_state.max()}")
        
        # Calcula métricas de entropia
        try:
            entropy_metrics = self.metrics.get_quantum_metrics(quantum_state)
            logger.debug(f"Entropy metrics: {entropy_metrics}")
        except Exception as e:
            logger.error(f"Error calculating quantum metrics: {e}")
            entropy_metrics = {'entropia': 0, 'complexidade': 0}
        
        # Seleciona estratégia de adaptação
        try:
            strategy_func = self._select_adaptation_strategy(entropy_metrics)
            logger.debug(f"Selected strategy: {strategy_func.__name__}")
        except Exception as e:
            logger.error(f"Error selecting adaptation strategy: {e}")
            strategy_func = self.adaptation_strategies['controlled_randomness']
        
        # Registra histórico de adaptação
        adaptation_result = {
            'metrics': entropy_metrics,
            'strategy': strategy_func.__name__,
            'strategy_func': strategy_func
        }
        self.adaptation_history.append(adaptation_result)
        
        return adaptation_result

    def _select_adaptation_strategy(self, entropy_metrics: Dict[str, float]) -> Callable:
        """
        Seleciona estratégia de adaptação com base em métricas de entropia
        
        Args:
            entropy_metrics: Métricas de entropia do estado quântico
        
        Returns:
            Função de estratégia de adaptação
        """
        # Extrai métricas
        entropia = entropy_metrics.get('entropia', 0)
        complexidade = entropy_metrics.get('complexidade', 0)
        coerencia = entropy_metrics.get('coerencia', 0)
        
        # Definição de limiares dinâmicos
        limiar_entropia_baixa = 1.0
        limiar_entropia_alta = 3.0
        limiar_complexidade_baixa = 1.0
        limiar_complexidade_alta = 3.0
        limiar_coerencia_baixa = 0.2
        limiar_coerencia_alta = 0.8
        
        # Lógica de seleção de estratégia
        if entropia < limiar_entropia_baixa and complexidade < limiar_complexidade_baixa:
            return self.adaptation_strategies['controlled_randomness']
        
        elif entropia > limiar_entropia_alta and complexidade > limiar_complexidade_alta:
            return self.adaptation_strategies['structural_compression']
        
        elif coerencia < limiar_coerencia_baixa:
            return self.adaptation_strategies['high_coherence']
        
        elif coerencia > limiar_coerencia_alta:
            return self.adaptation_strategies['controlled_randomness']
        
        # Estratégia padrão com probabilidade variável
        estrategias_possiveis = list(self.adaptation_strategies.values())
        pesos = [0.3, 0.2, 0.3, 0.2]  # Pesos para distribuição probabilística
        
        return np.random.choice(estrategias_possiveis, p=pesos)

    def _increase_quantum_coherence(self, quantum_state: np.ndarray) -> np.ndarray:
        """
        Estratégia para aumentar coerência quântica
        
        Args:
            quantum_state: Estado quântico atual
        
        Returns:
            Estado quântico modificado com maior coerência
        """
        # Aplica transformações para aumentar coerência
        phi = (1 + np.sqrt(5)) / 2  # Proporção Áurea
        return np.sin(phi * quantum_state) * np.cos(quantum_state)

    def _compress_kolmogorov_structure(self, quantum_state: np.ndarray) -> np.ndarray:
        """
        Estratégia para comprimir estrutura de Kolmogorov
        
        Args:
            quantum_state: Estado quântico atual
        
        Returns:
            Estado quântico com estrutura comprimida
        """
        # Compressão usando transformações não-lineares
        compressed = np.log(1 + np.abs(quantum_state)) * np.sign(quantum_state)
        return compressed

    def _modulate_shannon_diversity(self, quantum_state: np.ndarray) -> np.ndarray:
        """
        Estratégia para modular diversidade de Shannon
        
        Args:
            quantum_state: Estado quântico atual
        
        Returns:
            Estado quântico com diversidade modulada
        """
        # Adiciona ruído controlado para diversidade
        noise = np.random.normal(0, 0.1, quantum_state.shape)
        return quantum_state + noise * np.std(quantum_state)

    def get_adaptation_summary(self) -> Dict[str, Any]:
        """
        Gera um resumo das adaptações metacognitivas
        
        Returns:
            Dicionário com sumário de adaptações
        """
        if not self.adaptation_history:
            return {
                "total_adaptacoes": 0,
                "distribuicao_estrategias": {},
                "estatisticas_entropia": {
                    "media": 0,
                    "std": 0,
                    "min": 0,
                    "max": 0
                }
            }
        
        # Extrai valores de entropia
        entropy_values = [entry['metrics']['entropia'] for entry in self.adaptation_history]
        strategy_names = [entry['strategy'] for entry in self.adaptation_history]
        
        # Distribuição de estratégias
        estrategias_unicas = set(strategy_names)
        distribuicao_estrategias = {
            estrategia: strategy_names.count(estrategia) / len(strategy_names) * 100 
            for estrategia in estrategias_unicas
        }
        
        # Estatísticas de entropia
        estatisticas_entropia = {
            "media": np.mean(entropy_values),
            "std": np.std(entropy_values),
            "min": np.min(entropy_values),
            "max": np.max(entropy_values)
        }
        
        return {
            "total_adaptacoes": len(self.adaptation_history),
            "distribuicao_estrategias": distribuicao_estrategias,
            "estatisticas_entropia": estatisticas_entropia
        }
