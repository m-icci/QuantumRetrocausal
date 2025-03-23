"""
Implementação do módulo iSConsciousness com integração temporal e quântica.
"""

from typing import Dict, Any, List
import logging
import numpy as np

from .iconsciousness import IConsciousness
from .quantum_consciousness import QuantumConsciousness
from quantum.core.qtypes.qualia_types import QualiaState
from quantum.core.qtypes.system_behavior import SystemBehavior

def calculate_entropy(sequence, base=2, epsilon=1e-10):
    """
    Calcula a entropia de uma sequência com tratamento robusto de erros

    Args:
        sequence (list or np.ndarray): Sequência de entrada
        base (int): Base logarítmica para cálculo de entropia
        epsilon (float): Valor pequeno para prevenir log(0)

    Returns:
        float: Valor calculado da entropia
    """
    if not isinstance(sequence, np.ndarray):
        sequence = np.array(sequence)

    if sequence.size == 0:
        return 0.0

    unique, counts = np.unique(sequence, return_counts=True)
    probabilities = counts / len(sequence)

    if len(unique) <= 1 or np.max(probabilities) >= 0.99:
        return 0.0

    entropy = -np.sum(probabilities * np.log(probabilities + epsilon) / np.log(base))
    return 0.0 if entropy < epsilon else entropy

class iSConsciousnessModule:
    """
    Módulo iSConsciousness para Sistema de Consulta Temporal Quântico-Sincronizado

    Gerencia a tomada de decisão adaptativa e rastreamento de contexto para
    medições temporais quânticas.
    """

    def __init__(self, quantum_system=None):
        """
        Inicializa o Módulo iSConsciousness

        Args:
            quantum_system: Sistema quântico opcional para inicialização
        """
        self.quantum_system = quantum_system or QuantumConsciousness()
        self.consciousness = IConsciousness()

        # Configuração de logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)

        # Estado e histórico
        self.temporal_history: List[Dict[str, Any]] = []
        self.context_window: List[Dict[str, Any]] = []
        self.window_size = 100
        self.adaptive_threshold = 0.5

    def process_temporal_measurement(self, measurement: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processa uma medição temporal e atualiza o estado interno

        Args:
            measurement: Medição temporal a ser processada

        Returns:
            Dict[str, Any]: Resultado do processamento
        """
        # Atualiza histórico
        self.temporal_history.append(measurement)
        self.context_window.append(measurement)

        if len(self.context_window) > self.window_size:
            self.context_window.pop(0)

        # Calcula métricas
        entropy = calculate_entropy([m.get('value', 0) for m in self.context_window])
        quantum_state = self.quantum_system.measure_state()

        # Atualiza consciência
        behavior = SystemBehavior(
            entropy=entropy,
            coherence=quantum_state['coherence'],
            complexity=quantum_state['complexity']
        )
        self.consciousness.process_system_behavior(behavior)

        return {
            'entropy': entropy,
            'quantum_state': quantum_state,
            'qualia_state': self.consciousness.get_qualia_state()
        }

    def get_current_state(self) -> Dict[str, Any]:
        """
        Retorna o estado atual do sistema

        Returns:
            Dict[str, Any]: Estado atual
        """
        return {
            'temporal_metrics': {
                'history_length': len(self.temporal_history),
                'context_size': len(self.context_window)
            },
            'quantum_state': self.quantum_system.measure_state(),
            'consciousness_state': self.consciousness.get_qualia_state()
        }