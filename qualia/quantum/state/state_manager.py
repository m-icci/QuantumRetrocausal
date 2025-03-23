"""
QUALIA Quantum State Manager
--------------------------

Gerenciador de estados quânticos para o sistema QUALIA.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from .quantum_state import QuantumState
from ..utils.merge_utils import (
    calculate_quantum_coherence,
    calculate_entropy
)

class StateManager:
    """
    Gerencia estados quânticos do sistema QUALIA
    """
    def __init__(self):
        self.states: Dict[str, QuantumState] = {}
        self.state_history: List[Dict[str, Any]] = []
    
    def register_state(self, name: str, state: QuantumState) -> None:
        """
        Registra um novo estado quântico

        Args:
            name: Nome do estado
            state: Estado quântico
        """
        self.states[name] = state
        self._record_state_metrics(name, state)
    
    def get_state(self, name: str) -> Optional[QuantumState]:
        """
        Recupera um estado quântico

        Args:
            name: Nome do estado

        Returns:
            Estado quântico ou None se não encontrado
        """
        return self.states.get(name)
    
    def update_state(self, name: str, state: QuantumState) -> None:
        """
        Atualiza um estado existente

        Args:
            name: Nome do estado
            state: Novo estado
        """
        if name in self.states:
            self.states[name] = state
            self._record_state_metrics(name, state)
    
    def remove_state(self, name: str) -> None:
        """
        Remove um estado

        Args:
            name: Nome do estado
        """
        if name in self.states:
            del self.states[name]
    
    def _record_state_metrics(self, name: str, state: QuantumState) -> None:
        """
        Registra métricas de um estado

        Args:
            name: Nome do estado
            state: Estado quântico
        """
        metrics = {
            'name': name,
            'timestamp': datetime.now().timestamp(),
            'coherence': calculate_quantum_coherence(state.data),
            'entropy': calculate_entropy(state.data)
        }
        self.state_history.append(metrics)
    
    def get_state_metrics(self, name: str) -> Dict[str, Any]:
        """
        Obtém métricas de um estado

        Args:
            name: Nome do estado

        Returns:
            Dicionário com métricas do estado
        """
        state = self.states.get(name)
        if not state:
            return {}
            
        return {
            'coherence': calculate_quantum_coherence(state.data),
            'entropy': calculate_entropy(state.data)
        }
    
    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """
        Obtém histórico de métricas

        Returns:
            Lista com histórico de métricas
        """
        return self.state_history
