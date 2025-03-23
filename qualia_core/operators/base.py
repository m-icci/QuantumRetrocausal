"""
Operador quântico base
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from quantum.core.quantum_state import QuantumState

class BaseQuantumOperator(ABC):
    """Classe base para operadores quânticos"""
    
    def __init__(self, name: str):
        """
        Inicializa operador base
        
        Args:
            name: Nome do operador
        """
        self.name = name
        
    @abstractmethod
    def apply(self, state: QuantumState) -> QuantumState:
        """
        Aplica operador ao estado
        
        Args:
            state: Estado quântico
            
        Returns:
            Novo estado quântico
        """
        pass
        
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """
        Retorna métricas do operador
        
        Returns:
            Dicionário com métricas
        """
        pass
