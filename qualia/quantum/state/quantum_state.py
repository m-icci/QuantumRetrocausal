import numpy as np
from typing import Union, Optional

class QuantumState:
    """
    Representa um estado quântico com métodos para manipulação e análise
    """
    def __init__(self, initial_state: Union[np.ndarray, list], name: Optional[str] = None):
        """
        Inicializar estado quântico

        Args:
            initial_state: Estado quântico inicial como numpy array ou lista
            name: Nome opcional para identificação do estado
        """
        # Garantir que o estado seja um numpy array complexo
        self.quantum_state = np.asarray(initial_state, dtype=np.complex128)
        self.name = name or "Unnamed Quantum State"

    def set_quantum_state(self, new_state: Union[np.ndarray, list]):
        """
        Definir novo estado quântico

        Args:
            new_state: Novo estado quântico
        """
        self.quantum_state = np.asarray(new_state, dtype=np.complex128)

    def normalize(self):
        """
        Normalizar o estado quântico
        """
        norm = np.linalg.norm(self.quantum_state)
        if norm != 0:
            self.quantum_state /= norm

    def __repr__(self):
        """
        Representação em string do estado quântico

        Returns:
            String representando o estado quântico
        """
        return f"QuantumState(name={self.name}, state={self.quantum_state})"
