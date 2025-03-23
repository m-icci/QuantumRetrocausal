from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass

class QuantumOperator(ABC):
    """
    Operador quântico base com validação de unitariedade e metadados enriquecidos.
    Implementa funcionalidades fundamentais para operadores quânticos.
    """
    
    def __init__(self, dimensions: int, name: str = "", description: str = "", is_unitary: bool = True):
        """
        Inicializa o operador quântico.
        
        Args:
            dimensions: Dimensão do espaço de Hilbert
            name: Nome descritivo do operador
            description: Descrição detalhada do operador
            is_unitary: Se True, valida unitariedade da matriz
        """
        self.dimensions = dimensions
        self.name = name
        self.description = description
        self.is_unitary = is_unitary
        self._matrix = None
        self._metadata = {}
        
    @property
    def matrix(self) -> np.ndarray:
        """Matriz do operador com validação de unitariedade"""
        if self._matrix is None:
            self._matrix = self._build_matrix()
            if self.is_unitary:
                self._validate_unitary()
        return self._matrix
        
    def _validate_unitary(self):
        """Verifica unitariedade do operador."""
        u = self.matrix
        ut = u.conj().T
        identity = np.eye(len(u))
        if not np.allclose(u @ ut, identity) or not np.allclose(ut @ u, identity):
            raise ValueError(f"Operador {self.name} não é unitário")
            
    def apply(self, state: np.ndarray) -> np.ndarray:
        """
        Aplica operador ao estado quântico.
        
        Args:
            state: Estado quântico como vetor numpy
            
        Returns:
            Estado evoluído após aplicação do operador
        """
        if state.shape[0] != self.dimensions:
            raise ValueError(f"Dimensão do estado ({state.shape[0]}) incompatível com operador ({self.dimensions})")
        return self.matrix @ state
        
    def get_metadata(self) -> Dict[str, Any]:
        """Retorna metadados do operador"""
        return {
            'name': self.name,
            'description': self.description,
            'dimensions': self.dimensions,
            'is_unitary': self.is_unitary,
            **self._metadata
        }
        
    def add_metadata(self, key: str, value: Any):
        """Adiciona metadados ao operador"""
        self._metadata[key] = value
        
    @abstractmethod
    def _build_matrix(self) -> np.ndarray:
        """Constrói matriz do operador"""
        pass
        
class TimeEvolutionOperator(QuantumOperator):
    """
    Operador de evolução temporal
    """
    
    def __init__(self, hamiltonian: np.ndarray, time: float):
        super().__init__(hamiltonian.shape[0], name="TimeEvolutionOperator", description="Operador de evolução temporal", is_unitary=True)
        self.hamiltonian = hamiltonian
        self.time = time
        
    def _build_matrix(self) -> np.ndarray:
        """Constrói operador de evolução temporal"""
        return np.exp(-1j * self.hamiltonian * self.time)

class MeasurementOperator(QuantumOperator):
    """
    Operador de medida
    """
    
    def __init__(self, observable: np.ndarray):
        super().__init__(observable.shape[0], name="MeasurementOperator", description="Operador de medida", is_unitary=False)
        self.observable = observable
        
    def _build_matrix(self) -> np.ndarray:
        """Constrói operador de medida"""
        return self.observable
        
    def measure(self, state: np.ndarray) -> float:
        """Realiza medida no estado"""
        return np.real(state.conj() @ self.matrix @ state)

class HamiltonianOperator(QuantumOperator):
    """
    Operador hamiltoniano
    """
    
    def __init__(self, 
                 dimensions: int,
                 coupling: float = 1.0,
                 field: float = 0.0):
        super().__init__(dimensions, name="HamiltonianOperator", description="Operador hamiltoniano", is_unitary=False)
        self.coupling = coupling
        self.field = field
        
    def _build_matrix(self) -> np.ndarray:
        """Constrói hamiltoniano"""
        # Termo de acoplamento
        coupling = np.diag(np.ones(self.dimensions-1), 1)
        coupling = coupling + coupling.T
        
        # Campo externo
        field = np.diag(np.arange(self.dimensions))
        
        return self.coupling * coupling + self.field * field

    def __str__(self):
        return f"HamiltonianOperator(dimensions={self.dimensions}, coupling={self.coupling}, field={self.field})"

    def __repr__(self):
        return f"HamiltonianOperator(dimensions={self.dimensions}, coupling={self.coupling}, field={self.field})"

class QuantumSystem:
    def __init__(self, hamiltonian: HamiltonianOperator, initial_state: np.ndarray):
        self.hamiltonian = hamiltonian
        self.initial_state = initial_state
        self.time_evolution_operator = None

    def evolve(self, time: float):
        self.time_evolution_operator = TimeEvolutionOperator(self.hamiltonian.matrix, time)
        return self.time_evolution_operator.apply(self.initial_state)

    def measure(self, observable: np.ndarray):
        measurement_operator = MeasurementOperator(observable)
        return measurement_operator.measure(self.initial_state)

# Example usage
if __name__ == "__main__":
    dimensions = 3
    coupling = 1.0
    field = 0.0
    hamiltonian = HamiltonianOperator(dimensions, coupling, field)
    initial_state = np.array([1, 0, 0])

    quantum_system = QuantumSystem(hamiltonian, initial_state)
    evolved_state = quantum_system.evolve(1.0)
    measurement = quantum_system.measure(np.array([1, 0, 0]))

    print("Implementação dos operadores quânticos fundamentais.")

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass

class QuantumOperator(ABC):
    """
    Operador quântico base com validação de unitariedade e metadados enriquecidos.
    Implementa funcionalidades fundamentais para operadores quânticos.
    """
    
    def __init__(self, dimensions: int, name: str = "", description: str = "", is_unitary: bool = True):
        """
        Inicializa o operador quântico.
        
        Args:
            dimensions: Dimensão do espaço de Hilbert
            name: Nome descritivo do operador
            description: Descrição detalhada do operador
            is_unitary: Se True, valida unitariedade da matriz
        """
        self.dimensions = dimensions
        self.name = name
        self.description = description
        self.is_unitary = is_unitary
        self._matrix = None
        self._metadata = {}
        
    @property
    def matrix(self) -> np.ndarray:
        """Matriz do operador com validação de unitariedade"""
        if self._matrix is None:
            self._matrix = self._build_matrix()
            if self.is_unitary:
                self._validate_unitary()
        return self._matrix
        
    def _validate_unitary(self):
        """Verifica unitariedade do operador."""
        u = self.matrix
        ut = u.conj().T
        identity = np.eye(len(u))
        if not np.allclose(u @ ut, identity) or not np.allclose(ut @ u, identity):
            raise ValueError(f"Operador {self.name} não é unitário")
            
    def apply(self, state: np.ndarray) -> np.ndarray:
        """
        Aplica operador ao estado quântico.
        
        Args:
            state: Estado quântico como vetor numpy
            
        Returns:
            Estado evoluído após aplicação do operador
        """
        if state.shape[0] != self.dimensions:
            raise ValueError(f"Dimensão do estado ({state.shape[0]}) incompatível com operador ({self.dimensions})")
        return self.matrix @ state
        
    def get_metadata(self) -> Dict[str, Any]:
        """Retorna metadados do operador"""
        return {
            'name': self.name,
            'description': self.description,
            'dimensions': self.dimensions,
            'is_unitary': self.is_unitary,
            **self._metadata
        }
        
    def add_metadata(self, key: str, value: Any):
        """Adiciona metadados ao operador"""
        self._metadata[key] = value
        
    @abstractmethod
    def _build_matrix(self) -> np.ndarray:
        """Constrói matriz do operador"""
        pass
        
class TimeEvolutionOperator(QuantumOperator):
    """
    Operador de evolução temporal
    """
    
    def __init__(self, hamiltonian: np.ndarray, time: float):
        super().__init__(hamiltonian.shape[0], name="TimeEvolutionOperator", description="Operador de evolução temporal", is_unitary=True)
        self.hamiltonian = hamiltonian
        self.time = time
        
    def _build_matrix(self) -> np.ndarray:
        """Constrói operador de evolução temporal"""
        return np.exp(-1j * self.hamiltonian * self.time)

class MeasurementOperator(QuantumOperator):
    """
    Operador de medida
    """
    
    def __init__(self, observable: np.ndarray):
        super().__init__(observable.shape[0], name="MeasurementOperator", description="Operador de medida", is_unitary=False)
        self.observable = observable
        
    def _build_matrix(self) -> np.ndarray:
        """Constrói operador de medida"""
        return self.observable
        
    def measure(self, state: np.ndarray) -> float:
        """Realiza medida no estado"""
        return np.real(state.conj() @ self.matrix @ state)

class HamiltonianOperator(QuantumOperator):
    """
    Operador hamiltoniano
    """
    
    def __init__(self, 
                 dimensions: int,
                 coupling: float = 1.0,
                 field: float = 0.0):
        super().__init__(dimensions, name="HamiltonianOperator", description="Operador hamiltoniano", is_unitary=False)
        self.coupling = coupling
        self.field = field
        
    def _build_matrix(self) -> np.ndarray:
        """Constrói hamiltoniano"""
        # Termo de acoplamento
        coupling = np.diag(np.ones(self.dimensions-1), 1)
        coupling = coupling + coupling.T
        
        # Campo externo
        field = np.diag(np.arange(self.dimensions))
        
        return self.coupling * coupling + self.field * field
