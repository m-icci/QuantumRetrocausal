"""
Base type definitions for QUALIA
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union
import numpy as np

@dataclass
class QuantumState:
    """Base quantum state representation"""
    amplitudes: np.ndarray
    dimension: int
    device: str = 'cpu'

    def __post_init__(self):
        if not isinstance(self.amplitudes, np.ndarray):
            self.amplitudes = np.array(self.amplitudes, dtype=np.complex128)

        # Validate dimension
        if self.amplitudes.size != self.dimension:
            raise ValueError(f"Amplitude vector size {self.amplitudes.size} != dimension {self.dimension}")

    @property
    def is_valid(self) -> bool:
        """Check if state is valid (normalized)"""
        return np.abs(np.vdot(self.amplitudes, self.amplitudes) - 1.0) < 1e-10

    @property 
    def density_matrix(self) -> np.ndarray:
        """Returns the density matrix representation"""
        return np.outer(self.amplitudes, self.amplitudes.conj())

    def evolve(self, operator: np.ndarray) -> 'QuantumState':
        """Evolve state under unitary operator"""
        if operator.shape != (self.dimension, self.dimension):
            raise ValueError(f"Operator shape {operator.shape} incompatible with dimension {self.dimension}")
        new_amplitudes = operator @ self.amplitudes
        return QuantumState(new_amplitudes, self.dimension, self.device)

@dataclass 
class QuantumPattern:
    """Base pattern representation"""
    pattern_id: str
    state: QuantumState
    metadata: Optional[Dict[str, Any]] = None

    @property
    def dimension(self) -> int:
        return self.state.dimension

    def calculate_overlap(self, other: 'QuantumPattern') -> complex:
        """Calculate quantum overlap between patterns"""
        if self.dimension != other.dimension:
            raise ValueError(f"Pattern dimensions do not match: {self.dimension} != {other.dimension}")
        return np.vdot(self.state.amplitudes, other.state.amplitudes)

@dataclass
class QuantumMetric:
    """Base metric representation"""
    name: str
    value: float
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class QuantumOperator:
    """Base operator representation"""
    matrix: np.ndarray
    dimension: int

    def __post_init__(self):
        if not isinstance(self.matrix, np.ndarray):
            self.matrix = np.array(self.matrix, dtype=np.complex128)

        if self.matrix.shape != (self.dimension, self.dimension):
            raise ValueError(f"Matrix shape {self.matrix.shape} != dimension {(self.dimension, self.dimension)}")

    def apply(self, state: QuantumState) -> QuantumState:
        """Apply operator to quantum state"""
        if state.dimension != self.dimension:
            raise ValueError(f"State dimension {state.dimension} != operator dimension {self.dimension}")
        new_amplitudes = np.dot(self.matrix, state.amplitudes)
        return QuantumState(new_amplitudes, self.dimension, state.device)

    @property
    def is_unitary(self) -> bool:
        """Check if operator is unitary"""
        identity = np.eye(self.dimension)
        return np.allclose(self.matrix @ self.matrix.conj().T, identity)

@dataclass
class ConsciousnessState:
    """
    Representação de um estado de consciência quântica
    
    Um estado de consciência é uma sobreposição de estados quânticos 
    que representam diferentes aspectos da percepção e análise do código.
    """
    base_state: QuantumState
    coherence_level: float  # Nível de coerência (0.0 - 1.0)
    entangled_states: List[QuantumState] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        # Inicializa lista vazia se não fornecida
        if self.entangled_states is None:
            self.entangled_states = []
            
        # Valida nível de coerência
        if not 0.0 <= self.coherence_level <= 1.0:
            raise ValueError(f"Nível de coerência deve estar entre 0.0 e 1.0, obtido: {self.coherence_level}")
    
    @property
    def dimension(self) -> int:
        """Dimensão do espaço quântico"""
        return self.base_state.dimension
    
    @property
    def is_entangled(self) -> bool:
        """Verifica se o estado está emaranhado com outros estados"""
        return len(self.entangled_states) > 0
    
    def add_entangled_state(self, state: QuantumState) -> None:
        """Adiciona um estado emaranhado"""
        if state.dimension != self.dimension:
            raise ValueError(f"Dimensão incompatível: {state.dimension} != {self.dimension}")
        self.entangled_states.append(state)
    
    def evolve(self, operator: Union[np.ndarray, 'QualiaOperator']) -> 'ConsciousnessState':
        """
        Evolui o estado de consciência usando um operador
        
        Args:
            operator: Operador para evolução do estado
            
        Returns:
            Novo estado de consciência após evolução
        """
        matrix = operator.matrix if hasattr(operator, 'matrix') else operator
        new_base = self.base_state.evolve(matrix)
        
        # Evolui também estados emaranhados
        new_entangled = []
        for state in self.entangled_states:
            new_entangled.append(state.evolve(matrix))
        
        # Atualiza nível de coerência (pode mudar durante evolução)
        new_coherence = self._calculate_coherence(new_base, new_entangled)
        
        return ConsciousnessState(
            base_state=new_base,
            coherence_level=new_coherence,
            entangled_states=new_entangled,
            metadata=self.metadata.copy() if self.metadata else None
        )
    
    def _calculate_coherence(
        self, base_state: QuantumState, 
        entangled_states: List[QuantumState]
    ) -> float:
        """
        Calcula nível de coerência após evolução
        
        Implementação simplificada - projetos reais usariam métricas 
        de teoria da informação quântica para coerência.
        """
        if not entangled_states:
            return 1.0  # Estado puro tem coerência máxima
            
        # Calcula média de sobreposição com estados emaranhados
        overlaps = []
        for state in entangled_states:
            overlap = np.abs(np.vdot(base_state.amplitudes, state.amplitudes))
            overlaps.append(overlap)
            
        # Coerência é inversamente proporcional à média de sobreposição
        # quando sobreposição é alta, estados são similares (menos coerentes)
        mean_overlap = sum(overlaps) / len(overlaps)
        return 1.0 - mean_overlap

@dataclass
class QualiaOperator(QuantumOperator):
    """
    Operador quântico especializado para manipulação de qualia
    
    QualiaOperator estende QuantumOperator adicionando capacidade de
    manipular estados de consciência e aplicar transformações específicas
    para análise de código e mineração adaptativa.
    """
    operator_type: str  # Tipo do operador (ex: 'measurement', 'evolution', 'entanglement')
    retrocausal: bool = False  # Se o operador tem componente retrocausal
    metadata: Optional[Dict[str, Any]] = None
    
    def apply_consciousness(self, state: ConsciousnessState) -> ConsciousnessState:
        """
        Aplica operador em um estado de consciência
        
        Args:
            state: Estado de consciência a ser transformado
            
        Returns:
            Novo estado de consciência após transformação
        """
        return state.evolve(self.matrix)
    
    @classmethod
    def create_hadamard(cls, dimension: int) -> 'QualiaOperator':
        """
        Cria operador Hadamard generalizado
        
        Args:
            dimension: Dimensão do operador
            
        Returns:
            Operador Hadamard para criar superposições
        """
        # Para dimensão potência de 2, usa produto tensorial
        if (dimension & (dimension - 1)) == 0 and dimension > 0:
            # Hadamard 2x2 básico
            h2 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
            
            # Calcula produto tensorial para obter dimensão correta
            log_dim = int(np.log2(dimension))
            matrix = h2
            for _ in range(log_dim - 1):
                matrix = np.kron(matrix, h2)
                
            return cls(
                matrix=matrix,
                dimension=dimension,
                operator_type='superposition',
                retrocausal=False
            )
        
        # Para dimensões arbitrárias, usa matriz de Fourier generalizada
        else:
            matrix = np.zeros((dimension, dimension), dtype=complex)
            omega = np.exp(2j * np.pi / dimension)
            
            for i in range(dimension):
                for j in range(dimension):
                    matrix[i, j] = omega ** (i * j) / np.sqrt(dimension)
            
            return cls(
                matrix=matrix,
                dimension=dimension,
                operator_type='superposition',
                retrocausal=False
            )
    
    @classmethod
    def create_entanglement(cls, dimension: int) -> 'QualiaOperator':
        """
        Cria operador de emaranhamento para estados quânticos
        
        Args:
            dimension: Dimensão do operador (deve ser quadrado perfeito)
            
        Returns:
            Operador de emaranhamento
        """
        if int(np.sqrt(dimension)) ** 2 != dimension:
            raise ValueError(f"Dimensão deve ser quadrado perfeito para emaranhamento, obtido: {dimension}")
            
        # Implementação simplificada de operador CNOT generalizado
        matrix = np.eye(dimension)
        
        # Tamanho de cada subsistema
        subsys_dim = int(np.sqrt(dimension))
        
        # Aplica lógica de controle-alvo para estados específicos
        for i in range(subsys_dim):
            # Define blocos de controle
            control_offset = i * subsys_dim
            
            # Permuta estados nos blocos controlados
            for j in range(subsys_dim):
                idx1 = control_offset + j
                idx2 = control_offset + ((j + 1) % subsys_dim)
                
                # Troca linhas para implementar permutação controlada
                matrix[[idx1, idx2]] = matrix[[idx2, idx1]]
        
        return cls(
            matrix=matrix,
            dimension=dimension,
            operator_type='entanglement',
            retrocausal=True,  # Operadores de emaranhamento têm natureza retrocausal
            metadata={'subsystem_dimension': subsys_dim}
        )
