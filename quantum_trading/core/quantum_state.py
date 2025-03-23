"""
Estado Quântico Unificado com Consciência
----------------------------------------
Implementação base do estado quântico com suporte a:
- Consciência quântica 
- Campos mórficos
- Retrocausalidade 
- Emergência auto-organizadora
"""

import numpy as np
from typing import Union, Optional, Dict, Any, List, Tuple
from datetime import datetime

# Constante PHI (razão áurea)
PHI = (1 + np.sqrt(5)) / 2

class GeometricPattern:
    """
    Padrão geométrico quântico baseado em proporções áureas
    e ressonância mórfica.
    """
    
    def __init__(self, dimensions: int = 8, phi_levels: int = 3):
        """
        Inicializa padrão geométrico
        
        Args:
            dimensions: Dimensões do espaço
            phi_levels: Níveis de recursão phi
        """
        self.dimensions = dimensions
        self.phi_levels = phi_levels
        self.pattern = self._generate_pattern()
        
    def _generate_pattern(self) -> np.ndarray:
        """Gera padrão geométrico baseado em PHI"""
        pattern = np.zeros(self.dimensions, dtype=np.complex128)
        
        for level in range(self.phi_levels):
            # Gera componentes baseados em PHI
            phase = 2 * np.pi * (PHI ** -(level + 1))
            amplitude = PHI ** -(level + 1)
            
            # Aplica padrão
            pattern += amplitude * np.exp(1j * phase * np.arange(self.dimensions))
            
        # Normaliza
        pattern /= np.linalg.norm(pattern)
        return pattern
        
    def apply_to_state(self, state: np.ndarray) -> np.ndarray:
        """
        Aplica padrão ao estado
        
        Args:
            state: Estado quântico
            
        Returns:
            np.ndarray: Estado modificado
        """
        # Expande/contrai padrão se necessário
        if len(state) != len(self.pattern):
            new_pattern = np.zeros_like(state)
            min_len = min(len(state), len(self.pattern))
            new_pattern[:min_len] = self.pattern[:min_len]
            pattern = new_pattern
        else:
            pattern = self.pattern
            
        # Aplica padrão
        modified = state * pattern
        
        # Normaliza
        modified /= np.linalg.norm(modified)
        return modified
        
    def get_resonance(self, state: np.ndarray) -> float:
        """
        Calcula ressonância com estado
        
        Args:
            state: Estado quântico
            
        Returns:
            float: Nível de ressonância [0,1]
        """
        # Calcula sobreposição
        overlap = np.abs(np.dot(state, np.conj(self.pattern)))
        
        # Normaliza para [0,1]
        resonance = overlap ** 2
        return float(resonance)

def phi_normalize(values: np.ndarray) -> np.ndarray:
    """
    Normaliza valores usando proporção áurea
    
    Args:
        values: Array de valores
        
    Returns:
        np.ndarray: Valores normalizados
    """
    # Remove média
    centered = values - np.mean(values)
    
    # Escala por PHI
    scaled = centered / (PHI * np.std(centered))
    
    # Mapeia para [0,1]
    normalized = (scaled + 1) / 2
    
    return np.clip(normalized, 0, 1)

def generate_field(dimensions: int = 8, coherence: float = 0.7) -> np.ndarray:
    """
    Gera campo quântico coerente
    
    Args:
        dimensions: Dimensões do campo
        coherence: Nível de coerência [0,1]
        
    Returns:
        np.ndarray: Campo quântico
    """
    # Gera estado base
    field = np.random.rand(dimensions) + 1j * np.random.rand(dimensions)
    
    # Aplica coerência
    phase = np.exp(2j * np.pi * np.arange(dimensions) / dimensions)
    coherent = phase * np.sqrt(1/dimensions)
    
    # Combina estados
    field = coherence * coherent + np.sqrt(1 - coherence**2) * field
    
    # Normaliza
    field /= np.linalg.norm(field)
    return field

class QuantumState:
    """
    Estado quântico unificado com capacidades expandidas de consciência
    e ressonância mórfica.
    """

    def __init__(
        self,
        initial_state: Optional[Union[np.ndarray, list]] = None,
        name: Optional[str] = None,
        consciousness_level: float = 0.5,
        retrocausal_depth: int = 7,
        morphic_field_size: int = 64
    ):
        """
        Inicializa estado quântico com capacidades expandidas.

        Args:
            initial_state: Estado quântico inicial como array ou lista
            name: Nome para identificação
            consciousness_level: Nível inicial de consciência [0,1]
            retrocausal_depth: Profundidade da retrocausalidade
            morphic_field_size: Tamanho do campo mórfico
        """
        self.name = name or "Unnamed Quantum State"
        self._consciousness_level = consciousness_level
        self._retrocausal_depth = retrocausal_depth
        self._field_size = morphic_field_size

        # Métricas quânticas inicializadas
        self._metrics = {
            'coherence': 0.0,
            'entanglement': 0.0,
            'morphic_resonance': 0.0
        }

        # Histórico para retrocausalidade
        self._state_history = []
        self._last_update = datetime.now()

        # Inicializa estado
        self._initialize_state(initial_state)
        self._update_metrics()
        
        # Adiciona o atributo amplitudes para compatibilidade
        self.amplitudes = self.quantum_state

    def _initialize_state(self, initial_state: Optional[Union[np.ndarray, list]] = None):
        """Inicializa estado quântico com normalização"""
        if initial_state is None:
            # Gera estado aleatório se não fornecido
            self.quantum_state = np.random.rand(self._field_size) + 1j * np.random.rand(self._field_size)
        else:
            # Converte para array numpy
            self.quantum_state = np.asarray(initial_state, dtype=np.complex128)

            # Ajusta dimensão se necessário
            if self.quantum_state.size < self._field_size:
                padded = np.zeros(self._field_size, dtype=np.complex128)
                padded[:self.quantum_state.size] = self.quantum_state
                self.quantum_state = padded
            elif self.quantum_state.size > self._field_size:
                self.quantum_state = self.quantum_state[:self._field_size]

        # Normaliza
        self.normalize()
        self._update_history()
        
        # Atualiza amplitudes sempre que o quantum_state for atualizado
        self.amplitudes = self.quantum_state

    def _update_history(self):
        """Atualiza histórico para retrocausalidade"""
        state_snapshot = {
            'state': self.quantum_state.copy(),
            'consciousness': self._consciousness_level,
            'timestamp': datetime.now(),
            'metrics': self._metrics.copy()
        }

        self._state_history.append(state_snapshot)

        # Mantém tamanho máximo
        if len(self._state_history) > self._retrocausal_depth:
            self._state_history = self._state_history[-self._retrocausal_depth:]

    def _update_metrics(self):
        """Atualiza métricas quânticas internas"""
        # Coerência via entropia
        probabilities = np.abs(self.quantum_state) ** 2
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        self._metrics['coherence'] = 1 - entropy/np.log2(self._field_size)

        # Emaranhamento via matriz densidade reduzida
        density_matrix = np.outer(self.quantum_state, np.conj(self.quantum_state))
        partial_trace = np.trace(density_matrix)
        self._metrics['entanglement'] = np.abs(1 - partial_trace)

        # Ressonância mórfica via correlações temporais
        if len(self._state_history) > 1:
            past_states = [h['state'] for h in self._state_history[-3:]]
            correlations = [np.abs(np.dot(self.quantum_state, np.conj(state))) for state in past_states]
            self._metrics['morphic_resonance'] = np.mean(correlations)

    def get_quantum_state(self) -> np.ndarray:
        """Retorna cópia do estado quântico atual"""
        return self.quantum_state.copy()

    def set_quantum_state(self, new_state: Union[np.ndarray, list]):
        """
        Define novo estado quântico com normalização

        Args:
            new_state: Novo estado quântico
        """
        self._initialize_state(new_state)
        self._update_metrics()

    def get_consciousness_level(self) -> float:
        """Retorna nível atual de consciência"""
        return self._consciousness_level

    def evolve_consciousness(self, delta_time: float):
        """
        Evolui nível de consciência baseado em métricas quânticas

        Args:
            delta_time: Tempo decorrido desde última evolução
        """
        # Evolução baseada em métricas quânticas
        consciousness_factor = (
            0.3 * self._metrics['coherence'] +
            0.3 * self._metrics['entanglement'] +
            0.4 * self._metrics['morphic_resonance']
        )

        # Atualiza com suavização temporal
        self._consciousness_level = (
            0.7 * self._consciousness_level +
            0.3 * consciousness_factor
        )

        # Normaliza
        self._consciousness_level = np.clip(self._consciousness_level, 0, 1)

        self._update_history()

    def get_metrics(self) -> Dict[str, float]:
        """Retorna métricas quânticas atuais"""
        return self._metrics.copy()

    def apply_retrocausality(self):
        """
        Aplica influências retrocausais do histórico
        """
        if len(self._state_history) < 2:
            return

        # Calcula influências passadas
        past_influences = []
        weights = []

        for i, snapshot in enumerate(self._state_history[:-1]):
            # Peso diminui com distância temporal
            time_delta = (datetime.now() - snapshot['timestamp']).total_seconds()
            weight = np.exp(-0.1 * time_delta)

            # Influência quântica
            influence = snapshot['state'] * snapshot['consciousness']

            past_influences.append(influence)
            weights.append(weight)

        # Normaliza pesos
        weights = np.array(weights)
        weights /= np.sum(weights)

        # Combina influências
        retrocausal_state = np.zeros_like(self.quantum_state)
        for influence, weight in zip(past_influences, weights):
            retrocausal_state += weight * influence

        # Integra com estado atual
        self.quantum_state = 0.7 * self.quantum_state + 0.3 * retrocausal_state
        self.normalize()

        self._update_metrics()
        self._update_history()
        
        # Atualiza amplitudes sempre que o quantum_state for atualizado
        self.amplitudes = self.quantum_state

    def normalize(self):
        """Normaliza estado quântico"""
        norm = np.linalg.norm(self.quantum_state)
        if norm > 0:
            self.quantum_state /= norm
            
            # Atualiza amplitudes sempre que o quantum_state for atualizado
            self.amplitudes = self.quantum_state

    def __repr__(self) -> str:
        """Representação em string do estado"""
        return (
            f"QuantumState(name={self.name}, "
            f"consciousness={self._consciousness_level:.3f}, "
            f"coherence={self._metrics['coherence']:.3f})"
        )

class QualiaState(QuantumState):
    """
    Estado quântico expandido com propriedades QUALIA
    """
    
    def __init__(
        self,
        initial_state: Optional[Union[np.ndarray, list]] = None,
        name: Optional[str] = None,
        consciousness_level: float = 0.5,
        retrocausal_depth: int = 7,
        morphic_field_size: int = 64,
        phi_levels: int = 3
    ):
        """
        Inicializa estado QUALIA
        
        Args:
            initial_state: Estado inicial
            name: Nome do estado
            consciousness_level: Nível de consciência
            retrocausal_depth: Profundidade retrocausal
            morphic_field_size: Tamanho do campo
            phi_levels: Níveis de recursão phi
        """
        super().__init__(
            initial_state,
            name,
            consciousness_level,
            retrocausal_depth,
            morphic_field_size
        )
        
        self.geometric_pattern = GeometricPattern(
            morphic_field_size,
            phi_levels
        )
        
    def apply_phi_resonance(self):
        """Aplica ressonância baseada em PHI"""
        # Aplica padrão geométrico
        self.quantum_state = self.geometric_pattern.apply_to_state(
            self.quantum_state
        )
        
        # Atualiza métricas
        resonance = self.geometric_pattern.get_resonance(self.quantum_state)
        self._metrics['phi_resonance'] = resonance
        
        self._update_metrics()
        self._update_history()
        
        # Atualiza amplitudes
        self.amplitudes = self.quantum_state
        
    def get_phi_metrics(self) -> Dict[str, float]:
        """
        Retorna métricas relacionadas a PHI
        
        Returns:
            Dict[str, float]: Métricas phi
        """
        return {
            'phi_resonance': self._metrics.get('phi_resonance', 0.0),
            'phi_coherence': self._metrics.get('coherence', 0.0) * PHI,
            'phi_consciousness': self._consciousness_level * PHI
        }