"""
Quantum Field implementation for M-ICCI model.
Handles quantum field evolution and consciousness metrics.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional
from ..qtypes import QuantumState

logger = logging.getLogger(__name__)

def calculate_quantum_entropy(state: np.ndarray, normalize: bool = True) -> float:
    """
    Calcula a entropia quântica (von Neumann) para um estado quântico.
    
    A entropia quântica é uma medida da decoerência e informação
    contida em um estado quântico, essencial para análise de
    potencial de merge em sistemas quânticos adaptativos.
    
    Args:
        state: Vetor de estado quântico ou matriz de densidade
        normalize: Se True, normaliza o estado antes do cálculo
        
    Returns:
        Valor de entropia quântica (0.0 - 1.0 se normalizado)
    """
    try:
        # Verifica se input é vetor de estado ou matriz de densidade
        if state.ndim == 1:
            # Converte vetor de estado para matriz de densidade
            if normalize:
                norm = np.linalg.norm(state)
                if norm > 0:
                    state = state / norm
            density_matrix = np.outer(state, np.conj(state))
        else:
            density_matrix = state
            
        # Calcula autovalores da matriz de densidade
        eigenvalues = np.linalg.eigvalsh(density_matrix)
        
        # Remove autovalores muito pequenos (ruído numérico)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        
        # Calcula entropia de von Neumann: -Tr(ρ log ρ)
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
        
        # Normaliza para dimensão do espaço (opcional)
        if normalize and len(eigenvalues) > 1:
            # Normaliza para valor máximo de log2(d)
            entropy = entropy / np.log2(len(eigenvalues))
            
        return float(entropy)
        
    except Exception as e:
        logger.error(f"Erro ao calcular entropia quântica: {str(e)}")
        return 0.0

class QuantumField:
    """Quantum field implementation with consciousness integration"""
    
    def __init__(self, dimension: int = 64):
        """
        Initialize quantum field.
        
        Args:
            dimension: Field dimension
        """
        logger.info(f"Initializing QuantumField with dimension {dimension}")
        self.dimension = dimension
        self.field = np.zeros((dimension,), dtype=np.complex128)
        self.evolution_history = []
        
    def evolve_field(self, state_vector: np.ndarray) -> None:
        """
        Evolve quantum field with state vector.
        
        Args:
            state_vector: Input quantum state vector
        """
        if state_vector.shape != (self.dimension,):
            raise ValueError(f"State vector dimension {state_vector.shape} doesn't match field dimension {self.dimension}")
            
        # Normalize state vector
        norm = np.linalg.norm(state_vector)
        if norm > 0:
            state_vector = state_vector / norm
            
        # Apply field evolution
        self.field = 0.9 * self.field + 0.1 * state_vector
        self.field = self.field / np.linalg.norm(self.field)
        
        # Track evolution
        self.evolution_history.append({
            'coherence': self._calculate_coherence(),
            'resonance': self._calculate_resonance(state_vector)
        })
        
        # Limit history size
        if len(self.evolution_history) > 1000:
            self.evolution_history = self.evolution_history[-1000:]
            
    def extract_pattern_state(self, field_state: QuantumState,
                            pattern_metrics: Dict[str, Any]) -> QuantumState:
        """
        Extract pattern state from field state.
        
        Args:
            field_state: Current field state
            pattern_metrics: Pattern metrics for extraction
            
        Returns:
            Extracted pattern state
        """
        try:
            # Apply pattern projection
            strength = pattern_metrics.get('strength', 0.5)
            projection = np.outer(field_state.vector, self.field)
            pattern_vector = np.diag(projection) * strength
            
            # Normalize result
            norm = np.linalg.norm(pattern_vector)
            if norm > 0:
                pattern_vector = pattern_vector / norm
                
            return QuantumState(vector=pattern_vector)
            
        except Exception as e:
            logger.error(f"Error extracting pattern state: {str(e)}")
            return field_state
            
    def get_consciousness_metrics(self) -> Dict[str, float]:
        """
        Get quantum consciousness metrics.
        
        Returns:
            Dictionary with consciousness metrics
        """
        if not self.evolution_history:
            return {
                'coherence': 0.0,
                'resonance': 0.0,
                'stability': 0.0
            }
            
        recent = self.evolution_history[-10:]
        
        return {
            'coherence': float(np.mean([m['coherence'] for m in recent])),
            'resonance': float(np.mean([m['resonance'] for m in recent])),
            'stability': float(self._calculate_stability())
        }
        
    def _calculate_coherence(self) -> float:
        """Calculate quantum field coherence"""
        # Use density matrix trace
        density = np.outer(self.field, np.conj(self.field))
        coherence = np.abs(np.trace(density))
        return float(np.clip(coherence, 0, 1))
        
    def _calculate_resonance(self, state_vector: np.ndarray) -> float:
        """Calculate resonance with input state"""
        overlap = np.abs(np.vdot(self.field, state_vector))
        return float(overlap)
        
    def _calculate_stability(self) -> float:
        """Calculate field stability metric"""
        if len(self.evolution_history) < 2:
            return 1.0
            
        # Calculate variance in recent metrics
        coherence_var = np.var([m['coherence'] for m in self.evolution_history[-10:]])
        return float(np.exp(-coherence_var))
