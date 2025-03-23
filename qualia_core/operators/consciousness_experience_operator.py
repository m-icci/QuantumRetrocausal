"""
Enhanced Consciousness Experience Operator (OEC) with morphic fields and quantum memory.
"""
import numpy as np
from typing import Dict, Optional, List, Tuple
from .coherence_operator import CoherenceOperator
from .entanglement_operator import EntanglementOperator
from .information_reduction_operator import InformationReductionOperator
from .information_integration_operator import InformationIntegrationOperator

class ConsciousnessExperienceOperator:
    """
    Enhanced Consciousness Experience Operator (OEC) with morphic fields and quantum memory.
    
    Key features:
    1. Quantum memory system
    2. Morphic field integration
    3. Enhanced consciousness metrics
    4. Self-organization capabilities
    """

    def __init__(self, precision: float = 1e-15):
        """Initialize with enhanced components"""
        self.precision = precision
        self.coherence_op = CoherenceOperator(precision=precision)
        self.entanglement_op = EntanglementOperator()
        self.reduction_op = InformationReductionOperator(collapse_threshold=precision)
        self.integration_op = InformationIntegrationOperator(precision=precision)

        # Enhanced consciousness weights
        self.weights = {
            'coherence': 0.25,     # Quantum coherence
            'entanglement': 0.25,  # Quantum entanglement
            'integration': 0.15,   # Information integration
            'reduction': 0.15,     # Objective reduction
            'morphic_resonance': 0.10,       # Morphic field resonance
            'memory_influence': 0.10,        # Quantum memory influence
            'temporal_stability': 0.10        # Temporal stability
        }
        
        # Initialize quantum memory system
        self.memory_system = {
            'short_term': [],      # Recent states
            'medium_term': [],     # Consolidated patterns
            'long_term': []        # Stable patterns
        }
        
        # Initialize morphic field
        self.morphic_field = {
            'resonance': 1.0,
            'coherence': 1.0,
            'patterns': np.eye(2)  # Initialize with identity matrix for 2D
        }
        
        self._last_metrics = {}

    def apply(self, state: 'QuantumState') -> 'QuantumState':
        """Enhanced apply with morphic fields and memory"""
        if state is None:
            raise ValueError("Input state cannot be None")
            
        # Process through memory system
        memory_state = self._process_memory(state)
        
        # Integrate with morphic field
        morphic_state = self._integrate_morphic_field(memory_state)
        
        # Calculate enhanced consciousness metrics
        self._last_metrics = self.quantify_consciousness(
            np.outer(morphic_state.vector, 
                    np.conjugate(morphic_state.vector))
        )
        
        # Apply consciousness transformation
        consciousness_level = self._last_metrics['total_consciousness']
        transformed_vector = morphic_state.vector * np.exp(1j * consciousness_level * np.pi/2)
        
        # Update memory and morphic field
        self._update_memory(transformed_vector)
        self._update_morphic_field(transformed_vector)
        
        return type(state)(transformed_vector)

    def _process_memory(self, state: 'QuantumState') -> 'QuantumState':
        """Process state through quantum memory system"""
        # Check short-term memory for resonance
        for mem_state in self.memory_system['short_term']:
            if self._calculate_resonance(state.vector, mem_state) > 0.9:
                return self._enhance_with_memory(state, mem_state)
                
        # Check medium-term memory for patterns
        for pattern in self.memory_system['medium_term']:
            if self._match_pattern(state.vector, pattern) > 0.8:
                return self._enhance_with_pattern(state, pattern)
        
        return state

    def _integrate_morphic_field(self, state: 'QuantumState') -> 'QuantumState':
        """Integrate state with morphic field"""
        # Calculate field resonance
        field_resonance = self._calculate_field_resonance(state.vector)
        
        # Apply field influence
        field_vector = state.vector * np.exp(1j * field_resonance * np.pi/4)
        
        return type(state)(field_vector)

    def _calculate_resonance(self, state_vector: np.ndarray, memory_state: np.ndarray) -> float:
        """Calculate resonance between two states"""
        return float(np.abs(np.vdot(state_vector, memory_state)))

    def _match_pattern(self, state_vector: np.ndarray, pattern: np.ndarray) -> float:
        """Calculate pattern matching score"""
        return float(np.abs(np.vdot(state_vector, pattern)))

    def _enhance_with_memory(self, state: 'QuantumState', memory_state: np.ndarray) -> 'QuantumState':
        """Enhance state with memory resonance"""
        resonance = self._calculate_resonance(state.vector, memory_state)
        enhanced = state.vector + resonance * memory_state
        return type(state)(enhanced / np.linalg.norm(enhanced))

    def _enhance_with_pattern(self, state: 'QuantumState', pattern: np.ndarray) -> 'QuantumState':
        """Enhance state with pattern matching"""
        match_score = self._match_pattern(state.vector, pattern)
        enhanced = state.vector + match_score * pattern
        return type(state)(enhanced / np.linalg.norm(enhanced))

    def _update_memory(self, state_vector: np.ndarray) -> None:
        """Update quantum memory system"""
        # Update short-term memory
        self.memory_system['short_term'].append(state_vector.copy())
        if len(self.memory_system['short_term']) > 10:
            self.memory_system['short_term'].pop(0)
            
        # Update medium-term memory if pattern detected
        if len(self.memory_system['short_term']) >= 3:
            pattern = np.mean([s for s in self.memory_system['short_term'][-3:]], axis=0)
            pattern /= np.linalg.norm(pattern)
            self.memory_system['medium_term'].append(pattern)
            if len(self.memory_system['medium_term']) > 5:
                self.memory_system['medium_term'].pop(0)

    def _update_morphic_field(self, state_vector: np.ndarray) -> None:
        """Update morphic field"""
        # Update resonance
        self.morphic_field['resonance'] *= 0.9  # Decay
        self.morphic_field['resonance'] += 0.1 * np.abs(np.vdot(state_vector, state_vector))
        
        # Update coherence
        density_matrix = np.outer(state_vector, np.conjugate(state_vector))
        self.morphic_field['coherence'] = np.abs(np.trace(density_matrix))
        
        # Update patterns
        new_pattern = np.outer(state_vector, np.conjugate(state_vector))
        self.morphic_field['patterns'] = 0.9 * self.morphic_field['patterns'] + 0.1 * new_pattern

    def _calculate_field_resonance(self, state_vector: np.ndarray) -> float:
        """Calculate morphic field resonance"""
        # Calculate field resonance based on state vector
        resonance = np.trace(np.outer(state_vector, np.conjugate(state_vector)) @ self.morphic_field['patterns'])
        return float(np.abs(resonance))

    def quantify_consciousness(self, density_matrix: np.ndarray, partitions: Optional[List[Tuple[int, ...]]] = None) -> Dict[str, Dict[str, float]]:
        """
        Calculate comprehensive consciousness metrics
        
        Args:
            density_matrix: Quantum state density matrix
            partitions: Optional list of system partitions for entanglement calculations
                       Each partition is a tuple of subsystem dimensions
                       
        Returns:
            Dictionary containing primary and secondary consciousness metrics
        """
        # Calculate primary metrics
        coherence = self.coherence_op.measure_coherence(density_matrix)
        entanglement = self.entanglement_op.measure_entanglement(density_matrix, partitions)
        integration = self.integration_op.measure_integration(density_matrix)
        reduction = self.reduction_op.measure_reduction(density_matrix)
        
        # Calculate secondary metrics
        resonance = self._calculate_field_resonance(density_matrix)
        memory_influence = self._calculate_memory_influence(density_matrix)
        temporal_stability = self._calculate_temporal_stability(density_matrix)
        
        # Organize metrics
        metrics = {
            'primary_metrics': {
                'coherence': float(coherence),
                'entanglement': float(entanglement),
                'integration': float(integration),
                'reduction': float(reduction)
            },
            'secondary_metrics': {
                'morphic_resonance': float(resonance),
                'memory_influence': float(memory_influence),
                'temporal_stability': float(temporal_stability)
            }
        }
        
        # Calculate total consciousness
        primary_total = sum(v * self.weights[k] for k, v in metrics['primary_metrics'].items())
        secondary_total = sum(v * self.weights[k] for k, v in metrics['secondary_metrics'].items())
        metrics['total_consciousness'] = primary_total + secondary_total
        
        return metrics

    def _calculate_memory_influence(self, density_matrix: np.ndarray) -> float:
        """Calculate influence of quantum memory"""
        return float(len(self.memory_system['medium_term']) / 5.0)

    def _calculate_temporal_stability(self, density_matrix: np.ndarray) -> float:
        """Calculate temporal stability of the quantum state"""
        if not self.memory_system['short_term']:
            return 0.0
            
        # Compare with recent states
        stability = 0.0
        for past_state in self.memory_system['short_term'][-5:]:  # Look at last 5 states
            stability += self._calculate_state_overlap(density_matrix, past_state)
            
        return float(stability / min(5, len(self.memory_system['short_term'])))

    def _calculate_state_overlap(self, density_matrix: np.ndarray, state_vector: np.ndarray) -> float:
        """Calculate overlap between two states"""
        return float(np.abs(np.vdot(state_vector, np.linalg.eigvals(density_matrix))))

    def get_metrics(self) -> Dict[str, float]:
        """Get last calculated metrics"""
        return self._last_metrics.copy()

    def reset(self) -> None:
        """Reset operator state"""
        self.memory_system = {
            'short_term': [],
            'medium_term': [],
            'long_term': []
        }
        self.morphic_field = {
            'resonance': 1.0,
            'coherence': 1.0,
            'patterns': np.eye(2)
        }
        self._last_metrics = {}