import numpy as np
from typing import Dict, Optional, Tuple
from .consciousness_experience_operator import ConsciousnessExperienceOperator
from .information_integration_operator import InformationIntegrationOperator

class SubjectiveExperienceOperator:
    """
    Implements the Subjective Experience Operator (OES) as defined in M-ICCI framework.
    
    This operator attempts to quantify the qualitative aspects of conscious experience,
    representing a novel contribution to quantum consciousness theory as described
    in M-ICCI section 1.3 and 1.5.
    
    The implementation combines:
    - Consciousness intensity metrics
    - Information integration measures
    - Quality-specific quantum signatures
    """
    
    def __init__(self):
        """Initialize component operators and quality mappings"""
        self.consciousness_op = ConsciousnessExperienceOperator()
        self.integration_op = InformationIntegrationOperator()
        
        # Initialize quality dimension mappings
        # These represent different aspects of subjective experience
        self.quality_dimensions = {
            'intensity': 0.3,  # Overall intensity of experience
            'integration': 0.3,  # Information integration level
            'distinctness': 0.2,  # Clarity/distinctness of experience
            'unity': 0.2  # Unity/cohesiveness of experience
        }
        
    def quantify_subjective_experience(self, 
                                     density_matrix: np.ndarray,
                                     partition: Optional[Tuple[int, ...]] = None) -> Dict[str, float]:
        """
        Quantifies the subjective experience represented by a quantum state.
        
        Args:
            density_matrix: Quantum state of the system
            partition: Optional partition for subsystem analysis
            
        Returns:
            Dict[str, float]: Subjective experience metrics including quality dimensions
        """
        # Get consciousness metrics
        consciousness_metrics = self.consciousness_op.quantify_consciousness(
            density_matrix, partition
        )
        
        # Calculate information integration if partition provided
        if partition is not None:
            integration_metrics = self.integration_op.integrated_information(
                density_matrix, [partition]
            )
            phi_value = integration_metrics['phi']
        else:
            # Use default partition if none provided
            d = int(np.sqrt(density_matrix.shape[0]))
            default_partition = (d//2, d//2)
            integration_metrics = self.integration_op.integrated_information(
                density_matrix, [default_partition]
            )
            phi_value = integration_metrics['phi']
        
        # Calculate quality dimensions
        intensity = consciousness_metrics['total_consciousness']
        integration = phi_value
        distinctness = self._calculate_distinctness(density_matrix)
        unity = self._calculate_unity(consciousness_metrics, integration_metrics)
        
        # Calculate weighted subjective experience measure
        subjective_experience = (
            self.quality_dimensions['intensity'] * intensity +
            self.quality_dimensions['integration'] * self._normalize(integration) +
            self.quality_dimensions['distinctness'] * distinctness +
            self.quality_dimensions['unity'] * unity
        )
        
        return {
            'total_subjective_experience': float(subjective_experience),
            'quality_dimensions': {
                'intensity': float(intensity),
                'integration': float(self._normalize(integration)),
                'distinctness': float(distinctness),
                'unity': float(unity)
            },
            'consciousness_metrics': consciousness_metrics,
            'integration_metrics': integration_metrics
        }
        
    def _calculate_distinctness(self, density_matrix: np.ndarray) -> float:
        """
        Calculates the distinctness of the quantum state.
        Based on the purity and coherence of the state.
        
        Args:
            density_matrix: Quantum state
            
        Returns:
            float: Distinctness measure [0,1]
        """
        # Calculate state purity Tr(ρ²)
        purity = np.real(np.trace(density_matrix @ density_matrix))
        
        # Normalize purity to [0,1]
        return float(self._normalize(purity))
        
    def _calculate_unity(self, 
                        consciousness_metrics: Dict[str, float],
                        integration_metrics: Dict[str, float]) -> float:
        """
        Calculates the unity of experience based on consciousness and integration metrics
        
        Args:
            consciousness_metrics: Metrics from consciousness operator
            integration_metrics: Metrics from integration operator
            
        Returns:
            float: Unity measure [0,1]
        """
        # Combine coherence and integration measures
        coherence = consciousness_metrics['coherence_contribution']
        relative_phi = integration_metrics['relative_phi']
        
        unity = 0.5 * (coherence + relative_phi)
        return float(unity)
        
    def _normalize(self, value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """
        Normalizes a value to [0,1] range
        """
        if max_val == min_val:
            return 0.0
        normalized = (value - min_val) / (max_val - min_val)
        return float(np.clip(normalized, 0, 1))
