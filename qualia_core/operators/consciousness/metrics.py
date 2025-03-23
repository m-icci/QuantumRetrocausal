"""
Metrics module for quantum consciousness measurements
"""
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np

@dataclass
class UnifiedConsciousnessMetrics:
    """Unified metrics for consciousness measurement"""
    integration_value: float
    coherence_level: float
    information_content: float
    quantum_correlations: float
    entanglement_measure: float
    field_coupling_strength: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'integration': self.integration_value,
            'coherence': self.coherence_level,
            'information': self.information_content,
            'correlations': self.quantum_correlations,
            'entanglement': self.entanglement_measure,
            'field_coupling': self.field_coupling_strength
        }
    
    @classmethod
    def from_measurements(cls, measurements: Dict[str, float]) -> 'UnifiedConsciousnessMetrics':
        return cls(
            integration_value=measurements.get('integration', 0.0),
            coherence_level=measurements.get('coherence', 0.0),
            information_content=measurements.get('information', 0.0),
            quantum_correlations=measurements.get('correlations', 0.0),
            entanglement_measure=measurements.get('entanglement', 0.0),
            field_coupling_strength=measurements.get('field_coupling')
        )

@dataclass
class FieldOperatorMetrics:
    """Metrics for quantum field operations"""
    field_strength: float
    coupling_constants: List[float]
    interaction_potential: float
    field_gradients: np.ndarray
    energy_density: float
    
    def calculate_total_energy(self) -> float:
        """Calculate total field energy"""
        return float(np.sum(self.energy_density * np.prod(self.field_gradients.shape)))
    
    def get_coupling_strength(self) -> float:
        """Get average coupling strength"""
        return float(np.mean(self.coupling_constants))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'field_strength': self.field_strength,
            'coupling_constants': self.coupling_constants,
            'interaction_potential': self.interaction_potential,
            'field_gradients': self.field_gradients.tolist(),
            'energy_density': self.energy_density,
            'total_energy': self.calculate_total_energy(),
            'mean_coupling': self.get_coupling_strength()
        }
