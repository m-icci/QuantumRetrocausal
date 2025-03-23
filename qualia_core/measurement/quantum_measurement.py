"""
Quantum Measurement Module
------------------------

Define tipos e funcionalidades para medições quânticas.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np

from ..qtypes.quantum_state import QuantumState
from ..operators.base.quantum_operators import QuantumOperator

@dataclass
class QuantumMeasurement:
    """
    Resultado de medição quântica com metadados enriquecidos.
    Implementa funcionalidades para rastrear e analisar resultados de medições.
    """
    state: QuantumState
    observable: str
    value: float
    uncertainty: float
    basis: Optional[np.ndarray] = None
    operator: Optional[QuantumOperator] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validação pós-inicialização"""
        if self.uncertainty < 0:
            raise ValueError("Incerteza não pode ser negativa")
        if self.basis is not None and self.state.dimension != self.basis.shape[0]:
            raise ValueError("Dimensão da base incompatível com estado")
            
    def to_dict(self) -> Dict[str, Any]:
        """Converte medição para dicionário"""
        return {
            'observable': self.observable,
            'value': self.value,
            'uncertainty': self.uncertainty,
            'timestamp': self.timestamp.isoformat(),
            'state_dimension': self.state.dimension,
            'has_basis': self.basis is not None,
            'has_operator': self.operator is not None,
            **self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any], state: QuantumState) -> 'QuantumMeasurement':
        """Cria medição a partir de dicionário"""
        return cls(
            state=state,
            observable=data['observable'],
            value=data['value'],
            uncertainty=data['uncertainty'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata={k: v for k, v in data.items() if k not in 
                     ['observable', 'value', 'uncertainty', 'timestamp', 
                      'state_dimension', 'has_basis', 'has_operator']}
        )
        
    def add_metadata(self, key: str, value: Any):
        """Adiciona metadados à medição"""
        self.metadata[key] = value
        
@dataclass
class MeasurementSequence:
    """
    Sequência de medições quânticas relacionadas.
    Útil para rastrear evolução temporal ou conjunto de medições.
    """
    measurements: List[QuantumMeasurement] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_measurement(self, measurement: QuantumMeasurement):
        """Adiciona medição à sequência"""
        self.measurements.append(measurement)
        
    def get_values(self) -> List[float]:
        """Retorna valores das medições"""
        return [m.value for m in self.measurements]
        
    def get_uncertainties(self) -> List[float]:
        """Retorna incertezas das medições"""
        return [m.uncertainty for m in self.measurements]
        
    def get_timestamps(self) -> List[datetime]:
        """Retorna timestamps das medições"""
        return [m.timestamp for m in self.measurements]
        
    def to_dict(self) -> Dict[str, Any]:
        """Converte sequência para dicionário"""
        return {
            'measurements': [m.to_dict() for m in self.measurements],
            'metadata': self.metadata
        }
