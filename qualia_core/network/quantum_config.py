"""
Configuração de Parâmetros Quânticos para QualiaHawkingP2P
"""

from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np

@dataclass
class QuantumParameters:
    """Parâmetros quânticos do sistema"""
    # Dimensões do espaço quântico
    dimensions: int = 64
    
    # Número de realidades por camada
    realities_per_layer: int = 4
    
    # Número de camadas
    num_layers: int = 3
    
    # Proporção áurea (phi)
    phi: float = (1 + np.sqrt(5)) / 2
    
    # Taxa de decoerência
    decoherence_rate: float = 0.1
    
    # Limite de memória holográfica
    holographic_memory_limit: int = 1000
    
    # Taxa de entrelaçamento
    entanglement_rate: float = 0.5
    
    # Parâmetros de Hawking
    hawking_temperature: float = 0.1
    hawking_entropy: float = 0.5
    
    # Configurações de rede
    default_pub_port: int = 5555
    default_sub_ports: List[int] = None
    
    def __post_init__(self):
        if self.default_sub_ports is None:
            self.default_sub_ports = [5556, 5557, 5558]
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário"""
        return {
            'dimensions': self.dimensions,
            'realities_per_layer': self.realities_per_layer,
            'num_layers': self.num_layers,
            'phi': self.phi,
            'decoherence_rate': self.decoherence_rate,
            'holographic_memory_limit': self.holographic_memory_limit,
            'entanglement_rate': self.entanglement_rate,
            'hawking_temperature': self.hawking_temperature,
            'hawking_entropy': self.hawking_entropy,
            'default_pub_port': self.default_pub_port,
            'default_sub_ports': self.default_sub_ports
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuantumParameters':
        """Cria a partir de dicionário"""
        return cls(**data)

# Configuração padrão
DEFAULT_CONFIG = QuantumParameters() 