"""
Configuration settings for quantum consciousness system
"""
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class QuantumConfig:
    """Configuration parameters for quantum consciousness system."""
    
    # System dimensions
    HILBERT_SPACE_DIMENSION: int = 8
    
    # Physical parameters
    TEMPERATURE_KELVIN: float = 310.0  # Body temperature
    COHERENCE_THRESHOLD: float = 0.95
    DECOHERENCE_TIME_MS: float = 100.0
    
    # Computational settings
    USE_GPU: bool = True
    JIT_COMPILE: bool = True
    
    # Optimization parameters
    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 32
    
    # Protection settings
    MAX_RETRIES: int = 3
    RECOVERY_THRESHOLD: float = 0.8
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'QuantumConfig':
        """Create config from dictionary."""
        return cls(**{
            k: v for k, v in config_dict.items() 
            if k in cls.__annotations__
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            k: getattr(self, k) 
            for k in self.__annotations__
        }
