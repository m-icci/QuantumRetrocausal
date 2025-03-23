"""
Base operator module for quantum operations
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

class BaseOperator(ABC):
    """
    Base class for quantum operators.
    Defines interface for all quantum operators.
    """
    
    def __init__(self):
        """Initialize base operator"""
        self.properties: Dict[str, Any] = {}
        
    @abstractmethod
    def apply(self, state: np.ndarray) -> np.ndarray:
        """
        Apply operator to quantum state
        
        Args:
            state: Input quantum state
            
        Returns:
            Modified quantum state
        """
        pass
    
    def get_property(self, key: str) -> Any:
        """Get operator property"""
        return self.properties.get(key)
        
    def set_property(self, key: str, value: Any) -> None:
        """Set operator property"""
        self.properties[key] = value
