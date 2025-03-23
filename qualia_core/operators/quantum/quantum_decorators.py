"""
Decoradores para operadores quânticos
"""

import numpy as np
from typing import TypeVar, ParamSpec, Callable
from functools import wraps

T = TypeVar('T')
P = ParamSpec('P')

def quantum_operator(name: str, dimensions: int = 64):
    """Decorador para inicialização automática de operadores quânticos
    
    Implementa inicialização automática de operadores quânticos usando
    princípios de auto-organização e campos morfogenéticos.
    
    Args:
        name: Nome do operador
        dimensions: Dimensão do espaço de Hilbert
        
    Returns:
        Callable: Decorador configurado
        
    References:
        [1] Bohm, D. (1980). Wholeness and the Implicate Order
        [2] Sheldrake, R. (1981). A New Science of Life
    """
    def decorator(cls: T) -> T:
        original_init = cls.__init__
        
        @wraps(original_init)
        def wrapped_init(self, *args, **kwargs):
            # Inicializa operador base
            super(cls, self).__init__(dimensions, name)
            # Inicializa razão áurea como parâmetro natural
            self.phi = (1 + np.sqrt(5)) / 2
            # Chama inicializador original
            original_init(self, *args, **kwargs)
            
        cls.__init__ = wrapped_init
        return cls
    
    return decorator
