"""
System Behavior Module
Define os comportamentos do sistema quântico.
"""

from enum import Enum, auto

class SystemBehavior(Enum):
    """Comportamentos possíveis do sistema quântico."""
    
    # Comportamentos fundamentais
    COHERENT = auto()  # Sistema mantém coerência
    DECOHERENT = auto()  # Sistema perde coerência
    RESONANT = auto()  # Sistema em ressonância
    
    # Comportamentos emergentes
    SELF_ORGANIZING = auto()  # Sistema se auto-organiza
    ADAPTIVE = auto()  # Sistema se adapta
    LEARNING = auto()  # Sistema aprende
    
    # Comportamentos quânticos
    ENTANGLED = auto()  # Sistema emaranhado
    SUPERPOSED = auto()  # Sistema em superposição
    MEASURED = auto()  # Sistema medido
    
    # Comportamentos de consciência
    CONSCIOUS = auto()  # Sistema consciente
    UNCONSCIOUS = auto()  # Sistema inconsciente
    METACOGNITIVE = auto()  # Sistema metacognitivo
