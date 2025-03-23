"""
Pattern type definitions for quantum analysis.
"""
from enum import Enum, auto
from typing import Dict, Any
from dataclasses import dataclass, field

@dataclass
class PatternDescription:
    """Description of a pattern type

    Attributes:
        name (str): Pattern name
        description (str): Detailed description
        threshold (float): Detection threshold [0,1] 
        metadata (Dict[str, Any]): Additional metadata
    """
    name: str
    description: str
    threshold: float
    metadata: Dict[str, Any]

class PatternType(Enum):
    """Quantum pattern types"""

    # Resonance Patterns
    RESONANCE = auto()  # Morphic resonance
    COHERENCE = auto()  # Quantum coherence
    ENTANGLEMENT = auto()  # Entanglement

    # Topological Patterns  
    VORTEX = auto()  # Quantum vortex
    KNOT = auto()  # Topological knot
    BRAID = auto()  # Quantum braid

    # Emerging Patterns
    SOLITON = auto()  # Quantum soliton
    BREATHER = auto()  # Breather
    INSTANTON = auto()  # Instanton

    # Self-organization Patterns
    MORPHIC = auto()  # Morphic field
    HOLONOMIC = auto()  # Holonomy
    RECURSIVE = auto()  # φ recursion

    # Consciousness Patterns
    AWARENESS = auto()  # Consciousness
    INTEGRATION = auto()  # Integration
    EMERGENCE = auto()  # Emergence
    QUANTUM = auto()  # Fundamental quantum pattern

    @property
    def description(self) -> PatternDescription:
        """Get pattern description"""
        descriptions = {
            PatternType.RESONANCE: PatternDescription(
                name="Morphic Resonance",
                description="Non-local resonance patterns between quantum states",
                threshold=0.85,
                metadata={"requires_entanglement": True, "min_distance": 2}
            ),
            PatternType.COHERENCE: PatternDescription(
                name="Quantum Coherence",
                description="Coherence maintenance patterns in quantum states",
                threshold=0.90,
                metadata={"max_decoherence": 0.1, "min_fidelity": 0.9}
            ),
            PatternType.QUANTUM: PatternDescription(
                name="Quantum Pattern",
                description="Fundamental quantum pattern with universal properties",
                threshold=0.95,
                metadata={"universal": True, "fundamental": True}
            ),
            PatternType.ENTANGLEMENT: PatternDescription(
                name="Emaranhamento Quântico",
                description="Padrões de correlação quântica não-local",
                threshold=0.80,
                metadata={"min_correlation": 0.8, "max_separability": 0.2}
            ),
            PatternType.VORTEX: PatternDescription(
                name="Vórtice Quântico",
                description="Padrões de vórtice em campos quânticos",
                threshold=0.75,
                metadata={"min_size": 3, "max_energy": 10}
            ),
            PatternType.KNOT: PatternDescription(
                name="Nó Topológico",
                description="Padrões de nó em estruturas topológicas",
                threshold=0.70,
                metadata={"min_twist": 2, "max_turn": 5}
            ),
            PatternType.BRAID: PatternDescription(
                name="Trança Quântica",
                description="Padrões de trança em estruturas quânticas",
                threshold=0.65,
                metadata={"min_crossing": 2, "max_twist": 3}
            ),
            PatternType.SOLITON: PatternDescription(
                name="Sóliton Quântico",
                description="Padrões de sóliton em campos quânticos",
                threshold=0.60,
                metadata={"min_size": 2, "max_energy": 5}
            ),
            PatternType.BREATHER: PatternDescription(
                name="Respirador",
                description="Padrões de respirador em estruturas quânticas",
                threshold=0.55,
                metadata={"min_period": 2, "max_amplitude": 3}
            ),
            PatternType.INSTANTON: PatternDescription(
                name="Instanton",
                description="Padrões de instanton em campos quânticos",
                threshold=0.50,
                metadata={"min_size": 1, "max_energy": 2}
            ),
            PatternType.MORPHIC: PatternDescription(
                name="Campo Mórfico",
                description="Padrões de campo mórfico em estruturas quânticas",
                threshold=0.45,
                metadata={"min_size": 2, "max_energy": 3}
            ),
            PatternType.HOLONOMIC: PatternDescription(
                name="Holonomia",
                description="Padrões de holonomia em estruturas quânticas",
                threshold=0.40,
                metadata={"min_size": 1, "max_energy": 2}
            ),
            PatternType.RECURSIVE: PatternDescription(
                name="Recursão φ",
                description="Padrões de recursão φ em estruturas quânticas",
                threshold=0.35,
                metadata={"min_size": 2, "max_energy": 3}
            ),
            PatternType.AWARENESS: PatternDescription(
                name="Consciência",
                description="Padrões de consciência em estruturas quânticas",
                threshold=0.30,
                metadata={"min_size": 1, "max_energy": 2}
            ),
            PatternType.INTEGRATION: PatternDescription(
                name="Integração",
                description="Padrões de integração em estruturas quânticas",
                threshold=0.25,
                metadata={"min_size": 2, "max_energy": 3}
            ),
            PatternType.EMERGENCE: PatternDescription(
                name="Emergência",
                description="Padrões de emergência em estruturas quânticas",
                threshold=0.20,
                metadata={"min_size": 1, "max_energy": 2}
            )
        }
        return descriptions.get(self, PatternDescription(
            name=self.name,
            description=f"Pattern type {self.name}",
            threshold=0.5,
            metadata={}
        ))

    def validate_pattern(self, strength: float, size: int) -> bool:
        """Validate a pattern of this type

        Args:
            strength: Pattern strength [0,1]
            size: Pattern size 

        Returns:
            True if valid
        """
        desc = self.description
        if strength < desc.threshold:
            return False
        min_size = desc.metadata.get('min_size', 1)
        return size >= min_size