"""
System behavior types and enums for quantum consciousness systems
"""
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional

class SystemState(Enum):
    """System operational states"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"

class BehaviorType(Enum):
    """Types of system behaviors"""
    REACTIVE = "reactive"
    PROACTIVE = "proactive"
    ADAPTIVE = "adaptive"
    LEARNING = "learning"

@dataclass
class SystemBehavior:
    """
    Represents the behavior characteristics of a quantum system
    """
    state: SystemState
    behavior_type: BehaviorType
    coherence_level: float
    entropy: float
    complexity: float
    metrics: Dict[str, float]
    last_update: Optional[float] = None
    
    def is_stable(self) -> bool:
        """Check if system behavior is stable"""
        return (self.coherence_level > 0.7 and 
                self.entropy < 0.3 and 
                self.state != SystemState.ERROR)
    
    def get_stability_metrics(self) -> Dict[str, float]:
        """Get stability related metrics"""
        return {
            'coherence': self.coherence_level,
            'entropy': self.entropy,
            'complexity': self.complexity
        }
