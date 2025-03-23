"""
Economic Events Module for QUALIA Trading System
Manages real-time global economic event tracking and impact analysis
"""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class EventImpact(Enum):
    """Economic event impact classification"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"

class EventCategory(Enum):
    """Economic event categories"""
    MONETARY_POLICY = "monetary_policy"
    ECONOMIC_INDICATOR = "economic_indicator"
    GEOPOLITICAL = "geopolitical"
    EARNINGS = "earnings"
    MARKET_STRUCTURE = "market_structure"
    OTHER = "other"

@dataclass
class EconomicEvent:
    """Economic event data structure with quantum-aware impact metrics"""
    id: str
    title: str
    description: str
    timestamp: datetime
    category: EventCategory
    impact: EventImpact
    region: str
    source: str
    actual_value: Optional[float] = None
    forecast_value: Optional[float] = None
    previous_value: Optional[float] = None
    quantum_impact_score: float = 0.0
    consciousness_resonance: float = 0.0
    market_coherence_effect: float = 0.0
    related_assets: List[str] = None
    additional_data: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize optional fields and validate quantum metrics"""
        if self.related_assets is None:
            self.related_assets = []
        if self.additional_data is None:
            self.additional_data = {}

        # Ensure quantum metrics are within valid ranges
        self.quantum_impact_score = max(0.0, min(1.0, self.quantum_impact_score))
        self.consciousness_resonance = max(0.0, min(1.0, self.consciousness_resonance))
        self.market_coherence_effect = max(0.0, min(1.0, self.market_coherence_effect))

class EventTimelineManager:
    """Manages the economic event timeline and impact analysis"""

    def __init__(self, quantum_dimension: int = 64):
        self.quantum_dimension = quantum_dimension
        self.events: List[EconomicEvent] = []
        self.consciousness_threshold = 0.7
        self.coherence_threshold = 0.4  # Increased from previous value
        logger.info(f"Initialized EventTimelineManager with dimension {quantum_dimension}")

    def add_event(self, event: EconomicEvent) -> None:
        """Add new event to timeline with quantum impact analysis"""
        try:
            self._calculate_quantum_impact(event)
            self.events.append(event)
            self._update_market_coherence()
            logger.info(f"Added event: {event.title} with quantum impact {event.quantum_impact_score}")
        except Exception as e:
            logger.error(f"Failed to add event: {str(e)}")

    def _calculate_quantum_impact(self, event: EconomicEvent) -> None:
        """Calculate quantum impact metrics for the event with improved coherence handling"""
        try:
            # Calculate base impact score with higher minimum values
            base_impact = {
                EventImpact.HIGH: 0.8,
                EventImpact.MEDIUM: 0.5,
                EventImpact.LOW: 0.3,
                EventImpact.UNKNOWN: 0.2  # Increased from 0.1
            }[event.impact]

            # Calculate consciousness resonance with modified formula
            event.consciousness_resonance = min(
                base_impact * (1 + len(event.related_assets) * 0.2),  # Increased factor
                1.0
            )

            # Calculate quantum impact score with higher weights
            event.quantum_impact_score = (
                event.consciousness_resonance * 0.7 +  # Increased weight
                base_impact * 0.3
            )

            # Initialize market coherence effect with higher baseline
            event.market_coherence_effect = max(base_impact, 0.4)  # Ensure minimum coherence

        except Exception as e:
            logger.error(f"Error calculating quantum impact: {str(e)}")
            # Set safe default values above minimum thresholds
            event.quantum_impact_score = 0.4
            event.consciousness_resonance = 0.4
            event.market_coherence_effect = 0.4

    def _update_market_coherence(self) -> None:
        """Update market coherence based on recent events with improved stability"""
        try:
            recent_events = sorted(
                self.events,
                key=lambda x: x.timestamp,
                reverse=True
            )[:10]  # Consider last 10 events

            # Calculate aggregate coherence with minimum threshold
            if recent_events:
                coherence = sum(e.market_coherence_effect for e in recent_events) / len(recent_events)
                coherence = max(coherence, self.coherence_threshold)  # Enforce minimum threshold
                logger.info(f"Updated market coherence: {coherence:.3f}")

        except Exception as e:
            logger.error(f"Error updating market coherence: {str(e)}")

    def get_recent_events(self, limit: int = 10) -> List[EconomicEvent]:
        """Get recent events sorted by timestamp"""
        return sorted(
            self.events,
            key=lambda x: x.timestamp,
            reverse=True
        )[:limit]

    def get_high_impact_events(self) -> List[EconomicEvent]:
        """Get high impact events with quantum consciousness integration"""
        return [
            event for event in self.events
            if event.impact == EventImpact.HIGH and
            event.quantum_impact_score >= self.consciousness_threshold
        ]