"""
Holistic Risk Manager for QUALIA Trading System
Implements quantum risk assessment with morphic field integration.
"""

import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Risk metrics for holistic evaluation"""
    risk_score: float = 0.0
    field_risk: float = 0.0
    quantum_volatility: float = 0.0
    resonance_stability: float = 0.0
    morphic_coherence: float = 0.0

class HolisticRiskManager:
    """
    Holistic Risk Management System
    
    Implements:
    1. Quantum risk assessment
    2. Morphic field risk evaluation
    3. Dynamic risk threshold adjustment
    4. Non-local correlation analysis
    """
    
    def __init__(self, initial_risk_threshold: float = 0.5):
        """
        Initialize risk manager
        
        Args:
            initial_risk_threshold: Initial risk threshold
        """
        self.risk_threshold = initial_risk_threshold
        self.risk_history: List[RiskMetrics] = []
        self.max_history = 50
        
        # Risk weights
        self.weights = {
            'field_risk': 0.3,
            'quantum_volatility': 0.2,
            'resonance_stability': 0.3,
            'morphic_coherence': 0.2
        }
    
    def evaluate_risk(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate trading risk based on holistic analysis
        
        Args:
            analysis: Market analysis results
            
        Returns:
            Risk assessment
        """
        # Calculate risk metrics
        metrics = self._calculate_risk_metrics(analysis)
        
        # Calculate weighted risk score
        risk_score = (
            metrics.field_risk * self.weights['field_risk'] +
            metrics.quantum_volatility * self.weights['quantum_volatility'] +
            metrics.resonance_stability * self.weights['resonance_stability'] +
            metrics.morphic_coherence * self.weights['morphic_coherence']
        )
        metrics.risk_score = risk_score
        
        # Update risk history
        self.risk_history.append(metrics)
        if len(self.risk_history) > self.max_history:
            self.risk_history.pop(0)
        
        # Adjust risk threshold
        self._adjust_risk_threshold()
        
        return {
            'risk_score': risk_score,
            'is_acceptable': risk_score <= self.risk_threshold,
            'risk_threshold': self.risk_threshold,
            'metrics': metrics
        }
    
    def _calculate_risk_metrics(self, analysis: Dict[str, Any]) -> RiskMetrics:
        """
        Calculate detailed risk metrics
        
        Args:
            analysis: Market analysis results
            
        Returns:
            Risk metrics
        """
        metrics = RiskMetrics()
        
        # Field risk from holographic metrics
        if 'holographic_metrics' in analysis:
            holo = analysis['holographic_metrics']
            metrics.field_risk = 1 - holo.field_strength
            metrics.resonance_stability = 1 - holo.resonance_quality
            metrics.morphic_coherence = 1 - holo.coherence_level
        
        # Quantum volatility from market data
        if 'market_data' in analysis:
            data = analysis['market_data']
            if len(data) > 0:
                metrics.quantum_volatility = np.std(data) / np.mean(np.abs(data))
        
        return metrics
    
    def _adjust_risk_threshold(self):
        """
        Dynamically adjust risk threshold based on history
        """
        if len(self.risk_history) > 10:
            recent_risks = [m.risk_score for m in self.risk_history[-10:]]
            risk_volatility = np.std(recent_risks)
            
            # Adaptive threshold adjustment
            if risk_volatility > 0.2:
                # High volatility - reduce threshold
                self.risk_threshold *= 0.95
            elif risk_volatility < 0.1:
                # Low volatility - increase threshold
                self.risk_threshold *= 1.05
            
            # Ensure threshold stays within bounds
            self.risk_threshold = np.clip(self.risk_threshold, 0.1, 0.9)
    
    def get_risk_summary(self) -> Dict[str, float]:
        """
        Get summary of current risk state
        
        Returns:
            Risk summary statistics
        """
        if not self.risk_history:
            return {}
        
        recent_metrics = self.risk_history[-1]
        return {
            'current_risk_score': recent_metrics.risk_score,
            'risk_threshold': self.risk_threshold,
            'field_risk': recent_metrics.field_risk,
            'quantum_volatility': recent_metrics.quantum_volatility,
            'resonance_stability': recent_metrics.resonance_stability,
            'morphic_coherence': recent_metrics.morphic_coherence
        }
