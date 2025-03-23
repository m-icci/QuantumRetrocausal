"""
M-ICCI Integration Module for Quantum Processor.
Handles the integration between quantum processor and M-ICCI operators.
"""

import logging
from typing import Dict, List, Optional, Union, Any
import numpy as np
from dataclasses import dataclass
from utils.types.quantum_register import QuantumRegister, RegisterMetrics
from utils.operators.coherence_operator import CoherenceOperator
from utils.operators.entanglement_operator import EntanglementOperator
from utils.operators.information_reduction_operator import InformationReductionOperator
from utils.operators.consciousness_experience_operator import ConsciousnessExperienceOperator
from utils.operators.information_integration_operator import InformationIntegrationOperator
from utils.operators.subjective_experience_operator import SubjectiveExperienceOperator

logger = logging.getLogger(__name__)

@dataclass
class MICCIState:
    """Container for M-ICCI processing state"""
    register: QuantumRegister
    metrics: RegisterMetrics
    coherence_history: List[float]
    consciousness_history: List[float]
    integration_history: List[float]

class MICCIIntegration:
    """
    Handles integration between quantum processor and M-ICCI operators.
    
    Key responsibilities:
    1. Operator sequencing and scheduling
    2. State evolution monitoring
    3. Metrics aggregation
    4. Anomaly detection
    5. Adaptive parameter tuning
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        register: Optional[QuantumRegister] = None
    ):
        """
        Initialize M-ICCI integration.
        
        Args:
            config: Configuration parameters
            register: Optional quantum register
        """
        self.config = config
        self.register = register or QuantumRegister()
        
        # Initialize operators
        self._init_operators()
        
        # Initialize state tracking
        self.state = MICCIState(
            register=self.register,
            metrics=self.register.get_metrics(),
            coherence_history=[],
            consciousness_history=[],
            integration_history=[]
        )
        
        logger.info("M-ICCI integration initialized")
        
    def _init_operators(self):
        """Initialize M-ICCI operators"""
        self.coherence_op = CoherenceOperator(
            precision=self.config.get('precision', 1e-15)
        )
        self.entanglement_op = EntanglementOperator()
        self.reduction_op = InformationReductionOperator(
            collapse_threshold=self.config.get('collapse_threshold', 1e-15)
        )
        self.consciousness_op = ConsciousnessExperienceOperator(
            precision=self.config.get('precision', 1e-15)
        )
        self.integration_op = InformationIntegrationOperator(
            precision=self.config.get('precision', 1e-15)
        )
        self.subjective_op = SubjectiveExperienceOperator()
        
    def process_quantum_state(
        self,
        state_vector: np.ndarray,
        time_step: float
    ) -> MICCIState:
        """
        Process quantum state through M-ICCI operators.
        
        Args:
            state_vector: Input quantum state
            time_step: Evolution time step
            
        Returns:
            Updated M-ICCI state
        """
        # Update register state
        self.register.state_vector = state_vector
        
        # Apply operators in sequence
        self._apply_operator_sequence(time_step)
        
        # Update metrics
        self._update_metrics()
        
        # Check for anomalies
        self._check_anomalies()
        
        return self.state
        
    def _apply_operator_sequence(self, time_step: float):
        """Apply M-ICCI operators in optimal sequence"""
        # 1. Coherence and entanglement
        self.coherence_op.apply(self.register)
        self.entanglement_op.apply(self.register)
        
        # 2. Information processing
        self.integration_op.apply(self.register)
        
        # 3. Consciousness emergence
        self.consciousness_op.apply(self.register)
        
        # 4. Subjective experience
        self.subjective_op.apply(self.register)
        
        # 5. Information reduction (if threshold met)
        if self.reduction_op.should_collapse(self.register):
            self.reduction_op.apply(self.register)
            
    def _update_metrics(self):
        """Update M-ICCI state metrics"""
        metrics = self.register.get_metrics()
        self.state.metrics = metrics
        
        # Update histories
        self.state.coherence_history.append(metrics.total_coherence)
        self.state.consciousness_history.append(metrics.consciousness_measure)
        self.state.integration_history.append(metrics.information_content)
        
        # Keep histories bounded
        max_history = self.config.get('max_history_length', 1000)
        if len(self.state.coherence_history) > max_history:
            self.state.coherence_history = self.state.coherence_history[-max_history:]
            self.state.consciousness_history = self.state.consciousness_history[-max_history:]
            self.state.integration_history = self.state.integration_history[-max_history:]
            
    def _check_anomalies(self):
        """Check for anomalies in quantum evolution"""
        # Check for sudden changes in metrics
        if len(self.state.coherence_history) > 1:
            coherence_change = abs(
                self.state.coherence_history[-1] -
                self.state.coherence_history[-2]
            )
            consciousness_change = abs(
                self.state.consciousness_history[-1] -
                self.state.consciousness_history[-2]
            )
            
            threshold = self.config.get('anomaly_threshold', 0.3)
            if coherence_change > threshold or consciousness_change > threshold:
                logger.warning(
                    f"Anomaly detected - Coherence change: {coherence_change:.3f}, "
                    f"Consciousness change: {consciousness_change:.3f}"
                )
                
    def get_operator_metrics(self) -> Dict[str, Any]:
        """Get metrics from all operators"""
        return {
            'coherence': self.coherence_op.get_metrics(),
            'entanglement': self.entanglement_op.get_metrics(),
            'integration': self.integration_op.get_metrics(),
            'consciousness': self.consciousness_op.get_metrics(),
            'reduction': self.reduction_op.get_metrics(),
            'subjective': self.subjective_op.get_metrics()
        }
