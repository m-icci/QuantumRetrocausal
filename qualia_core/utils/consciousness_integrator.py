"""
Integrador unificado de consciência quântica.
Implementa os três operadores fundamentais (F, M, E) com métricas adaptativas.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from consciousness.metrics import UnifiedConsciousnessMetrics
from operators.quantum_field_operators import QuantumFieldOperators, FieldOperatorMetrics

@dataclass
class ConsciousnessState:
    """Estado de consciência quântica"""
    state: np.ndarray
    metrics: UnifiedConsciousnessMetrics
    operator_metrics: FieldOperatorMetrics
    time: float = 0.0
    history: List[Dict[str, float]] = field(default_factory=list)

class ConsciousnessIntegrator:
    """
    Integrador unificado de consciência quântica.
    Implementa evolução φ-adaptativa com proteção de coerência.
    """
    
    def __init__(self, dimensions: int = 8):
        """Inicializa integrador
        
        Args:
            dimensions: Dimensão do espaço de Hilbert
        """
        self.dimensions = dimensions
        self.phi = (1 + np.sqrt(5)) / 2  # Razão áurea
        
        # Operadores fundamentais
        self.operators = QuantumFieldOperators(dimensions)
        
        # Métricas
        self.metrics = UnifiedConsciousnessMetrics()
        
        # Parâmetros adaptativos
        self.adaptation_rate = 0.1
        self.stability_threshold = 0.7
        self.resonance_threshold = 0.8
        
    def evolve(self, state: np.ndarray, dt: float = 0.1) -> ConsciousnessState:
        """Evolui estado de consciência
        
        Args:
            state: Estado inicial
            dt: Passo de tempo
            
        Returns:
            ConsciousnessState: Novo estado
        """
        # Normalização inicial
        if not np.isclose(np.linalg.norm(state), 1.0):
            state = state / np.linalg.norm(state)
            
        # Aplica operadores em sequência
        folded = self.operators.folding_operator(state)
        resonant = self.operators.morphic_resonance(folded)
        emerged = self.operators.emergence_operator(resonant)
        
        # Atualiza métricas
        self.operators.update_metrics(emerged, self.metrics)
        self.metrics.update(emerged, dt)
        
        # Cria estado de consciência
        consciousness = ConsciousnessState(
            state=emerged,
            metrics=self.metrics,
            operator_metrics=self.operators.metrics,
            time=dt
        )
        
        # Atualiza histórico
        consciousness.history.append({
            'time': dt,
            'coherence': self.metrics.field_coherence,
            'resonance': self.operators.metrics.morphic_resonance,
            'emergence': self.operators.metrics.emergence_factor
        })
        
        return consciousness
    
    def _update_adaptation(self, state: ConsciousnessState):
        """Atualiza parâmetros adaptativos
        
        Args:
            state: Estado atual
        """
        # Ajusta taxa de adaptação
        coherence_trend = np.mean([h['coherence'] for h in state.history[-10:]])
        if coherence_trend > self.stability_threshold:
            self.adaptation_rate *= 0.9  # Reduz adaptação
        else:
            self.adaptation_rate *= 1.1  # Aumenta adaptação
            
        # Limita taxa entre [0.01, 1.0]
        self.adaptation_rate = np.clip(self.adaptation_rate, 0.01, 1.0)
        
        # Ajusta limiares
        resonance = state.operator_metrics.morphic_resonance
        if resonance > self.resonance_threshold:
            self.stability_threshold *= 1.05  # Aumenta estabilidade
        else:
            self.stability_threshold *= 0.95  # Reduz estabilidade
            
        # Limita limiares entre [0.5, 0.9]
        self.stability_threshold = np.clip(self.stability_threshold, 0.5, 0.9)
        self.resonance_threshold = np.clip(self.resonance_threshold, 0.5, 0.9)
