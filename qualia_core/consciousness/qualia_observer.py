"""
QualiaObserver para monitoramento de estados qualitativos e consciência
"""
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

class QualiaObserver:
    def __init__(self):
        """Inicializa observador de estados qualitativos"""
        self.consciousness_level = 0.8  # Nível base de consciência
        self.qualia_intensity = 0.5    # Intensidade inicial
        
        # Histórico de estados
        self.state_history = []
        self.last_update = datetime.now()
        
        # Parâmetros de acoplamento
        self.coherence_weight = 0.4
        self.field_weight = 0.3
        self.consciousness_weight = 0.3
        
    def observe_state(self, 
                     coherence: float,
                     field_strength: float,
                     entropy: float) -> Dict[str, float]:
        """
        Observa e atualiza estado qualitativo do sistema
        
        Args:
            coherence: Coerência quântica [0-1]
            field_strength: Força do campo mórfico [0-1]
            entropy: Entropia do sistema [0-1]
        """
        # Atualiza consciência baseado em coerência e campo
        consciousness_delta = (
            0.2 * (coherence - 0.5) +     # Impacto da coerência
            0.1 * (field_strength - 0.3)   # Impacto do campo
        )
        
        self.consciousness_level = min(1.0, max(0.3,
            self.consciousness_level + consciousness_delta
        ))
        
        # Calcula nova intensidade qualia
        qualia_delta = (
            0.3 * (coherence - self.qualia_intensity) +  # Mudança por coerência
            0.2 * (1 - entropy)                          # Impacto da entropia
        )
        
        self.qualia_intensity = min(1.0, max(0.0,
            self.qualia_intensity + qualia_delta
        ))
        
        # Registra estado
        current_state = {
            'timestamp': datetime.now(),
            'coherence': coherence,
            'field_strength': field_strength,
            'entropy': entropy,
            'consciousness_level': self.consciousness_level,
            'qualia_intensity': self.qualia_intensity
        }
        
        self.state_history.append(current_state)
        self._trim_history()
        
        return {
            'consciousness_level': float(self.consciousness_level),
            'qualia_intensity': float(self.qualia_intensity)
        }
        
    def get_recent_state_average(self, window: int = 10) -> Dict[str, float]:
        """Calcula média dos estados recentes"""
        if not self.state_history:
            return {
                'consciousness_level': self.consciousness_level,
                'qualia_intensity': self.qualia_intensity
            }
            
        recent_states = self.state_history[-window:]
        
        avg_consciousness = np.mean([
            s['consciousness_level'] for s in recent_states
        ])
        avg_qualia = np.mean([
            s['qualia_intensity'] for s in recent_states
        ])
        
        return {
            'consciousness_level': float(avg_consciousness),
            'qualia_intensity': float(avg_qualia)
        }
        
    def _trim_history(self, max_size: int = 1000):
        """Mantém tamanho do histórico controlado"""
        if len(self.state_history) > max_size:
            self.state_history = self.state_history[-max_size:]
            
    def calculate_field_coupling(self, coherence: float) -> float:
        """Calcula acoplamento com campo mórfico"""
        base_coupling = 0.618  # Razão áurea como base
        
        # Ajusta acoplamento baseado em coerência
        coupling = base_coupling * (1 + (coherence - 0.5))
        
        return min(0.95, max(0.3, coupling))
