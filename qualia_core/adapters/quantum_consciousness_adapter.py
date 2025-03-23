"""
Quantum Consciousness Adapter Module
Adapta interfaces de consciência quântica.
"""

from typing import Dict, Optional, List
import numpy as np

from quantum.core.consciousness.quantum_consciousness import (
    QualiaState,
    SystemBehavior,
    ConsciousnessObservation,
    QuantumConsciousness
)

class QuantumConsciousnessAdapter:
    """
    Adaptador para interface com sistema de consciência quântica
    """
    
    def __init__(self, dimensions: int = 8):
        self.consciousness = QuantumConsciousness(dimensions)
        
    def adapt_market_state(self, 
                          price_data: Dict[str, np.ndarray],
                          field_strength: float) -> ConsciousnessObservation:
        """
        Adapta estado do mercado para observação de consciência
        
        Args:
            price_data: Dados de preço OHLCV
            field_strength: Força do campo mórfico
            
        Returns:
            Observação de consciência adaptada
        """
        # Calcula métricas base
        returns = np.diff(price_data['close']) / price_data['close'][:-1]
        volatility = np.std(returns)
        volume = price_data['volume']
        volume_ma = np.mean(volume)
        
        # Cria estado de qualia
        qualia = QualiaState(
            intensity=min(1.0, volatility * 10),  # Maior volatilidade = maior intensidade
            complexity=min(1.0, len(returns) / 1000),  # Mais dados = maior complexidade
            coherence=1.0 - min(1.0, volatility * 5)  # Menor volatilidade = maior coerência
        )
        
        # Cria comportamento do sistema
        behavior = SystemBehavior(
            pattern_type="market",
            frequency=1.0,  # Frequência base
            stability=1.0 - min(1.0, volatility * 3),  # Menor volatilidade = maior estabilidade
            resonance=min(1.0, volume[-1] / volume_ma),  # Volume relativo à média
            field_strength=field_strength
        )
        
        # Cria estado quântico sintético
        quantum_state = np.array([
            np.sqrt(1 - volatility),  # Amplitude base
            np.sqrt(volatility)  # Excitação
        ])
        
        return ConsciousnessObservation(
            qualia=qualia,
            behavior=behavior,
            quantum_state=quantum_state
        )
