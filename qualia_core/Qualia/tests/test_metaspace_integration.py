"""
Teste de Integração do Metaespaço
Valida a integração entre QuantumVoid, QuantumConsciousnessOperator e QuantumDance
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Optional, Tuple, List, Any

from core.fields.quantum_void import QuantumVoid
from core.fields.quantum_dance import QuantumDance
from core.consciousness.consciousness_operator import QuantumConsciousnessOperator
from core.constants import QualiaConstants

class MetaspaceVisualizer:
    """Visualizador do Metaespaço e suas interações"""
    
    def __init__(self, size: int = 64, granularity: int = 21):
        """
        Inicializa visualizador
        
        Args:
            size: Tamanho do campo
            granularity: Granularidade (3, 21 ou 42 bits)
        """
        if not QualiaConstants.validate_granularity(granularity):
            raise ValueError(f"Granularidade {granularity} inválida. Use 3, 21 ou 42 bits.")
            
        self.size = size
        self.granularity = granularity
        self.void = QuantumVoid(size, granularity)
        self.dance = QuantumDance(size, granularity)
        self.consciousness = QuantumConsciousnessOperator()
        
        # Histórico para visualização
        self.void_history: List[Dict[str, Any]] = []
        self.dance_history: List[Dict[str, Any]] = []
        self.consciousness_history: List[Dict[str, float]] = []
        
    def evolve(self, steps: int = 100) -> None:
        """Evolui o sistema integrado"""
        
        print(f"Evoluindo sistema integrado (granularidade: {self.granularity} bits)...")
        for step in range(steps):
            # Gera dados sintéticos de mercado
            market_data = np.random.normal(0, 1, self.size)
            
            # Evolui componentes
            void_state = self.void.evolve(market_data)
            dance_state = self.dance.evolve(market_data)
            
            # Integra estados no operador de consciência
            consciousness_state = self.consciousness.apply_consciousness(
                market_data,
                morphic_field=dance_state['field']
            )
            
            # Extrai métricas de consciência
            if isinstance(consciousness_state, tuple):
                state, metrics = consciousness_state
                consciousness_metrics = {
                    'coherence': metrics.coherence_level if hasattr(metrics, 'coherence_level') else 0.0,
                    'resonance': metrics.resonance if hasattr(metrics, 'resonance') else 0.0
                }
            else:
                consciousness_metrics = {
                    'coherence': consciousness_state.coherence_level if hasattr(consciousness_state, 'coherence_level') else 0.0,
                    'resonance': consciousness_state.resonance if hasattr(consciousness_state, 'resonance') else 0.0
                }
            
            # Armazena histórico
            self.void_history.append(void_state)
            self.dance_history.append(dance_state)
            self.consciousness_history.append(consciousness_metrics)
            
            if step % 10 == 0:
                print(f"Passo {step}: Energia do Vazio = {void_state['energy']:.2f}, "
                      f"Coerência da Dança = {dance_state['coherence']:.2f}")
                
    def plot_metaspace(self) -> None:
        """Plota visualização do Metaespaço"""
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        # Plot do Vazio Quântico
        void_energy = [state['energy'] for state in self.void_history]
        ax1.plot(void_energy, label='Energia do Vazio')
        ax1.set_title(f'Evolução do Vazio Quântico (G={self.granularity})')
        ax1.set_xlabel('Passo')
        ax1.set_ylabel('Energia')
        ax1.grid(True)
        ax1.legend()
        
        # Plot da Dança Quântica
        dance_coherence = [state['coherence'] for state in self.dance_history]
        ax2.plot(dance_coherence, label='Coerência da Dança')
        ax2.set_title(f'Evolução da Dança Quântica (G={self.granularity})')
        ax2.set_xlabel('Passo')
        ax2.set_ylabel('Coerência')
        ax2.grid(True)
        ax2.legend()
        
        # Plot do Operador de Consciência
        consciousness_coherence = [state['coherence'] for state in self.consciousness_history]
        consciousness_resonance = [state['resonance'] for state in self.consciousness_history]
        ax3.plot(consciousness_coherence, label='Nível de Coerência')
        ax3.plot(consciousness_resonance, label='Ressonância', linestyle='--')
        ax3.set_title(f'Evolução do Operador de Consciência (G={self.granularity})')
        ax3.set_xlabel('Passo')
        ax3.set_ylabel('Métrica')
        ax3.grid(True)
        ax3.legend()
        
        plt.tight_layout()
        plt.show()
        
    def analyze_integration(self) -> Dict[str, float]:
        """Analisa métricas de integração entre componentes"""
        
        # Calcula correlações entre componentes
        void_energy = np.array([state['energy'] for state in self.void_history])
        dance_coherence = np.array([state['coherence'] for state in self.dance_history])
        consciousness_coherence = np.array([state['coherence'] for state in self.consciousness_history])
        
        void_dance_corr = np.corrcoef(void_energy, dance_coherence)[0,1]
        void_consciousness_corr = np.corrcoef(void_energy, consciousness_coherence)[0,1]
        dance_consciousness_corr = np.corrcoef(dance_coherence, consciousness_coherence)[0,1]
        
        # Calcula métricas específicas da granularidade
        weight = QualiaConstants.get_granularity_weight(self.granularity)
        coherence_factor = QualiaConstants.get_coherence_factor(self.granularity)
        
        return {
            'void_dance_correlation': void_dance_corr,
            'void_consciousness_correlation': void_consciousness_corr,
            'dance_consciousness_correlation': dance_consciousness_corr,
            'granularity_weight': weight,
            'coherence_factor': coherence_factor,
            'mean_void_energy': float(np.mean(void_energy)),
            'mean_dance_coherence': float(np.mean(dance_coherence)),
            'mean_consciousness_coherence': float(np.mean(consciousness_coherence))
        }

def test_granularity(granularity: int) -> None:
    """
    Testa sistema com granularidade específica
    
    Args:
        granularity: Granularidade a testar (3, 21 ou 42 bits)
    """
    print(f"\nTestando granularidade {granularity} bits...")
    
    # Cria visualizador
    visualizer = MetaspaceVisualizer(size=64, granularity=granularity)
    
    # Evolui sistema
    visualizer.evolve(steps=100)
    
    # Analisa integração
    metrics = visualizer.analyze_integration()
    print("\nMétricas de Integração:")
    for key, value in metrics.items():
        print(f"{key}: {value:.3f}")
    
    # Plota resultados
    print("\nGerando visualização do Metaespaço...")
    visualizer.plot_metaspace()

def main():
    """Executa teste de integração do Metaespaço"""
    
    print("Iniciando teste de integração do Metaespaço...")
    
    # Testa todas as granularidades ativas
    for granularity in QualiaConstants.ACTIVE_GRANULARITIES:
        test_granularity(granularity)
    
if __name__ == "__main__":
    main()
