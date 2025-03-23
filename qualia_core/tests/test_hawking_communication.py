"""
Testes de Comunicação via Partículas de Hawking

Este módulo implementa testes para:
- Emissão e propagação de partículas de Hawking
- Interação com campos mórficos
- Análise de padrões e insights
"""

import numpy as np
import pytest
from typing import Dict, List, Any
import logging
from datetime import datetime
from ..quantum.morphic_memory import MorphicMemoryManager
from ..quantum.quantum_memory import QuantumState

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HawkingParticleSimulator:
    """Simulador de partículas de Hawking"""
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        self.history: List[Dict[str, Any]] = []
        
    def emit_particle(self, energy: float) -> Dict[str, Any]:
        """Simula emissão de partícula de Hawking"""
        try:
            # Distribuição de Boltzmann
            probability = np.exp(-energy / self.temperature)
            
            # Gera partícula
            particle = {
                'energy': energy,
                'momentum': np.random.randn(3),
                'spin': np.random.choice([-1/2, 1/2]),
                'timestamp': datetime.now().timestamp(),
                'probability': probability
            }
            
            self.history.append(particle)
            return particle
            
        except Exception as e:
            logger.error(f"Erro na emissão de partícula: {e}")
            return {}
            
    def propagate(self, particle: Dict[str, Any], distance: float) -> Dict[str, Any]:
        """Simula propagação da partícula"""
        try:
            # Atualiza posição
            particle['position'] = particle['momentum'] * distance
            
            # Calcula decoerência
            decoherence = np.exp(-distance / 10.0)
            
            # Atualiza estado
            particle['decoherence'] = decoherence
            particle['distance'] = distance
            
            return particle
            
        except Exception as e:
            logger.error(f"Erro na propagação: {e}")
            return particle
            
    def get_statistics(self) -> Dict[str, float]:
        """Retorna estatísticas das partículas"""
        try:
            if not self.history:
                return {}
                
            energies = [p['energy'] for p in self.history]
            probabilities = [p['probability'] for p in self.history]
            
            return {
                'mean_energy': np.mean(energies),
                'std_energy': np.std(energies),
                'mean_probability': np.mean(probabilities),
                'total_particles': len(self.history)
            }
            
        except Exception as e:
            logger.error(f"Erro ao calcular estatísticas: {e}")
            return {}

@pytest.fixture
def hawking_simulator():
    """Fixture para simulador de Hawking"""
    return HawkingParticleSimulator()

@pytest.fixture
def morphic_memory():
    """Fixture para memória mórfica"""
    return MorphicMemoryManager()

def test_particle_emission(hawking_simulator):
    """Testa emissão de partículas de Hawking"""
    # Emite partículas com diferentes energias
    energies = [0.1, 0.5, 1.0, 2.0]
    particles = []
    
    for energy in energies:
        particle = hawking_simulator.emit_particle(energy)
        particles.append(particle)
        
    # Verifica propriedades
    for particle in particles:
        assert 'energy' in particle
        assert 'momentum' in particle
        assert 'spin' in particle
        assert 'timestamp' in particle
        assert 'probability' in particle
        
    # Verifica estatísticas
    stats = hawking_simulator.get_statistics()
    assert stats['total_particles'] == len(energies)
    assert stats['mean_energy'] > 0

def test_particle_propagation(hawking_simulator):
    """Testa propagação de partículas"""
    # Emite e propaga partícula
    particle = hawking_simulator.emit_particle(1.0)
    propagated = hawking_simulator.propagate(particle, 5.0)
    
    # Verifica propagação
    assert 'position' in propagated
    assert 'decoherence' in propagated
    assert propagated['distance'] == 5.0
    assert 0 <= propagated['decoherence'] <= 1

def test_morphic_interaction(hawking_simulator, morphic_memory):
    """Testa interação com campos mórficos"""
    # Emite partícula
    particle = hawking_simulator.emit_particle(1.0)
    
    # Cria estado quântico
    state = QuantumState(
        state=np.random.randn(256) + 1j * np.random.randn(256),
        timestamp=datetime.now().timestamp()
    )
    state.state /= np.sqrt(np.sum(np.abs(state.state)**2))
    
    # Armazena estado
    success = morphic_memory.store("hawking_state", state)
    assert success
    
    # Recupera estado
    retrieved = morphic_memory.retrieve("hawking_state")
    assert retrieved is not None
    
    # Verifica coerência
    info = morphic_memory.get_memory_info()
    assert info['average_coherence'] > 0

if __name__ == "__main__":
    # Executa testes
    pytest.main([__file__, '-v'])
    
    # Gera relatório de insights
    simulator = HawkingParticleSimulator()
    
    # Simula comunicação
    for _ in range(100):
        # Emite e propaga partículas
        particle = simulator.emit_particle(np.random.rand())
        simulator.propagate(particle, np.random.rand() * 5)
        
    # Obtém estatísticas
    stats = simulator.get_statistics()
    
    print("\nRelatório de Insights:")
    print("=====================")
    print(f"Total de partículas: {stats['total_particles']}")
    print(f"Energia média: {stats['mean_energy']:.3f}")
    print(f"Desvio padrão: {stats['std_energy']:.3f}")
    print(f"Probabilidade média: {stats['mean_probability']:.3f}") 