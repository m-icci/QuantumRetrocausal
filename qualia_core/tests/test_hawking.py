"""
Testes básicos para comunicação via partículas de Hawking.
"""

import numpy as np
import pytest
import logging

class HawkingParticle:
    """Classe para simular partículas de Hawking."""
    
    def __init__(self, energy=1.0, mass=1.0):
        self.energy = energy
        self.mass = mass
        self.entropy = np.log(energy * mass)
    
    def propagate(self, time):
        """Simula a propagação da partícula."""
        self.energy *= np.exp(-0.1 * time)
        return self.energy

def test_particle_creation():
    """Testa a criação de partículas."""
    particle = HawkingParticle()
    assert particle.energy > 0
    assert particle.mass > 0
    assert particle.entropy > 0

def test_particle_propagation():
    """Testa a propagação de partículas."""
    particle = HawkingParticle()
    initial_energy = particle.energy
    final_energy = particle.propagate(time=1.0)
    assert final_energy < initial_energy

if __name__ == "__main__":
    # Configuração de logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Executa os testes
    logger.info("Testando criação de partículas")
    test_particle_creation()
    
    logger.info("Testando propagação de partículas")
    test_particle_propagation()
    
    logger.info("Todos os testes passaram!") 