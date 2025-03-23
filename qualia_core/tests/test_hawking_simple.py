"""
Testes simples para comunicação via partículas de Hawking.
"""

import numpy as np
import pytest
import logging
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

@dataclass
class TestConfig:
    """Configurações para os testes."""
    num_particles: int = 100
    energy: float = 1.0
    mass: float = 1.0
    time_step: float = 0.01

class HawkingParticle:
    """Classe para simular partículas de Hawking."""
    
    def __init__(self, energy=1.0, mass=1.0):
        self.energy = max(energy, 1e-10)  # Garante energia positiva
        self.mass = max(mass, 1e-10)      # Garante massa positiva
        # Garante que o produto energia * massa seja maior que e para log positivo
        self.entropy = np.log(max(self.energy * self.mass, np.e))
        self.timestamp = datetime.now()
        
        logging.debug(f"Partícula criada: energia={self.energy}, massa={self.mass}, entropia={self.entropy}")
    
    def propagate(self, time):
        """Simula a propagação da partícula."""
        decay_rate = 0.1
        old_energy = self.energy
        self.energy *= np.exp(-decay_rate * time)
        # Atualiza a entropia após a propagação
        self.entropy = np.log(max(self.energy * self.mass, np.e))
        
        logging.debug(f"Partícula propagada: energia={old_energy}->{self.energy}, entropia={self.entropy}")
        return self.energy

def run_tests(config: TestConfig):
    """Executa os testes com a configuração fornecida."""
    logger = logging.getLogger(__name__)
    
    # Teste de criação
    logger.info("Testando criação de partículas")
    particles = []
    for i in tqdm(range(config.num_particles), desc="Criando partículas"):
        particle = HawkingParticle(energy=config.energy, mass=config.mass)
        logger.debug(f"Partícula {i}: energia={particle.energy}, massa={particle.mass}, entropia={particle.entropy}")
        assert particle.energy > 0, f"Energia deve ser positiva, mas é {particle.energy}"
        assert particle.mass > 0, f"Massa deve ser positiva, mas é {particle.mass}"
        assert particle.entropy > 0, f"Entropia deve ser positiva, mas é {particle.entropy}"
        particles.append(particle)
    
    # Teste de propagação
    logger.info("Testando propagação de partículas")
    for i, particle in enumerate(tqdm(particles, desc="Propagando partículas")):
        initial_energy = particle.energy
        final_energy = particle.propagate(time=config.time_step)
        logger.debug(f"Partícula {i}: energia inicial={initial_energy}, energia final={final_energy}")
        assert final_energy < initial_energy, f"Energia final ({final_energy}) deve ser menor que inicial ({initial_energy})"
    
    # Salva resultados
    logger.info("Salvando resultados...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    
    # Salva configuração
    with open(output_dir / f"config_{timestamp}.json", "w") as f:
        json.dump(config.__dict__, f, indent=4)
    
    # Salva resultados
    results = {
        "num_particles": len(particles),
        "mean_energy": float(np.mean([p.energy for p in particles])),
        "mean_entropy": float(np.mean([p.entropy for p in particles]))
    }
    
    with open(output_dir / f"results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Resultados salvos em {output_dir}")
    return results

if __name__ == "__main__":
    # Configuração de logging
    logging.basicConfig(
        level=logging.DEBUG,  # Alterado para DEBUG para mostrar mais informações
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Mostra logs no console
            logging.FileHandler("test_hawking.log")  # Salva logs em arquivo
        ]
    )
    
    # Carrega configuração se existir
    try:
        with open("test_config.json", "r") as f:
            config_data = json.load(f)
            config = TestConfig(**config_data)
    except FileNotFoundError:
        config = TestConfig()
    
    # Executa os testes
    results = run_tests(config)
    
    # Exibe resultados
    print("\nResultados dos testes:")
    for key, value in results.items():
        print(f"{key}: {value}") 