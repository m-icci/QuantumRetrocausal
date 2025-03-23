"""
Configurações para os testes de comunicação via partículas de Hawking.
"""

from dataclasses import dataclass
from typing import Dict, Any
import json
import os
import logging

@dataclass
class TestConfig:
    """Configurações para os testes de comunicação via partículas de Hawking."""
    
    # Parâmetros de simulação
    num_iterations: int = 100
    time_step: float = 0.01
    temperature: float = 1.0
    
    # Parâmetros de partículas
    particle_energy: float = 1.0
    particle_mass: float = 1.0
    particle_charge: float = 0.0
    
    # Parâmetros de memória mórfica
    morphic_resonance: float = 0.8
    morphic_coherence: float = 0.9
    
    # Parâmetros de análise
    coherence_threshold: float = 0.7
    efficiency_threshold: float = 0.6
    stability_threshold: float = 0.8
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte a configuração para um dicionário."""
        return {
            "num_iterations": self.num_iterations,
            "time_step": self.time_step,
            "temperature": self.temperature,
            "particle_energy": self.particle_energy,
            "particle_mass": self.particle_mass,
            "particle_charge": self.particle_charge,
            "morphic_resonance": self.morphic_resonance,
            "morphic_coherence": self.morphic_coherence,
            "coherence_threshold": self.coherence_threshold,
            "efficiency_threshold": self.efficiency_threshold,
            "stability_threshold": self.stability_threshold
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestConfig':
        """Cria uma instância de TestConfig a partir de um dicionário."""
        return cls(**data)
    
    @classmethod
    def load(cls, filepath: str) -> 'TestConfig':
        """Carrega a configuração de um arquivo JSON."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            logging.error(f"Erro ao carregar configuração: {e}")
            return cls()
    
    def save(self, filepath: str) -> None:
        """Salva a configuração em um arquivo JSON."""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=4)
        except Exception as e:
            logging.error(f"Erro ao salvar configuração: {e}")
    
    def update(self, **kwargs) -> None:
        """Atualiza os parâmetros da configuração."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def validate(self) -> bool:
        """Valida os parâmetros da configuração."""
        try:
            assert self.num_iterations > 0
            assert self.time_step > 0
            assert self.temperature >= 0
            assert self.particle_energy >= 0
            assert self.particle_mass >= 0
            assert 0 <= self.morphic_resonance <= 1
            assert 0 <= self.morphic_coherence <= 1
            assert 0 <= self.coherence_threshold <= 1
            assert 0 <= self.efficiency_threshold <= 1
            assert 0 <= self.stability_threshold <= 1
            return True
        except AssertionError:
            return False

# Configuração padrão
DEFAULT_CONFIG = TestConfig()

if __name__ == "__main__":
    # Exemplo de uso
    config = TestConfig()
    print("Configuração padrão:")
    print(json.dumps(config.to_dict(), indent=4))
    
    # Atualiza alguns parâmetros
    config.update(
        num_iterations=200,
        temperature=1.5,
        morphic_resonance=0.9
    )
    print("\nConfiguração atualizada:")
    print(json.dumps(config.to_dict(), indent=4))
    
    # Salva a configuração
    config.save("test_config.json")
    
    # Carrega a configuração
    loaded_config = TestConfig.load("test_config.json")
    print("\nConfiguração carregada:")
    print(json.dumps(loaded_config.to_dict(), indent=4)) 