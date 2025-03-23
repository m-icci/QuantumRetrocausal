"""
Configuração do Sistema QUALIA

Este módulo implementa a configuração do sistema QUALIA, incluindo parâmetros
quânticos, cosmológicos e de processamento.
"""

import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class QuantumConfig:
    """Configuração do sistema quântico"""
    
    # Dimensões e qubits
    dimensions: int = 256
    num_qubits: int = 8
    
    # Parâmetros quânticos
    phi: float = 0.618033988749895  # Proporção áurea
    temperature: float = 0.1
    
    # Limites e capacidades
    holographic_memory_limit: int = 1000
    max_parallel_tasks: int = 100
    
    # Parâmetros de rede
    pub_port: int = 5555
    sub_port: int = 5556
    
    # Parâmetros cosmológicos
    cosmology_alpha: float = 1.0
    cosmology_lambda_0: float = 1e-52
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte configuração para dicionário"""
        return {
            'dimensions': self.dimensions,
            'num_qubits': self.num_qubits,
            'phi': self.phi,
            'temperature': self.temperature,
            'holographic_memory_limit': self.holographic_memory_limit,
            'max_parallel_tasks': self.max_parallel_tasks,
            'pub_port': self.pub_port,
            'sub_port': self.sub_port,
            'cosmology_alpha': self.cosmology_alpha,
            'cosmology_lambda_0': self.cosmology_lambda_0
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuantumConfig':
        """Cria configuração a partir de dicionário"""
        return cls(**data)
        
    @classmethod
    def load(cls, config_path: str = "qualia_config.json") -> 'QuantumConfig':
        """Carrega configuração de arquivo"""
        try:
            if not os.path.exists(config_path):
                logger.warning(f"Arquivo de configuração {config_path} não encontrado")
                return cls()
                
            with open(config_path, 'r') as f:
                data = json.load(f)
                
            return cls.from_dict(data)
            
        except Exception as e:
            logger.error(f"Erro ao carregar configuração: {e}")
            return cls()
            
    def save(self, config_path: str = "qualia_config.json") -> bool:
        """Salva configuração em arquivo"""
        try:
            with open(config_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=4)
                
            logger.info(f"Configuração salva em {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao salvar configuração: {e}")
            return False
            
    def update(self, **kwargs) -> None:
        """Atualiza configuração com novos valores"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
    def validate(self) -> bool:
        """Valida configuração"""
        try:
            # Verifica dimensões
            if self.dimensions <= 0:
                logger.error("Dimensões devem ser positivas")
                return False
                
            # Verifica qubits
            if self.num_qubits <= 0:
                logger.error("Número de qubits deve ser positivo")
                return False
                
            # Verifica phi
            if not 0 < self.phi < 1:
                logger.error("Phi deve estar entre 0 e 1")
                return False
                
            # Verifica temperatura
            if self.temperature < 0:
                logger.error("Temperatura não pode ser negativa")
                return False
                
            # Verifica limites
            if self.holographic_memory_limit <= 0:
                logger.error("Limite de memória holográfica deve ser positivo")
                return False
                
            if self.max_parallel_tasks <= 0:
                logger.error("Número máximo de tarefas paralelas deve ser positivo")
                return False
                
            # Verifica portas
            if not 1024 <= self.pub_port <= 65535:
                logger.error("Porta PUB deve estar entre 1024 e 65535")
                return False
                
            if not 1024 <= self.sub_port <= 65535:
                logger.error("Porta SUB deve estar entre 1024 e 65535")
                return False
                
            # Verifica parâmetros cosmológicos
            if self.cosmology_alpha <= 0:
                logger.error("Alpha cosmológico deve ser positivo")
                return False
                
            if self.cosmology_lambda_0 <= 0:
                logger.error("Lambda cosmológico deve ser positivo")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Erro ao validar configuração: {e}")
            return False

if __name__ == "__main__":
    # Exemplo de uso
    config = QuantumConfig()
    
    # Atualiza alguns parâmetros
    config.update(
        dimensions=512,
        num_qubits=16,
        temperature=0.2
    )
    
    # Valida configuração
    if config.validate():
        # Salva configuração
        config.save()
        
        # Carrega configuração
        loaded_config = QuantumConfig.load()
        print("Configuração carregada:", loaded_config.to_dict())
