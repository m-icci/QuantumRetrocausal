"""
Configuração da mineração.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import json
import os

@dataclass
class MiningConfig:
    """Configuração da mineração."""
    
    # Configurações gerais
    log_level: str = "INFO"
    metrics_interval: float = 1.0  # Intervalo para atualização de métricas em segundos
    
    # Configurações do minerador
    miner_type: str = "xmr-stak"  # Tipo de minerador (xmr-stak, xmrig, etc)
    threads: int = 4  # Número de threads
    intensity: int = 100  # Intensidade da mineração (0-100)
    priority: int = 2  # Prioridade do processo (0-5)
    
    # Configurações da pool
    pool_url: str = ""  # URL da pool
    pool_port: int = 3333  # Porta da pool
    wallet_address: str = ""  # Endereço da carteira
    pool_password: str = "x"  # Senha da pool (geralmente "x")
    
    # Configurações de hardware
    cpu_threads: int = 4  # Número de threads CPU
    gpu_threads: int = 2  # Número de threads GPU
    gpu_platform: int = 0  # Plataforma GPU
    gpu_device: int = 0  # Dispositivo GPU
    
    # Configurações de otimização
    auto_optimize: bool = True  # Se deve otimizar automaticamente
    optimization_interval: float = 300.0  # Intervalo para otimização em segundos
    min_hashrate: float = 0.0  # Hashrate mínimo aceitável
    max_power: float = 100.0  # Potência máxima em watts
    max_temperature: float = 80.0  # Temperatura máxima em Celsius
    
    # Configurações de pools
    pool_config_file: str = "pools.json"  # Arquivo de configuração das pools
    min_pool_latency: float = 1.0  # Latência mínima aceitável em segundos
    pool_switch_interval: float = 300.0  # Intervalo mínimo entre trocas de pool em segundos
    
    def __post_init__(self):
        """Validação após inicialização."""
        self._validate()
    
    def _validate(self) -> bool:
        """
        Valida a configuração.
        
        Returns:
            True se válida.
        """
        # Validações básicas
        if self.threads < 1:
            raise ValueError("Número de threads deve ser positivo")
        
        if not 0 <= self.intensity <= 100:
            raise ValueError("Intensidade deve estar entre 0 e 100")
        
        if not 0 <= self.priority <= 5:
            raise ValueError("Prioridade deve estar entre 0 e 5")
        
        # Validações de hardware
        if self.cpu_threads < 1:
            raise ValueError("Número de threads CPU deve ser positivo")
        
        if self.gpu_threads < 1:
            raise ValueError("Número de threads GPU deve ser positivo")
        
        if self.gpu_platform < 0:
            raise ValueError("Plataforma GPU deve ser não negativa")
        
        if self.gpu_device < 0:
            raise ValueError("Dispositivo GPU deve ser não negativo")
        
        # Validações de otimização
        if self.optimization_interval < 1.0:
            raise ValueError("Intervalo de otimização deve ser positivo")
        
        if self.min_hashrate < 0.0:
            raise ValueError("Hashrate mínimo deve ser não negativo")
        
        if self.max_power <= 0.0:
            raise ValueError("Potência máxima deve ser positiva")
        
        if self.max_temperature <= 0.0:
            raise ValueError("Temperatura máxima deve ser positiva")
        
        # Validações de pools
        if self.min_pool_latency <= 0.0:
            raise ValueError("Latência mínima da pool deve ser positiva")
        
        if self.pool_switch_interval < 1.0:
            raise ValueError("Intervalo de troca de pool deve ser positivo")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converte para dicionário.
        
        Returns:
            Dicionário com a configuração.
        """
        return {
            'log_level': self.log_level,
            'metrics_interval': self.metrics_interval,
            'miner_type': self.miner_type,
            'threads': self.threads,
            'intensity': self.intensity,
            'priority': self.priority,
            'pool_url': self.pool_url,
            'pool_port': self.pool_port,
            'wallet_address': self.wallet_address,
            'pool_password': self.pool_password,
            'cpu_threads': self.cpu_threads,
            'gpu_threads': self.gpu_threads,
            'gpu_platform': self.gpu_platform,
            'gpu_device': self.gpu_device,
            'auto_optimize': self.auto_optimize,
            'optimization_interval': self.optimization_interval,
            'min_hashrate': self.min_hashrate,
            'max_power': self.max_power,
            'max_temperature': self.max_temperature,
            'pool_config_file': self.pool_config_file,
            'min_pool_latency': self.min_pool_latency,
            'pool_switch_interval': self.pool_switch_interval
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MiningConfig':
        """
        Cria configuração a partir de dicionário.
        
        Args:
            data: Dicionário com a configuração.
            
        Returns:
            Configuração criada.
        """
        return cls(**data)
    
    def save(self, filepath: str) -> bool:
        """
        Salva configuração em arquivo.
        
        Args:
            filepath: Caminho do arquivo.
            
        Returns:
            True se salvo com sucesso.
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=4)
            return True
        except Exception:
            return False
    
    @classmethod
    def load(cls, filepath: str) -> Optional['MiningConfig']:
        """
        Carrega configuração de arquivo.
        
        Args:
            filepath: Caminho do arquivo.
            
        Returns:
            Configuração carregada ou None se erro.
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception:
            return None 