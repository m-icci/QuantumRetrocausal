"""
Configuração central unificada do sistema
"""
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class SystemPaths:
    """Caminhos do sistema"""
    root: Path = Path(__file__).parent.parent.parent
    data: Path = root / "data"
    logs: Path = root / "logs"
    config: Path = root / "config"

@dataclass
class ExchangeConfig:
    """Configuração de exchange"""
    api_key: str = os.environ.get("KRAKEN_API_KEY", "")
    api_secret: str = os.environ.get("KRAKEN_SECRET_KEY", "")
    base_url: str = "https://api.kraken.com"
    timeout: int = 30

@dataclass
class DatabaseConfig:
    """Configuração do banco de dados"""
    url: str = os.environ.get("DATABASE_URL", "")
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30

@dataclass
class QuantumConfig:
    """Configuração do sistema quântico"""
    
    num_qubits: int = 8
    """Número de qubits no sistema"""
    
    temperature: float = 0.1
    """Temperatura do sistema quântico"""
    
    coherence_time: float = 1.0
    """Tempo de coerência quântica"""
    
    entanglement_threshold: float = 0.5
    """Limiar de emaranhamento"""
    
    def validate(self) -> bool:
        """Valida a configuração"""
        if not 1 <= self.num_qubits <= 64:
            return False
        if not 0 <= self.temperature <= 1:
            return False
        if not 0 < self.coherence_time <= 10:
            return False
        if not 0 <= self.entanglement_threshold <= 1:
            return False
        return True

class SystemConfig:
    """Configuração central do sistema"""
    def __init__(self):
        self.paths = SystemPaths()
        self.exchange = ExchangeConfig()
        self.database = DatabaseConfig()
        
        # Cria diretórios necessários
        self.paths.data.mkdir(exist_ok=True)
        self.paths.logs.mkdir(exist_ok=True)
        self.paths.config.mkdir(exist_ok=True)
        
    def get_exchange_config(self) -> Dict[str, Any]:
        """Retorna configuração da exchange"""
        return {
            "api_key": self.exchange.api_key,
            "api_secret": self.exchange.api_secret,
            "base_url": self.exchange.base_url,
            "timeout": self.exchange.timeout
        }
        
    def get_database_config(self) -> Dict[str, Any]:
        """Retorna configuração do banco de dados"""
        return {
            "url": self.database.url,
            "pool_size": self.database.pool_size,
            "max_overflow": self.database.max_overflow,
            "pool_timeout": self.database.pool_timeout
        }

# Instância global de configuração
system_config = SystemConfig()
