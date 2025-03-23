"""
Configuração de Memória do Sistema QUALIA

Este módulo implementa a configuração de memória do sistema QUALIA,
incluindo gerenciamento de memória virtual e cache em disco.
"""

import os
import logging
from pathlib import Path
from typing import Optional

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MemoryConfig:
    """Configuração de memória do sistema"""
    
    def __init__(
        self,
        swap_dir: str = "D:/qualia_swap",
        temp_dir: str = "D:/qualia_temp",
        max_memory_gb: float = 8.0,
        cache_size_mb: int = 512
    ):
        self.swap_dir = Path(swap_dir)
        self.temp_dir = Path(temp_dir)
        self.max_memory_gb = max_memory_gb
        self.cache_size_mb = cache_size_mb
        
        # Cria diretórios se não existirem
        self._setup_directories()
        
    def _setup_directories(self) -> None:
        """Configura diretórios de swap e temporários"""
        try:
            # Cria diretório de swap
            self.swap_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Diretório de swap criado: {self.swap_dir}")
            
            # Cria diretório temporário
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Diretório temporário criado: {self.temp_dir}")
            
        except Exception as e:
            logger.error(f"Erro ao criar diretórios: {e}")
            raise
            
    def get_swap_path(self, filename: str) -> Path:
        """Retorna caminho para arquivo de swap"""
        return self.swap_dir / filename
        
    def get_temp_path(self, filename: str) -> Path:
        """Retorna caminho para arquivo temporário"""
        return self.temp_dir / filename
        
    def cleanup_temp_files(self) -> None:
        """Limpa arquivos temporários"""
        try:
            for file in self.temp_dir.glob("*"):
                try:
                    file.unlink()
                except Exception as e:
                    logger.warning(f"Erro ao remover arquivo temporário {file}: {e}")
                    
            logger.info("Arquivos temporários removidos")
            
        except Exception as e:
            logger.error(f"Erro ao limpar arquivos temporários: {e}")
            
    def cleanup_swap_files(self) -> None:
        """Limpa arquivos de swap"""
        try:
            for file in self.swap_dir.glob("*"):
                try:
                    file.unlink()
                except Exception as e:
                    logger.warning(f"Erro ao remover arquivo de swap {file}: {e}")
                    
            logger.info("Arquivos de swap removidos")
            
        except Exception as e:
            logger.error(f"Erro ao limpar arquivos de swap: {e}")
            
    def __enter__(self):
        """Contexto de entrada"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Contexto de saída"""
        self.cleanup_temp_files()
        self.cleanup_swap_files() 