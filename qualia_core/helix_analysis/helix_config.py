#!/usr/bin/env python3
"""
HelixConfig: Configuração para o sistema de análise Helix
Define parâmetros essenciais para a inicialização e evolução do campo quântico.
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class HelixConfig:
    """
    Configuração para o Analisador Helix.
    Define parâmetros para dimensões do campo, características quânticas, 
    e granularidade da análise fractal.
    """
    
    # Dimensões do campo da hélice
    dimensions: int = 256
    
    # Características quânticas
    num_qubits: int = 8
    
    # Constantes físicas
    phi: float = 0.618  # Proporção áurea para evolução quasi-periódica
    temperature: float = 0.1  # Temperatura para flutuações quânticas
    
    # Parâmetros de processamento
    batch_size: int = 1024  # Tamanho do lote para histórico
    max_field_size: int = 1024  # Tamanho máximo do campo
    
    def validate(self) -> bool:
        """
        Valida a configuração.
        
        Returns:
            True se a configuração é válida, False caso contrário
        """
        if self.dimensions <= 0:
            return False
        
        if self.num_qubits <= 0:
            return False
        
        if self.phi <= 0 or self.phi >= 1:
            return False
        
        if self.temperature < 0:
            return False
        
        if self.batch_size <= 0:
            return False
        
        if self.max_field_size <= 0:
            return False
        
        return True
    
    def __post_init__(self):
        """Validação automática após inicialização"""
        if not self.validate():
            raise ValueError("Configuração da hélice inválida") 