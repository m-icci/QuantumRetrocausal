"""
Proxy para qualia_core.security.decoherence_protection
"""

import sys
import os
from pathlib import Path

# Adicionar caminho do projeto ao sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

# Importar diretamente do módulo original
try:
    from qualia_core.security.decoherence_protection import *
except ImportError as e:
    import warnings
    warnings.warn(f"Não foi possível importar qualia_core.security.decoherence_protection: {e}")
    
    # Criar stub para a classe QuantumShield para evitar erros
    class QuantumShield:
        """Stub para QuantumShield (não foi possível importar o original)"""
        def __init__(self, *args, **kwargs):
            print("AVISO: Esta é uma implementação stub de QuantumShield - funcionalidade limitada")
            
        def protect(self, *args, **kwargs):
            """Função stub para protect"""
            return args[0] if args else None
            
        def activate(self, *args, **kwargs):
            """Função stub para activate"""
            return True 