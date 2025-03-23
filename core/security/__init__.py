"""
Módulo proxy para qualia_core.security

Este módulo redireciona importações de core.security.* para qualia_core.security.*
"""

import sys
import os
from pathlib import Path

# Adicionar caminhos dos módulos ao sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

# Importar de qualia_core.security
try:
    from qualia_core.security import *
    
    # Tentar importar submódulos específicos que sabemos que são referenciados
    submods = [
        'decoherence_protection'
    ]
    
    # Importar dinamicamente
    for submod in submods:
        try:
            mod = __import__(f'qualia_core.security.{submod}', fromlist=[submod])
            globals()[submod] = mod
            # Exportar as classes/funções importantes
            if hasattr(mod, 'QuantumShield'):
                globals()['QuantumShield'] = mod.QuantumShield
            # Exportar todos os atributos não privados se não houver __all__
            if not hasattr(mod, '__all__'):
                for attr in dir(mod):
                    if not attr.startswith('_'):
                        globals()[attr] = getattr(mod, attr)
        except ImportError as e:
            print(f"Aviso: Não foi possível importar qualia_core.security.{submod}: {e}")
            
except ImportError as e:
    print(f"Aviso: Não foi possível importar qualia_core.security: {e}") 