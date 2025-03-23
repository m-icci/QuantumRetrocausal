"""
Módulo proxy para qualia_core.fields

Este módulo redireciona importações de core.fields.* para qualia_core.fields.*
"""

import sys
import os
from pathlib import Path

# Adicionar caminhos dos módulos ao sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

# Importar de qualia_core.fields
try:
    from qualia_core.fields import *
    
    # Tentar importar submódulos específicos que sabemos que são referenciados
    submods = [
        'morphic_field',
        'quantum_void', 
        'quantum_dance',
        'retrocausal_dance',
        'conscious_black_hole'
    ]
    
    # Importar dinamicamente
    for submod in submods:
        try:
            mod = __import__(f'qualia_core.fields.{submod}', fromlist=[submod])
            globals()[submod] = mod
            if not hasattr(mod, '__all__'):
                for attr in dir(mod):
                    if not attr.startswith('_'):
                        globals()[attr] = getattr(mod, attr)
        except ImportError as e:
            print(f"Aviso: Não foi possível importar qualia_core.fields.{submod}: {e}")
            
except ImportError as e:
    print(f"Aviso: Não foi possível importar qualia_core.fields: {e}") 