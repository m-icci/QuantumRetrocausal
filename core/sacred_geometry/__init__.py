"""
Módulo proxy para core.sacred_geometry

Este módulo redireciona importações para C:\\Users\\Natalia\\Documents\\Miner\\core\\sacred_geometry
"""

import sys
import os
from pathlib import Path

# Adicionar o caminho do módulo original ao sys.path
miner_core_path = r"C:\Users\Natalia\Documents\Miner\core"
sys.path.append(miner_core_path)

# Tentar importar do módulo original
try:
    # Importar sacred_geometry do caminho original
    # Primeiro verificamos se o diretório existe
    if os.path.exists(os.path.join(miner_core_path, 'sacred_geometry')):
        # Ajustamos sys.path temporariamente para importar do local correto
        original_path = sys.path.copy()
        sys.path.insert(0, miner_core_path)
        
        # Agora tentamos importar
        try:
            from sacred_geometry import *
            print("Importação de sacred_geometry bem-sucedida")
        except ImportError as e:
            print(f"Erro ao importar sacred_geometry: {e}")
        
        # Restauramos sys.path
        sys.path = original_path
    else:
        print(f"Diretório sacred_geometry não encontrado em {miner_core_path}")
        
    # Tentar importar sacred_geometry.py diretamente
    sacred_geometry_path = os.path.join(miner_core_path, 'sacred_geometry', 'sacred_geometry.py')
    if os.path.exists(sacred_geometry_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("sacred_geometry", sacred_geometry_path)
        sacred_geometry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sacred_geometry)
        
        # Exportar todas as definições do módulo
        for attr in dir(sacred_geometry):
            if not attr.startswith('_'):
                globals()[attr] = getattr(sacred_geometry, attr)
        
        print(f"Módulo sacred_geometry.py carregado diretamente de {sacred_geometry_path}")
    else:
        print(f"Arquivo sacred_geometry.py não encontrado em {sacred_geometry_path}")
        
except Exception as e:
    print(f"Erro ao configurar importação de sacred_geometry: {e}")
    
    # Definir stubs para as funções principais para evitar erros
    def phi_encode(*args, **kwargs):
        """Stub para phi_encode"""
        return args[0] if args else None
        
    class SacredGeometry:
        """Stub para SacredGeometry"""
        def __init__(self, *args, **kwargs):
            print("AVISO: Esta é uma implementação stub de SacredGeometry")
        
        def calculate(self, *args, **kwargs):
            """Função stub"""
            return 1.618 # Valor phi 