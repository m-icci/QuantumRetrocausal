"""
Módulo Core para QUALIA
=======================

Este módulo serve como um proxy para os submódulos distribuídos pelo sistema.
Ele foi criado para resolver problemas de importação onde partes do código 
tentam importar de 'core.*' mas os módulos estão em outros locais.
"""

import sys
import os
import importlib
from pathlib import Path

# Adicionar caminhos dos módulos ao sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

# Adicionar o caminho para o core original do Miner
miner_core_path = r"C:\Users\Natalia\Documents\Miner\core"
if os.path.exists(miner_core_path):
    sys.path.append(miner_core_path)
    print(f"Adicionado caminho para Miner core: {miner_core_path}")

# Lista de submódulos que sabemos que são importados
SUBMODS = [
    'fields',
    'quantum',
    'security',
    'consciousness',
    'sacred_geometry',
    'dark_finance',
    'logging',
    'hardware',
    'config',
    'trading',
    'Code_analyzer',
    'code_analyzer',
    'adaptive_granularity'
]

# Exportar os submódulos para o namespace deste pacote
__all__ = SUBMODS

# Função para tentar importar um submódulo de diferentes locais
def try_import_from_multiple_locations(module_name):
    # Locais onde procurar o módulo, em ordem de prioridade
    locations = [
        f"C:\\Users\\Natalia\\Documents\\Miner\\core\\{module_name}",
        f"qualia_core.{module_name}",
        f"quantum_trading.core.{module_name}"
    ]
    
    for location in locations:
        try:
            if location.startswith("C:"):
                # É um caminho de arquivo, verificamos se existe
                if os.path.exists(location):
                    print(f"Módulo {module_name} encontrado em {location}")
                    return True
            else:
                # É um módulo Python, tentamos importar
                importlib.import_module(location)
                print(f"Módulo {module_name} importado de {location}")
                return True
        except (ImportError, ModuleNotFoundError):
            continue
    
    print(f"Módulo {module_name} não encontrado em nenhum local conhecido")
    return False

# Classe para lazy loading dinâmico de submódulos
class LazyLoader:
    def __init__(self, module_name):
        self.module_name = module_name
        self.module = None
        self.searched = False
        
    def __getattr__(self, name):
        if self.module is None and not self.searched:
            self.searched = True
            # Tentar importar do local correspondente
            
            # 1. Tentar do diretório Miner
            miner_module = os.path.join(miner_core_path, self.module_name)
            if os.path.exists(miner_module):
                # Módulo existe no Miner, importamos do proxy que já criamos
                try:
                    self.module = importlib.import_module(f"core.{self.module_name}")
                    return getattr(self.module, name)
                except (ImportError, AttributeError) as e:
                    print(f"Erro ao importar {self.module_name} de core.{self.module_name}: {e}")
            
            # 2. Tentar de qualia_core
            try:
                self.module = importlib.import_module(f"qualia_core.{self.module_name}")
                return getattr(self.module, name)
            except (ImportError, AttributeError):
                pass
                
            # 3. Tentar de quantum_trading.core
            try:
                self.module = importlib.import_module(f"quantum_trading.core.{self.module_name}")
                return getattr(self.module, name)
            except (ImportError, AttributeError):
                pass
                
            # Se chegamos aqui, não conseguimos importar
            raise ImportError(f"Não foi possível importar o módulo '{self.module_name}' de nenhum local conhecido")
        
        # Se o módulo foi carregado, retornamos o atributo solicitado
        if self.module:
            return getattr(self.module, name)
            
        # Se já tentamos e não conseguimos carregar, lançamos erro
        raise ImportError(f"Não foi possível importar o módulo '{self.module_name}'")

# Registrar os lazy loaders para cada submódulo
for submod in SUBMODS:
    globals()[submod] = LazyLoader(submod)

# Importar explicitamente os submódulos que já sabemos onde estão
try:
    from qualia_core.fields import *
except ImportError:
    pass

# Para debugging
if 'QUALIA_DEBUG' in os.environ:
    print("core módulo inicializado com sucesso") 