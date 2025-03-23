"""
Módulo proxy para core.code_analyzer

Este módulo redireciona importações para C:\\Users\\Natalia\\Documents\\Miner\\core\\quantum\\Code_analyzer
"""

import sys
import os
from pathlib import Path

# Adicionar o caminho do módulo original ao sys.path
miner_core_path = r"C:\Users\Natalia\Documents\Miner\core"
quantum_path = os.path.join(miner_core_path, 'quantum')
code_analyzer_path = os.path.join(quantum_path, 'Code_analyzer')
sys.path.append(quantum_path)

# Tentar importar do módulo original
try:
    # Verificar se o diretório Code_analyzer existe
    if os.path.exists(code_analyzer_path):
        # Ajustar sys.path temporariamente
        original_path = sys.path.copy()
        sys.path.insert(0, quantum_path)
        
        # Tentar importar
        try:
            from Code_analyzer import *
            print("Importação de Code_analyzer bem-sucedida")
        except ImportError as e:
            print(f"Erro ao importar Code_analyzer: {e}")
        
        # Restaurar sys.path
        sys.path = original_path
    else:
        print(f"Diretório Code_analyzer não encontrado em {code_analyzer_path}")
    
    # Tentar importar o arquivo principal do Code_analyzer diretamente
    code_analyzer_main_path = os.path.join(code_analyzer_path, 'code_analyzer.py')
    if os.path.exists(code_analyzer_main_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("code_analyzer", code_analyzer_main_path)
        code_analyzer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(code_analyzer)
        
        # Exportar as classes e funções
        for attr in dir(code_analyzer):
            if not attr.startswith('_'):
                globals()[attr] = getattr(code_analyzer, attr)
                
        print(f"Módulo code_analyzer.py carregado diretamente de {code_analyzer_main_path}")
    else:
        # Tentar importar qualquer arquivo .py encontrado no diretório
        if os.path.exists(code_analyzer_path):
            for filename in os.listdir(code_analyzer_path):
                if filename.endswith('.py') and not filename.startswith('__'):
                    file_path = os.path.join(code_analyzer_path, filename)
                    module_name = filename[:-3]  # Remover '.py'
                    
                    import importlib.util
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Exportar as classes e funções
                    for attr in dir(module):
                        if not attr.startswith('_'):
                            globals()[attr] = getattr(module, attr)
                    
                    print(f"Módulo {filename} carregado diretamente de {file_path}")
        else:
            print(f"Nenhum arquivo Python encontrado em {code_analyzer_path}")
        
except Exception as e:
    print(f"Erro ao configurar importação de code_analyzer: {e}")
    
    # Criar stubs para classes importantes
    class CodeAnalyzer:
        """Stub para CodeAnalyzer"""
        def __init__(self, *args, **kwargs):
            print("AVISO: Esta é uma implementação stub de CodeAnalyzer")
            
        def analyze(self, *args, **kwargs):
            """Função stub"""
            return {"status": "simulado", "resultado": {}}
            
        def process(self, *args, **kwargs):
            """Função stub"""
            return args[0] if args else None 