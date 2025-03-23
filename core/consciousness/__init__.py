"""
Módulo proxy para core.consciousness

Este módulo redireciona importações para C:\\Users\\Natalia\\Documents\\Miner\\core\\consciousness
"""

import sys
import os
from pathlib import Path

# Adicionar o caminho do módulo original ao sys.path
miner_core_path = r"C:\Users\Natalia\Documents\Miner\core"
sys.path.append(miner_core_path)

# Tentar importar do módulo original
try:
    # Verificar se o diretório consciousness existe
    if os.path.exists(os.path.join(miner_core_path, 'consciousness')):
        # Ajustar sys.path temporariamente
        original_path = sys.path.copy()
        sys.path.insert(0, miner_core_path)
        
        # Tentar importar
        try:
            from consciousness import *
            print("Importação de consciousness bem-sucedida")
        except ImportError as e:
            print(f"Erro ao importar consciousness: {e}")
        
        # Restaurar sys.path
        sys.path = original_path
    else:
        print(f"Diretório consciousness não encontrado em {miner_core_path}")
    
    # Tentar importar o arquivo do operador de consciência diretamente
    consciousness_operator_path = os.path.join(miner_core_path, 'consciousness', 'consciousness_operator.py')
    if os.path.exists(consciousness_operator_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("consciousness_operator", consciousness_operator_path)
        consciousness_operator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(consciousness_operator)
        
        # Exportar as classes e funções
        for attr in dir(consciousness_operator):
            if not attr.startswith('_'):
                globals()[attr] = getattr(consciousness_operator, attr)
                
        # Verificar se a classe principal existe e exportá-la
        if hasattr(consciousness_operator, 'QuantumConsciousnessOperator'):
            globals()['QuantumConsciousnessOperator'] = consciousness_operator.QuantumConsciousnessOperator
            
        print(f"Módulo consciousness_operator.py carregado diretamente de {consciousness_operator_path}")
    else:
        print(f"Arquivo consciousness_operator.py não encontrado em {consciousness_operator_path}")
        
except Exception as e:
    print(f"Erro ao configurar importação de consciousness: {e}")
    
    # Criar stubs para classes importantes
    class QuantumConsciousnessOperator:
        """Stub para QuantumConsciousnessOperator"""
        def __init__(self, *args, **kwargs):
            print("AVISO: Esta é uma implementação stub de QuantumConsciousnessOperator")
            
        def process(self, *args, **kwargs):
            """Função stub"""
            return args[0] if args else None
            
        def analyze(self, *args, **kwargs):
            """Função stub"""
            return {"estado": "simulado", "valor": 0.5} 