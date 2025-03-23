import os
import subprocess
import sys

def run_tests():
    """
    Script para executar testes de forma padronizada
    """
    # Definir diretórios de testes
    test_dirs = [
        'quantum_trading/tests',
        'qualia/tests'
    ]
    
    # Configurações de execução
    pytest_args = [
        '-v',  # Verbose
        '--disable-warnings',  # Desabilitar warnings
        '--no-header',  # Remover cabeçalho
        '--tb=short',  # Traceback curto
        '--maxfail=10'  # Parar após 10 falhas
    ]
    
    # Adicionar diretórios de testes
    pytest_args.extend(test_dirs)
    
    try:
        # Executar pytest
        result = subprocess.run([sys.executable, '-m', 'pytest'] + pytest_args, 
                                capture_output=True, 
                                text=True)
        
        # Imprimir resultados
        print("Saída dos testes:")
        print(result.stdout)
        
        # Imprimir erros, se houver
        if result.stderr:
            print("\nErros encontrados:")
            print(result.stderr)
        
        # Retornar código de saída
        return result.returncode
    
    except Exception as e:
        print(f"Erro ao executar testes: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(run_tests())
