"""
Exemplo simplificado de uso do módulo de análise da hélice do QUALIA

Este exemplo demonstra como utilizar o módulo de análise da hélice para:
1. Realizar análise da hélice
2. Visualizar resultados e métricas
"""

import asyncio
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import logging
import sys
import os
from tqdm import tqdm
from datetime import datetime
import gc
import traceback

# Configuração de logging mais simples
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Adiciona o diretório raiz ao path do Python
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(root_dir)
print(f"Diretório raiz: {root_dir}")
print(f"Python path: {sys.path}")

try:
    print("Importando módulos QUALIA...")
    from qualia_core.helix_analysis import HelixAnalyzer, HelixConfig
    print("Módulos importados com sucesso")
except ImportError as e:
    print(f"Erro ao importar módulos: {e}")
    print(f"Traceback completo:\n{traceback.format_exc()}")
    sys.exit(1)

async def run_helix_analysis(max_time: int = 300):
    """Executa análise da hélice com feedback visual"""
    try:
        start_time = datetime.now()
        print("\nIniciando análise da hélice...")
        
        # Configuração do analisador da hélice
        print("Configurando analisador da hélice...")
        helix_config = HelixConfig(
            dimensions=256,
            num_qubits=8,
            phi=0.618,
            temperature=0.1,
            batch_size=1024,
            max_field_size=1024
        )
        helix_analyzer = HelixAnalyzer(helix_config)
        print("Analisador da hélice configurado")
        
        # Inicialização do campo da hélice
        print("Inicializando campo da hélice...")
        helix_analyzer.initialize_helix()
        print("Campo da hélice inicializado")
        
        # Evolução e análise com barra de progresso
        steps = 200
        print("\nEvoluindo hélice...")
        results = []
        
        for step in tqdm(range(steps), desc="Progresso da Hélice"):
            # Verifica tempo máximo
            if (datetime.now() - start_time).total_seconds() > max_time:
                print("\nTempo máximo de execução atingido!")
                break
                
            try:
                # Evolui hélice
                step_result = helix_analyzer.evolve_helix(steps=1)
                if step_result:
                    results.append(step_result)
                    print(f"Passo {step}: Resultado obtido")
                else:
                    print(f"Passo {step}: Sem resultado")
                
                # Limpeza de memória periódica
                if step % 10 == 0:
                    gc.collect()
                    
                # Pequena pausa para não sobrecarregar
                await asyncio.sleep(0.01)
                
            except Exception as step_error:
                print(f"Erro no passo {step}: {step_error}")
                print(f"Traceback:\n{traceback.format_exc()}")
                continue
        
        if not results:
            print("Nenhum resultado obtido da evolução da hélice")
            return
            
        # Análise de padrões quânticos
        print("\nAnalisando padrões quânticos...")
        try:
            quantum_patterns = helix_analyzer.get_quantum_patterns()
            print(f"Padrões quânticos obtidos: {quantum_patterns}")
        except Exception as e:
            print(f"Erro ao obter padrões quânticos: {e}")
            quantum_patterns = {}
        
        # Visualização dos resultados
        print("\nGerando visualizações...")
        try:
            visualize_results(results, quantum_patterns)
        except Exception as e:
            print(f"Erro ao visualizar resultados: {e}")
            print(f"Traceback:\n{traceback.format_exc()}")
            
    except Exception as e:
        print(f"Erro na análise da hélice: {e}")
        print(f"Traceback completo:\n{traceback.format_exc()}")
        raise

def visualize_results(results: List[Dict], quantum_patterns: Dict):
    """Visualiza resultados da análise"""
    try:
        print("Iniciando visualização dos resultados...")
        print(f"Dados recebidos - Results: {type(results)}, Quantum Patterns: {type(quantum_patterns)}")
        
        # Extração de dados
        fractal_analysis = []
        for result in results:
            if isinstance(result, dict) and 'fractal_analysis' in result:
                fractal_analysis.extend(result['fractal_analysis'])
            
        if not fractal_analysis:
            print("Dados de análise fractal não encontrados")
            return
            
        times = []
        fractal_factors = []
        lambda_couplings = []
        avg_intensities = []
        
        for data in fractal_analysis:
            if isinstance(data, dict):
                times.append(data.get('t', 0))
                fractal_factors.append(data.get('fractal_factor', 0))
                lambda_couplings.append(data.get('lambda_coupling', 0))
                avg_intensities.append(data.get('avg_intensity', 0))
        
        print(f"Dados extraídos - Times: {len(times)}, Factors: {len(fractal_factors)}")
        
        if not times:
            print("Nenhum dado temporal encontrado para visualização")
            return
        
        # Criação dos gráficos
        print("\nCriando gráficos...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Gráfico 1: Fator Fractal
        axes[0, 0].plot(times, fractal_factors, 'b-', label='Fator Fractal')
        axes[0, 0].axhline(y=1.8, color='r', linestyle='--', label='Limiar')
        axes[0, 0].set_title('Evolução do Fator Fractal')
        axes[0, 0].set_xlabel('Tempo')
        axes[0, 0].set_ylabel('Fator Fractal')
        axes[0, 0].legend()
        
        # Gráfico 2: Acoplamento λ
        axes[0, 1].plot(times, lambda_couplings, 'g-', label='λ Acoplamento')
        axes[0, 1].set_title('Evolução do Acoplamento λ')
        axes[0, 1].set_xlabel('Tempo')
        axes[0, 1].set_ylabel('λ')
        axes[0, 1].legend()
        
        # Gráfico 3: Intensidade Média
        axes[1, 0].plot(times, avg_intensities, 'c-', label='Intensidade Média')
        axes[1, 0].set_title('Evolução da Intensidade Média')
        axes[1, 0].set_xlabel('Tempo')
        axes[1, 0].set_ylabel('Intensidade')
        axes[1, 0].legend()
        
        # Gráfico 4: Padrões Quânticos
        quantum_metrics = ['entanglement', 'coherence', 'superposition', 'decoherence']
        for i, metric in enumerate(quantum_metrics):
            if metric in quantum_patterns:
                value = quantum_patterns[metric]
                if isinstance(value, (int, float)):
                    axes[1, 1].plot(times, [value] * len(times), 
                                  label=metric.capitalize())
                else:
                    print(f"Valor inválido para métrica {metric}: {value}")
            else:
                print(f"Métrica {metric} não encontrada nos padrões quânticos")
        
        axes[1, 1].set_title('Padrões Quânticos')
        axes[1, 1].set_xlabel('Tempo')
        axes[1, 1].set_ylabel('Valor')
        axes[1, 1].legend()
        
        # Ajuste do layout
        plt.tight_layout()
        
        # Salvamento dos gráficos
        print("\nSalvando gráficos...")
        plt.savefig('helix_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Gráficos salvos com sucesso em 'helix_analysis_results.png'")
        
        # Limpeza de memória
        del fig, axes
        gc.collect()
        
    except Exception as e:
        print(f"Erro na visualização dos resultados: {e}")
        print(f"Traceback completo:\n{traceback.format_exc()}")
        raise

if __name__ == "__main__":
    try:
        print("Iniciando exemplo de análise da hélice...")
        asyncio.run(run_helix_analysis())
        print("Exemplo concluído com sucesso!")
    except Exception as e:
        print(f"Erro na execução do exemplo: {e}")
        print(f"Traceback completo:\n{traceback.format_exc()}")
        sys.exit(1) 