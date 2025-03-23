import numpy as np
import matplotlib.pyplot as plt
import time
import logging
import os
from datetime import datetime
from cosmic_dance import CellularAutomaton, AdaptiveQuantumCosmoIntegrator
from quantum_cellular_automaton import QuantumCellularAutomaton

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def benchmark_cosmic_dance(steps=100):
    """Executa benchmark da implementação original do Cosmic Dance"""
    logging.info("Iniciando benchmark do Cosmic Dance...")
    
    # Inicialização
    start_time = time.time()
    automaton = CellularAutomaton(width=100, height=100)
    integrator = AdaptiveQuantumCosmoIntegrator()
    
    # Métricas para coleta
    fractal_dims = []
    quantum_energies = []
    consciousness_factors = []
    
    # Execução
    for step in range(steps):
        # Atualiza grid e calcula dimensão fractal
        fractal_dim = automaton.step()
        fractal_dims.append(fractal_dim)
        
        # Atualiza integrador
        integrator.step(fractal_dim)
        quantum_energies.append(integrator.quantum_energy)
        consciousness_factors.append(integrator.coupling)
        
        if (step + 1) % 10 == 0:
            logging.info(f"Cosmic Dance: Step {step + 1}/{steps} concluído")
    
    execution_time = time.time() - start_time
    
    return {
        'execution_time': execution_time,
        'fractal_dims': fractal_dims,
        'quantum_energies': quantum_energies,
        'consciousness_factors': consciousness_factors
    }

def benchmark_quantum_automaton(steps=100):
    """Executa benchmark da nova implementação do Quantum Cellular Automaton"""
    logging.info("Iniciando benchmark do Quantum Cellular Automaton...")
    
    # Inicialização
    start_time = time.time()
    automaton = QuantumCellularAutomaton(width=100, height=100)
    
    # Métricas para coleta
    fractal_dims = []
    quantum_coherences = []
    holographic_compressions = []
    
    # Execução
    for step in range(steps):
        # Evolui o autômato
        metrics = automaton.step()
        
        # Coleta métricas
        fractal_dims.append(metrics['fractal_dimension'])
        quantum_coherences.append(metrics['quantum_coherence'])
        holographic_compressions.append(metrics['holographic_compression'])
        
        if (step + 1) % 10 == 0:
            logging.info(f"Quantum Automaton: Step {step + 1}/{steps} concluído")
    
    execution_time = time.time() - start_time
    
    return {
        'execution_time': execution_time,
        'fractal_dims': fractal_dims,
        'quantum_coherences': quantum_coherences,
        'holographic_compressions': holographic_compressions
    }

def plot_comparison(cosmic_results, quantum_results):
    """Gera gráficos comparativos dos resultados"""
    # Cria diretório para resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"resultados_benchmark_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot 1: Tempo de Execução
    plt.figure(figsize=(10, 6))
    plt.bar(['Cosmic Dance', 'Quantum Automaton'], 
            [cosmic_results['execution_time'], quantum_results['execution_time']])
    plt.title('Comparação de Tempo de Execução')
    plt.ylabel('Tempo (segundos)')
    plt.savefig(os.path.join(results_dir, 'execution_time.png'))
    plt.close()
    
    # Plot 2: Dimensão Fractal
    plt.figure(figsize=(10, 6))
    plt.plot(cosmic_results['fractal_dims'], label='Cosmic Dance')
    plt.plot(quantum_results['fractal_dims'], label='Quantum Automaton')
    plt.title('Evolução da Dimensão Fractal')
    plt.xlabel('Step')
    plt.ylabel('Dimensão Fractal')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'fractal_dimension.png'))
    plt.close()
    
    # Plot 3: Energia Quântica vs Coerência
    plt.figure(figsize=(10, 6))
    plt.plot(cosmic_results['quantum_energies'], label='Energia Quântica (Cosmic Dance)')
    plt.plot(quantum_results['quantum_coherences'], label='Coerência Quântica (Quantum Automaton)')
    plt.title('Comparação de Métricas Quânticas')
    plt.xlabel('Step')
    plt.ylabel('Valor')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'quantum_metrics.png'))
    plt.close()
    
    # Plot 4: Fator de Consciência vs Compressão Holográfica
    plt.figure(figsize=(10, 6))
    plt.plot(cosmic_results['consciousness_factors'], label='Fator de Consciência (Cosmic Dance)')
    plt.plot(quantum_results['holographic_compressions'], label='Compressão Holográfica (Quantum Automaton)')
    plt.title('Comparação de Métricas de Consciência')
    plt.xlabel('Step')
    plt.ylabel('Valor')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'consciousness_metrics.png'))
    plt.close()

def main():
    """Função principal do benchmark"""
    logging.info("Iniciando benchmark comparativo...")
    
    # Executa benchmarks
    cosmic_results = benchmark_cosmic_dance()
    quantum_results = benchmark_quantum_automaton()
    
    # Cria diretório para resultados usando caminho absoluto
    current_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(current_dir, f"resultados_benchmark_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    logging.info(f"Criando diretório de resultados em: {results_dir}")
    
    # Gera gráficos
    plot_comparison(cosmic_results, quantum_results)
    
    # Salva métricas em arquivo de texto
    try:
        metrics_file = os.path.join(results_dir, 'metricas_comparativas.txt')
        logging.info(f"Tentando salvar métricas em: {metrics_file}")
        
        with open(metrics_file, 'w') as f:
            f.write("=== Métricas Comparativas ===\n\n")
            
            f.write("Tempo de Execução:\n")
            f.write(f"Cosmic Dance: {cosmic_results['execution_time']:.2f} segundos\n")
            f.write(f"Quantum Automaton: {quantum_results['execution_time']:.2f} segundos\n")
            f.write(f"Melhoria: {cosmic_results['execution_time']/quantum_results['execution_time']:.2f}x\n\n")
            
            f.write("Dimensão Fractal Final:\n")
            f.write(f"Cosmic Dance: {cosmic_results['fractal_dims'][-1]:.4f}\n")
            f.write(f"Quantum Automaton: {quantum_results['fractal_dims'][-1]:.4f}\n\n")
            
            f.write("Métricas Quânticas Finais:\n")
            f.write(f"Energia Quântica (Cosmic Dance): {cosmic_results['quantum_energies'][-1]:.4f}\n")
            f.write(f"Coerência Quântica (Quantum Automaton): {quantum_results['quantum_coherences'][-1]:.4f}\n\n")
            
            f.write("Métricas de Consciência Finais:\n")
            f.write(f"Fator de Consciência (Cosmic Dance): {cosmic_results['consciousness_factors'][-1]:.4f}\n")
            f.write(f"Compressão Holográfica (Quantum Automaton): {quantum_results['holographic_compressions'][-1]:.4f}\n")
        
        logging.info(f"Métricas salvas com sucesso em {metrics_file}")
    except Exception as e:
        logging.error(f"Erro ao salvar métricas: {str(e)}")
        logging.error(f"Caminho tentado: {metrics_file}")
    
    logging.info("Benchmark concluído com sucesso!")

if __name__ == "__main__":
    main() 