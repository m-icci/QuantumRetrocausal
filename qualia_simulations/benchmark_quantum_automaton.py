import time
import numpy as np
from qualia_simulations.quantum_cellular_automaton import QuantumCellularAutomaton

def benchmark_automaton_initialization():
    """Benchmark da inicialização do autômato"""
    print("\n=== Benchmark de Inicialização do Autômato ===")
    
    sizes = [10, 20, 50, 100]
    results = []
    
    for size in sizes:
        start_time = time.time()
        automaton = QuantumCellularAutomaton(size, size)
        end_time = time.time()
        
        results.append({
            'size': size,
            'time': end_time - start_time,
            'memory': automaton.grid.nbytes / (1024 * 1024)  # MB
        })
        
        print(f"Grade {size}x{size}:")
        print(f"  Tempo: {results[-1]['time']:.4f}s")
        print(f"  Memória: {results[-1]['memory']:.2f}MB")
    
    return results

def benchmark_automaton_evolution():
    """Benchmark da evolução do autômato"""
    print("\n=== Benchmark de Evolução do Autômato ===")
    
    size = 50
    steps = [10, 50, 100, 200]
    results = []
    
    automaton = QuantumCellularAutomaton(size, size)
    
    for n_steps in steps:
        start_time = time.time()
        for _ in range(n_steps):
            automaton.step()
        end_time = time.time()
        
        results.append({
            'steps': n_steps,
            'time': end_time - start_time,
            'time_per_step': (end_time - start_time) / n_steps
        })
        
        print(f"{n_steps} passos:")
        print(f"  Tempo total: {results[-1]['time']:.4f}s")
        print(f"  Tempo por passo: {results[-1]['time_per_step']:.6f}s")
    
    return results

def run_all_benchmarks():
    """Executa todos os benchmarks"""
    print("Iniciando benchmarks do QuantumCellularAutomaton...")
    
    results = {
        'initialization': benchmark_automaton_initialization(),
        'evolution': benchmark_automaton_evolution()
    }
    
    print("\n=== Resumo dos Benchmarks ===")
    for benchmark, data in results.items():
        print(f"\n{benchmark}:")
        for result in data:
            print(f"  {result}")
    
    return results

if __name__ == "__main__":
    run_all_benchmarks() 