"""
Benchmark para operadores quânticos otimizados
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from core.fields.quantum_void import QuantumVoid
from core.fields.quantum_dance import QuantumDance

def run_benchmark(size: int = 64, steps: int = 1000):
    """Executa benchmark dos operadores"""
    print(f"\nExecutando benchmark com tamanho {size} e {steps} iterações...")
    
    # Inicializa operadores
    void = QuantumVoid(size=size)
    dance = QuantumDance(size=size)
    
    # Benchmark QuantumVoid
    start = time.time()
    for _ in range(steps):
        market_data = np.random.normal(0, 0.1, size)
        void.evolve(market_data=market_data)
    void_time = time.time() - start
    print(f"QuantumVoid: {void_time:.3f}s ({steps/void_time:.1f} it/s)")
    
    # Benchmark QuantumDance
    start = time.time()
    for _ in range(steps):
        void_state = void.state
        dance.evolve(void_state=void_state)
    dance_time = time.time() - start
    print(f"QuantumDance: {dance_time:.3f}s ({steps/dance_time:.1f} it/s)")
    
    return void_time, dance_time

if __name__ == "__main__":
    # Testa diferentes tamanhos
    sizes = [32, 64, 128, 256]
    steps = 1000
    
    results = []
    for size in sizes:
        void_time, dance_time = run_benchmark(size, steps)
        results.append((size, void_time, dance_time))
    
    print("\nResultados finais:")
    print("Size | QuantumVoid (it/s) | QuantumDance (it/s)")
    print("-" * 45)
    for size, vt, dt in results:
        print(f"{size:4d} | {steps/vt:16.1f} | {steps/dt:16.1f}")
