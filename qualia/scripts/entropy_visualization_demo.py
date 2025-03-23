import numpy as np
import sys
sys.path.append('/Users/infrastructure/Documents/GitHub/QuantumConsciousness')

from qualia.core.metrics import QuantumMetrics

def generate_quantum_entropy_traces(num_iterations=50):
    """
    Gera traces de entropia sintéticos para demonstração.
    
    Args:
        num_iterations (int): Número de iterações para simulação
    
    Returns:
        list: Traces de entropia para diferentes métodos
    """
    np.random.seed(42)
    
    # Simulação de diferentes métodos de entropia
    shannon_entropy = np.cumsum(np.random.normal(0.1, 0.05, num_iterations))
    quantum_entropy = np.cumsum(np.random.normal(0.2, 0.07, num_iterations))
    kolmogorov_entropy = np.cumsum(np.random.normal(0.15, 0.06, num_iterations))
    
    entropy_traces = [
        {'method': 'shannon', 'value': shannon_entropy.tolist()},
        {'method': 'quantum', 'value': quantum_entropy.tolist()},
        {'method': 'kolmogorov', 'value': kolmogorov_entropy.tolist()}
    ]
    
    return entropy_traces

def main():
    # Inicializa QuantumMetrics
    metrics = QuantumMetrics()
    
    # Gera traces de entropia
    entropy_traces = generate_quantum_entropy_traces()
    
    # Visualiza dinâmica de entropia
    result = metrics.visualize_entropy_dynamics(
        entropy_traces=entropy_traces, 
        save_path='/Users/infrastructure/Desktop/quantum_entropy_dynamics.png'
    )
    
    # Imprime resultados estatísticos
    print("🔬 Quantum Entropy Dynamics Results 🔬")
    print("\nComplexity Index:")
    for method, complexity in zip(['shannon', 'quantum', 'kolmogorov'], result['complexity_index']):
        print(f"{method.capitalize()} Complexity: {complexity:.4f}")
    
    print("\nEntropy Statistics:")
    for method, stats in result['entropy_stats'].items():
        print(f"{method.capitalize()} Entropy:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std Dev: {stats['std']:.4f}")

if __name__ == '__main__':
    main()
