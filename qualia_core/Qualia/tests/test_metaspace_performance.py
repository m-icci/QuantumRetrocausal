"""
Teste de Performance do Metaespaço
Valida otimizações NumPy e analisa métricas de desempenho
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
from typing import Dict, List, Any
from dataclasses import dataclass

from core.fields.quantum_void import QuantumVoid
from core.fields.quantum_dance import QuantumDance
from core.consciousness.consciousness_operator import QuantumConsciousnessOperator
from core.constants import QualiaConstants

@dataclass
class PerformanceMetrics:
    """Métricas de performance"""
    execution_time: float
    memory_usage: float
    operation_count: int
    coherence_stability: float
    resonance_quality: float
    pattern_strength: float

class MetaspacePerformanceTest:
    """Testa performance do Metaespaço"""
    
    def __init__(self, size: int = 64, granularity: int = 21):
        """
        Inicializa teste de performance
        
        Args:
            size: Tamanho do campo
            granularity: Granularidade (3, 21 ou 42 bits)
        """
        self.size = size
        self.granularity = granularity
        
        # Componentes do Metaespaço
        self.void = QuantumVoid(size, granularity)
        self.dance = QuantumDance(size, granularity)
        self.consciousness = QuantumConsciousnessOperator()
        
        # Métricas
        self.metrics_history: List[Dict[str, float]] = []
        self.performance_history: List[PerformanceMetrics] = []
        
    def run_benchmark(self, steps: int = 100) -> None:
        """
        Executa benchmark de performance
        
        Args:
            steps: Número de iterações
        """
        print(f"\nExecutando benchmark ({steps} passos)...")
        start_time = time.time()
        
        for step in range(steps):
            # Gera dados de mercado
            market_data = np.random.normal(0, 1, self.size)
            
            # Mede tempo de execução de cada componente
            t0 = time.time()
            void_state = self.void.evolve(market_data)
            t1 = time.time()
            dance_state = self.dance.evolve(market_data)
            t2 = time.time()
            consciousness_state = self.consciousness.apply_consciousness(
                market_data, 
                morphic_field=dance_state['field']
            )
            t3 = time.time()
            
            # Calcula métricas de performance
            void_time = t1 - t0
            dance_time = t2 - t1
            consciousness_time = t3 - t2
            total_time = t3 - t0
            
            # Analisa estabilidade e padrões
            coherence_stability = dance_state['coherence']
            resonance_quality = dance_state.get('resonance', 0.0)
            pattern_strength = dance_state.get('pattern_similarity', 0.0)
            
            # Registra métricas
            metrics = PerformanceMetrics(
                execution_time=total_time,
                memory_usage=0.0,  # TODO: Implementar medição de memória
                operation_count=self.size * 3,  # Operações básicas por componente
                coherence_stability=coherence_stability,
                resonance_quality=resonance_quality,
                pattern_strength=pattern_strength
            )
            
            self.performance_history.append(metrics)
            
            if step % 10 == 0:
                print(f"Passo {step}: Tempo={total_time:.3f}s, "
                      f"Coerência={coherence_stability:.2f}, "
                      f"Ressonância={resonance_quality:.2f}")
                
    def plot_performance(self) -> None:
        """Plota métricas de performance"""
        times = [m.execution_time for m in self.performance_history]
        coherence = [m.coherence_stability for m in self.performance_history]
        resonance = [m.resonance_quality for m in self.performance_history]
        patterns = [m.pattern_strength for m in self.performance_history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot de performance
        ax1.plot(times, label='Tempo de Execução')
        ax1.set_title('Performance Computacional')
        ax1.set_xlabel('Iteração')
        ax1.set_ylabel('Tempo (s)')
        ax1.grid(True)
        ax1.legend()
        
        # Plot de qualidade
        ax2.plot(coherence, label='Estabilidade de Coerência')
        ax2.plot(resonance, label='Qualidade de Ressonância')
        ax2.plot(patterns, label='Força dos Padrões')
        ax2.set_title('Métricas de Qualidade')
        ax2.set_xlabel('Iteração')
        ax2.set_ylabel('Valor')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
    def analyze_results(self) -> Dict[str, float]:
        """Analisa resultados do benchmark"""
        metrics = self.performance_history
        
        return {
            'mean_execution_time': np.mean([m.execution_time for m in metrics]),
            'std_execution_time': np.std([m.execution_time for m in metrics]),
            'mean_coherence': np.mean([m.coherence_stability for m in metrics]),
            'mean_resonance': np.mean([m.resonance_quality for m in metrics]),
            'mean_pattern_strength': np.mean([m.pattern_strength for m in metrics]),
            'operations_per_second': self.size * 3 / np.mean([m.execution_time for m in metrics])
        }

def test_granularity(granularity: int, steps: int = 100) -> None:
    """
    Testa performance com granularidade específica
    
    Args:
        granularity: Granularidade a testar
        steps: Número de passos
    """
    print(f"\nTestando performance - Granularidade {granularity} bits")
    
    # Executa benchmark
    test = MetaspacePerformanceTest(size=64, granularity=granularity)
    test.run_benchmark(steps)
    
    # Analisa resultados
    results = test.analyze_results()
    print("\nResultados:")
    for key, value in results.items():
        print(f"{key}: {value:.3f}")
        
    # Plota resultados
    test.plot_performance()

def main():
    """Executa teste de performance do Metaespaço"""
    
    print("Iniciando teste de performance do Metaespaço...")
    
    # Testa todas as granularidades
    for granularity in QualiaConstants.ACTIVE_GRANULARITIES:
        test_granularity(granularity)
    
if __name__ == "__main__":
    main()
