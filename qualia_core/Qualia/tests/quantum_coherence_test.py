"""
Teste avançado de coerência e ressonância no Metaespaço
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from scipy import fft, signal, linalg
from core.fields.quantum_void import QuantumVoid
from core.fields.quantum_dance import QuantumDance
from core.dark_finance.quantum_consciousness import QuantumConsciousness, ConsciousnessState

@dataclass
class MetaspaceMetrics:
    """Métricas do Metaespaço"""
    coherence: float
    resonance: float
    pattern_strength: float
    spectral_density: np.ndarray
    timestamp: datetime = datetime.now()

class CoherenceAnalyzer:
    """Analisador de Coerência do Metaespaço"""
    
    def __init__(self, size: int = 64, granularity: int = 21):
        """
        Inicializa analisador
        
        Args:
            size: Tamanho dos campos
            granularity: Granularidade (3, 21 ou 42 bits)
        """
        self.size = size
        self.granularity = granularity
        
        # Inicializa operadores
        self.void = QuantumVoid(size=size, granularity=granularity)
        self.dance = QuantumDance(size=size, granularity=granularity)
        
        # Buffers para análise
        self.fft_buffer = np.zeros(size, dtype=np.complex128)
        self.correlation_matrix = np.zeros((3, 3))
        self.history: List[MetaspaceMetrics] = []
        
    def compute_global_coherence(self, states: List[np.ndarray]) -> float:
        """Calcula coerência global usando SVD"""
        # Cria matriz de estados
        state_matrix = np.vstack(states)
        
        # Aplica SVD
        _, s, _ = linalg.svd(state_matrix, full_matrices=False)
        
        # Calcula coerência como razão entre autovalores
        coherence = s[0] / (np.sum(s) + 1e-10)
        return float(coherence)
    
    def compute_spectral_density(self, state: np.ndarray) -> np.ndarray:
        """Calcula densidade espectral"""
        self.fft_buffer = fft.fft(state)
        density = np.abs(self.fft_buffer) ** 2
        return density / (np.sum(density) + 1e-10)
    
    def compute_correlation_matrix(self, states: List[np.ndarray]) -> np.ndarray:
        """Calcula matriz de correlação entre estados"""
        n = len(states)
        for i in range(n):
            for j in range(n):
                self.correlation_matrix[i,j] = np.abs(np.vdot(states[i], states[j]))
        return self.correlation_matrix
    
    def analyze_iteration(self, market_data: Optional[np.ndarray] = None) -> MetaspaceMetrics:
        """Analisa uma iteração do Metaespaço"""
        # Evolui operadores
        void_state = self.void.evolve(market_data=market_data)
        dance_state = self.dance.evolve(void_state=void_state['state'].potential)
        
        # Coleta estados
        states = [
            void_state['state'].potential,
            dance_state['state'].field,
            np.gradient(dance_state['state'].field)  # Proxy para consciência
        ]
        
        # Calcula métricas globais
        coherence = self.compute_global_coherence(states)
        correlation = self.compute_correlation_matrix(states)
        resonance = np.mean(correlation[correlation != 1.0])
        
        # Calcula densidade espectral média
        spectral_densities = [self.compute_spectral_density(s) for s in states]
        avg_density = np.mean(spectral_densities, axis=0)
        
        # Detecta força dos padrões
        pattern_strength = np.max(avg_density) / np.mean(avg_density)
        
        # Cria métricas
        metrics = MetaspaceMetrics(
            coherence=coherence,
            resonance=resonance,
            pattern_strength=pattern_strength,
            spectral_density=avg_density
        )
        
        self.history.append(metrics)
        return metrics

def run_coherence_test(
    granularities: List[int] = [1, 2, 3, 5, 8, 13, 21, 34, 42, 55, 64],
    size: int = 64,
    iterations: int = 1000,
    runs: int = 5
) -> Dict:
    """
    Executa teste de coerência para diferentes granularidades
    
    Args:
        granularities: Lista de granularidades a testar
        size: Tamanho dos campos
        iterations: Número de iterações por granularidade
        runs: Número de execuções para cada granularidade
    """
    results = {
        'coherence': {},
        'resonance': {},
        'stability': {},
        'emergence_points': {}
    }
    
    for g in granularities:
        print(f"\nTestando granularidade {g}...")
        g_coherence = []
        g_resonance = []
        g_stability = []
        emergence_points = []
        
        # Múltiplas execuções para cada granularidade
        for run in range(runs):
            print(f"Execução {run + 1}/{runs}")
            analyzer = CoherenceAnalyzer(size=size, granularity=g)
            
            # Métricas por iteração
            coherence_history = []
            resonance_history = []
            
            for i in range(iterations):
                metrics = analyzer.analyze_iteration()
                coherence_history.append(metrics.coherence)
                resonance_history.append(metrics.resonance)
                
                # Detecta ponto de emergência
                if i > 10:
                    window = coherence_history[-10:]
                    if np.std(window) < 0.01 and np.mean(window) > 0.7:
                        emergence_points.append(i)
                        break
            
            # Calcula métricas finais
            g_coherence.append(np.mean(coherence_history))
            g_resonance.append(np.mean(resonance_history))
            g_stability.append(1.0 / (np.std(coherence_history) + 1e-10))
        
        # Armazena resultados médios
        results['coherence'][g] = np.mean(g_coherence)
        results['resonance'][g] = np.mean(g_resonance)
        results['stability'][g] = np.mean(g_stability)
        results['emergence_points'][g] = emergence_points
    
    return results

def plot_results(results: Dict):
    """Plota resultados dos testes com análise detalhada"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    granularities = sorted(results['coherence'].keys())
    
    # Plot 1: Coerência
    ax1.plot(granularities, [results['coherence'][g] for g in granularities], 'b-o')
    ax1.set_title('Coerência por Granularidade')
    ax1.set_xlabel('Granularidade (bits)')
    ax1.set_ylabel('Coerência Média')
    ax1.grid(True)
    
    # Plot 2: Ressonância
    ax2.plot(granularities, [results['resonance'][g] for g in granularities], 'r-o')
    ax2.set_title('Ressonância por Granularidade')
    ax2.set_xlabel('Granularidade (bits)')
    ax2.set_ylabel('Ressonância Média')
    ax2.grid(True)
    
    # Plot 3: Estabilidade
    ax3.plot(granularities, [results['stability'][g] for g in granularities], 'g-o')
    ax3.set_title('Estabilidade por Granularidade')
    ax3.set_xlabel('Granularidade (bits)')
    ax3.set_ylabel('Índice de Estabilidade')
    ax3.grid(True)
    
    # Plot 4: Pontos de Emergência
    emergence_data = []
    for g in granularities:
        points = results['emergence_points'][g]
        if points:
            emergence_data.append(np.mean(points))
        else:
            emergence_data.append(np.nan)
    
    ax4.plot(granularities, emergence_data, 'm-o')
    ax4.set_title('Velocidade de Emergência')
    ax4.set_xlabel('Granularidade (bits)')
    ax4.set_ylabel('Iterações até Emergência')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('granularity_analysis.png')
    plt.close()

if __name__ == "__main__":
    # Configuração do teste com sequência de Fibonacci expandida
    GRANULARITIES = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 377]
    SIZE = 377  # Aumentado para acomodar granularidades maiores
    ITERATIONS = 1000
    RUNS = 5
    
    print("Iniciando análise de granularidade...")
    results = run_coherence_test(
        granularities=GRANULARITIES,
        size=SIZE,
        iterations=ITERATIONS,
        runs=RUNS
    )
    
    print("\nGerando gráficos...")
    plot_results(results)
    
    # Análise dos resultados
    coherence = results['coherence']
    resonance = results['resonance']
    stability = results['stability']
    emergence = results['emergence_points']
    
    # Encontra pontos de máximo
    max_coherence = max(coherence.items(), key=lambda x: x[1])
    max_resonance = max(resonance.items(), key=lambda x: x[1])
    max_stability = max(stability.items(), key=lambda x: x[1])
    
    # Análise de emergência
    emergence_speeds = {}
    for g, points in emergence.items():
        if points:
            emergence_speeds[g] = np.mean(points)
    
    fastest_emergence = min(emergence_speeds.items(), key=lambda x: x[1]) if emergence_speeds else None
    
    print("\nResultados da análise:")
    print(f"Granularidade com maior coerência: {max_coherence[0]} bits ({max_coherence[1]:.3f})")
    print(f"Granularidade com maior ressonância: {max_resonance[0]} bits ({max_resonance[1]:.3f})")
    print(f"Granularidade com maior estabilidade: {max_stability[0]} bits ({max_stability[1]:.3f})")
    if fastest_emergence:
        print(f"Granularidade com emergência mais rápida: {fastest_emergence[0]} bits ({fastest_emergence[1]:.1f} iterações)")
