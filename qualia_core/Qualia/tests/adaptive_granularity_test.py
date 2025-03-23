"""
Testes do sistema de granularidade adaptativa
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from core.adaptive_granularity import (
    AdaptiveGranularity,
    ProcessingMode,
    GranularityMetrics
)
from core.constants import QualiaConstants

class TestAdaptiveGranularity(unittest.TestCase):
    """Testes unitários para o sistema de granularidade adaptativa"""
    
    def setUp(self):
        self.adapter = AdaptiveGranularity()
        
    def test_atomic_mode_selection(self):
        """Testa seleção de granularidade para modo atômico"""
        # Simula alta carga de hardware
        granularity = self.adapter.select_granularity(
            ProcessingMode.ATOMIC,
            hardware_load=0.9
        )
        self.assertIn(granularity, QualiaConstants.ATOMIC_GRANULARITIES)
        
    def test_balanced_mode_selection(self):
        """Testa seleção de granularidade para modo balanceado"""
        # Simula carga moderada
        granularity = self.adapter.select_granularity(
            ProcessingMode.BALANCED,
            hardware_load=0.5
        )
        self.assertIn(granularity, QualiaConstants.BALANCED_GRANULARITIES)
        
    def test_emergent_mode_selection(self):
        """Testa seleção de granularidade para modo emergente"""
        # Simula baixa carga
        granularity = self.adapter.select_granularity(
            ProcessingMode.EMERGENT,
            hardware_load=0.2
        )
        self.assertIn(granularity, QualiaConstants.EMERGENT_GRANULARITIES)
        
    def test_metrics_update(self):
        """Testa atualização de métricas"""
        metrics = GranularityMetrics(
            coherence=0.95,
            resonance=0.92,
            entropy=0.85,
            stability=0.88,
            execution_time=0.05
        )
        self.adapter.update_metrics(21, metrics)
        self.assertIn(21, self.adapter.metrics_cache)
        self.assertEqual(len(self.adapter.metrics_cache[21]), 1)
        
    def test_performance_report(self):
        """Testa geração de relatório de desempenho"""
        # Adiciona algumas métricas
        for g in [1, 21, 144]:
            metrics = GranularityMetrics(
                coherence=0.9,
                resonance=0.85,
                entropy=0.8,
                stability=0.82,
                execution_time=0.1
            )
            self.adapter.update_metrics(g, metrics)
            
        report = self.adapter.get_performance_report()
        self.assertIn("current_mode", report)
        self.assertIn("metrics_by_granularity", report)
        
    def test_entropy_based_selection(self):
        """Verifica seleção de granularidade com alto valor de entropia"""
        ag = AdaptiveGranularity()
    
        # Simula métricas com alta entropia
        for g in ag.balanced_granularities:
            ag.metrics_cache[g] = [
                GranularityMetrics(
                    coherence=0.9,
                    resonance=0.7,
                    entropy=0.9,  # Entropia alta
                    stability=0.85,
                    execution_time=0.5
                )
            ]
    
        print(f"\nBalanced Granularities: {ag.balanced_granularities}")
        print(f"Hardware Load: 0.3")
        print(f"Average Entropy: 0.9")
        
        # Calcula scores para todas as granularidades
        for g in ag.balanced_granularities:
            ag.current_granularity = g
            metrics = ag.metrics_cache[g][-1]
            score = ag._calculate_efficiency_score(metrics, 0.3)
            weight = QualiaConstants.get_granularity_weight(g)
            print(f"Granularity {g}: score={score:.2f}, weight={weight:.2f}")
        
        selected_g = ag.select_granularity(ProcessingMode.BALANCED, 0.3)
        
        # Calcula score esperado para a granularidade selecionada
        ag.current_granularity = selected_g
        metrics = ag.metrics_cache[selected_g][-1]
        expected_score = ag._calculate_efficiency_score(metrics, 0.3)
        
        print(f"Selected Granularity: {selected_g}")
        print(f"Expected Score: {expected_score:.2f}")
        
        # Com alta entropia e baixa carga, deve selecionar uma das maiores granularidades
        assert selected_g in ag.balanced_granularities[-3:], \
            f"Com alta entropia (0.9) e baixa carga (0.3), deveria selecionar uma das maiores granularidades"
        
def run_benchmark_test(iterations: int = 1000):
    """
    Executa teste de benchmark comparando diferentes modos
    
    Args:
        iterations: Número de iterações por modo
    """
    adapter = AdaptiveGranularity()
    results = {mode: [] for mode in ProcessingMode}
    
    # Simula diferentes cargas de hardware
    hardware_loads = np.linspace(0.1, 0.9, iterations)
    
    for mode in ProcessingMode:
        print(f"\nTestando modo {mode.value}...")
        
        for i, load in enumerate(hardware_loads):
            # Simula métricas baseadas na carga
            metrics = GranularityMetrics(
                coherence=0.95 - (load * 0.1),
                resonance=0.92 - (load * 0.08),
                stability=0.88 - (load * 0.05),
                execution_time=0.05 + (load * 0.1)
            )
            
            # Seleciona granularidade
            g = adapter.select_granularity(mode, load)
            adapter.update_metrics(g, metrics)
            
            results[mode].append({
                "granularity": g,
                "hardware_load": load,
                "metrics": metrics
            })
            
            if (i + 1) % 100 == 0:
                print(f"Progresso: {(i + 1) / iterations * 100:.1f}%")
                
    return results

def plot_benchmark_results(results):
    """Plota resultados do benchmark"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Análise de Granularidade Adaptativa", fontsize=14)
    
    for mode in ProcessingMode:
        data = results[mode]
        loads = [d["hardware_load"] for d in data]
        granularities = [d["granularity"] for d in data]
        coherence = [d["metrics"].coherence for d in data]
        execution_times = [d["metrics"].execution_time for d in data]
        
        # Plot granularidade vs carga
        axes[0, 0].scatter(loads, granularities, label=mode.value, alpha=0.5)
        axes[0, 0].set_xlabel("Carga do Hardware")
        axes[0, 0].set_ylabel("Granularidade")
        axes[0, 0].set_title("Adaptação de Granularidade")
        axes[0, 0].legend()
        
        # Plot coerência vs granularidade
        axes[0, 1].scatter(granularities, coherence, label=mode.value, alpha=0.5)
        axes[0, 1].set_xlabel("Granularidade")
        axes[0, 1].set_ylabel("Coerência")
        axes[0, 1].set_title("Coerência por Granularidade")
        axes[0, 1].legend()
        
        # Plot tempo de execução vs granularidade
        axes[1, 0].scatter(granularities, execution_times, label=mode.value, alpha=0.5)
        axes[1, 0].set_xlabel("Granularidade")
        axes[1, 0].set_ylabel("Tempo de Execução")
        axes[1, 0].set_title("Performance por Granularidade")
        axes[1, 0].legend()
        
        # Plot coerência vs tempo de execução
        axes[1, 1].scatter(execution_times, coherence, label=mode.value, alpha=0.5)
        axes[1, 1].set_xlabel("Tempo de Execução")
        axes[1, 1].set_ylabel("Coerência")
        axes[1, 1].set_title("Trade-off Coerência vs Performance")
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig("adaptive_granularity_analysis.png")
    plt.close()

if __name__ == "__main__":
    # Executa testes unitários
    unittest.main(argv=[''], exit=False)
    
    print("\nExecutando benchmark de granularidade adaptativa...")
    results = run_benchmark_test()
    
    print("\nGerando gráficos de análise...")
    plot_benchmark_results(results)
    
    print("\nAnálise completa! Resultados salvos em 'adaptive_granularity_analysis.png'")
