"""
Script para executar testes de comunicação via partículas de Hawking e gerar insights.
"""

import numpy as np
import logging
from datetime import datetime
from pathlib import Path
import pytest
from ..quantum.insights_analyzer import InsightsAnalyzer
from .test_hawking_communication import (
    HawkingParticleSimulator,
    test_particle_emission,
    test_particle_propagation,
    test_morphic_interaction
)

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_tests_and_analyze():
    """Executa os testes e gera insights."""
    logger.info("Iniciando testes de comunicação via partículas de Hawking")
    
    # Inicializa o simulador
    simulator = HawkingParticleSimulator()
    analyzer = InsightsAnalyzer()
    
    # Executa os testes
    logger.info("Executando teste de emissão de partículas")
    test_particle_emission(simulator)
    
    logger.info("Executando teste de propagação de partículas")
    test_particle_propagation(simulator)
    
    logger.info("Executando teste de interação com memória mórfica")
    test_morphic_interaction(simulator)
    
    # Analisa os resultados
    logger.info("Analisando resultados")
    for _ in range(100):  # Simula 100 iterações
        pattern = simulator.emit_particle()
        analyzer.analyze_pattern(pattern)
    
    # Gera insights
    logger.info("Gerando insights")
    insights = analyzer.generate_insights()
    
    # Salva os resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    
    # Salva insights
    insights_file = output_dir / f"hawking_insights_{timestamp}.txt"
    with open(insights_file, "w") as f:
        f.write("Insights da Comunicação via Partículas de Hawking\n")
        f.write("=" * 50 + "\n\n")
        for insight in insights:
            f.write(f"- {insight}\n")
    
    # Gera visualizações
    analyzer.visualize_insights(output_dir / f"hawking_visualization_{timestamp}.png")
    
    logger.info(f"Resultados salvos em {output_dir}")
    return insights

if __name__ == "__main__":
    insights = run_tests_and_analyze()
    print("\nInsights Gerados:")
    for insight in insights:
        print(f"- {insight}") 