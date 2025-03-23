"""
Script para executar os testes e gerar um relatório.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from test_hawking_simple import run_tests, TestConfig

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_report(results_dir: Path, config: TestConfig):
    """Gera um relatório visual dos resultados."""
    # Encontra os arquivos de resultados mais recentes
    result_files = list(results_dir.glob("results_*.json"))
    if not result_files:
        logger.error("Nenhum arquivo de resultados encontrado")
        return
    
    latest_result = max(result_files, key=lambda x: x.stat().st_mtime)
    
    # Carrega os resultados
    with open(latest_result, "r") as f:
        results = json.load(f)
    
    # Cria o gráfico
    plt.figure(figsize=(10, 6))
    
    # Plot de energia vs. entropia
    plt.subplot(1, 2, 1)
    plt.scatter(results["mean_energy"], results["mean_entropy"])
    plt.xlabel("Energia Média")
    plt.ylabel("Entropia Média")
    plt.title("Energia vs. Entropia")
    
    # Plot de distribuição de energia
    plt.subplot(1, 2, 2)
    plt.hist([results["mean_energy"]], bins=20)
    plt.xlabel("Energia")
    plt.ylabel("Frequência")
    plt.title("Distribuição de Energia")
    
    # Salva o gráfico
    plt.tight_layout()
    plt.savefig(results_dir / "report.png")
    plt.close()
    
    # Gera relatório em texto
    report_file = results_dir / "report.txt"
    with open(report_file, "w") as f:
        f.write("Relatório de Testes de Partículas de Hawking\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Configuração:\n")
        f.write("-" * 20 + "\n")
        for key, value in config.__dict__.items():
            f.write(f"{key}: {value}\n")
        f.write("\nResultados:\n")
        f.write("-" * 20 + "\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    logger.info(f"Relatório gerado em {results_dir}")

def main():
    """Função principal."""
    logger.info("Iniciando execução dos testes")
    
    # Carrega configuração
    try:
        with open("test_config.json", "r") as f:
            config_data = json.load(f)
            config = TestConfig(**config_data)
    except FileNotFoundError:
        config = TestConfig()
    
    # Executa os testes
    results = run_tests(config)
    
    # Gera relatório
    results_dir = Path("test_results")
    generate_report(results_dir, config)
    
    logger.info("Execução concluída")

if __name__ == "__main__":
    main() 