#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exemplo de uso da simulação de mineração QUALIA vs ASIC
Este script demonstra como utilizar os diferentes métodos de análise implementados
na classe MiningSimulation para avaliar o desempenho do paradigma QUALIA.
"""

import os
import numpy as np
from mining_simulation import MiningSimulation

def main():
    # Criar diretório para resultados
    output_dir = "resultados_simulacao"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Iniciando Simulação de Mineração QUALIA vs ASIC ===")
    
    # Criar instância da simulação com cenário básico
    sim = MiningSimulation(simulation_days=365)
    
    # Definir parâmetros específicos para um cenário realista
    sim.monero_price = 160  # Preço atual do Monero em USD
    sim.qualia_resonance_factor = 1.15  # Ajustar ressonância quântica
    sim.qualia_learning_rate = 0.02  # Ajustar taxa de aprendizado
    
    # Imprimir configuração inicial
    print("\nConfiguração Inicial:")
    print(f"Preço do Monero: ${sim.monero_price}")
    print(f"Hashrate ASIC: {sim.asic_hashrate/1000:.2f} kH/s")
    print(f"Hashrate QUALIA: {sim.qualia_hashrate/1000:.2f} kH/s")
    print(f"Dias da simulação: {sim.simulation_days}")
    
    # 1. Executar simulação básica
    print("\n[1/4] Executando simulação básica...")
    sim.run_simulation()
    
    # Gerar visualizações padrão
    print("\nGerando visualizações...")
    sim.generate_all_charts(os.path.join(output_dir, "simulacao_basica"))
    
    # Imprimir resumo dos resultados
    print("\nResumo dos Resultados:")
    print(f"Lucro Final ASIC: ${sim.summary['asic_final_profit']:.2f}")
    print(f"Lucro Final QUALIA: ${sim.summary['qualia_final_profit']:.2f}")
    print(f"Diferença de Lucro: ${sim.summary['profit_difference']:.2f}")
    if 'asic_roi_days' in sim.summary and sim.summary['asic_roi_days'] is not None:
        print(f"ROI ASIC: {sim.summary['asic_roi_days']:.1f} dias")
    if 'qualia_roi_days' in sim.summary and sim.summary['qualia_roi_days'] is not None:
        print(f"ROI QUALIA: {sim.summary['qualia_roi_days']:.1f} dias")
    
    # 2. Executar simulação de Monte Carlo
    print("\n[2/4] Executando simulação de Monte Carlo (20 iterações)...")
    mc_df = sim.run_monte_carlo_simulation(
        num_simulations=10,  # Reduzindo para 10 para teste mais rápido
        output_dir=os.path.join(output_dir, "monte_carlo")
    )
    
    # Imprimir estatísticas da simulação Monte Carlo
    print("\nEstatísticas de Monte Carlo:")
    print(f"Lucro médio ASIC: ${mc_df['asic_final_profit'].mean():.2f}")
    print(f"Lucro médio QUALIA: ${mc_df['qualia_final_profit'].mean():.2f}")
    print(f"Probabilidade QUALIA ser mais lucrativo: {(mc_df['profit_difference'] > 0).mean() * 100:.1f}%")
    
    # 3. Análise de sensibilidade para ressonância quântica
    print("\n[3/4] Análise de sensibilidade: Fator de Ressonância Quântica...")
    resonance_range = np.linspace(1.0, 1.3, 5)  # Testar 5 valores de 1.0 a 1.3
    
    sensitivity_results = sim.run_sensitivity_analysis(
        parameter_ranges={"qualia_resonance_factor": resonance_range},
        output_dir=os.path.join(output_dir, "sensibilidade_ressonancia")
    )
    
    # 4. Análise de sensibilidade para preço do Monero
    print("\n[4/4] Análise de sensibilidade: Preço do Monero...")
    price_range = np.linspace(100, 300, 5)  # Testar 5 valores de $100 a $300
    
    sensitivity_results.update(
        sim.run_sensitivity_analysis(
            parameter_ranges={"monero_price": price_range},
            output_dir=os.path.join(output_dir, "sensibilidade_preco")
        )
    )
    
    print("\n=== Simulação Concluída ===")
    print(f"Resultados e gráficos salvos em: {os.path.abspath(output_dir)}")
    
    # Sugestões de análise
    print("\nSugestões para análise dos resultados:")
    print("1. Verifique a probabilidade do QUALIA superar os ASICs em diferentes cenários")
    print("2. Analise como o fator de ressonância quântica impacta o lucro final")
    print("3. Observe o impacto das flutuações de preço do Monero na viabilidade dos sistemas")
    print("4. Compare os tempos de ROI entre QUALIA e ASICs em diferentes condições")


if __name__ == "__main__":
    main()
