#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Demonstração do Simulador Cosmológico-Quântico Retrocausal

Este script demonstra a utilização do framework quântico-cosmológico retrocausal,
executando uma simulação completa e gerando visualizações dos resultados.

O sistema implementa um modelo onde processos quânticos locais (não-lineares e sujeitos 
a decoerência) influenciam grandezas cosmológicas (H(t) e Λ(t)), ao mesmo tempo em que 
há um "banho" informacional latente retroalimentando a evolução.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import pandas as pd
from typing import Dict, List, Tuple
from quantum_cosmological_simulator import QuantumFieldSimulator
from cosmological_evolution import CosmologicalEvolution

# Configuração de logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuantumCosmologicalSimulation:
    def __init__(self,
                 num_steps: int = 1000,
                 dt: float = 0.01,
                 quantum_gravity_coupling: float = 0.1,
                 latent_dimensions: int = 3):
        """
        Inicializa a simulação cosmológica quântica expandida.
    
    Args:
            num_steps: Número de passos de tempo
            dt: Tamanho do passo de tempo
            quantum_gravity_coupling: Força do acoplamento com gravitação quântica
            latent_dimensions: Número de dimensões da informação latente
        """
        self.num_steps = num_steps
        self.dt = dt
        
        # Inicializa simuladores com parâmetros expandidos
        self.quantum_sim = QuantumFieldSimulator(
            dt=dt,
            quantum_gravity_coupling=quantum_gravity_coupling,
            latent_dimensions=latent_dimensions
        )
        
        self.cosmo_sim = CosmologicalEvolution(
            dt=dt,
            quantum_gravity_coupling=quantum_gravity_coupling,
            latent_dimensions=latent_dimensions
        )
        
        # Cria diretório para resultados se não existir
        self.results_dir = Path("simulation_results")
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info("Simulação cosmológica quântica expandida inicializada")
    
    def run_simulation(self) -> Dict[str, List[float]]:
        """
        Executa a simulação cosmológica quântica expandida.
        
        Returns:
            Dicionário com histórico completo da simulação
        """
        logger.info("Iniciando simulação cosmológica quântica expandida...")
        
        for step in range(self.num_steps):
            # 1. Evolui o campo quântico
            quantum_state = self.quantum_sim.evolve_step()
            
            # 2. Extrai informações do estado quântico
            quantum_energy = quantum_state['energy_density']
            latent_info = quantum_state['latent_info_rate']
            
            # 3. Evolui as variáveis cosmológicas
            self.cosmo_sim.evolve_step(quantum_energy, latent_info)
            
            # Log de progresso a cada 10% da simulação
            if step % (self.num_steps // 10) == 0:
                progress = (step / self.num_steps) * 100
                logger.info(f"Progresso da simulação: {progress:.1f}%")
        
        logger.info("Simulação concluída com sucesso")
        return self.cosmo_sim.history
    
    def analyze_results(self, history: Dict[str, List[float]]):
        """
        Analisa e visualiza os resultados da simulação expandida.
    
    Args:
            history: Histórico completo da simulação
        """
        logger.info("Iniciando análise dos resultados...")
        
        # 1. Salva dados em CSV
        df = pd.DataFrame(history)
        df.to_csv(self.results_dir / "simulation_data.csv", index=False)
        logger.info("Dados salvos em CSV")
        
        # 2. Configura estilo dos gráficos
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
        # 3. Plot da evolução temporal
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Evolução Cosmológica Quântica', fontsize=16)
        
        # Hubble Rate
        axes[0,0].plot(history['times'], history['hubble'])
        axes[0,0].set_title('Taxa de Hubble H(t)')
        axes[0,0].set_xlabel('Tempo')
        axes[0,0].set_ylabel('H(t)')
        
        # Cosmological Constant
        axes[0,1].plot(history['times'], history['lambda'])
        axes[0,1].set_title('Constante Cosmológica Λ(t)')
        axes[0,1].set_xlabel('Tempo')
        axes[0,1].set_ylabel('Λ(t)')
        
        # Scale Factor
        axes[1,0].plot(history['times'], history['scale_factor'])
        axes[1,0].set_title('Fator de Escala a(t)')
        axes[1,0].set_xlabel('Tempo')
        axes[1,0].set_ylabel('a(t)')
        
        # Quantum Gravity Potential
        axes[1,1].plot(history['times'], history['quantum_gravity_potential'])
        axes[1,1].set_title('Potencial Gravitacional Quântico')
        axes[1,1].set_xlabel('Tempo')
        axes[1,1].set_ylabel('V_q(t)')
        
        # Spacetime Curvature
        axes[2,0].plot(history['times'], history['spacetime_curvature'])
        axes[2,0].set_title('Curvatura do Espaço-tempo')
        axes[2,0].set_xlabel('Tempo')
        axes[2,0].set_ylabel('R(t)')
        
        # Latent Information
        axes[2,1].plot(history['times'], history['latent_info_integral'])
        axes[2,1].set_title('Informação Latente Acumulada')
        axes[2,1].set_xlabel('Tempo')
        axes[2,1].set_ylabel('∫Φ_latent(t)dt')
        
    plt.tight_layout()
        plt.savefig(self.results_dir / "evolution_plots.png")
        plt.close()
        logger.info("Gráficos de evolução temporal salvos")
        
        # 4. Análise de correlações
        correlation_matrix = df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Matriz de Correlação entre Variáveis')
        plt.tight_layout()
        plt.savefig(self.results_dir / "correlation_plot.png")
        plt.close()
        logger.info("Matriz de correlação salva")
        
        # 5. Análise de correlações específicas
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Análise de Correlações Específicas', fontsize=16)
        
        # H(t) vs Λ(t)
        axes[0,0].scatter(history['hubble'], history['lambda'], alpha=0.5)
        axes[0,0].set_title('H(t) vs Λ(t)')
        axes[0,0].set_xlabel('H(t)')
        axes[0,0].set_ylabel('Λ(t)')
        
        # Curvatura vs Potencial Quântico
        axes[0,1].scatter(history['spacetime_curvature'], 
                         history['quantum_gravity_potential'], 
                         alpha=0.5)
        axes[0,1].set_title('Curvatura vs Potencial Quântico')
        axes[0,1].set_xlabel('R(t)')
        axes[0,1].set_ylabel('V_q(t)')
        
        # Informação Latente vs Λ(t)
        axes[1,0].scatter(history['latent_info_integral'], 
                         history['lambda'], 
                         alpha=0.5)
        axes[1,0].set_title('Informação Latente vs Λ(t)')
        axes[1,0].set_xlabel('∫Φ_latent(t)dt')
        axes[1,0].set_ylabel('Λ(t)')
        
        # Flutuações Quânticas vs H(t)
        axes[1,1].scatter(history['quantum_fluctuations'], 
                         history['hubble'], 
                         alpha=0.5)
        axes[1,1].set_title('Flutuações Quânticas vs H(t)')
        axes[1,1].set_xlabel('δq(t)')
        axes[1,1].set_ylabel('H(t)')
        
    plt.tight_layout()
        plt.savefig(self.results_dir / "correlation_analysis.png")
        plt.close()
        logger.info("Análise de correlações específicas salva")
        
        # 6. Análise do espaço de fase retrocausal
        plt.figure(figsize=(10, 8))
        plt.plot(history['hubble'], 
                np.gradient(history['lambda'], history['times']),
                'b-', alpha=0.5)
        plt.title('Espaço de Fase Retrocausal')
        plt.xlabel('H(t)')
        plt.ylabel('dΛ/dt')
        plt.grid(True)
        plt.savefig(self.results_dir / "retrocausal_phase_space.png")
        plt.close()
        logger.info("Análise do espaço de fase retrocausal salva")
        
        logger.info("Análise de resultados concluída com sucesso")

def main():
    """Função principal para executar a simulação cosmológica quântica expandida."""
    # Parâmetros da simulação
    num_steps = 1000
    dt = 0.01
    quantum_gravity_coupling = 0.1
    latent_dimensions = 3
    
    # Cria e executa a simulação
    simulation = QuantumCosmologicalSimulation(
        num_steps=num_steps,
        dt=dt,
        quantum_gravity_coupling=quantum_gravity_coupling,
        latent_dimensions=latent_dimensions
    )
    
    # Executa a simulação e obtém resultados
    history = simulation.run_simulation()
    
    # Analisa e visualiza os resultados
    simulation.analyze_results(history)
    
    logger.info("Simulação cosmológica quântica expandida concluída com sucesso")

if __name__ == "__main__":
    main()
