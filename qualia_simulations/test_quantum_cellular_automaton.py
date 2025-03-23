#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste do Autômato Celular Quântico Integrado com QUALIA
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from quantum_cellular_automaton import QuantumCellularAutomaton
import logging

# Configuração do logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_grid(grid: np.ndarray, title: str, save_path: Path = None):
    """Plota grid do autômato"""
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap='binary')
    plt.title(title)
    plt.colorbar()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_metrics(metrics_history: list, save_path: Path = None):
    """Plota histórico de métricas"""
    plt.figure(figsize=(12, 8))
    
    # Extrai métricas
    fractal_dims = [m['fractal_dimension'] for m in metrics_history]
    quantum_coherence = [m['quantum_coherence'] for m in metrics_history]
    holographic_compression = [m['holographic_compression'] for m in metrics_history]
    retrocausal_influence = [m['retrocausal_influence'] for m in metrics_history]
    parallel_efficiency = [m['parallel_efficiency'] for m in metrics_history]
    
    # Plota cada métrica
    plt.subplot(2, 3, 1)
    plt.plot(fractal_dims)
    plt.title('Dimensão Fractal')
    
    plt.subplot(2, 3, 2)
    plt.plot(quantum_coherence)
    plt.title('Coerência Quântica')
    
    plt.subplot(2, 3, 3)
    plt.plot(holographic_compression)
    plt.title('Compressão Holográfica')
    
    plt.subplot(2, 3, 4)
    plt.plot(retrocausal_influence)
    plt.title('Influência Retrocausal')
    
    plt.subplot(2, 3, 5)
    plt.plot(parallel_efficiency)
    plt.title('Eficiência Paralela')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def main():
    """Executa teste do autômato celular quântico"""
    # Cria diretório para resultados
    output_dir = Path('resultados_quantum_automaton')
    output_dir.mkdir(exist_ok=True)
    
    # Inicializa autômato
    logger.info("Inicializando autômato celular quântico...")
    automaton = QuantumCellularAutomaton(width=64, height=64, p_init=0.3)
    
    # Salva estado inicial
    plot_grid(automaton.grid, 'Estado Inicial', 
              output_dir / 'estado_inicial.png')
    
    # Executa evolução
    num_steps = 100
    metrics_history = []
    
    logger.info(f"Executando {num_steps} passos de evolução...")
    for step in range(num_steps):
        # Executa passo
        fractal_dim = automaton.step()
        
        # Coleta métricas
        metrics = automaton.get_evolution_metrics()
        metrics_history.append(metrics)
        
        # Salva estado a cada 20 passos
        if (step + 1) % 20 == 0:
            plot_grid(automaton.grid, f'Estado após {step+1} passos',
                     output_dir / f'estado_passo_{step+1}.png')
            
        # Log de progresso
        if (step + 1) % 10 == 0:
            logger.info(f"Passo {step+1}/{num_steps} - "
                       f"Dimensão Fractal: {fractal_dim:.4f}")
    
    # Plota histórico de métricas
    plot_metrics(metrics_history, output_dir / 'metricas_evolucao.png')
    
    # Salva estado final
    plot_grid(automaton.grid, 'Estado Final',
              output_dir / 'estado_final.png')
    
    # Salva estado completo
    automaton.save_state(output_dir / 'estado_completo')
    
    logger.info("Teste concluído com sucesso!")
    logger.info(f"Resultados salvos em: {output_dir}")

if __name__ == '__main__':
    main() 