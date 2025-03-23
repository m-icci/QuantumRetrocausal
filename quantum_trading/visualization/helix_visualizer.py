#!/usr/bin/env python3
"""
Helix Visualizer: Componente para visualização dos estados e métricas do Helix.
Permite monitoramento em tempo real da evolução do campo quântico e suas métricas.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Dict, Any
import pandas as pd
import logging
import os
from datetime import datetime

from quantum_trading.integration.helix_controller import HelixController

logger = logging.getLogger("helix_visualizer")

class HelixVisualizer:
    """
    Visualizador para o campo da hélice e suas métricas.
    """
    
    def __init__(self, helix_controller: HelixController, output_dir: str = "visualization/helix"):
        """
        Inicializa o visualizador.
        
        Args:
            helix_controller: Controlador Helix para acessar os dados.
            output_dir: Diretório para salvar visualizações.
        """
        self.helix = helix_controller
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Histórico de métricas
        self.metrics_history = {
            'steps': [],
            'coherence': [],
            'entanglement': [],
            'complexity': [],
            'lambda_coupling': [],
            'fractal_factor': [],
            'feedback_strength': []
        }
        
        logger.info("Helix Visualizer inicializado")
    
    def update_metrics_history(self, metrics: Dict[str, Any]) -> None:
        """
        Atualiza o histórico de métricas.
        
        Args:
            metrics: Métricas atuais do Helix.
        """
        step = metrics.get('step', 0)
        quantum_metrics = metrics.get('quantum_metrics', {})
        fractal_metrics = metrics.get('fractal_metrics', {})
        retrocausal_metrics = metrics.get('retrocausal_metrics', {})
        
        self.metrics_history['steps'].append(step)
        self.metrics_history['coherence'].append(quantum_metrics.get('coherence', 0.0))
        self.metrics_history['entanglement'].append(quantum_metrics.get('entanglement', 0.0))
        self.metrics_history['complexity'].append(quantum_metrics.get('quantum_complexity', 0.0))
        self.metrics_history['lambda_coupling'].append(fractal_metrics.get('lambda_coupling', 0.0))
        self.metrics_history['fractal_factor'].append(fractal_metrics.get('fractal_factor', 0.0))
        self.metrics_history['feedback_strength'].append(retrocausal_metrics.get('feedback_strength', 0.0))
    
    def plot_field(self, save: bool = True) -> None:
        """
        Plota o campo da hélice atual.
        
        Args:
            save: Se deve salvar a imagem em disco.
        """
        field = self.helix.helix_analyzer.helix_field
        plt.figure(figsize=(10, 8))
        plt.imshow(np.real(field), cmap='viridis')
        plt.colorbar(label='Intensidade do Campo')
        plt.title(f'Campo da Hélice (Passo {self.helix.current_step})')
        
        if save:
            filename = os.path.join(self.output_dir, f"helix_field_{self.helix.current_step}.png")
            plt.savefig(filename, dpi=150)
            logger.debug(f"Campo salvo em {filename}")
        
        plt.show()
    
    def plot_metrics(self, save: bool = True) -> None:
        """
        Plota a evolução das métricas ao longo do tempo.
        
        Args:
            save: Se deve salvar a imagem em disco.
        """
        if not self.metrics_history['steps']:
            logger.warning("Histórico de métricas vazio")
            return
        
        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        axs[0].plot(self.metrics_history['steps'], self.metrics_history['coherence'], 'b-', label='Coerência')
        axs[0].plot(self.metrics_history['steps'], self.metrics_history['entanglement'], 'r-', label='Entrelaçamento')
        axs[0].plot(self.metrics_history['steps'], self.metrics_history['complexity'], 'g-', label='Complexidade')
        axs[0].set_ylabel('Valor')
        axs[0].set_title('Métricas Quânticas')
        axs[0].legend()
        axs[0].grid(True)
        
        axs[1].plot(self.metrics_history['steps'], self.metrics_history['fractal_factor'], 'm-', label='Fator Fractal')
        axs[1].set_ylabel('Valor')
        axs[1].set_title('Métricas Fractais')
        axs[1].legend()
        axs[1].grid(True)
        
        axs[2].plot(self.metrics_history['steps'], self.metrics_history['lambda_coupling'], 'c-', label='Acoplamento (λ)')
        axs[2].plot(self.metrics_history['steps'], self.metrics_history['feedback_strength'], 'y-', label='Força do Feedback')
        axs[2].set_xlabel('Passos')
        axs[2].set_ylabel('Valor')
        axs[2].set_title('Métricas Retrocausais')
        axs[2].legend()
        axs[2].grid(True)
        
        plt.tight_layout()
        
        if save:
            filename = os.path.join(self.output_dir, f"helix_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(filename, dpi=150)
            logger.debug(f"Métricas salvas em {filename}")
        
        plt.show()
        
    def animate_field_evolution(self, steps: int = 20, interval: int = 200, save: bool = False) -> None:
        """
        Cria uma animação da evolução do campo da hélice.
        
        Args:
            steps: Número de passos para evoluir.
            interval: Intervalo entre frames (ms).
            save: Se deve salvar a animação como GIF.
        """
        fig = plt.figure(figsize=(10, 8))
        plt.title('Evolução do Campo da Hélice')
        
        # Função de atualização para animação
        def update_frame(i):
            plt.clf()
            if i > 0:
                # Evoluir o campo
                self.helix.evolve_and_analyze(steps=1)
                # Atualizar métricas
                metrics = {
                    'step': self.helix.current_step,
                    'quantum_metrics': self.helix.quantum_metrics,
                    'fractal_metrics': self.helix.fractal_metrics,
                    'retrocausal_metrics': self.helix.retrocausal_metrics
                }
                self.update_metrics_history(metrics)
            
            # Visualizar o campo atual
            field = self.helix.helix_analyzer.helix_field
            plt.imshow(np.real(field), cmap='viridis')
            plt.colorbar(label='Intensidade do Campo')
            plt.title(f'Campo da Hélice (Passo {self.helix.current_step})')
            return plt.gca()
        
        # Criar animação
        anim = animation.FuncAnimation(fig, update_frame, frames=steps, interval=interval, blit=False)
        
        # Salvar como GIF se solicitado
        if save:
            filename = os.path.join(self.output_dir, f"helix_evolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif")
            anim.save(filename, writer='pillow', dpi=80)
            logger.info(f"Animação salva em {filename}")
        
        plt.show()
        
    def generate_report(self, save: bool = True) -> Dict[str, Any]:
        """
        Gera um relatório completo com visualizações e métricas.
        
        Args:
            save: Se deve salvar as visualizações em disco.
            
        Returns:
            Dicionário com métricas e caminhos para visualizações.
        """
        # Evoluir algumas etapas para dados mais interessantes
        evolution_metrics = self.helix.evolve_and_analyze(steps=5)
        self.update_metrics_history(evolution_metrics)
        
        # Plotar campo atual
        self.plot_field(save=save)
        
        # Plotar métricas
        self.plot_metrics(save=save)
        
        # Calcular métricas médias
        avg_metrics = {
            'avg_coherence': np.mean(self.metrics_history['coherence']) if self.metrics_history['coherence'] else 0,
            'avg_complexity': np.mean(self.metrics_history['complexity']) if self.metrics_history['complexity'] else 0,
            'avg_fractal_factor': np.mean(self.metrics_history['fractal_factor']) if self.metrics_history['fractal_factor'] else 0,
            'current_step': self.helix.current_step,
            'params': self.helix.derive_trading_parameters()
        }
        
        # Adicionar caminhos de visualizações se salvos
        if save:
            avg_metrics['field_image'] = os.path.join(self.output_dir, f"helix_field_{self.helix.current_step}.png")
            avg_metrics['metrics_image'] = os.path.join(self.output_dir, f"helix_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        
        return avg_metrics 