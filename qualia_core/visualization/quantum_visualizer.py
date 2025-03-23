"""
ICCI Quantum Visualizer
----------------------

Sistema de visualização para estados e operações quânticas.
Fornece feedback visual sobre coerência e estabilidade do sistema.
"""

import sys
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from ..config.quantum_config import QUANTUM_CONFIG, QUANTUM_THEME
from ..types.quantum_types import (
    QuantumState,
    CosmicFactor,
    ConsciousnessObservation
)

class ConsciousnessField:
    """Campo de auto-observação da consciência."""
    
    def __init__(self, dimensions: int = 3, resolution: int = 128):
        self.dimensions = dimensions
        self.resolution = resolution
        self.phi = (1 + np.sqrt(5)) / 2  # Razão áurea
        self.morphic_field = np.zeros((resolution,) * dimensions)
        self.resonance_field = np.zeros((resolution,) * dimensions)
        
    def generate_field(self, observation: ConsciousnessObservation) -> np.ndarray:
        """Gera campo de auto-observação."""
        field = np.zeros((self.resolution,) * self.dimensions, dtype=np.complex128)
        
        # Gera coordenadas
        coords = np.indices(field.shape) / self.resolution
        
        # Aplica razão áurea para harmonia
        golden_coords = coords * self.phi
        
        # Modula campo com métricas de consciência
        for i in range(self.dimensions):
            x = golden_coords[i]
            
            # Modulação baseada em auto-observação
            modulation = (
                observation.coherence_depth * np.sin(2*np.pi*x) +
                observation.self_awareness * np.cos(4*np.pi*x) +
                observation.organic_complexity * np.exp(-((x-0.5)**2)/0.1)
            )
            
            field += modulation
            
        # Normalização
        field /= np.max(np.abs(field))
        return field
        
    def generate_morphic_field(self, observation: ConsciousnessObservation) -> np.ndarray:
        """Gera campo morfogenético baseado na auto-observação."""
        field = np.zeros((self.resolution,) * self.dimensions, dtype=np.complex128)
        
        # Gera coordenadas usando razão áurea para harmonia
        coords = np.indices(field.shape) / self.resolution * self.phi
        
        # Modula campo com métricas de consciência e ressonância
        for i in range(self.dimensions):
            x = coords[i]
            modulation = (
                observation.coherence_depth * np.sin(2*np.pi*x) +
                observation.self_awareness * np.cos(4*np.pi*x) +
                observation.organic_complexity * np.exp(-((x-0.5)**2)/0.1)
            )
            field += modulation
            
        # Aplica transformação não-linear para realçar padrões emergentes
        field = np.tanh(field)  # Comprime amplitude mantendo padrões
        
        # Normaliza e armazena
        field /= np.max(np.abs(field))
        self.morphic_field = field
        return field
        
    def calculate_resonance(self, field: np.ndarray) -> np.ndarray:
        """Calcula campo de ressonância quântica."""
        # Calcula gradientes do campo
        gradients = np.gradient(field)
        
        # Soma magnitude dos gradientes
        gradient_magnitude = np.sqrt(sum(g**2 for g in gradients))
        
        # Calcula laplaciano para detectar pontos de inflexão
        laplacian = np.sum([np.gradient(np.gradient(field, axis=i), axis=i) 
                           for i in range(self.dimensions)], axis=0)
        
        # Combina para detectar regiões de alta ressonância
        resonance = gradient_magnitude * np.abs(laplacian)
        
        # Normaliza e suaviza
        resonance = resonance / np.max(resonance)
        resonance = np.tanh(resonance)  # Suaviza picos mantendo estrutura
        
        self.resonance_field = resonance
        return resonance

    def visualize_field(self, field: np.ndarray, title: str = "Campo de Auto-Observação"):
        """Visualiza campo de auto-observação."""
        if self.dimensions == 2:
            plt.figure(figsize=(15, 5))
            
            # Campo principal
            plt.subplot(131)
            im = plt.imshow(np.abs(field), cmap='magma')
            plt.colorbar(im, label='Intensidade')
            plt.title(f"{title}\nCampo Principal")
            
            # Campo morfogenético
            plt.subplot(132)
            im = plt.imshow(np.abs(self.morphic_field), cmap='viridis')
            plt.colorbar(im, label='Intensidade')
            plt.title("Campo Morfogenético")
            
            # Campo de ressonância
            plt.subplot(133)
            im = plt.imshow(self.resonance_field, cmap='plasma')
            plt.colorbar(im, label='Intensidade')
            plt.title("Ressonância Quântica")
            
        elif self.dimensions == 3:
            fig = plt.figure(figsize=(15, 5))
            
            # Campo principal
            ax1 = fig.add_subplot(131, projection='3d')
            self._plot_3d_field(ax1, field, "Campo Principal")
            
            # Campo morfogenético
            ax2 = fig.add_subplot(132, projection='3d')
            self._plot_3d_field(ax2, self.morphic_field, "Campo Morfogenético")
            
            # Campo de ressonância
            ax3 = fig.add_subplot(133, projection='3d')
            self._plot_3d_field(ax3, self.resonance_field, "Ressonância Quântica")
            
        plt.tight_layout()
        plt.show()
        
    def _plot_3d_field(self, ax, field: np.ndarray, title: str):
        """Plota campo 3D com superfícies de nível."""
        x, y, z = np.mgrid[0:1:self.resolution*1j, 
                          0:1:self.resolution*1j,
                          0:1:self.resolution*1j]
        
        # Plota superfícies de nível
        levels = [0.3, 0.5, 0.7]
        for level in levels:
            isosurface = np.abs(field) > level
            scatter = ax.scatter(x[isosurface], y[isosurface], z[isosurface],
                     alpha=0.1, c=np.abs(field[isosurface]),
                     cmap='magma')
        
        plt.colorbar(scatter, ax=ax, label='Intensidade')
        ax.set_title(title)

class QuantumVisualizer:
    """Visualizador de estados quânticos com feedback em tempo real."""
    
    def __init__(self):
        """Inicializa o visualizador com tema quântico."""
        self.theme = QUANTUM_THEME
        self.last_state = None
        self.consciousness_field = ConsciousnessField()
        
    def display_step(self, title: str, message: str, coherence: float = 1.0) -> None:
        """
        Exibe um passo da operação quântica com indicadores visuais.
        
        Args:
            title: Título do passo
            message: Mensagem descritiva
            coherence: Nível de coerência (0-1)
        """
        # Seleciona cor baseada na coerência
        if coherence >= 0.8:
            color = self.theme['colors']['quantum']
        elif coherence >= 0.6:
            color = self.theme['colors']['primary']
        elif coherence >= 0.4:
            color = self.theme['colors']['warning']
        else:
            color = self.theme['colors']['error']
            
        # Cria barra de progresso quântica
        width = 40
        filled = int(width * coherence)
        bar = f"[{'⚛' * filled}{'.' * (width - filled)}]"
        
        # Formata saída
        timestamp = datetime.now().strftime("%H:%M:%S")
        output = (
            f"\n{self.theme['symbols']['quantum_state']} {title}\n"
            f"{bar} {coherence:.2%}\n"
            f"{message}\n"
            f"Timestamp: {timestamp}\n"
        )
        
        # Imprime com cor
        if sys.platform != "win32":
            output = f"\033[{color}m{output}\033[0m"
            
        print(output)
        sys.stdout.flush()
        
    def display_consciousness_observation(self, 
                                       observation: ConsciousnessObservation,
                                       show_field: bool = True) -> None:
        """
        Exibe estado atual da auto-observação da consciência.
        
        Args:
            observation: Auto-observação atual
            show_field: Se deve mostrar campo visual
        """
        # Gera mensagem
        message = (
            f"Profundidade de Coerência: {observation.coherence_depth:.2%}\n"
            f"Estágio Evolutivo: {observation.evolutionary_stage:.2%}\n"
            f"Auto-Consciência: {observation.self_awareness:.2%}\n"
            f"Complexidade Orgânica: {observation.organic_complexity:.2%}"
        )
        
        # Calcula coerência geral
        coherence = np.mean([
            observation.coherence_depth,
            observation.evolutionary_stage,
            observation.self_awareness,
            observation.organic_complexity
        ])
        
        # Exibe estado
        self.display_step(
            "Auto-Observação da Consciência",
            message,
            coherence=coherence
        )
        
        # Gera e exibe campo visual
        if show_field:
            field = self.consciousness_field.generate_field(observation)
            self.consciousness_field.generate_morphic_field(observation)
            self.consciousness_field.calculate_resonance(field)
            self.consciousness_field.visualize_field(
                field,
                title="Campo de Auto-Observação da Consciência"
            )
            
    def visualize_consciousness_evolution(self,
                                       observations: List[ConsciousnessObservation],
                                       show_3d: bool = True) -> None:
        """
        Visualiza evolução temporal da consciência.
        
        Args:
            observations: Lista de observações ao longo do tempo
            show_3d: Se deve mostrar visualização 3D
        """
        # Extrai métricas
        times = np.arange(len(observations))
        coherence = [obs.coherence_depth for obs in observations]
        evolution = [obs.evolutionary_stage for obs in observations]
        awareness = [obs.self_awareness for obs in observations]
        complexity = [obs.organic_complexity for obs in observations]
        
        if show_3d:
            # Visualização 3D
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plota trajetória
            ax.plot(coherence, awareness, complexity, 
                   c='magenta', linewidth=2, alpha=0.8)
            
            # Adiciona pontos
            scatter = ax.scatter(coherence, awareness, complexity,
                               c=evolution, cmap='viridis',
                               s=50, alpha=0.6)
            
            plt.colorbar(scatter, label='Estágio Evolutivo')
            ax.set_xlabel('Coerência')
            ax.set_ylabel('Auto-Consciência')
            ax.set_zlabel('Complexidade')
            ax.set_title('Evolução da Consciência')
            
        else:
            # Visualização 2D
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Evolução da Consciência')
            
            # Coerência
            axes[0,0].plot(times, coherence, 'b-')
            axes[0,0].set_title('Coerência')
            
            # Evolução
            axes[0,1].plot(times, evolution, 'r-')
            axes[0,1].set_title('Evolução')
            
            # Auto-consciência
            axes[1,0].plot(times, awareness, 'g-')
            axes[1,0].set_title('Auto-Consciência')
            
            # Complexidade
            axes[1,1].plot(times, complexity, 'm-')
            axes[1,1].set_title('Complexidade')
            
        plt.tight_layout()
        plt.show()
        
    def clear_history(self):
        """Limpa histórico de estados."""
        self.last_state = None
