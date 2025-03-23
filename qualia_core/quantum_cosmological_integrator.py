#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrador Principal do Simulador Cosmológico-Quântico Retrocausal

Este módulo orquestra a interação entre:
1. Campo quântico não-linear com decoerência
2. Evolução cosmológica dinâmica
3. Campo de informação latente

Implementando um sistema coerente onde processos quânticos locais
influenciam grandezas cosmológicas através de loops de feedback mediados
por fluxos informacionais com efeitos retrocausais.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable, Union
import logging
from pathlib import Path
import time
import os

# Importa os componentes principais
from qualia_unified.utils.quantum_cosmological_simulator import QuantumFieldSimulator
from qualia_unified.utils.cosmological_evolution import CosmologicalEvolution
from qualia_unified.utils.latent_information_field import LatentInformationField

# Configuração de logging
logger = logging.getLogger("QUALIA-COSMOQUANTUM.integrator")

class QuantumCosmologicalIntegrator:
    """
    Integrador principal que coordena a evolução acoplada do sistema quântico-cosmológico.
    
    Implementa os loops de feedback:
    - Campo quântico → Densidade de energia → H(t)
    - H(t) → Λ(t) → Potencial no campo quântico
    - Entropia quântica → Informação latente → Λ(t)
    """
    
    def __init__(self,
                 # Parâmetros do campo quântico
                 grid_size: int = 64,
                 spatial_dim: int = 1,
                 dx: float = 0.1,
                 dt: float = 0.01,
                 hbar: float = 1.0,
                 mass: float = 1.0,
                 nonlinearity_strength: float = 1.0,
                 decoherence_rate: float = 0.1,
                 use_density_matrix: bool = False,
                 
                 # Parâmetros cosmológicos
                 initial_hubble: float = 70.0,
                 initial_lambda: float = 1.0e-35,
                 matter_density: float = 0.3,
                 beta_coupling: float = 0.1,
                 
                 # Parâmetros de informação latente
                 latent_amplitude: float = 1.0e-38,
                 oscillation_period: float = 10.0,
                 latent_noise_level: float = 0.1,
                 latent_mode: str = 'mixed',
                 
                 # Parâmetros de acoplamento
                 quantum_cosmo_coupling: float = 0.5,
                 cosmo_quantum_coupling: float = 0.5,
                 entropy_latent_coupling: float = 0.5):
        """
        Inicializa o integrador quântico-cosmológico.
        
        Args:
            # Parâmetros do campo quântico
            grid_size: Número de pontos na grade espacial
            spatial_dim: Dimensão espacial da simulação (1, 2 ou 3)
            dx: Espaçamento da grade espacial
            dt: Passo de tempo para evolução
            hbar: Constante de Planck reduzida
            mass: Massa da partícula
            nonlinearity_strength: Força do termo não-linear (g)
            decoherence_rate: Taxa base de decoerência
            use_density_matrix: Se True, usa formalismo de matriz densidade
            
            # Parâmetros cosmológicos
            initial_hubble: Valor inicial da taxa de Hubble H(0)
            initial_lambda: Valor inicial da constante cosmológica Λ(0)
            matter_density: Densidade relativa de matéria no universo
            beta_coupling: Acoplamento entre derivada de H e Λ
            
            # Parâmetros de informação latente
            latent_amplitude: Amplitude base do fluxo informacional
            oscillation_period: Período base para modos oscilatórios
            latent_noise_level: Nível relativo de ruído estocástico
            latent_mode: Tipo de padrão para geração
            
            # Parâmetros de acoplamento
            quantum_cosmo_coupling: Força do acoplamento campo → cosmologia
            cosmo_quantum_coupling: Força do acoplamento cosmologia → campo
            entropy_latent_coupling: Força do acoplamento entropia → informação latente
        """
        # Armazena parâmetros de acoplamento
        self.quantum_cosmo_coupling = quantum_cosmo_coupling
        self.cosmo_quantum_coupling = cosmo_quantum_coupling
        self.entropy_latent_coupling = entropy_latent_coupling
        self.dt = dt
        
        # Inicializa componentes
        logger.info("Inicializando os subsistemas do integrador quântico-cosmológico...")
        
        # 1. Inicializa o simulador de campo quântico
        self.quantum_field = QuantumFieldSimulator(
            grid_size=grid_size,
            spatial_dim=spatial_dim,
            dx=dx,
            dt=dt,
            hbar=hbar,
            mass=mass,
            nonlinearity_strength=nonlinearity_strength,
            decoherence_rate=decoherence_rate,
            use_density_matrix=use_density_matrix
        )
        
        # 2. Inicializa o simulador cosmológico
        self.cosmology = CosmologicalEvolution(
            initial_hubble=initial_hubble,
            initial_lambda=initial_lambda,
            matter_density=matter_density,
            beta_coupling=beta_coupling,
            dt=dt
        )
        
        # 3. Inicializa o gerador de informação latente
        self.latent_field = LatentInformationField(
            base_amplitude=latent_amplitude,
            oscillation_period=oscillation_period,
            noise_level=latent_noise_level,
            mode=latent_mode,
            dt=dt
        )
        
        # Inicializa histórico global
        self.current_step = 0
        self.history = {
            'step': [0],
            'time': [0.0],
            'quantum_energy': [self.quantum_field.calculate_total_energy()],
            'quantum_entropy': [self.quantum_field.calculate_entropy()],
            'hubble': [self.cosmology.current_hubble],
            'lambda': [self.cosmology.current_lambda],
            'latent_info': [self.latent_field.current_value],
            'latent_accum': [0.0]
        }
        
        logger.info("Integrador quântico-cosmológico inicializado com sucesso.")
    
    def evolve_step(self):
        """
        Evolui o sistema completo por um passo de tempo dt, implementando os loops de feedback.
        """
        # 1. Obtém o valor atual da constante cosmológica
        current_lambda = self.cosmology.current_lambda
        
        # 2. Evolui o campo quântico com o termo de acoplamento cosmológico
        # O acoplamento cosmológico entra como um potencial adicional no campo
        # Λ(t) → termo adicional no campo
        scaled_lambda = current_lambda * self.cosmo_quantum_coupling
        self.quantum_field.evolve_step(cosmological_constant=scaled_lambda)
        
        # 3. Calcula a densidade de energia do campo quântico
        # A energia do campo influencia a equação de Friedmann
        quantum_energy = self.quantum_field.calculate_total_energy()
        quantum_energy_density = quantum_energy * self.quantum_cosmo_coupling
        
        # 4. Calcula a entropia do campo quântico
        quantum_entropy = self.quantum_field.calculate_entropy()
        
        # 5. Evolui o campo de informação latente com feedback da entropia quântica
        # A entropia do campo modula o fluxo de informação latente
        # S_quantum → modulação de Φ_latent
        scaled_entropy = quantum_entropy * self.entropy_latent_coupling
        latent_info_rate = self.latent_field.evolve_step(quantum_entropy=scaled_entropy)
        
        # 6. Evolui a cosmologia com feedback do campo quântico e informação latente
        # A densidade de energia quântica e informação latente influenciam H(t) e Λ(t)
        # ρ_quantum → H(t) → Λ(t)
        # Φ_latent → Λ(t)
        self.cosmology.evolve_step(
            quantum_energy_density=quantum_energy_density,
            latent_info_rate=latent_info_rate
        )
        
        # 7. Atualiza histórico global
        self.current_step += 1
        current_time = self.history['time'][-1] + self.dt
        
        self.history['step'].append(self.current_step)
        self.history['time'].append(current_time)
        self.history['quantum_energy'].append(quantum_energy)
        self.history['quantum_entropy'].append(quantum_entropy)
        self.history['hubble'].append(self.cosmology.current_hubble)
        self.history['lambda'].append(self.cosmology.current_lambda)
        self.history['latent_info'].append(latent_info_rate)
        self.history['latent_accum'].append(self.latent_field.accumulated_value)
        
        return {
            'quantum_energy': quantum_energy,
            'quantum_entropy': quantum_entropy,
            'hubble': self.cosmology.current_hubble,
            'lambda': self.cosmology.current_lambda,
            'latent_info': latent_info_rate
        }
    
    def run_simulation(self, num_steps: int, callback: Optional[Callable] = None):
        """
        Executa a simulação por um número específico de passos.
        
        Args:
            num_steps: Número de passos de tempo para evoluir
            callback: Função opcional a ser chamada após cada passo
        
        Returns:
            Dicionário com histórico completo da simulação
        """
        logger.info(f"Iniciando simulação com {num_steps} passos...")
        start_time = time.time()
        
        for step in range(num_steps):
            # Evolui um passo
            results = self.evolve_step()
            
            # Executa callback se fornecido
            if callback is not None:
                callback(step, results)
            
            # Log a cada 10% da simulação
            if step % max(1, num_steps // 10) == 0:
                progress = 100 * step / num_steps
                elapsed = time.time() - start_time
                logger.info(f"Progresso: {progress:.1f}% | Passo: {step}/{num_steps} | Tempo: {elapsed:.2f}s")
        
        total_time = time.time() - start_time
        logger.info(f"Simulação concluída em {total_time:.2f} segundos.")
        
        return self.history
    
    def plot_results(self, save_dir: Optional[str] = None):
        """
        Gera visualizações dos resultados da simulação.
        
        Args:
            save_dir: Diretório opcional para salvar as figuras
        """
        if save_dir is not None:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        
        # Configura o estilo de plot
        plt.style.use('dark_background')
        
        # 1. Plot da evolução do campo quântico
        self._plot_quantum_evolution(save_dir)
        
        # 2. Plot da evolução cosmológica
        self._plot_cosmological_evolution(save_dir)
        
        # 3. Plot da informação latente
        self._plot_latent_information(save_dir)
        
        # 4. Plot integrado dos loops de feedback
        self._plot_feedback_loops(save_dir)
    
    def _plot_quantum_evolution(self, save_dir: Optional[str] = None):
        """Gera visualizações da evolução do campo quântico."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot da energia quântica
        time_array = np.array(self.history['time'])
        energy_array = np.array(self.history['quantum_energy'])
        
        ax1.plot(time_array, energy_array, 'c-', linewidth=2)
        ax1.set_title('Evolução da Energia do Campo Quântico', fontsize=16)
        ax1.set_xlabel('Tempo', fontsize=14)
        ax1.set_ylabel('Energia Total', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Plot da entropia quântica
        entropy_array = np.array(self.history['quantum_entropy'])
        
        ax2.plot(time_array, entropy_array, 'm-', linewidth=2)
        ax2.set_title('Evolução da Entropia do Campo Quântico', fontsize=16)
        ax2.set_xlabel('Tempo', fontsize=14)
        ax2.set_ylabel('Entropia de von Neumann', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, 'quantum_evolution.png'), dpi=300)
            plt.close()
        else:
            plt.show()
    
    def _plot_cosmological_evolution(self, save_dir: Optional[str] = None):
        """Gera visualizações da evolução cosmológica."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot da taxa de Hubble
        time_array = np.array(self.history['time'])
        hubble_array = np.array(self.history['hubble'])
        
        ax1.plot(time_array, hubble_array, 'r-', linewidth=2)
        ax1.set_title('Evolução da Taxa de Hubble H(t)', fontsize=16)
        ax1.set_xlabel('Tempo', fontsize=14)
        ax1.set_ylabel('H(t)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Plot da constante cosmológica
        lambda_array = np.array(self.history['lambda'])
        
        ax2.plot(time_array, lambda_array, 'g-', linewidth=2)
        ax2.set_title('Evolução da Constante Cosmológica Λ(t)', fontsize=16)
        ax2.set_xlabel('Tempo', fontsize=14)
        ax2.set_ylabel('Λ(t)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, 'cosmological_evolution.png'), dpi=300)
            plt.close()
        else:
            plt.show()
    
    def _plot_latent_information(self, save_dir: Optional[str] = None):
        """Gera visualizações do fluxo de informação latente."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot da taxa de fluxo de informação
        time_array = np.array(self.history['time'])
        latent_array = np.array(self.history['latent_info'])
        
        ax1.plot(time_array, latent_array, 'y-', linewidth=2)
        ax1.set_title('Fluxo de Informação Latente Φ_latent(t)', fontsize=16)
        ax1.set_xlabel('Tempo', fontsize=14)
        ax1.set_ylabel('Φ_latent(t)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Plot da integral acumulada
        latent_accum_array = np.array(self.history['latent_accum'])
        
        ax2.plot(time_array, latent_accum_array, 'orange', linewidth=2)
        ax2.set_title('Integral Acumulada de Informação Latente', fontsize=16)
        ax2.set_xlabel('Tempo', fontsize=14)
        ax2.set_ylabel('∫₀ᵗ Φ_latent(τ) dτ', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, 'latent_information.png'), dpi=300)
            plt.close()
        else:
            plt.show()
    
    def _plot_feedback_loops(self, save_dir: Optional[str] = None):
        """Gera visualizações dos loops de feedback."""
        fig, ax = plt.subplots(figsize=(14, 10))
        
        time_array = np.array(self.history['time'])
        
        # Normaliza as diferentes grandezas para mesma escala
        energy_norm = np.array(self.history['quantum_energy']) / np.max(np.abs(self.history['quantum_energy'])) if np.max(np.abs(self.history['quantum_energy'])) > 0 else np.zeros_like(time_array)
        entropy_norm = np.array(self.history['quantum_entropy']) / np.max(np.abs(self.history['quantum_entropy'])) if np.max(np.abs(self.history['quantum_entropy'])) > 0 else np.zeros_like(time_array)
        hubble_norm = np.array(self.history['hubble']) / np.max(np.abs(self.history['hubble'])) if np.max(np.abs(self.history['hubble'])) > 0 else np.zeros_like(time_array)
        lambda_norm = np.array(self.history['lambda']) / np.max(np.abs(self.history['lambda'])) if np.max(np.abs(self.history['lambda'])) > 0 else np.zeros_like(time_array)
        latent_norm = np.array(self.history['latent_info']) / np.max(np.abs(self.history['latent_info'])) if np.max(np.abs(self.history['latent_info'])) > 0 else np.zeros_like(time_array)
        
        # Plot todas as grandezas normalizadas
        ax.plot(time_array, energy_norm, 'c-', linewidth=2, label='Energia Quântica')
        ax.plot(time_array, entropy_norm, 'm-', linewidth=2, label='Entropia Quântica')
        ax.plot(time_array, hubble_norm, 'r-', linewidth=2, label='Taxa de Hubble H(t)')
        ax.plot(time_array, lambda_norm, 'g-', linewidth=2, label='Constante Cosmológica Λ(t)')
        ax.plot(time_array, latent_norm, 'y-', linewidth=2, label='Informação Latente Φ(t)')
        
        ax.set_title('Loops de Feedback no Sistema Quântico-Cosmológico', fontsize=18)
        ax.set_xlabel('Tempo', fontsize=14)
        ax.set_ylabel('Valores Normalizados', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, 'feedback_loops.png'), dpi=300)
            plt.close()
        else:
            plt.show()
    
    def reset(self):
        """Reinicia todos os componentes para o estado inicial."""
        logger.info("Reiniciando o sistema quântico-cosmológico para o estado inicial...")
        
        # Reinicia cada componente
        self.quantum_field._initialize_wavefunction()
        self.cosmology.reset()
        self.latent_field.reset()
        
        # Reinicia o histórico global
        self.current_step = 0
        self.history = {
            'step': [0],
            'time': [0.0],
            'quantum_energy': [self.quantum_field.calculate_total_energy()],
            'quantum_entropy': [self.quantum_field.calculate_entropy()],
            'hubble': [self.cosmology.current_hubble],
            'lambda': [self.cosmology.current_lambda],
            'latent_info': [self.latent_field.current_value],
            'latent_accum': [0.0]
        }
        
        logger.info("Sistema reiniciado com sucesso.")
