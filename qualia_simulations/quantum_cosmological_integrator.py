#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrador Quântico-Cosmológico Retrocausal

Este módulo implementa o integrador principal que coordena a evolução
do campo quântico e das grandezas cosmológicas, incluindo os efeitos
retrocausais mediados por fluxos informacionais.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Callable
import logging
from pathlib import Path
import os

# Importa os componentes principais
from .quantum_cosmological_simulator import QuantumFieldSimulator
from .cosmological_evolution import CosmologicalEvolution

# Configuração de logging
logger = logging.getLogger("QUALIA-COSMOQUANTUM.integrator")

class QuantumCosmologicalIntegrator:
    """
    Integrador que coordena a evolução do campo quântico e das grandezas cosmológicas.
    
    Implementa o acoplamento bidirecional entre:
    1. Campo quântico não-linear com decoerência
    2. Evolução cosmológica dinâmica (H(t) e Λ(t))
    3. Fluxos de informação latente com efeitos retrocausais
    """
    
    def __init__(self, **params):
        """
        Inicializa o integrador com os parâmetros especificados.
        
        Args:
            **params: Dicionário com todos os parâmetros necessários
        """
        # Parâmetros do campo quântico
        self.quantum_params = {
            'grid_size': params.get('grid_size', 64),
            'spatial_dimensions': params.get('spatial_dimensions', 3),
            'dx': params.get('dx', 0.1),
            'dt': params.get('dt', 0.01),
            'hbar': params.get('hbar', 1.0),
            'mass': params.get('mass', 1.0),
            'nonlinearity_strength': params.get('nonlinearity_strength', 2.0),
            'decoherence_rate': params.get('decoherence_rate', 0.05),
            'use_density_matrix': params.get('use_density_matrix', False),
            'potential_strength': params.get('potential_strength', 1.0),
            'quantum_gravity_coupling': params.get('quantum_gravity_coupling', 0.1),
            'latent_dimensions': params.get('latent_dimensions', 3),
            'entanglement_strength': params.get('entanglement_strength', 0.5)
        }
        
        # Parâmetros cosmológicos
        self.cosmo_params = {
            'initial_hubble': params.get('initial_hubble', 70.0),
            'initial_lambda': params.get('initial_lambda', 1.0e-35),
            'matter_density': params.get('matter_density', 0.3),
            'beta_coupling': params.get('beta_coupling', 0.2)
        }
        
        # Parâmetros de informação latente
        self.latent_params = {
            'amplitude': params.get('latent_amplitude', 5.0e-37),
            'oscillation_period': params.get('oscillation_period', 10.0),
            'noise_level': params.get('latent_noise_level', 0.2),
            'mode': params.get('latent_mode', 'mixed')
        }
        
        # Parâmetros de acoplamento
        self.coupling_params = {
            'quantum_cosmo': params.get('quantum_cosmo_coupling', 0.7),
            'cosmo_quantum': params.get('cosmo_quantum_coupling', 0.5),
            'entropy_latent': params.get('entropy_latent_coupling', 0.3)
        }
        
        # Inicializa os componentes
        self.quantum_simulator = QuantumFieldSimulator(**self.quantum_params)
        self.cosmo_evolution = CosmologicalEvolution(**self.cosmo_params)
        
        # Histórico da simulação
        self.history = {
            'step': [],
            'time': [],
            'quantum_energy': [],
            'quantum_entropy': [],
            'hubble': [],
            'lambda': [],
            'latent_info': [],
            'latent_accum': []
        }
        
        logger.info("Integrador quântico-cosmológico inicializado com sucesso")
    
    def _calculate_latent_info(self, step: int, time: float) -> float:
        """
        Calcula a taxa de informação latente Φ(t) baseada no modo configurado.
        
        Args:
            step: Passo atual da simulação
            time: Tempo atual
            
        Returns:
            Taxa de informação latente Φ(t)
        """
        if self.latent_params['mode'] == 'oscillation':
            # Modo oscilatório puro
            return self.latent_params['amplitude'] * np.sin(2 * np.pi * time / self.latent_params['oscillation_period'])
        
        elif self.latent_params['mode'] == 'noise':
            # Modo ruído puro
            return self.latent_params['amplitude'] * np.random.normal(0, self.latent_params['noise_level'])
        
        else:  # 'mixed'
            # Modo misto: oscilação + ruído
            oscillation = np.sin(2 * np.pi * time / self.latent_params['oscillation_period'])
            noise = np.random.normal(0, self.latent_params['noise_level'])
            return self.latent_params['amplitude'] * (oscillation + noise)
    
    def _update_history(self, step: int, time: float):
        """Atualiza o histórico da simulação com os valores atuais."""
        self.history['step'].append(step)
        self.history['time'].append(time)
        self.history['quantum_energy'].append(self.quantum_simulator.calculate_total_energy())
        self.history['quantum_entropy'].append(self.quantum_simulator.calculate_entropy())
        self.history['hubble'].append(self.cosmo_evolution.current_hubble)
        self.history['lambda'].append(self.cosmo_evolution.current_lambda)
        self.history['latent_info'].append(self._calculate_latent_info(step, time))
        self.history['latent_accum'].append(self.cosmo_evolution.latent_info_integral)
    
    def run_simulation(self, num_steps: int, callback: Optional[Callable] = None) -> Dict:
        """
        Executa a simulação completa por um número especificado de passos.
        
        Args:
            num_steps: Número de passos para simular
            callback: Função opcional para monitoramento em tempo real
            
        Returns:
            Histórico completo da simulação
        """
        logger.info(f"Iniciando simulação com {num_steps} passos")
        
        # Reinicia os componentes
        self.quantum_simulator = QuantumFieldSimulator(**self.quantum_params)
        self.cosmo_evolution = CosmologicalEvolution(**self.cosmo_params)
        self.history = {key: [] for key in self.history.keys()}
        
        # Loop principal da simulação
        for step in range(num_steps):
            time = step * self.quantum_params['dt']
            
            # 1. Calcula informação latente
            latent_info = self._calculate_latent_info(step, time)
            
            # 2. Evolui o campo quântico com acoplamento cosmológico
            quantum_energy = self.quantum_simulator.calculate_total_energy()
            self.quantum_simulator.evolve_step(
                cosmological_constant=self.cosmo_evolution.current_lambda * self.coupling_params['quantum_cosmo']
            )
            
            # 3. Evolui as grandezas cosmológicas com acoplamento quântico
            self.cosmo_evolution.evolve_step(
                quantum_energy_density=quantum_energy * self.coupling_params['cosmo_quantum'],
                latent_info_rate=latent_info
            )
            
            # 4. Atualiza histórico
            self._update_history(step, time)
            
            # 5. Chama callback se fornecido
            if callback is not None:
                callback(step, {
                    'quantum_energy': self.history['quantum_energy'][-1],
                    'lambda': self.history['lambda'][-1],
                    'hubble': self.history['hubble'][-1]
                })
        
        logger.info("Simulação concluída com sucesso")
        return self.history
    
    def plot_results(self, save_dir: Optional[str] = None):
        """
        Gera visualizações dos resultados da simulação.
        
        Args:
            save_dir: Diretório opcional para salvar as figuras
        """
        # 1. Evolução temporal das grandezas principais
        plt.figure(figsize=(15, 10))
        
        # Energia quântica
        plt.subplot(2, 2, 1)
        plt.plot(self.history['time'], self.history['quantum_energy'])
        plt.title('Energia Quântica vs Tempo')
        plt.xlabel('Tempo')
        plt.ylabel('Energia')
        plt.grid(True)
        
        # Taxa de Hubble
        plt.subplot(2, 2, 2)
        plt.plot(self.history['time'], self.history['hubble'])
        plt.title('Taxa de Hubble vs Tempo')
        plt.xlabel('Tempo')
        plt.ylabel('H(t)')
        plt.grid(True)
        
        # Constante cosmológica
        plt.subplot(2, 2, 3)
        plt.plot(self.history['time'], self.history['lambda'])
        plt.title('Constante Cosmológica vs Tempo')
        plt.xlabel('Tempo')
        plt.ylabel('Λ(t)')
        plt.grid(True)
        
        # Entropia quântica
        plt.subplot(2, 2, 4)
        plt.plot(self.history['time'], self.history['quantum_entropy'])
        plt.title('Entropia Quântica vs Tempo')
        plt.xlabel('Tempo')
        plt.ylabel('Entropia')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, 'evolution_plots.png'), dpi=300)
            plt.close()
        else:
            plt.show()
        
        # 2. Correção entre grandezas
        plt.figure(figsize=(12, 8))
        
        # Normaliza as grandezas para melhor visualização
        quantum_energy_norm = np.array(self.history['quantum_energy'])
        quantum_energy_norm = (quantum_energy_norm - np.min(quantum_energy_norm)) / (np.max(quantum_energy_norm) - np.min(quantum_energy_norm))
        
        hubble_norm = np.array(self.history['hubble'])
        hubble_norm = (hubble_norm - np.min(hubble_norm)) / (np.max(hubble_norm) - np.min(hubble_norm))
        
        lambda_norm = np.array(self.history['lambda'])
        lambda_norm = (lambda_norm - np.min(lambda_norm)) / (np.max(lambda_norm) - np.min(lambda_norm))
        
        # Plota as grandezas normalizadas
        plt.plot(self.history['time'], quantum_energy_norm, label='Energia Quântica')
        plt.plot(self.history['time'], hubble_norm, label='Taxa de Hubble')
        plt.plot(self.history['time'], lambda_norm, label='Constante Cosmológica')
        
        plt.title('Correlação entre Grandezas do Sistema')
        plt.xlabel('Tempo')
        plt.ylabel('Valor Normalizado')
        plt.legend()
        plt.grid(True)
        
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, 'correlation_plot.png'), dpi=300)
            plt.close()
        else:
            plt.show() 