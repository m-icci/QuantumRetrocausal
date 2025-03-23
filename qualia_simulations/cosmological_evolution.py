#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Evolução Cosmológica para o Simulador Cosmológico-Quântico

Implementa a evolução dinâmica das grandezas cosmológicas:
- H(t): taxa de Hubble
- Λ(t): constante cosmológica dinâmica

Com base nas equações:
- H²(t) = (8πG/3) * (ρ_m + ρ_campo + ρ_Λ)
- Λ(t) = Λ_0 - β * [Ḣ(t)/H³(t)] + ∫₀ᵗ Φ_latent(τ) dτ
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Callable, Union
from .quantum_cosmological_simulator import QuantumFieldSimulator

# Configuração de logging
logger = logging.getLogger("QUALIA-COSMOQUANTUM.cosmology")

class CosmologicalEvolution:
    """
    Simulador da evolução cosmológica dinâmica.
    
    Implementa a evolução das grandezas cosmológicas:
    - H(t): taxa de Hubble
    - Λ(t): constante cosmológica dinâmica
    
    Com base nas equações:
    - H²(t) = (8πG/3) * (ρ_m + ρ_campo + ρ_Λ)
    - Λ(t) = Λ_0 - β * [Ḣ(t)/H³(t)] + ∫₀ᵗ Φ_latent(τ) dτ
    """
    
    def __init__(self, 
                 initial_hubble: float = 70.0,  # km/s/Mpc (valor atual)
                 initial_lambda: float = 1.0e-35,  # Unidades naturais
                 matter_density: float = 0.3,  # Densidade relativa
                 beta_coupling: float = 0.1,  # Acoplamento entre H e Λ
                 dt: float = 0.01,
                 quantum_gravity_coupling: float = 0.1,  # Acoplamento com gravitação quântica
                 latent_dimensions: int = 3):  # Dimensões da informação latente
        """
        Inicializa o simulador cosmológico expandido.
        
        Args:
            initial_hubble: Valor inicial da taxa de Hubble H(0)
            initial_lambda: Valor inicial da constante cosmológica Λ(0)
            matter_density: Densidade relativa de matéria no universo
            beta_coupling: Força do acoplamento entre derivada de H e Λ
            dt: Passo de tempo para evolução
            quantum_gravity_coupling: Força do acoplamento com gravitação quântica
            latent_dimensions: Número de dimensões da informação latente
        """
        # Constantes e parâmetros originais
        self.G = 6.67430e-11  # Constante gravitacional (m³/kg/s²)
        self.c = 299792458.0  # Velocidade da luz (m/s)
        
        # Conversão para unidades naturais onde c = 1
        self.H0 = initial_hubble * 1000 / (3.086e22)  # Conversão para 1/s
        self.Lambda0 = initial_lambda
        self.matter_density = matter_density
        self.beta = beta_coupling
        self.dt = dt
        
        # Novos parâmetros para gravitação quântica e informação latente
        self.quantum_gravity_coupling = quantum_gravity_coupling
        self.latent_dimensions = latent_dimensions
        
        # Estado atual
        self.current_hubble = self.H0
        self.current_lambda = self.Lambda0
        self.previous_hubble = self.H0  # Para cálculo de derivadas
        
        # Densidade crítica
        self.critical_density = 3 * self.H0**2 / (8 * np.pi * self.G)
        
        # Histórico expandido para análise
        self.history = {
            'times': [0.0],
            'hubble': [self.current_hubble],
            'lambda': [self.current_lambda],
            'scale_factor': [1.0],  # a(t) normalizado para a(0) = 1
            'latent_info_integral': [0.0],  # Integral de Φ_latent
            'quantum_gravity_potential': [0.0],  # Potencial gravitacional quântico
            'spacetime_curvature': [0.0],  # Curvatura do espaço-tempo
            'quantum_fluctuations': [0.0]  # Flutuações quânticas do vácuo
        }
        
        # Integral acumulada de informação latente
        self.latent_info_integral = 0.0
        
        # Inicializa estruturas para gravitação quântica
        self._setup_quantum_gravity()
        
        # Inicializa estruturas para informação latente multidimensional
        self._setup_latent_info()
        
        logger.info(f"Simulador cosmológico expandido inicializado: H0={initial_hubble} km/s/Mpc, Λ0={initial_lambda}")
    
    def _setup_quantum_gravity(self):
        """Configura as estruturas para gravitação quântica."""
        # Inicializa o tensor de curvatura do espaço-tempo
        self.spacetime_curvature = 0.0
        
        # Inicializa o potencial gravitacional quântico
        self.quantum_gravity_potential = 0.0
        
        # Inicializa as flutuações quânticas do vácuo
        self.vacuum_fluctuations = np.random.normal(0, 1e-10)
    
    def _setup_latent_info(self):
        """Configura as estruturas para informação latente multidimensional."""
        # Cria tensor de informação latente
        self.latent_tensor = np.zeros(self.latent_dimensions)
        
        # Inicializa com padrões fractais
        for dim in range(self.latent_dimensions):
            self.latent_tensor[dim] = np.sin(2 * np.pi * dim / self.latent_dimensions)
        
        # Normaliza o tensor
        self.latent_tensor /= np.max(np.abs(self.latent_tensor))
    
    def _calculate_quantum_corrections(self, quantum_energy_density: float) -> float:
        """
        Calcula correções quânticas para a evolução cosmológica.
        
        Args:
            quantum_energy_density: Densidade de energia do campo quântico
            
        Returns:
            Correção quântica total
        """
        # Calcula flutuações quânticas do vácuo
        vacuum_energy = self.vacuum_fluctuations * quantum_energy_density
        
        # Calcula efeitos de gravitação quântica
        gravity_correction = self.quantum_gravity_coupling * (
            self.quantum_gravity_potential + 
            self.spacetime_curvature * self.current_hubble**2
        )
        
        # Combina as correções
        return vacuum_energy + gravity_correction
    
    def _update_quantum_gravity(self, quantum_energy_density: float):
        """
        Atualiza as variáveis de gravitação quântica.
        
        Args:
            quantum_energy_density: Densidade de energia do campo quântico
        """
        # Atualiza o potencial gravitacional quântico
        self.quantum_gravity_potential = (
            self.quantum_gravity_coupling * 
            quantum_energy_density * 
            self.current_hubble**2
        )
        
        # Atualiza a curvatura do espaço-tempo
        self.spacetime_curvature = (
            self.current_lambda / 3 + 
            8 * np.pi * self.G * quantum_energy_density / (3 * self.c**4)
        )
        
        # Atualiza as flutuações quânticas do vácuo
        self.vacuum_fluctuations = np.random.normal(0, 1e-10)
    
    def evolve_step(self, quantum_energy_density: float, latent_info_rate: float):
        """
        Evolui as grandezas cosmológicas por um passo de tempo dt.
        
        Args:
            quantum_energy_density: Densidade de energia do campo quântico ρ_campo
            latent_info_rate: Taxa atual de informação latente Φ_latent(t)
        """
        # 1. Armazena valor atual para cálculo de derivadas
        self.previous_hubble = self.current_hubble
        
        # 2. Calcula densidade de energia da constante cosmológica
        lambda_energy_density = self.current_lambda * self.c**4 / (8 * np.pi * self.G)
        
        # 3. Atualiza variáveis de gravitação quântica
        self._update_quantum_gravity(quantum_energy_density)
        
        # 4. Calcula correções quânticas
        quantum_corrections = self._calculate_quantum_corrections(quantum_energy_density)
        
        # Proteção: Se a densidade de energia quântica for NaN ou infinita, substitua por um valor padrão pequeno
        if np.isnan(quantum_energy_density) or np.isinf(quantum_energy_density):
            logger.warning(f"Valor inválido detectado para densidade de energia quântica: {quantum_energy_density}")
            quantum_energy_density = 1.0e-10  # Valor pequeno positivo 
        
        # 5. Atualiza H(t) com base na equação de Friedmann expandida
        matter_contribution = self.matter_density * self.critical_density
        total_energy_density = (
            matter_contribution + 
            quantum_energy_density + 
            lambda_energy_density + 
            quantum_corrections
        )
        
        # Proteção: Garante que a densidade total seja sempre positiva
        if total_energy_density <= 0:
            logger.warning(f"Densidade de energia total negativa detectada: {total_energy_density}")
            logger.warning(f"Componentes: matéria={matter_contribution}, quântico={quantum_energy_density}, lambda={lambda_energy_density}, correções={quantum_corrections}")
            # Ajusta para um valor mínimo positivo
            total_energy_density = 1.0e-10  # Valor pequeno positivo
        
        # H²(t) = (8πG/3) * ρ_total
        self.current_hubble = np.sqrt(8 * np.pi * self.G * total_energy_density / 3)
        
        # 6. Calcula Ḣ(t) usando diferenças finitas
        hubble_derivative = (self.current_hubble - self.previous_hubble) / self.dt
        
        # 7. Atualiza a integral da informação latente multidimensional
        # Calcula a contribuição de cada dimensão
        latent_contribution = np.sum(self.latent_tensor * latent_info_rate)
        # Proteção: limita o valor da informação latente para evitar divergências
        bounded_latent_rate = np.clip(latent_contribution, -1.0e-35, 1.0e-35)
        self.latent_info_integral += bounded_latent_rate * self.dt
        
        # 8. Atualiza Λ(t) com base na equação dinâmica expandida
        # Λ(t) = Λ_0 - β * [Ḣ(t)/H³(t)] + ∫₀ᵗ Φ_latent(τ) dτ + correções quânticas
        retrocausal_term = 0.0
        if abs(self.current_hubble) > 1e-10:  # Evita divisão por zero
            retrocausal_term = self.beta * hubble_derivative / (self.current_hubble**3)
            # Limita o termo retrocausal para evitar instabilidades
            retrocausal_term = np.clip(retrocausal_term, -1.0e-35, 1.0e-35)
        
        self.current_lambda = (
            self.Lambda0 - 
            retrocausal_term + 
            self.latent_info_integral +
            quantum_corrections * self.quantum_gravity_coupling
        )
        
        # Garante que lambda não se torne negativo (evita violações de energia)
        if self.current_lambda < 0:
            logger.warning(f"Lambda negativo detectado: {self.current_lambda}")
            self.current_lambda = 1.0e-40  # Valor pequeno positivo
        
        # 9. Atualiza o fator de escala a(t) com base em H(t)
        # a(t+dt) = a(t) * exp(H(t) * dt)
        current_time = self.history['times'][-1] + self.dt
        current_scale = self.history['scale_factor'][-1] * np.exp(self.current_hubble * self.dt)
        
        # 10. Atualiza histórico expandido
        self.history['times'].append(current_time)
        self.history['hubble'].append(self.current_hubble)
        self.history['lambda'].append(self.current_lambda)
        self.history['scale_factor'].append(current_scale)
        self.history['latent_info_integral'].append(self.latent_info_integral)
        self.history['quantum_gravity_potential'].append(self.quantum_gravity_potential)
        self.history['spacetime_curvature'].append(self.spacetime_curvature)
        self.history['quantum_fluctuations'].append(self.vacuum_fluctuations)
        
        return self.current_lambda
    
    def get_current_values(self) -> Dict[str, float]:
        """Retorna os valores atuais das grandezas cosmológicas."""
        return {
            'hubble': self.current_hubble,
            'lambda': self.current_lambda,
            'scale_factor': self.history['scale_factor'][-1],
            'time': self.history['times'][-1]
        }
    
    def calculate_energy_density_from_lambda(self) -> float:
        """Calcula a densidade de energia associada à constante cosmológica."""
        return self.current_lambda * self.c**4 / (8 * np.pi * self.G)
    
    def reset(self):
        """Reinicia a simulação para os valores iniciais."""
        self.current_hubble = self.H0
        self.current_lambda = self.Lambda0
        self.previous_hubble = self.H0
        self.latent_info_integral = 0.0
        
        self.history = {
            'times': [0.0],
            'hubble': [self.current_hubble],
            'lambda': [self.current_lambda],
            'scale_factor': [1.0],
            'latent_info_integral': [0.0],
            'quantum_gravity_potential': [0.0],
            'spacetime_curvature': [0.0],
            'quantum_fluctuations': [0.0]
        }
        
        self._setup_quantum_gravity()
        self._setup_latent_info()
        
        logger.info("Simulador cosmológico reiniciado para valores iniciais.")
