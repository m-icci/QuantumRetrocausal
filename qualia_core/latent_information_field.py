#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Campo de Informação Latente para o Simulador Cosmológico-Quântico

Implementa o fluxo de informação latente Φ_latent(t) que influencia
cumulativamente a constante cosmológica Λ(t) através da integral:

∫₀ᵗ Φ_latent(τ) dτ

Este fluxo pode ser interpretado como uma influência "externa" ou 
emergente que retroalimenta o sistema quântico-cosmológico.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Callable, Union

# Configuração de logging
logger = logging.getLogger("QUALIA-COSMOQUANTUM.latent_info")

class LatentInformationField:
    """
    Gerador do fluxo de informação latente que influencia a constante cosmológica.
    
    Implementa diversos modelos para Φ_latent(t), incluindo:
    - Oscilações harmônicas
    - Ruído quântico
    - Padrões fractais
    - Dinâmica caótica
    - Feedback baseado na entropia do campo quântico
    """
    
    def __init__(self, 
                 base_amplitude: float = 1.0e-38,  # Amplitude base
                 oscillation_period: float = 10.0,  # Período base para oscilações
                 noise_level: float = 0.1,  # Nível de ruído
                 mode: str = 'mixed',  # Modo de geração (oscillatory, noise, fractal, chaotic, mixed)
                 dt: float = 0.01):
        """
        Inicializa o gerador de informação latente.
        
        Args:
            base_amplitude: Amplitude base do fluxo
            oscillation_period: Período base para modos oscilatórios
            noise_level: Nível relativo de ruído estocástico
            mode: Tipo de padrão para geração (oscillatory, noise, fractal, chaotic, mixed)
            dt: Passo de tempo para evolução
        """
        self.amplitude = base_amplitude
        self.period = oscillation_period
        self.noise_level = noise_level
        self.mode = mode
        self.dt = dt
        
        # Parâmetros para modos específicos
        self.phase = 0.0  # Fase atual para oscilações
        self.chaotic_state = np.random.rand(3)  # Estado inicial para dinâmica caótica
        
        # Parâmetros para modelo de Lorenz (caótico)
        self.lorenz_sigma = 10.0
        self.lorenz_rho = 28.0
        self.lorenz_beta = 8.0/3.0
        
        # Valores atuais e histórico
        self.current_value = self._generate_initial_value()
        
        self.history = {
            'times': [0.0],
            'values': [self.current_value],
            'accumulated': [0.0]
        }
        
        # Integral acumulada
        self.accumulated_value = 0.0
        
        logger.info(f"Gerador de informação latente inicializado: modo={mode}, amplitude={base_amplitude}")
    
    def _generate_initial_value(self) -> float:
        """Gera o valor inicial com base no modo selecionado."""
        if self.mode == 'oscillatory':
            return self.amplitude * np.sin(self.phase)
        elif self.mode == 'noise':
            return self.amplitude * (2 * np.random.rand() - 1) * self.noise_level
        elif self.mode == 'fractal':
            # Implementação simplificada de ruído 1/f (rosa)
            return self._generate_fractal_noise()
        elif self.mode == 'chaotic':
            # Usa o primeiro componente do sistema de Lorenz
            return self.amplitude * self.chaotic_state[0] / 30.0  # Normalizado
        elif self.mode == 'mixed':
            # Combinação de oscilação e ruído
            oscillation = self.amplitude * np.sin(self.phase)
            noise = self.amplitude * (2 * np.random.rand() - 1) * self.noise_level
            return oscillation + noise
        else:
            # Modo padrão: oscilação simples
            return self.amplitude * np.sin(self.phase)
    
    def evolve_step(self, quantum_entropy: float = 0.0) -> float:
        """
        Evolui o fluxo de informação latente por um passo de tempo dt.
        
        Args:
            quantum_entropy: Entropia atual do campo quântico, para feedback
            
        Returns:
            Valor atual do fluxo Φ_latent(t)
        """
        # Evolução depende do modo selecionado
        if self.mode == 'oscillatory':
            self._evolve_oscillatory(quantum_entropy)
        elif self.mode == 'noise':
            self._evolve_noise(quantum_entropy)
        elif self.mode == 'fractal':
            self._evolve_fractal(quantum_entropy)
        elif self.mode == 'chaotic':
            self._evolve_chaotic(quantum_entropy)
        elif self.mode == 'mixed':
            self._evolve_mixed(quantum_entropy)
        else:
            self._evolve_oscillatory(quantum_entropy)
        
        # Atualiza integral acumulada
        self.accumulated_value += self.current_value * self.dt
        
        # Atualiza histórico
        current_time = self.history['times'][-1] + self.dt
        self.history['times'].append(current_time)
        self.history['values'].append(self.current_value)
        self.history['accumulated'].append(self.accumulated_value)
        
        return self.current_value
    
    def _evolve_oscillatory(self, quantum_entropy: float):
        """Evolui modo oscilatório."""
        # Atualiza fase
        self.phase += 2 * np.pi * self.dt / self.period
        
        # Permite que a entropia do campo quântico module a amplitude
        entropy_factor = 1.0 + 0.2 * quantum_entropy  # Modulação de 20%
        
        # Atualiza valor atual
        self.current_value = self.amplitude * entropy_factor * np.sin(self.phase)
    
    def _evolve_noise(self, quantum_entropy: float):
        """Evolui modo de ruído."""
        # Ruído branco simples
        base_noise = (2 * np.random.rand() - 1)
        
        # Modula amplitude do ruído pela entropia quântica
        entropy_factor = 1.0 + 0.2 * quantum_entropy
        
        self.current_value = self.amplitude * entropy_factor * base_noise * self.noise_level
    
    def _evolve_fractal(self, quantum_entropy: float):
        """Evolui modo fractal (ruído 1/f)."""
        # Implementação simplificada de ruído 1/f
        self.current_value = self._generate_fractal_noise()
        
        # Modula pela entropia
        entropy_factor = 1.0 + 0.2 * quantum_entropy
        self.current_value *= entropy_factor
    
    def _generate_fractal_noise(self) -> float:
        """Gera ruído com características fractais (1/f)."""
        # Implementação simplificada - para simulação real usaríamos
        # transformada de Fourier com espectro 1/f
        
        # Esta é uma aproximação simples
        num_octaves = 4
        noise_value = 0.0
        amplitude = 1.0
        frequency = 1.0
        
        for _ in range(num_octaves):
            noise_value += amplitude * (2 * np.random.rand() - 1)
            amplitude *= 0.5
            frequency *= 2.0
            
        return self.amplitude * noise_value * self.noise_level
    
    def _evolve_chaotic(self, quantum_entropy: float):
        """Evolui modo caótico usando sistema de Lorenz."""
        # Sistema dinâmico de Lorenz (caótico)
        x, y, z = self.chaotic_state
        
        # Derivadas do sistema de Lorenz
        dx = self.lorenz_sigma * (y - x)
        dy = x * (self.lorenz_rho - z) - y
        dz = x * y - self.lorenz_beta * z
        
        # Modula o sistema pela entropia quântica
        entropy_factor = 1.0 + 0.1 * quantum_entropy
        dx *= entropy_factor
        dy *= entropy_factor
        dz *= entropy_factor
        
        # Atualiza o estado
        x += dx * self.dt
        y += dy * self.dt
        z += dz * self.dt
        
        self.chaotic_state = np.array([x, y, z])
        
        # Normaliza para ficar na escala apropriada
        self.current_value = self.amplitude * x / 30.0
    
    def _evolve_mixed(self, quantum_entropy: float):
        """Evolui modo misto (combinação de oscilação e ruído)."""
        # Atualiza fase
        self.phase += 2 * np.pi * self.dt / self.period
        
        # Componente oscilatória
        oscillation = np.sin(self.phase)
        
        # Componente de ruído
        noise = (2 * np.random.rand() - 1) * self.noise_level
        
        # Modula pela entropia quântica
        entropy_factor = 1.0 + 0.2 * quantum_entropy
        
        # Combina os componentes
        self.current_value = self.amplitude * entropy_factor * (oscillation + noise)
    
    def get_accumulated_value(self) -> float:
        """Retorna o valor acumulado da integral ∫₀ᵗ Φ_latent(τ) dτ."""
        return self.accumulated_value
    
    def reset(self):
        """Reinicia o gerador para o estado inicial."""
        self.phase = 0.0
        self.chaotic_state = np.random.rand(3)
        self.current_value = self._generate_initial_value()
        self.accumulated_value = 0.0
        
        self.history = {
            'times': [0.0],
            'values': [self.current_value],
            'accumulated': [0.0]
        }
        
        logger.info("Gerador de informação latente reiniciado para valores iniciais.")
