#!/usr/bin/env python3
"""
HelixAnalyzer: Módulo responsável pela análise e evolução do campo da hélice quântica.
Implementa a evolução do campo quântico e extrai métricas fractais relevantes.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from qualia_core.helix_analysis.helix_config import HelixConfig

logger = logging.getLogger("helix_analyzer")

class HelixAnalyzer:
    """
    Analisador que evolui o campo da hélice quântica e extrai métricas fractais.
    Implementa a dinâmica quântica e a análise de padrões emergentes.
    """
    
    def __init__(self, config: HelixConfig):
        """
        Inicializa o analisador da hélice.
        
        Args:
            config: Configuração da hélice
        """
        self.config = config
        self.helix_field = None
        self.quantum_state = None
        self.field_history = []
        self.metrics_history = []
        
        # Constantes internas
        self._phi = self.config.phi  # Proporção áurea
        self._iterations = 0
        
        logger.info(f"HelixAnalyzer inicializado com dimensões: {self.config.dimensions}, qubits: {self.config.num_qubits}")
    
    def initialize_helix(self) -> None:
        """
        Inicializa o campo da hélice e o estado quântico.
        Cria estrutura quântica inicial com padrões fractais.
        """
        # Criar campo da hélice com dimensões adequadas
        dimensions = self.config.dimensions
        field_size = min(self.config.max_field_size, dimensions * 2)
        
        # Inicializar campo com padrão fractal usando a sequência de Fibonacci
        self.helix_field = np.zeros((field_size, field_size), dtype=np.complex128)
        
        # Criar um padrão fractal inicial
        center = field_size // 2
        for i in range(field_size):
            for j in range(field_size):
                # Distância do centro usando norma L1 (Manhattan)
                dist = abs(i - center) + abs(j - center)
                # Valores iniciais baseados na sequência de Fibonacci módulo phi
                val = np.exp(1j * 2 * np.pi * (dist * self._phi % 1.0))
                self.helix_field[i, j] = val
        
        # Normalizar o campo
        self.helix_field /= np.sqrt(np.sum(np.abs(self.helix_field)**2))
        
        # Inicializar estado quântico (espaço de dimensão 2^num_qubits)
        qubit_dim = 2**self.config.num_qubits
        self.quantum_state = np.zeros((qubit_dim, qubit_dim), dtype=np.complex128)
        
        # Inicializar com estado de superposição
        hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        state = np.array([1, 0], dtype=np.complex128)  # |0⟩
        
        # Aplicar Hadamard em cada qubit para criar superposição
        for _ in range(self.config.num_qubits):
            state = np.kron(state, hadamard @ np.array([1, 0]))
        
        # Remodelar para matriz densidade
        state = state.reshape(-1, 1)
        self.quantum_state = np.outer(state, state.conj())
        
        self._iterations = 0
        self.field_history = [self.helix_field.copy()]
        
        logger.info("Campo da hélice inicializado com padrão fractal")
    
    def evolve_helix(self, steps: int = 1) -> Dict[str, Any]:
        """
        Evolui o campo da hélice por um número especificado de passos.
        Implementa a evolução quântica e extrai métricas a cada passo.
        
        Args:
            steps: Número de passos para evolução
            
        Returns:
            Dicionário com métricas da evolução
        """
        if self.helix_field is None:
            logger.warning("Campo da hélice não inicializado. Inicializando agora.")
            self.initialize_helix()
        
        evolution_metrics = {
            'timestamp': datetime.now().isoformat(),
            'initial_step': self._iterations,
            'final_step': self._iterations + steps,
            'fractal_analysis': []
        }
        
        for _ in range(steps):
            # Aplicar operador de evolução no campo da hélice
            self._evolve_field_step()
            
            # Extrair métricas fractais
            fractal_metrics = self._analyze_fractal_pattern()
            evolution_metrics['fractal_analysis'].append(fractal_metrics)
            
            # Atualizar estado quântico com informações do campo
            self._update_quantum_state()
            
            # Incrementar contador de iterações
            self._iterations += 1
            
            # Armazenar histórico (limitado pelo tamanho do batch)
            if len(self.field_history) >= self.config.batch_size:
                self.field_history.pop(0)
            self.field_history.append(self.helix_field.copy())
        
        logger.debug(f"Campo da hélice evoluído por {steps} passos, iteração atual: {self._iterations}")
        return evolution_metrics
    
    def visualize_field(self) -> np.ndarray:
        """
        Gera uma representação visual do campo da hélice.
        
        Returns:
            Array 2D para visualização
        """
        if self.helix_field is None:
            logger.warning("Campo da hélice não inicializado para visualização")
            return np.zeros((10, 10))
        
        # Obter magnitude do campo para visualização
        magnitude = np.abs(self.helix_field)
        phase = np.angle(self.helix_field) / (2 * np.pi) + 0.5  # Normalizado entre 0 e 1
        
        # Criar visualização colorida (magnitude * fase)
        visualization = magnitude * phase
        
        # Normalizar para visualização
        if np.max(visualization) > 0:
            visualization = visualization / np.max(visualization)
        
        return visualization
    
    def reset(self) -> None:
        """Reseta o analisador da hélice para estado inicial"""
        self.initialize_helix()
        self.metrics_history = []
        logger.info("HelixAnalyzer resetado para estado inicial")
    
    def _evolve_field_step(self) -> None:
        """Aplica um passo de evolução no campo da hélice"""
        # Obter tamanho do campo
        field_size = self.helix_field.shape[0]
        
        # Criar cópia do campo para atualização
        new_field = np.zeros_like(self.helix_field)
        
        # Aplicar regra de evolução baseada em autômatos celulares quânticos
        for i in range(field_size):
            for j in range(field_size):
                # Obter células vizinhas (com condições de contorno periódicas)
                neighbors = [
                    self.helix_field[(i-1) % field_size, j],
                    self.helix_field[(i+1) % field_size, j],
                    self.helix_field[i, (j-1) % field_size],
                    self.helix_field[i, (j+1) % field_size]
                ]
                
                # Calcular média dos vizinhos
                avg = sum(neighbors) / 4.0
                
                # Aplicar regra de evolução não-linear
                # Usar a proporção áurea para criar comportamento quasi-periódico
                phase_shift = 2 * np.pi * self._phi * self._iterations
                evolution_factor = np.exp(1j * phase_shift)
                
                # Atualizar campo combinando valor atual e média dos vizinhos
                current = self.helix_field[i, j]
                new_val = (current + avg * evolution_factor) / (1 + np.abs(avg))
                
                # Adicionar flutuação quântica (ruído térmico)
                if self.config.temperature > 0:
                    noise = np.random.normal(0, self.config.temperature, 2)
                    noise_complex = noise[0] + 1j * noise[1]
                    new_val += noise_complex * self.config.temperature
                
                new_field[i, j] = new_val
        
        # Normalizar o novo campo
        new_field /= np.sqrt(np.sum(np.abs(new_field)**2))
        
        # Atualizar o campo da hélice
        self.helix_field = new_field
    
    def _analyze_fractal_pattern(self) -> Dict[str, float]:
        """
        Analisa padrões fractais no campo da hélice.
        
        Returns:
            Métricas fractais extraídas
        """
        if self.helix_field is None:
            return {}
        
        # Calcular energia total do campo
        field_energy = np.sum(np.abs(self.helix_field)**2)
                
        # Calcular distribuição de frequências usando FFT 2D
        fft = np.fft.fft2(self.helix_field)
        fft_shifted = np.fft.fftshift(fft)
        fft_magnitude = np.abs(fft_shifted)
        
        # Calcular espectro de potência
        power_spectrum = fft_magnitude**2
        
        # Calcular dimensão fractal usando método box-counting aproximado
        # Aplicar limiar para binarizar o campo
        threshold = np.mean(np.abs(self.helix_field))
        binary_field = np.abs(self.helix_field) > threshold
        
        # Contar caixas em diferentes escalas
        scales = [2, 4, 8, 16]
        counts = []
        
        for scale in scales:
            if scale < min(binary_field.shape):
                boxcount = 0
                for i in range(0, binary_field.shape[0], scale):
                    for j in range(0, binary_field.shape[1], scale):
                        i_max = min(i + scale, binary_field.shape[0])
                        j_max = min(j + scale, binary_field.shape[1])
                        if np.any(binary_field[i:i_max, j:j_max]):
                            boxcount += 1
                counts.append(boxcount)
        
        # Estimar dimensão fractal se tivermos contagens suficientes
        fractal_factor = 1.0
        if len(counts) > 2:
            # Usar regressão linear para estimar dimensão fractal
            log_scales = np.log([s for s in scales[:len(counts)]])
            log_counts = np.log(counts)
            
            # Calcular inclinação (dimensão fractal)
            try:
                coef = np.polyfit(log_scales, log_counts, 1)
                fractal_factor = -coef[0]  # Negativo da inclinação
            except:
                # Em caso de erro, usar valor padrão
                fractal_factor = 1.0
        
        # Calcular entropia de Shannon do espectro de potência normalizado
        if np.sum(power_spectrum) > 0:
            p = power_spectrum / np.sum(power_spectrum)
            entropy = -np.sum(p * np.log2(p + 1e-10))
        else:
            entropy = 0
        
        # Calcular lambda coupling (medida de acoplamento entre escalas)
        lambda_coupling = self._calculate_lambda_coupling()
        
        # Calcular dimensão de correlação (aproximação)
        correlation_dim = self._calculate_correlation_dimension()
        
        metrics = {
            'step': self._iterations,
            'field_energy': float(field_energy),
                    'fractal_factor': float(fractal_factor),
            'entropy': float(entropy),
                    'lambda_coupling': float(lambda_coupling),
            'correlation_dimension': float(correlation_dim)
        }
        
        return metrics
    
    def _update_quantum_state(self) -> None:
        """Atualiza o estado quântico com base no campo da hélice"""
        if self.helix_field is None or self.quantum_state is None:
            return
        
        # Extrair características relevantes do campo
        field_flattened = self.helix_field.flatten()
        field_size = len(field_flattened)
        quantum_size = self.quantum_state.shape[0]
        
        # Redimensionar campo para corresponder ao estado quântico
        if field_size > quantum_size**2:
            # Selecionar características mais importantes (centro do campo)
            center = field_size // 2
            radius = int(np.sqrt(quantum_size**2 / 4))
            field_size_1d = int(np.sqrt(field_size))
            center_2d = field_size_1d // 2
            
            # Extrair região central
            selected_indices = []
            for i in range(max(0, center_2d - radius), min(field_size_1d, center_2d + radius)):
                for j in range(max(0, center_2d - radius), min(field_size_1d, center_2d + radius)):
                    idx = i * field_size_1d + j
                    if idx < field_size:
                        selected_indices.append(idx)
            
            # Limitar ao tamanho do estado quântico
            selected_indices = selected_indices[:quantum_size**2]
            selected_values = field_flattened[selected_indices]
        else:
            # Repetir valores para preencher o estado quântico
            repeats = int(np.ceil(quantum_size**2 / field_size))
            selected_values = np.tile(field_flattened, repeats)[:quantum_size**2]
        
        # Remodelar para matriz
        selected_matrix = selected_values.reshape(quantum_size, quantum_size)
        
        # Remodelar preservando a hermiticidade
        new_state = (selected_matrix + selected_matrix.conj().T) / 2
        
        # Normalizar traço
        if np.trace(new_state) > 0:
            new_state = new_state / np.trace(new_state)
        
        # Aplicar evolução unitária com o novo estado
        # Mesclar gradualmente para preservar coerência
        alpha = 0.2  # Taxa de mistura
        self.quantum_state = (1 - alpha) * self.quantum_state + alpha * new_state
        
        # Renormalizar
        if np.trace(self.quantum_state) > 0:
            self.quantum_state = self.quantum_state / np.trace(self.quantum_state)
    
    def _calculate_lambda_coupling(self) -> float:
        """
        Calcula o acoplamento lambda entre diferentes escalas no campo.
        
        Returns:
            Valor do acoplamento lambda
        """
        if len(self.field_history) < 2:
            return 0.0
        
        # Calcular correlação entre campo atual e anterior
        prev_field = self.field_history[-2]
        current_field = self.field_history[-1]
        
        # Correlação normalizada
        corr_num = np.sum(np.abs(prev_field * current_field.conj()))
        corr_denom = np.sqrt(np.sum(np.abs(prev_field)**2) * np.sum(np.abs(current_field)**2))
        
        if corr_denom > 0:
            correlation = corr_num / corr_denom
        else:
            correlation = 0
        
        # Calcular espectros de potência
        fft_prev = np.fft.fft2(prev_field)
        fft_curr = np.fft.fft2(current_field)
        
        power_prev = np.abs(fft_prev)**2
        power_curr = np.abs(fft_curr)**2
        
        # Dividir em bandas de frequência
        bands = 4
        rows, cols = power_prev.shape
        band_power_prev = []
        band_power_curr = []
        
        for b in range(bands):
            r_start = int(rows * b / bands)
            r_end = int(rows * (b + 1) / bands)
            c_start = int(cols * b / bands)
            c_end = int(cols * (b + 1) / bands)
            
            band_prev = np.sum(power_prev[r_start:r_end, c_start:c_end])
            band_curr = np.sum(power_curr[r_start:r_end, c_start:c_end])
            
            band_power_prev.append(band_prev)
            band_power_curr.append(band_curr)
        
        # Calcular correlação entre bandas
        if np.sum(band_power_prev) > 0 and np.sum(band_power_curr) > 0:
            band_power_prev = np.array(band_power_prev) / np.sum(band_power_prev)
            band_power_curr = np.array(band_power_curr) / np.sum(band_power_curr)
            
            # Calcular divergência KL como medida de transferência de informação
            kl_div = np.sum(band_power_curr * np.log((band_power_curr + 1e-10) / (band_power_prev + 1e-10)))
            
            # Lambda coupling combina correlação temporal e transferência entre escalas
            lambda_coupling = correlation * (1 - np.tanh(kl_div))
        else:
            lambda_coupling = 0
        
        return abs(lambda_coupling)
    
    def _calculate_correlation_dimension(self) -> float:
        """
        Calcula a dimensão de correlação do campo.
        
        Returns:
            Dimensão de correlação
        """
        history_len = len(self.field_history)
        if history_len < 3:
            return 1.0
        
        # Usar apenas os últimos campos (limitado por desempenho)
        max_samples = min(10, history_len)
        recent_fields = self.field_history[-max_samples:]
        
        # Achatar os campos para vetores
        flattened_fields = [field.flatten() for field in recent_fields]
        
        # Calcular matriz de distâncias
        num_fields = len(flattened_fields)
        distances = np.zeros((num_fields, num_fields))
        
        for i in range(num_fields):
            for j in range(i+1, num_fields):
                # Distância euclidiana entre campos
                dist = np.sqrt(np.sum(np.abs(flattened_fields[i] - flattened_fields[j])**2))
                distances[i, j] = distances[j, i] = dist
        
        # Calcular função de correlação para diferentes raios
        radii = np.linspace(0.01, 1.0, 10) * np.max(distances)
        correlations = []
        
        for r in radii:
            # Contar pares com distância menor que r
            count = np.sum(distances < r) - num_fields  # Excluir diagonais
            correlation = count / (num_fields * (num_fields - 1))
            correlations.append(correlation)
        
        # Estimar dimensão de correlação por regressão
        valid_indices = [i for i, c in enumerate(correlations) if c > 0]
        if len(valid_indices) > 1:
            valid_radii = np.log(radii[valid_indices])
            valid_correlations = np.log([correlations[i] for i in valid_indices])
            
            try:
                coef = np.polyfit(valid_radii, valid_correlations, 1)
                correlation_dim = coef[0]  # Inclinação
            except:
                correlation_dim = 1.0
        else:
            correlation_dim = 1.0
        
        return max(0.5, min(3.0, abs(correlation_dim))) 