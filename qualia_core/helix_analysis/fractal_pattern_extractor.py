"""
FractalPatternExtractor - Extrator de padrões fractais

Esta classe implementa a extração e análise de padrões fractais no campo da hélice,
utilizando transformadas wavelet e técnicas de decomposição.
"""

import numpy as np
import pywt
from typing import Dict, List, Optional
import logging
import gc

class FractalPatternExtractor:
    """Extrator de padrões fractais"""
    
    def __init__(self, max_signal_size: int = 1024):
        """
        Inicializa o extrator de padrões fractais
        
        Args:
            max_signal_size: Tamanho máximo do sinal para análise
        """
        self.logger = logging.getLogger(__name__)
        self.max_signal_size = max_signal_size
        self.scales = np.arange(1, 41, dtype=np.float32)
        
    def extract(self, signal: np.ndarray, threshold: float = 1.8) -> float:
        """
        Extrai o fator fractal do sinal usando transformada wavelet contínua
        
        Args:
            signal: Sinal temporal para análise
            threshold: Limiar para classificação de padrões fractais
            
        Returns:
            float: Fator fractal calculado
        """
        try:
            # Reduz a dimensão do sinal se necessário
            signal = self._reduce_signal_dimension(signal)
            
            # Garante que o sinal está em float32
            signal = signal.astype(np.float32)
            
            # Aplicação da transformada wavelet contínua
            coeffs, _ = pywt.cwt(signal, self.scales, 'mexh')
            
            # Cálculo das somas de coeficientes em diferentes escalas
            high_sum = np.sum(np.abs(coeffs[25:, :]))
            low_sum = np.sum(np.abs(coeffs[:10, :]))
            
            # Cálculo do fator fractal
            fractal_factor = high_sum / (low_sum + 1e-6)
            
            # Análise de padrões
            patterns = self._analyze_patterns(coeffs)
            
            # Limpeza de memória
            del coeffs
            gc.collect()
            
            self.logger.debug(f"Fator fractal calculado: {fractal_factor:.2f}")
            return fractal_factor
            
        except Exception as e:
            self.logger.error(f"Erro na extração de padrões fractais: {str(e)}")
            return 0.0
            
    def _analyze_patterns(self, coeffs: np.ndarray) -> Dict:
        """
        Analisa padrões específicos nos coeficientes wavelet
        
        Args:
            coeffs: Coeficientes da transformada wavelet
            
        Returns:
            Dict: Dicionário com informações sobre os padrões encontrados
        """
        # Análise de auto-similaridade
        self_similarity = self._calculate_self_similarity(coeffs)
        
        # Análise de complexidade
        complexity = self._calculate_complexity(coeffs)
        
        # Análise de ressonância
        resonance = self._calculate_resonance(coeffs)
        
        return {
            "self_similarity": self_similarity,
            "complexity": complexity,
            "resonance": resonance
        }
        
    def _calculate_self_similarity(self, coeffs: np.ndarray) -> float:
        """Calcula o grau de auto-similaridade nos coeficientes"""
        # Reduz a dimensão dos coeficientes para análise
        reduced_coeffs = self._reduce_signal_dimension(coeffs)
        return np.mean(np.corrcoef(reduced_coeffs))
        
    def _calculate_complexity(self, coeffs: np.ndarray) -> float:
        """Calcula a complexidade dos padrões"""
        # Reduz a dimensão dos coeficientes para análise
        reduced_coeffs = self._reduce_signal_dimension(coeffs)
        return -np.sum(np.abs(reduced_coeffs) * np.log2(np.abs(reduced_coeffs) + 1e-10))
        
    def _calculate_resonance(self, coeffs: np.ndarray) -> float:
        """Calcula o grau de ressonância nos padrões"""
        # Reduz a dimensão dos coeficientes para análise
        reduced_coeffs = self._reduce_signal_dimension(coeffs)
        return np.mean(np.abs(np.fft.fft(reduced_coeffs)))
        
    def _reduce_signal_dimension(self, signal: np.ndarray) -> np.ndarray:
        """Reduz a dimensão do sinal para análise"""
        if signal.size <= self.max_signal_size:
            return signal
            
        # Reduz dimensão mantendo a estrutura do sinal
        if len(signal.shape) == 1:
            indices = np.linspace(0, len(signal)-1, self.max_signal_size, dtype=int)
            return signal[indices]
        else:
            indices = np.linspace(0, signal.shape[0]-1, self.max_signal_size, dtype=int)
            return signal[indices, :] 