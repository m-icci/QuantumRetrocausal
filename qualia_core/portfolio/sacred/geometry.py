"""
Sacred Geometry for Quantum Portfolio Optimization
-----------------------------------------------
Implementa princípios de geometria sagrada para otimização de portfólio.
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class SacredPattern:
    """Padrão geométrico sagrado identificado"""
    name: str
    ratio: float
    resonance: float
    harmony: float
    timestamp: float = None

class SacredGeometry:
    """
    Implementa geometria sagrada para otimização de portfólio.
    Utiliza razão áurea e sequências de Fibonacci.
    """
    
    def __init__(self, dimensions: int = 8):
        """
        Inicializa geometria sagrada
        
        Args:
            dimensions: Dimensões do espaço geométrico
        """
        self.dimensions = dimensions
        self.phi = (1 + np.sqrt(5)) / 2
        self.fibonacci_cache = {0: 0, 1: 1}
        
    def generate_sacred_matrix(self, size: int) -> np.ndarray:
        """
        Gera matriz de transformação sagrada
        
        Args:
            size: Tamanho da matriz
            
        Returns:
            Matriz de transformação
        """
        # Gera sequência Fibonacci
        fib_sequence = [self.fibonacci(i) for i in range(size)]
        
        # Cria matriz base
        matrix = np.zeros((size, size))
        
        # Preenche com padrões sagrados
        for i in range(size):
            for j in range(size):
                matrix[i,j] = (
                    fib_sequence[(i+j) % size] * 
                    np.exp(1j * 2 * np.pi * self.phi * (i*j/size))
                )
                
        # Normaliza
        matrix = matrix / np.linalg.norm(matrix)
        return matrix
    
    def fibonacci(self, n: int) -> int:
        """Calcula n-ésimo número de Fibonacci com cache"""
        if n in self.fibonacci_cache:
            return self.fibonacci_cache[n]
            
        self.fibonacci_cache[n] = self.fibonacci(n-1) + self.fibonacci(n-2)
        return self.fibonacci_cache[n]
        
    def calculate_harmony(self, weights: np.ndarray) -> float:
        """
        Calcula harmonia geométrica dos pesos
        
        Args:
            weights: Pesos do portfólio
            
        Returns:
            Índice de harmonia
        """
        # Projeta pesos em espaço sagrado
        projected = np.exp(1j * 2 * np.pi * self.phi * weights)
        
        # Calcula harmonia como coerência da projeção
        harmony = np.abs(np.sum(projected)) / len(weights)
        return float(harmony)
        
    def identify_patterns(self, 
                         weights: np.ndarray,
                         returns: np.ndarray) -> List[SacredPattern]:
        """
        Identifica padrões sagrados nos pesos e retornos
        
        Args:
            weights: Pesos do portfólio
            returns: Série de retornos
            
        Returns:
            Lista de padrões identificados
        """
        patterns = []
        
        # Calcula razões entre componentes adjacentes
        ratios = weights[1:] / weights[:-1]
        
        # Busca padrões próximos a razão áurea
        phi_patterns = np.abs(ratios - self.phi) < 0.2  # Aumenta tolerância
        if np.any(phi_patterns):
            patterns.append(
                SacredPattern(
                    name="golden_ratio",
                    ratio=float(np.mean(ratios[phi_patterns])),
                    resonance=float(np.std(ratios[phi_patterns])),
                    harmony=self.calculate_harmony(weights)
                )
            )
            
        # Busca sequências Fibonacci
        fib_ratios = np.array([self.fibonacci(i+1)/self.fibonacci(i)
                              for i in range(2, len(ratios)+2)])
        fib_patterns = np.abs(ratios - fib_ratios) < 0.1
        if np.any(fib_patterns):
            patterns.append(
                SacredPattern(
                    name="fibonacci",
                    ratio=float(np.mean(ratios[fib_patterns])),
                    resonance=float(np.std(ratios[fib_patterns])),
                    harmony=self.calculate_harmony(weights)
                )
            )
            
        return patterns
        
    def optimize_harmony(self, 
                        weights: np.ndarray,
                        returns: np.ndarray,
                        iterations: int = 100) -> np.ndarray:
        """
        Otimiza pesos para máxima harmonia geométrica
        
        Args:
            weights: Pesos iniciais
            returns: Série de retornos
            iterations: Número de iterações
            
        Returns:
            Pesos otimizados
        """
        best_weights = weights.copy()
        best_harmony = self.calculate_harmony(weights)
        
        for _ in range(iterations):
            # Perturbação aleatória
            perturbed = weights + np.random.normal(0, 0.1, size=len(weights))
            perturbed = perturbed / np.sum(np.abs(perturbed))  # Normaliza
            
            # Avalia harmonia
            harmony = self.calculate_harmony(perturbed)
            if harmony > best_harmony:
                best_weights = perturbed
                best_harmony = harmony
                
        return best_weights
        
    def calculate_metrics(self, data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calcula métricas de geometria sagrada para dados de mercado
        
        Args:
            data: Dicionário com dados OHLCV
            
        Returns:
            Dicionário com métricas calculadas
        """
        # Extrai dados
        close = data['close']
        volume = data.get('volume', np.ones_like(close))
        
        # Calcula retornos
        returns = np.diff(close) / close[:-1]
        
        # Calcula pesos normalizados do volume
        weights = volume / np.sum(volume)
        
        # Calcula harmonia base
        harmony = self.calculate_harmony(weights)
        
        # Identifica padrões
        patterns = self.identify_patterns(weights, returns)
        
        # Calcula ressonância como média das ressonâncias dos padrões
        resonance = np.mean([p.resonance for p in patterns]) if patterns else 0.0
        
        # Calcula força do campo como produto de harmonia e ressonância
        field_strength = harmony * (1 + resonance)
        
        return {
            'harmony': float(harmony),
            'resonance': float(resonance),
            'field_strength': float(field_strength),
            'phi_ratio': float(self.phi)
        }
