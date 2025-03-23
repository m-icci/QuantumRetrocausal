"""Sacred Geometry para trading quântico"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
from .sacred_patterns import SacredPatternAnalyzer, SacredPatternMetrics

@dataclass
class GeometryMetrics:
    """Métricas da geometria sagrada"""
    phi_ratio: float = 1.618033988749895  # Proporção áurea
    fibonacci_level: float = 0.0
    harmonic_resonance: float = 0.0
    symmetry_factor: float = 0.0
    fractal_dimension: float = 0.0
    pattern_metrics: SacredPatternMetrics = None

class SacredGeometry:
    """Geometria sagrada para análise de mercado"""

    def __init__(self):
        """Inicializa geometria sagrada"""
        self.metrics = GeometryMetrics()
        self.pattern_analyzer = SacredPatternAnalyzer()

    def analyze_pattern(self, prices: np.ndarray) -> GeometryMetrics:
        """Analisa padrão de preços usando geometria sagrada"""
        if len(prices) < 2:
            return self.metrics

        # Calcula níveis de Fibonacci
        price_range = np.max(prices) - np.min(prices)
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        closest_level = min(fib_levels, key=lambda x: abs(
            (prices[-1] - np.min(prices)) / price_range - x
        ))
        self.metrics.fibonacci_level = closest_level

        # Calcula ressonância harmônica
        fft = np.fft.fft(prices)
        harmonics = np.abs(fft[1:len(fft)//2])
        if len(harmonics) > 0:
            self.metrics.harmonic_resonance = float(
                np.max(harmonics) / np.mean(harmonics)
            )

        # Calcula fator de simetria
        diffs = np.diff(prices)
        pos_moves = np.sum(diffs > 0)
        neg_moves = np.sum(diffs < 0)
        total_moves = pos_moves + neg_moves
        if total_moves > 0:
            self.metrics.symmetry_factor = float(
                min(pos_moves, neg_moves) / total_moves
            )

        # Calcula dimensão fractal
        if len(prices) > 2:
            # Método de contagem de caixas simplificado
            scales = np.array([2, 4, 8, 16])
            counts = []
            for scale in scales:
                if scale < len(prices):
                    boxes = len(prices) // scale
                    count = 0
                    for i in range(boxes):
                        box_prices = prices[i*scale:(i+1)*scale]
                        if np.max(box_prices) - np.min(box_prices) > 0:
                            count += 1
                    counts.append(count)

            if len(counts) > 1:
                scales = scales[:len(counts)]
                log_scales = np.log(1/scales)
                log_counts = np.log(counts)
                coeffs = np.polyfit(log_scales, log_counts, 1)
                self.metrics.fractal_dimension = float(coeffs[0])

        # Analisa padrões sagrados
        self.metrics.pattern_metrics = self.pattern_analyzer.analyze_pattern(
            prices,
            self.metrics
        )

        return self.metrics