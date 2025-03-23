"""
Sistema QUALIAS com percepção quântica de mercado
Implementa análise avançada de padrões de mercado usando princípios quânticos
"""

import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from .qualia_system import QualiaMind, QualiaState
from ..logging.quantum_logger import quantum_logger

@dataclass
class MarketQualiaState(QualiaState):
    """Estado quântico do mercado"""
    market_coherence: float    # Coerência do mercado
    price_momentum: float      # Momento quântico dos preços
    volume_resonance: float    # Ressonância dos volumes
    pattern_emergence: float   # Taxa de emergência de padrões
    market_entropy: float      # Entropia do mercado
    quantum_trend: float       # Tendência quântica

class QualiaMarketPerception:
    """
    Sistema QUALIAS para percepção quântica do mercado
    Implementa análise avançada de padrões usando princípios quânticos
    """
    
    def __init__(
        self,
        consciousness_level: float = 0.8,
        perception_depth: int = 64,
        qualia_sensitivity: float = 0.618,
        temporal_resolution: int = 12
    ):
        self.consciousness = QualiaMind(dimensions=perception_depth)
        self.perception_depth = perception_depth
        self.sensitivity = qualia_sensitivity
        self.temporal_resolution = temporal_resolution
        self.market_memory: List[MarketQualiaState] = []
        self.quantum_patterns: Dict[str, np.ndarray] = {}
        
    def perceive_market(
        self,
        price_data: np.ndarray,
        volume_data: np.ndarray,
        timestamp: datetime = None
    ) -> MarketQualiaState:
        """
        Analisa o mercado usando percepção quântica
        
        Args:
            price_data: Série temporal de preços
            volume_data: Série temporal de volumes
            timestamp: Momento da análise
        
        Returns:
            Estado quântico do mercado
        """
        try:
            # Normaliza dados
            norm_prices = self._normalize_data(price_data)
            norm_volumes = self._normalize_data(volume_data)
            
            # Análise quântica básica
            base_state = self.consciousness.measure_market_consciousness(
                norm_prices,
                norm_volumes,
                timestamp or datetime.now()
            )
            
            # Calcula métricas quânticas avançadas
            market_metrics = self._calculate_quantum_metrics(norm_prices, norm_volumes)
            
            # Gera estado quântico do mercado
            market_state = MarketQualiaState(
                consciousness_level=base_state.consciousness_level,
                coherence=base_state.coherence,
                entanglement=base_state.entanglement,
                field_strength=base_state.field_strength,
                timestamp=base_state.timestamp,
                market_coherence=market_metrics['coherence'],
                price_momentum=market_metrics['momentum'],
                volume_resonance=market_metrics['resonance'],
                pattern_emergence=market_metrics['emergence'],
                market_entropy=market_metrics['entropy'],
                quantum_trend=market_metrics['trend']
            )
            
            # Atualiza memória quântica
            self._update_market_memory(market_state)
            
            # Atualiza padrões quânticos
            self._update_quantum_patterns(norm_prices, norm_volumes)
            
            return market_state
            
        except Exception as e:
            quantum_logger.error(f"Erro na percepção quântica: {str(e)}")
            return self._get_default_state()
            
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normaliza dados preservando características quânticas"""
        if len(data) < 2:
            return np.zeros(self.perception_depth)
            
        # Remove outliers
        mean = np.mean(data)
        std = np.std(data)
        clean_data = data[np.abs(data - mean) <= 3 * std]
        
        if len(clean_data) < 2:
            return np.zeros(self.perception_depth)
            
        # Normaliza para [-1, 1]
        normalized = (clean_data - np.mean(clean_data)) / np.std(clean_data)
        
        # Ajusta dimensão
        if len(normalized) > self.perception_depth:
            return normalized[-self.perception_depth:]
        elif len(normalized) < self.perception_depth:
            padding = np.zeros(self.perception_depth - len(normalized))
            return np.concatenate([padding, normalized])
        return normalized
        
    def _calculate_quantum_metrics(
        self,
        prices: np.ndarray,
        volumes: np.ndarray
    ) -> Dict[str, float]:
        """Calcula métricas quânticas avançadas do mercado"""
        try:
            # Análise de Fourier para coerência
            price_fft = np.fft.fft(prices)
            vol_fft = np.fft.fft(volumes)
            
            coherence = np.abs(np.mean(price_fft * np.conj(vol_fft)))
            coherence = min(0.99, max(0.01, coherence))
            
            # Momento quântico dos preços
            momentum = np.mean(np.diff(prices)) * self.sensitivity
            
            # Ressonância dos volumes
            resonance = np.corrcoef(prices, volumes)[0,1]
            resonance = min(0.99, max(0.01, (resonance + 1) / 2))
            
            # Taxa de emergência de padrões
            emergence = self._calculate_emergence(prices)
            
            # Entropia do mercado
            entropy = self._calculate_entropy(prices, volumes)
            
            # Tendência quântica
            trend = self._calculate_quantum_trend(prices)
            
            return {
                'coherence': float(coherence),
                'momentum': float(momentum),
                'resonance': float(resonance),
                'emergence': float(emergence),
                'entropy': float(entropy),
                'trend': float(trend)
            }
            
        except Exception as e:
            quantum_logger.error(f"Erro calculando métricas quânticas: {str(e)}")
            return {
                'coherence': 0.5,
                'momentum': 0.0,
                'resonance': 0.5,
                'emergence': 0.3,
                'entropy': 0.5,
                'trend': 0.0
            }
            
    def _calculate_emergence(self, data: np.ndarray) -> float:
        """Calcula taxa de emergência de padrões"""
        try:
            # Análise de auto-correlação
            acf = np.correlate(data, data, mode='full') / len(data)
            center = len(acf) // 2
            acf = acf[center:]
            
            # Identifica picos
            peaks = (acf[1:-1] > acf[:-2]) & (acf[1:-1] > acf[2:])
            emergence = np.sum(peaks) / len(peaks) if len(peaks) > 0 else 0.3
            
            return min(0.99, max(0.01, emergence))
            
        except:
            return 0.3
            
    def _calculate_entropy(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """Calcula entropia do mercado"""
        try:
            # Combina preços e volumes
            combined = np.column_stack([prices, volumes])
            
            # Calcula matriz de covariância
            cov = np.cov(combined.T)
            
            # Entropia baseada em autovalores
            eigenvals = np.linalg.eigvals(cov)
            eigenvals = eigenvals[eigenvals > 0]
            entropy = -np.sum(eigenvals * np.log(eigenvals))
            
            # Normaliza
            return min(0.99, max(0.01, entropy / (2 * np.log(2))))
            
        except:
            return 0.5
            
    def _calculate_quantum_trend(self, prices: np.ndarray) -> float:
        """Calcula tendência quântica"""
        try:
            # Diferenças de preços
            returns = np.diff(prices)
            
            # Média móvel exponencial
            alpha = 0.1
            ema = 0
            for r in returns:
                ema = alpha * r + (1 - alpha) * ema
                
            # Normaliza para [-1, 1]
            trend = np.tanh(ema * self.sensitivity)
            
            return float(trend)
            
        except:
            return 0.0
            
    def _update_market_memory(self, state: MarketQualiaState):
        """Atualiza memória do mercado"""
        self.market_memory.append(state)
        
        # Mantém tamanho máximo
        max_memory = 1000
        if len(self.market_memory) > max_memory:
            self.market_memory = self.market_memory[-max_memory:]
            
    def _update_quantum_patterns(self, prices: np.ndarray, volumes: np.ndarray):
        """Atualiza padrões quânticos identificados"""
        try:
            # Análise wavelet para identificar padrões
            pattern = np.column_stack([prices, volumes])
            
            # Atualiza dicionário de padrões
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.quantum_patterns[timestamp] = pattern
            
            # Limita número de padrões
            max_patterns = 100
            if len(self.quantum_patterns) > max_patterns:
                oldest = min(self.quantum_patterns.keys())
                del self.quantum_patterns[oldest]
                
        except Exception as e:
            quantum_logger.error(f"Erro atualizando padrões: {str(e)}")
            
    def _get_default_state(self) -> MarketQualiaState:
        """Retorna estado default seguro"""
        return MarketQualiaState(
            consciousness_level=0.5,
            coherence=0.5,
            entanglement=0.3,
            field_strength=0.3,
            timestamp=datetime.now(),
            market_coherence=0.5,
            price_momentum=0.0,
            volume_resonance=0.5,
            pattern_emergence=0.3,
            market_entropy=0.5,
            quantum_trend=0.0
        )
