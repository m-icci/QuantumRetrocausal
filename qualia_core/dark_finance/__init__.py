"""
Dark Finance Module
-----------------
Análise de dark pools e fluxos ocultos de liquidez
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
from collections import deque

@dataclass
class DarkLiquidityMetrics:
    """Métricas de liquidez dark"""
    timestamp: datetime
    liquidity: float  # Volume em dark pools
    flow: float      # Direção do fluxo (-1 a 1)
    confidence: float  # Confiança da análise
    reliability: float  # Confiabilidade dos dados
    
class DarkPoolAnalyzer:
    """Analisa dark pools e fluxos ocultos"""
    
    def __init__(self,
                 window_size: int = 100,
                 phi_weight: float = 0.618):
        """
        Inicializa analisador
        
        Args:
            window_size: Tamanho da janela de análise
            phi_weight: Peso da razão áurea
        """
        self.window_size = window_size
        self.phi = (1 + np.sqrt(5)) / 2
        self.phi_weight = phi_weight
        
        # Buffers
        self.liquidity_buffer = deque(maxlen=window_size)
        self.flow_buffer = deque(maxlen=window_size)
        
    def analyze(self, market_data: Dict[str, float]) -> DarkLiquidityMetrics:
        """
        Analisa dados de mercado
        
        Args:
            market_data: Dados de mercado
            
        Returns:
            Métricas de liquidez dark
        """
        # Extrai volume dark (estimado)
        dark_volume = self._estimate_dark_volume(market_data)
        self.liquidity_buffer.append(dark_volume)
        
        # Analisa fluxo
        flow = self._analyze_flow()
        self.flow_buffer.append(flow)
        
        # Calcula métricas
        metrics = DarkLiquidityMetrics(
            timestamp=datetime.now(),
            liquidity=np.mean(self.liquidity_buffer),
            flow=flow,
            confidence=self._calculate_confidence(),
            reliability=self._calculate_reliability()
        )
        
        return metrics
        
    def _estimate_dark_volume(self, market_data: Dict[str, float]) -> float:
        """
        Estima volume em dark pools
        
        Args:
            market_data: Dados de mercado
            
        Returns:
            Volume estimado
        """
        # Volume total
        total_volume = market_data.get('volume', 0)
        
        # Estima volume dark usando razão áurea
        dark_ratio = 1 - (1 / self.phi)
        dark_volume = total_volume * dark_ratio
        
        return dark_volume
        
    def _analyze_flow(self) -> float:
        """
        Analisa direção do fluxo
        
        Returns:
            Direção do fluxo (-1 a 1)
        """
        if len(self.liquidity_buffer) < 2:
            return 0.0
            
        # Calcula variação
        delta = np.diff(list(self.liquidity_buffer))
        
        # Normaliza
        flow = np.tanh(np.mean(delta))
        
        return flow
        
    def _calculate_confidence(self) -> float:
        """
        Calcula confiança da análise
        
        Returns:
            Nível de confiança (0-1)
        """
        if len(self.liquidity_buffer) < self.window_size:
            return 0.5
            
        # Usa desvio padrão normalizado
        std = np.std(self.liquidity_buffer)
        mean = np.mean(self.liquidity_buffer)
        
        if mean == 0:
            return 0.5
            
        cv = std / mean
        confidence = 1 / (1 + cv)
        
        return confidence
        
    def _calculate_reliability(self) -> float:
        """
        Calcula confiabilidade dos dados
        
        Returns:
            Nível de confiabilidade (0-1)
        """
        if len(self.liquidity_buffer) < self.window_size:
            return len(self.liquidity_buffer) / self.window_size
            
        return 1.0

def analyze_dark_flow(
    dark_metrics: DarkLiquidityMetrics,
    flow_buffer: deque,
    phi: float = 1.618) -> Dict[str, float]:
    """
    Analisa fluxo em dark pools
    
    Args:
        dark_metrics: Métricas atuais
        flow_buffer: Histórico de fluxo
        phi: Razão áurea
        
    Returns:
        Análise do fluxo
    """
    flow_list = list(flow_buffer)
    
    if len(flow_list) < 2:
        return {
            'trend': 0.0,
            'momentum': 0.0,
            'reversal': 0.0
        }
        
    # Analisa tendência
    trend = np.mean(flow_list)
    
    # Analisa momentum
    momentum = flow_list[-1] - flow_list[0]
    
    # Analisa reversão
    mean = np.mean(flow_list)
    std = np.std(flow_list)
    
    if std == 0:
        reversal = 0.0
    else:
        z_score = (flow_list[-1] - mean) / std
        reversal = -np.tanh(z_score)  # Negativo indica reversão
        
    return {
        'trend': trend,
        'momentum': momentum,
        'reversal': reversal
    }

def detect_dark_patterns(
    dark_metrics: DarkLiquidityMetrics,
    flow_buffer: deque,
    phi: float = 1.618) -> Dict[str, float]:
    """
    Detecta padrões em dark pools
    
    Args:
        dark_metrics: Métricas atuais
        flow_buffer: Histórico de fluxo
        phi: Razão áurea
        
    Returns:
        Padrões detectados
    """
    # Analisa fluxo
    flow = analyze_dark_flow(dark_metrics, flow_buffer, phi)
    
    # Detecta acumulação
    accumulation = np.clip(
        (flow['trend'] + flow['momentum']) * dark_metrics.confidence,
        0, 1
    )
    
    # Detecta distribuição
    distribution = np.clip(
        (-flow['trend'] - flow['momentum']) * dark_metrics.confidence,
        0, 1
    )
    
    # Detecta manipulação
    manipulation = 1 - dark_metrics.reliability
    
    return {
        'accumulation': accumulation,
        'distribution': distribution,
        'manipulation': manipulation
    }
