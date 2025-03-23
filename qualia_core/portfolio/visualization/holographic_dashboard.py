"""
Holographic Dashboard Module
--------------------------
Visualização holográfica de padrões quânticos e campos mórficos.
"""

import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional

from quantum.core.portfolio.sacred.geometry import SacredGeometry
from quantum.core.portfolio.sacred.dark_integration import DarkPortfolioIntegrator
from quantum.core.portfolio.sacred.retrocausal_trading import RetrocausalTrader
from quantum.core.portfolio.exploration.market_explorer import MarketExplorer

@dataclass
class VisualizationMetrics:
    """Métricas de visualização holográfica"""
    pattern_strength: float  # Força do padrão
    field_coherence: float  # Coerência do campo
    market_potential: float  # Potencial de mercado
    timestamp: datetime = datetime.now()

class HolographicDashboard:
    """
    Dashboard holográfico para visualização de padrões quânticos.
    Segue o mantra: Investigar, Integrar, Inovar
    """
    
    def __init__(self,
                 dimensions: int = 8,
                 markets: List[str] = ["genomics", "climate", "emerging"]):
        """
        Inicializa dashboard
        
        Args:
            dimensions: Dimensões do espaço quântico
            markets: Lista de mercados para análise
        """
        self.dimensions = dimensions
        self.markets = markets
        self.sacred_geometry = SacredGeometry(dimensions)
        self.dark_integrator = DarkPortfolioIntegrator(dimensions)
        self.retrocausal_trader = RetrocausalTrader(dimensions)
        self.market_explorer = MarketExplorer(dimensions, markets)
        self.metrics_history: Dict[str, List[VisualizationMetrics]] = {
            m: [] for m in markets
        }
        
    def update_visualization(self,
                           market_data: Dict[str, np.ndarray],
                           quantum_state: Optional[QuantumState] = None) -> Dict[str, VisualizationMetrics]:
        """
        Atualiza visualização holográfica
        
        Args:
            market_data: Dados de mercado por tipo
            quantum_state: Estado quântico opcional
            
        Returns:
            Métricas de visualização por mercado
        """
        metrics = {}
        
        # Analisa padrões de mercado
        market_patterns = self.market_explorer.analyze_market(
            market_data, quantum_state
        )
        
        # Obtém recomendações
        recommendations = self.market_explorer.get_market_recommendations()
        
        for market in self.markets:
            if market not in market_data:
                continue
                
            pattern = market_patterns[market]
            weight = recommendations[market]
            
            viz_metrics = VisualizationMetrics(
                pattern_strength=pattern.strength,
                field_coherence=pattern.coherence,
                market_potential=weight
            )
            
            self.metrics_history[market].append(viz_metrics)
            metrics[market] = viz_metrics
            
        return metrics
        
    def get_portfolio_allocation(self) -> Dict[str, float]:
        """
        Obtém alocação recomendada de portfólio
        
        Returns:
            Dicionário com pesos por mercado
        """
        return self.market_explorer.get_market_recommendations()
        
    def get_market_metrics(self, market: str) -> List[VisualizationMetrics]:
        """
        Obtém histórico de métricas para um mercado
        
        Args:
            market: Nome do mercado
            
        Returns:
            Lista de métricas históricas
        """
        return self.metrics_history.get(market, [])
