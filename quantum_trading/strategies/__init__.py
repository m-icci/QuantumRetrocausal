"""
Módulo de estratégias para o projeto QUALIA.

Este módulo fornece diferentes estratégias de negociação, incluindo estratégias
quânticas, de tendência, reversão à média, volatilidade e arbitragem adaptativa.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import json
import os
from abc import ABC, abstractmethod
import logging

from ..exceptions import StrategyError

logger = logging.getLogger(__name__)

# Lista de estratégias disponíveis
available_strategies = [
    "quantum",
    "trend",
    "mean_reversion",
    "volatility",
    "wave",
    "quantum_scalping",
    "phi_pattern",
    "retrocausal_arbitrage"
]

class BaseStrategy(ABC):
    """Classe base para todas as estratégias."""
    
    def __init__(self):
        """Inicializa a estratégia base."""
        self.state = {}
        self.indicators = {}
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Gera sinais de negociação.
        
        Args:
            data: Dados de mercado.
            
        Returns:
            Série com sinais de negociação.
        """
        pass
    
    def get_state(self) -> Dict[str, Any]:
        """
        Retorna o estado atual da estratégia.
        
        Returns:
            Estado da estratégia.
        """
        return self.state
    
    def save_state(self, filepath: str) -> None:
        """
        Salva o estado da estratégia em arquivo.
        
        Args:
            filepath: Caminho do arquivo para salvar.
        """
        with open(filepath, "w") as f:
            json.dump(self.state, f, indent=2)
    
    def load_state(self, filepath: str) -> None:
        """
        Carrega o estado da estratégia de arquivo.
        
        Args:
            filepath: Caminho do arquivo para carregar.
        """
        with open(filepath, "r") as f:
            self.state = json.load(f)

class QuantumStrategy(BaseStrategy):
    """Estratégia baseada em computação quântica."""
    
    def __init__(self, config: Dict[str, Any], window_size: int = 20):
        """
        Inicializa a estratégia quântica.
        
        Args:
            config: Configuração quântica.
            window_size: Tamanho da janela para análise.
        """
        super().__init__()
        self.config = config
        self.window_size = window_size
        self.state = {
            "quantum_state": None,
            "circuit_params": None
        }
    
    def train(self, data: pd.DataFrame) -> None:
        """
        Treina a estratégia com dados históricos.
        
        Args:
            data: Dados históricos para treinamento.
        """
        # Implementa treinamento quântico
        # Este é um placeholder - a implementação real dependerá do framework quântico usado
        self.state["quantum_state"] = np.random.rand(2**self.config["num_qubits"])
        self.state["circuit_params"] = np.random.rand(self.config["num_layers"] * 3)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Gera sinais de negociação usando o estado quântico.
        
        Args:
            data: Dados de mercado.
            
        Returns:
            Série com sinais de negociação.
        """
        if self.state["quantum_state"] is None:
            raise StrategyError("Estratégia não treinada")
        
        # Calcula indicadores
        close = data["close"]
        returns = close.pct_change()
        volatility = returns.rolling(window=self.window_size).std()
        
        # Usa estado quântico para gerar sinais
        # Este é um placeholder - a implementação real dependerá do framework quântico usado
        signals = pd.Series(0, index=data.index)
        for i in range(len(data)):
            if i < self.window_size:
                continue
            
            # Simula medição quântica
            measurement = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
            
            # Ajusta sinal baseado em volatilidade
            if abs(volatility.iloc[i]) > 0.02:  # Alta volatilidade
                signals.iloc[i] = measurement
            else:  # Baixa volatilidade
                signals.iloc[i] = 0
        
        return signals

class TrendStrategy(BaseStrategy):
    """Estratégia baseada em tendência."""
    
    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 50,
        momentum_period: int = 14
    ):
        """
        Inicializa a estratégia de tendência.
        
        Args:
            short_window: Janela curta para média móvel.
            long_window: Janela longa para média móvel.
            momentum_period: Período para cálculo de momentum.
        """
        super().__init__()
        if short_window >= long_window:
            raise StrategyError("Janela curta deve ser menor que janela longa")
        
        self.short_window = short_window
        self.long_window = long_window
        self.momentum_period = momentum_period
        self.indicators = {}
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Gera sinais de negociação baseados em tendência.
        
        Args:
            data: Dados de mercado.
            
        Returns:
            Série com sinais de negociação.
        """
        if data is None:
            raise StrategyError("Dados não podem ser nulos")
        
        # Calcula médias móveis
        close = data["close"]
        sma_short = close.rolling(window=self.short_window).mean()
        sma_long = close.rolling(window=self.long_window).mean()
        
        # Calcula momentum
        momentum = close.pct_change(periods=self.momentum_period)
        
        # Armazena indicadores
        self.indicators = {
            "sma_short": sma_short,
            "sma_long": sma_long,
            "momentum": momentum
        }
        
        # Gera sinais
        signals = pd.Series(0, index=data.index)
        
        # Sinais baseados em cruzamento de médias
        signals[sma_short > sma_long] = 1
        signals[sma_short < sma_long] = -1
        
        # Ajusta sinais baseado em momentum
        signals[momentum < -0.02] = -1  # Momentum negativo forte
        signals[momentum > 0.02] = 1    # Momentum positivo forte
        
        return signals
    
    def get_indicators(self) -> Dict[str, pd.Series]:
        """
        Retorna os indicadores calculados.
        
        Returns:
            Dicionário com indicadores.
        """
        return self.indicators

class MeanReversionStrategy(BaseStrategy):
    """Estratégia baseada em reversão à média."""
    
    def __init__(
        self,
        window: int = 20,
        std_dev: float = 2.0,
        rsi_period: int = 14
    ):
        """
        Inicializa a estratégia de reversão à média.
        
        Args:
            window: Janela para cálculo de média e desvio padrão.
            std_dev: Número de desvios padrão para bandas.
            rsi_period: Período para cálculo do RSI.
        """
        super().__init__()
        self.window = window
        self.std_dev = std_dev
        self.rsi_period = rsi_period
        self.bands = {}
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Gera sinais de negociação baseados em reversão à média.
        
        Args:
            data: Dados de mercado.
            
        Returns:
            Série com sinais de negociação.
        """
        if data is None:
            raise StrategyError("Dados não podem ser nulos")
        
        # Calcula médias e desvios
        close = data["close"]
        middle = close.rolling(window=self.window).mean()
        std = close.rolling(window=self.window).std()
        
        # Calcula bandas
        upper = middle + self.std_dev * std
        lower = middle - self.std_dev * std
        
        # Calcula RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Armazena bandas
        self.bands = {
            "upper": upper,
            "lower": lower,
            "middle": middle
        }
        
        # Gera sinais
        signals = pd.Series(0, index=data.index)
        
        # Sinais baseados em bandas
        signals[close > upper] = -1  # Vende quando preço está acima da banda superior
        signals[close < lower] = 1   # Compra quando preço está abaixo da banda inferior
        
        # Ajusta sinais baseado em RSI
        signals[rsi > 70] = -1  # Vende quando RSI está sobrecomprado
        signals[rsi < 30] = 1   # Compra quando RSI está sobrevendido
        
        return signals
    
    def get_bands(self) -> Dict[str, pd.Series]:
        """
        Retorna as bandas calculadas.
        
        Returns:
            Dicionário com bandas.
        """
        return self.bands

class VolatilityStrategy(BaseStrategy):
    """Estratégia baseada em volatilidade."""
    
    def __init__(
        self,
        atr_period: int = 14,
        volatility_window: int = 20,
        threshold: float = 1.5
    ):
        """
        Inicializa a estratégia de volatilidade.
        
        Args:
            atr_period: Período para cálculo do ATR.
            volatility_window: Janela para cálculo de volatilidade.
            threshold: Limiar para sinais de volatilidade.
        """
        super().__init__()
        self.atr_period = atr_period
        self.volatility_window = volatility_window
        self.threshold = threshold
        self.metrics = {}
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Gera sinais de negociação baseados em volatilidade.
        
        Args:
            data: Dados de mercado.
            
        Returns:
            Série com sinais de negociação.
        """
        if data is None:
            raise StrategyError("Dados não podem ser nulos")
        
        # Calcula ATR
        high = data["high"]
        low = data["low"]
        close = data["close"]
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean()
        
        # Calcula volatilidade histórica
        returns = close.pct_change()
        historical_vol = returns.rolling(window=self.volatility_window).std() * np.sqrt(252)
        
        # Calcula razão de volatilidade
        vol_ratio = historical_vol / historical_vol.rolling(window=self.volatility_window).mean()
        
        # Armazena métricas
        self.metrics = {
            "atr": atr,
            "historical_volatility": historical_vol,
            "volatility_ratio": vol_ratio
        }
        
        # Gera sinais
        signals = pd.Series(0, index=data.index)
        
        # Sinais baseados em volatilidade
        signals[vol_ratio > self.threshold] = -1  # Vende em alta volatilidade
        signals[vol_ratio < 1/self.threshold] = 1  # Compra em baixa volatilidade
        
        # Ajusta sinais baseado em ATR
        signals[atr > atr.rolling(window=self.atr_period).mean() * 1.5] = -1
        signals[atr < atr.rolling(window=self.atr_period).mean() * 0.5] = 1
        
        return signals
    
    def get_metrics(self) -> Dict[str, pd.Series]:
        """
        Retorna as métricas calculadas.
        
        Returns:
            Dicionário com métricas.
        """
        return self.metrics

class StrategyFactory:
    """Fábrica para criar diferentes tipos de estratégias."""
    
    @staticmethod
    def create(strategy_type: str, **kwargs) -> BaseStrategy:
        """
        Cria uma instância de estratégia.
        
        Args:
            strategy_type: Tipo de estratégia a criar.
            **kwargs: Argumentos para a estratégia.
            
        Returns:
            Instância da estratégia.
        """
        if strategy_type == "quantum":
            return QuantumStrategy(**kwargs)
        elif strategy_type == "trend":
            return TrendStrategy(**kwargs)
        elif strategy_type == "mean_reversion":
            return MeanReversionStrategy(**kwargs)
        elif strategy_type == "volatility":
            return VolatilityStrategy(**kwargs)
        else:
            raise StrategyError(f"Tipo de estratégia não suportado: {strategy_type}")

# Importar a estratégia WAVE
from .wave_strategy import WAVEStrategy

# Mapeamento de estratégias
strategy_mapping = {
    "quantum": QuantumStrategy,
    "trend": TrendStrategy,
    "mean_reversion": None,  # Placeholder para implementação futura
    "volatility": None,      # Placeholder para implementação futura
    "wave": WAVEStrategy
}

def get_strategy(strategy_name: str, **kwargs) -> BaseStrategy:
    """
    Retorna uma instância da estratégia solicitada.
    
    Args:
        strategy_name: Nome da estratégia.
        **kwargs: Parâmetros adicionais para a estratégia.
        
    Returns:
        Instância da estratégia.
        
    Raises:
        StrategyError: Se a estratégia não existe ou não está implementada.
    """
    if strategy_name not in strategy_mapping:
        raise StrategyError(f"Estratégia {strategy_name} não encontrada")
    
    strategy_class = strategy_mapping[strategy_name]
    
    if strategy_class is None:
        raise StrategyError(f"Estratégia {strategy_name} não implementada")
    
    return strategy_class(**kwargs)

__all__ = [
    'BaseStrategy',
    'QuantumStrategy',
    'TrendStrategy',
    'MeanReversionStrategy',
    'VolatilityStrategy',
    'StrategyFactory',
    'get_strategy'
]

def get_strategy_from_name(name: str, config: Dict[str, Any] = None) -> BaseStrategy:
    """
    Retorna uma instância da estratégia solicitada
    
    Args:
        name: Nome da estratégia
        config: Configurações da estratégia
        
    Returns:
        Instância da estratégia
    """
    if config is None:
        config = {}
        
    if name not in available_strategies:
        logger.warning(f"Strategy {name} not found")
        return None
        
    try:
        if name == 'quantum_scalping':
            from .quantum_scalping import QuantumScalpingStrategy
            return QuantumScalpingStrategy(config)
        elif name == 'wave_strategy':
            from .wave_strategy import WaveStrategy
            return WaveStrategy(config)
        elif name == 'phi_pattern':
            from .phi_pattern import PhiPatternStrategy
            return PhiPatternStrategy(config)
        elif name == 'retrocausal_arbitrage':
            from .retrocausal_arbitrage import RetrocausalArbitrageStrategy
            return RetrocausalArbitrageStrategy(config)
    except ImportError as e:
        logger.error(f"Error importing strategy {name}: {str(e)}")
        return None
        
    return None 