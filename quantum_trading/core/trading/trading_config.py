"""
Configuração de trading.
"""

import logging
import json
import os
from typing import Dict, Any, Optional
from datetime import datetime

class TradingConfig:
    """Configuração de trading."""
    
    def __init__(
        self,
        exchange: str = 'binance',
        api_key: str = '',
        api_secret: str = '',
        symbol: str = 'BTC/USDT',
        timeframe: str = '1h',
        leverage: int = 1,
        max_positions: int = 3,
        daily_trades_limit: int = 10,
        daily_loss_limit: float = 0.02,
        min_confidence: float = 0.7,
        position_size: float = 0.1,
        min_position_size: float = 0.01,
        max_position_size: float = 0.5,
        stop_loss: float = 0.02,
        take_profit: float = 0.04,
        risk_per_trade: float = 0.01,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        log_level: str = 'INFO',
        # Novos parâmetros de taxas
        maker_fee: float = 0.001,  # 0.1%
        taker_fee: float = 0.001,  # 0.1%
        withdrawal_fee: float = 0.0005,  # 0.05%
        min_trade_profit: float = 0.002  # 0.2% lucro mínimo após taxas
    ):
        """
        Inicializa configuração.
        
        Args:
            exchange: Exchange.
            api_key: Chave da API.
            api_secret: Segredo da API.
            symbol: Símbolo.
            timeframe: Timeframe.
            leverage: Alavancagem.
            max_positions: Máximo de posições.
            daily_trades_limit: Limite diário de trades.
            daily_loss_limit: Limite diário de perda.
            min_confidence: Confiança mínima.
            position_size: Tamanho da posição.
            min_position_size: Tamanho mínimo da posição.
            max_position_size: Tamanho máximo da posição.
            stop_loss: Stop loss.
            take_profit: Take profit.
            risk_per_trade: Risco por trade.
            rsi_period: Período do RSI.
            rsi_oversold: RSI sobrevendido.
            rsi_overbought: RSI sobrecomprado.
            macd_fast: MACD rápido.
            macd_slow: MACD lento.
            macd_signal: MACD sinal.
            bb_period: Período das Bandas de Bollinger.
            bb_std: Desvio padrão das Bandas de Bollinger.
            atr_period: Período do ATR.
            atr_multiplier: Multiplicador do ATR.
            log_level: Nível de logging.
            maker_fee: Taxa de maker (ordem limit).
            taker_fee: Taxa de taker (ordem market).
            withdrawal_fee: Taxa de retirada.
            min_trade_profit: Lucro mínimo após taxas.
        """
        self.logger = logging.getLogger('TradingConfig')
        
        # Exchange
        self.exchange = exchange
        self.exchange_url = self._get_exchange_url()
        self.api_key = api_key
        self.api_secret = api_secret
        
        # Trading
        self.symbol = symbol
        self.timeframe = timeframe
        self.leverage = leverage
        self.max_positions = max_positions
        self.daily_trades_limit = daily_trades_limit
        self.daily_loss_limit = daily_loss_limit
        self.min_confidence = min_confidence
        self.position_size = position_size
        self.min_position_size = min_position_size
        self.max_position_size = max_position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.risk_per_trade = risk_per_trade
        
        # Indicadores
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        
        # Taxas
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.withdrawal_fee = withdrawal_fee
        self.min_trade_profit = min_trade_profit
        
        # Logging
        self.log_level = log_level
        
        # Valida configuração
        self._validate()
    
    def _validate(self) -> None:
        """Valida configuração."""
        try:
            # Exchange
            if self.exchange not in ['binance', 'kucoin', 'kraken']:
                raise ValueError(f"Exchange não suportada: {self.exchange}")
            
            if not self.api_key or not self.api_secret:
                raise ValueError("Chaves da API não fornecidas")
            
            # Trading
            if not self.symbol:
                raise ValueError("Símbolo não fornecido")
            
            if not self.timeframe:
                raise ValueError("Timeframe não fornecido")
            
            if self.leverage < 1:
                raise ValueError("Alavancagem deve ser maior que 0")
            
            if self.max_positions < 1:
                raise ValueError("Máximo de posições deve ser maior que 0")
            
            if self.daily_trades_limit < 1:
                raise ValueError("Limite diário de trades deve ser maior que 0")
            
            # Tratando daily_loss_limit como percentual (0-1) ou valor absoluto (>1)
            if self.daily_loss_limit <= 0:
                raise ValueError("Limite diário de perda deve ser maior que 0")
            
            if self.min_confidence <= 0 or self.min_confidence >= 1:
                raise ValueError("Confiança mínima deve estar entre 0 e 1")
            
            if self.position_size <= 0:
                raise ValueError("Tamanho da posição deve ser maior que 0")
            
            if self.min_position_size <= 0:
                raise ValueError("Tamanho mínimo da posição deve ser maior que 0")
            
            if self.max_position_size <= 0:
                raise ValueError("Tamanho máximo da posição deve ser maior que 0")
            
            if self.stop_loss <= 0 or self.stop_loss >= 1:
                raise ValueError("Stop loss deve estar entre 0 e 1")
            
            if self.take_profit <= 0 or self.take_profit >= 1:
                raise ValueError("Take profit deve estar entre 0 e 1")
            
            if self.risk_per_trade <= 0 or self.risk_per_trade >= 1:
                raise ValueError("Risco por trade deve estar entre 0 e 1")
            
            # Indicadores
            if self.rsi_period < 1:
                raise ValueError("Período do RSI deve ser maior que 0")
            
            if self.rsi_oversold <= 0 or self.rsi_oversold >= 100:
                raise ValueError("RSI sobrevendido deve estar entre 0 e 100")
            
            if self.rsi_overbought <= 0 or self.rsi_overbought >= 100:
                raise ValueError("RSI sobrecomprado deve estar entre 0 e 100")
            
            if self.macd_fast < 1:
                raise ValueError("MACD rápido deve ser maior que 0")
            
            if self.macd_slow < 1:
                raise ValueError("MACD lento deve ser maior que 0")
            
            if self.macd_signal < 1:
                raise ValueError("MACD sinal deve ser maior que 0")
            
            if self.bb_period < 1:
                raise ValueError("Período das Bandas de Bollinger deve ser maior que 0")
            
            if self.bb_std <= 0:
                raise ValueError("Desvio padrão das Bandas de Bollinger deve ser maior que 0")
            
            if self.atr_period < 1:
                raise ValueError("Período do ATR deve ser maior que 0")
            
            if self.atr_multiplier <= 0:
                raise ValueError("Multiplicador do ATR deve ser maior que 0")
            
            # Taxas
            if self.maker_fee < 0 or self.maker_fee >= 1:
                raise ValueError("Taxa de maker deve estar entre 0 e 1")
            
            if self.taker_fee < 0 or self.taker_fee >= 1:
                raise ValueError("Taxa de taker deve estar entre 0 e 1")
            
            if self.withdrawal_fee < 0 or self.withdrawal_fee >= 1:
                raise ValueError("Taxa de retirada deve estar entre 0 e 1")
            
            if self.min_trade_profit <= 0 or self.min_trade_profit >= 1:
                raise ValueError("Lucro mínimo por trade deve estar entre 0 e 1")
            
        except Exception as e:
            self.logger.error(f"Erro ao validar configuração: {str(e)}")
            raise
    
    def _get_exchange_url(self) -> str:
        """
        Obtém URL da exchange.
        
        Returns:
            URL da exchange.
        """
        if self.exchange == 'binance':
            return 'https://api.binance.com'
        elif self.exchange == 'kucoin':
            return 'https://api.kucoin.com'
        elif self.exchange == 'kraken':
            return 'https://api.kraken.com/0'
        else:
            raise ValueError(f"Exchange não suportada: {self.exchange}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TradingConfig':
        """
        Cria configuração a partir de dicionário.
        
        Args:
            config_dict: Dicionário de configuração.
            
        Returns:
            Configuração.
        """
        return cls(**config_dict)
    
    @classmethod
    def from_file(cls, config_file: str) -> 'TradingConfig':
        """
        Cria configuração a partir de arquivo.
        
        Args:
            config_file: Arquivo de configuração.
            
        Returns:
            Configuração.
        """
        try:
            # Verifica arquivo
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"Arquivo não encontrado: {config_file}")
            
            # Lê arquivo
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
            
            # Cria configuração
            return cls.from_dict(config_dict)
            
        except Exception as e:
            logging.error(f"Erro ao carregar configuração: {str(e)}")
            raise
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converte para dicionário.
        
        Returns:
            Dicionário.
        """
        return {
            # Exchange
            'exchange': self.exchange,
            'api_key': self.api_key,
            'api_secret': self.api_secret,
            
            # Trading
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'leverage': self.leverage,
            'max_positions': self.max_positions,
            'daily_trades_limit': self.daily_trades_limit,
            'daily_loss_limit': self.daily_loss_limit,
            'min_confidence': self.min_confidence,
            'position_size': self.position_size,
            'min_position_size': self.min_position_size,
            'max_position_size': self.max_position_size,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'risk_per_trade': self.risk_per_trade,
            
            # Indicadores
            'rsi_period': self.rsi_period,
            'rsi_oversold': self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought,
            'macd_fast': self.macd_fast,
            'macd_slow': self.macd_slow,
            'macd_signal': self.macd_signal,
            'bb_period': self.bb_period,
            'bb_std': self.bb_std,
            'atr_period': self.atr_period,
            'atr_multiplier': self.atr_multiplier,
            
            # Taxas
            'maker_fee': self.maker_fee,
            'taker_fee': self.taker_fee,
            'withdrawal_fee': self.withdrawal_fee,
            'min_trade_profit': self.min_trade_profit,
            
            # Logging
            'log_level': self.log_level
        }
    
    def save(self, config_file: str) -> None:
        """
        Salva em arquivo.
        
        Args:
            config_file: Arquivo de configuração.
        """
        try:
            # Converte para dicionário
            config_dict = self.to_dict()
            
            # Salva arquivo
            with open(config_file, 'w') as f:
                json.dump(config_dict, f, indent=4)
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar configuração: {str(e)}")
            raise 