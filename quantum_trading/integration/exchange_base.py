#!/usr/bin/env python3
"""
Exchange Base
=============
Classe base para todos os adaptadores de exchange, fornecendo
uma interface comum para interação com diferentes exchanges.
"""

import os
import json
import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

import ccxt
import ccxt.async_support as ccxt_async

logger = logging.getLogger("exchange_base")

class ExchangeBase:
    """Classe base para adaptadores de exchange que utiliza a biblioteca ccxt."""
    
    def __init__(self, exchange_id: str, config_path: str = "exchange_config.json"):
        """
        Inicializa o adaptador de exchange.
        
        Args:
            exchange_id: ID da exchange (e.g., 'kraken', 'kucoin')
            config_path: Caminho para o arquivo de configuração das exchanges
        """
        self.exchange_id = exchange_id.lower()
        self.config_path = config_path
        self.config = self._load_config()
        
        # Extrai configurações específicas da exchange
        exchange_config = self.config.get("exchanges", {}).get(self.exchange_id, {})
        global_settings = self.config.get("global_settings", {})
        
        if not exchange_config:
            raise ValueError(f"Configuração não encontrada para a exchange {exchange_id}")
        
        if not exchange_config.get("enabled", False):
            raise ValueError(f"Exchange {exchange_id} está desabilitada na configuração")
        
        # Combina configurações globais e específicas
        self.settings = {**global_settings, **exchange_config}
        
        # Inicializa instância ccxt
        self.exchange = self._initialize_exchange()
        self.async_exchange = self._initialize_async_exchange()
        
        logger.info(f"Exchange {exchange_id} inicializada.")
    
    def _load_config(self) -> Dict[str, Any]:
        """Carrega a configuração das exchanges do arquivo."""
        try:
            config_path = Path(self.config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Arquivo de configuração não encontrado: {self.config_path}")
            
            with open(config_path, 'r') as f:
                config = json.load(f)
                logger.debug(f"Configuração carregada de {self.config_path}")
                return config
        except Exception as e:
            logger.error(f"Erro ao carregar configuração: {e}")
            raise
    
    def _initialize_exchange(self) -> ccxt.Exchange:
        """Inicializa a instância sync da exchange ccxt."""
        exchange_class = getattr(ccxt, self.exchange_id)
        return exchange_class(self.settings)
    
    def _initialize_async_exchange(self) -> ccxt_async.Exchange:
        """Inicializa a instância async da exchange ccxt."""
        exchange_class = getattr(ccxt_async, self.exchange_id)
        return exchange_class(self.settings)
    
    def get_balance(self, asset: str = 'USDT') -> float:
        """
        Obtém o saldo disponível de um ativo.
        
        Args:
            asset: Símbolo do ativo a verificar
            
        Returns:
            Saldo disponível
        """
        try:
            balance = self.exchange.fetch_balance()
            asset = asset.upper()
            return float(balance.get(asset, {}).get('free', 0))
        except Exception as e:
            logger.error(f"Erro ao obter saldo de {asset}: {e}")
            return 0.0
    
    async def get_balance_async(self, asset: str = 'USDT') -> float:
        """Versão assíncrona de get_balance."""
        try:
            balance = await self.async_exchange.fetch_balance()
            asset = asset.upper()
            return float(balance.get(asset, {}).get('free', 0))
        except Exception as e:
            logger.error(f"Erro ao obter saldo de {asset} (async): {e}")
            return 0.0
    
    def get_price(self, pair: str) -> float:
        """
        Obtém o preço atual de um par.
        
        Args:
            pair: Par de trading (e.g., 'BTC/USDT')
            
        Returns:
            Preço atual
        """
        try:
            ticker = self.exchange.fetch_ticker(pair)
            return float(ticker['last'])
        except Exception as e:
            logger.error(f"Erro ao obter preço de {pair}: {e}")
            return 0.0
    
    async def get_price_async(self, pair: str) -> float:
        """Versão assíncrona de get_price."""
        try:
            ticker = await self.async_exchange.fetch_ticker(pair)
            return float(ticker['last'])
        except Exception as e:
            logger.error(f"Erro ao obter preço de {pair} (async): {e}")
            return 0.0
    
    def get_ohlcv(self, pair: str, timeframe: str = '1h', limit: int = 100) -> List[List[Any]]:
        """
        Obtém dados OHLCV para um par.
        
        Args:
            pair: Par de trading (e.g., 'BTC/USDT')
            timeframe: Intervalo de tempo (e.g., '1m', '1h', '1d')
            limit: Número de candles a obter
            
        Returns:
            Lista de candles [timestamp, open, high, low, close, volume]
        """
        try:
            return self.exchange.fetch_ohlcv(pair, timeframe, limit=limit)
        except Exception as e:
            logger.error(f"Erro ao obter OHLCV para {pair}: {e}")
            return []
    
    async def get_ohlcv_async(self, pair: str, timeframe: str = '1h', limit: int = 100) -> List[List[Any]]:
        """Versão assíncrona de get_ohlcv."""
        try:
            return await self.async_exchange.fetch_ohlcv(pair, timeframe, limit=limit)
        except Exception as e:
            logger.error(f"Erro ao obter OHLCV para {pair} (async): {e}")
            return []
    
    def get_24h_volume(self, pair: str) -> float:
        """
        Obtém o volume de 24h para um par.
        
        Args:
            pair: Par de trading (e.g., 'BTC/USDT')
            
        Returns:
            Volume de 24h
        """
        try:
            ticker = self.exchange.fetch_ticker(pair)
            return float(ticker['quoteVolume'])
        except Exception as e:
            logger.error(f"Erro ao obter volume de 24h para {pair}: {e}")
            return 0.0
    
    def create_market_buy_order(self, pair: str, amount: float) -> Dict[str, Any]:
        """
        Cria uma ordem de compra a mercado.
        
        Args:
            pair: Par de trading (e.g., 'BTC/USDT')
            amount: Quantidade a comprar
            
        Returns:
            Detalhes da ordem
        """
        try:
            return self.exchange.create_market_buy_order(pair, amount)
        except Exception as e:
            logger.error(f"Erro ao criar ordem de compra para {pair}: {e}")
            return {"error": str(e)}
    
    def create_market_sell_order(self, pair: str, amount: float) -> Dict[str, Any]:
        """
        Cria uma ordem de venda a mercado.
        
        Args:
            pair: Par de trading (e.g., 'BTC/USDT')
            amount: Quantidade a vender
            
        Returns:
            Detalhes da ordem
        """
        try:
            return self.exchange.create_market_sell_order(pair, amount)
        except Exception as e:
            logger.error(f"Erro ao criar ordem de venda para {pair}: {e}")
            return {"error": str(e)}
    
    async def close_async(self):
        """Fecha a conexão assíncrona."""
        await self.async_exchange.close()
    
    def __del__(self):
        """Destructor para garantir que recursos são liberados."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.close_async())
            else:
                asyncio.run(self.close_async())
        except Exception:
            pass 