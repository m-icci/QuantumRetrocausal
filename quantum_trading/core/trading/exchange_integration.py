"""
Integração com exchange.
"""

import logging
import asyncio
import aiohttp
import hmac
import hashlib
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from .trading_config import TradingConfig

class ExchangeIntegration:
    """Integração com exchange."""
    
    def __init__(self, config: TradingConfig):
        """
        Inicializa integração.
        
        Args:
            config: Configuração.
        """
        self.logger = logging.getLogger('ExchangeIntegration')
        
        # Configuração
        self.config = config
        
        # Sessão HTTP
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Caches
        self._orderbook_cache: Dict[str, Any] = {}
        self._trades_cache: Dict[str, Any] = {}
        self._candles_cache: Dict[str, Any] = {}
        self._ticker_cache: Dict[str, Any] = {}
        self._balance_cache: Dict[str, Any] = {}
        
        # TTL dos caches (segundos)
        self._orderbook_ttl = 1
        self._trades_ttl = 1
        self._candles_ttl = 60
        self._ticker_ttl = 1
        self._balance_ttl = 5
        
        # Controle
        self._running = False
        self._update_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Inicia integração."""
        try:
            # Cria sessão
            self._session = aiohttp.ClientSession()
            
            # Inicia loop de atualização
            self._running = True
            self._update_task = asyncio.create_task(self._update_loop())
            
            self.logger.info("Integração iniciada")
            
        except Exception as e:
            self.logger.error(f"Erro ao iniciar integração: {str(e)}")
            raise
    
    async def stop(self) -> None:
        """Para integração."""
        try:
            # Para loop de atualização
            self._running = False
            if self._update_task:
                await self._update_task
                self._update_task = None
            
            # Fecha sessão
            if self._session:
                await self._session.close()
                self._session = None
            
            self.logger.info("Integração parada")
            
        except Exception as e:
            self.logger.error(f"Erro ao parar integração: {str(e)}")
            raise
    
    async def _update_loop(self) -> None:
        """Loop de atualização."""
        try:
            while self._running:
                # Atualiza caches
                await self._update_caches()
                
                # Aguarda próximo ciclo
                await asyncio.sleep(1)
                
        except Exception as e:
            self.logger.error(f"Erro no loop de atualização: {str(e)}")
            raise
    
    async def _update_caches(self) -> None:
        """Atualiza caches."""
        try:
            # Atualiza orderbook
            if time.time() - self._orderbook_cache.get('timestamp', 0) >= self._orderbook_ttl:
                orderbook = await self._fetch_orderbook()
                if orderbook:
                    self._orderbook_cache = {
                        'data': orderbook,
                        'timestamp': time.time()
                    }
            
            # Atualiza trades
            if time.time() - self._trades_cache.get('timestamp', 0) >= self._trades_ttl:
                trades = await self._fetch_trades()
                if trades:
                    self._trades_cache = {
                        'data': trades,
                        'timestamp': time.time()
                    }
            
            # Atualiza candles
            if time.time() - self._candles_cache.get('timestamp', 0) >= self._candles_ttl:
                candles = await self._fetch_candles()
                if candles:
                    self._candles_cache = {
                        'data': candles,
                        'timestamp': time.time()
                    }
            
            # Atualiza ticker
            if time.time() - self._ticker_cache.get('timestamp', 0) >= self._ticker_ttl:
                ticker = await self._fetch_ticker()
                if ticker:
                    self._ticker_cache = {
                        'data': ticker,
                        'timestamp': time.time()
                    }
            
            # Atualiza balance
            if time.time() - self._balance_cache.get('timestamp', 0) >= self._balance_ttl:
                balance = await self._fetch_balance()
                if balance:
                    self._balance_cache = {
                        'data': balance,
                        'timestamp': time.time()
                    }
            
        except Exception as e:
            self.logger.error(f"Erro ao atualizar caches: {str(e)}")
    
    async def _fetch_orderbook(self) -> Optional[Dict[str, Any]]:
        """
        Busca orderbook.
        
        Returns:
            Orderbook.
        """
        try:
            # Prepara URL
            url = f"{self.config.exchange_url}/api/v3/depth"
            
            # Prepara parâmetros
            params = {
                'symbol': self.config.symbol,
                'limit': 100
            }
            
            # Faz requisição
            async with self._session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    self.logger.error(f"Erro ao buscar orderbook: {response.status}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Erro ao buscar orderbook: {str(e)}")
            return None
    
    async def _fetch_trades(self) -> Optional[List[Dict[str, Any]]]:
        """
        Busca trades.
        
        Returns:
            Trades.
        """
        try:
            # Prepara URL
            url = f"{self.config.exchange_url}/api/v3/trades"
            
            # Prepara parâmetros
            params = {
                'symbol': self.config.symbol,
                'limit': 100
            }
            
            # Faz requisição
            async with self._session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    self.logger.error(f"Erro ao buscar trades: {response.status}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Erro ao buscar trades: {str(e)}")
            return None
    
    async def _fetch_candles(self) -> Optional[List[Dict[str, Any]]]:
        """
        Busca candles.
        
        Returns:
            Candles.
        """
        try:
            # Prepara URL
            url = f"{self.config.exchange_url}/api/v3/klines"
            
            # Prepara parâmetros
            params = {
                'symbol': self.config.symbol,
                'interval': self.config.timeframe,
                'limit': 100
            }
            
            # Faz requisição
            async with self._session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return [
                        {
                            'timestamp': candle[0],
                            'open': float(candle[1]),
                            'high': float(candle[2]),
                            'low': float(candle[3]),
                            'close': float(candle[4]),
                            'volume': float(candle[5])
                        }
                        for candle in data
                    ]
                else:
                    self.logger.error(f"Erro ao buscar candles: {response.status}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Erro ao buscar candles: {str(e)}")
            return None
    
    async def _fetch_ticker(self) -> Optional[Dict[str, Any]]:
        """
        Busca ticker.
        
        Returns:
            Ticker.
        """
        try:
            # Prepara URL
            url = f"{self.config.exchange_url}/api/v3/ticker/24hr"
            
            # Prepara parâmetros
            params = {
                'symbol': self.config.symbol
            }
            
            # Faz requisição
            async with self._session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    self.logger.error(f"Erro ao buscar ticker: {response.status}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Erro ao buscar ticker: {str(e)}")
            return None
    
    async def _fetch_balance(self) -> Optional[Dict[str, Any]]:
        """
        Busca balance.
        
        Returns:
            Balance.
        """
        try:
            # Prepara URL
            url = f"{self.config.exchange_url}/api/v3/account"
            
            # Prepara timestamp
            timestamp = int(time.time() * 1000)
            
            # Prepara query string
            query_string = f"timestamp={timestamp}"
            
            # Prepara assinatura
            signature = hmac.new(
                self.config.api_secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            # Prepara headers
            headers = {
                'X-MBX-APIKEY': self.config.api_key,
                'X-MBX-SIGNATURE': signature
            }
            
            # Faz requisição
            async with self._session.get(url, params={'timestamp': timestamp}, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    self.logger.error(f"Erro ao buscar balance: {response.status}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Erro ao buscar balance: {str(e)}")
            return None
    
    async def get_orderbook(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Obtém orderbook.
        
        Args:
            symbol: Símbolo.
            
        Returns:
            Orderbook.
        """
        return self._orderbook_cache.get('data')
    
    async def get_trades(self, symbol: str) -> Optional[List[Dict[str, Any]]]:
        """
        Obtém trades.
        
        Args:
            symbol: Símbolo.
            
        Returns:
            Trades.
        """
        return self._trades_cache.get('data')
    
    async def get_candles(self, symbol: str, timeframe: str) -> Optional[List[Dict[str, Any]]]:
        """
        Obtém candles.
        
        Args:
            symbol: Símbolo.
            timeframe: Timeframe.
            
        Returns:
            Candles.
        """
        return self._candles_cache.get('data')
    
    async def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Obtém ticker.
        
        Args:
            symbol: Símbolo.
            
        Returns:
            Ticker.
        """
        return self._ticker_cache.get('data')
    
    async def get_balance(self) -> Optional[Dict[str, Any]]:
        """
        Obtém balance.
        
        Returns:
            Balance.
        """
        return self._balance_cache.get('data')
    
    async def place_order(
        self,
        symbol: str,
        side: str,
        type: str,
        quantity: float,
        price: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Coloca ordem.
        
        Args:
            symbol: Símbolo.
            side: Lado (buy/sell).
            type: Tipo (market/limit).
            quantity: Quantidade.
            price: Preço (opcional).
            
        Returns:
            Ordem.
        """
        try:
            # Prepara URL
            url = f"{self.config.exchange_url}/api/v3/order"
            
            # Prepara timestamp
            timestamp = int(time.time() * 1000)
            
            # Prepara parâmetros
            params = {
                'symbol': symbol,
                'side': side,
                'type': type,
                'quantity': quantity,
                'timestamp': timestamp
            }
            
            # Adiciona preço se for ordem limit
            if price:
                params['price'] = price
            
            # Prepara query string
            query_string = '&'.join(f"{k}={v}" for k, v in params.items())
            
            # Prepara assinatura
            signature = hmac.new(
                self.config.api_secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            # Prepara headers
            headers = {
                'X-MBX-APIKEY': self.config.api_key,
                'X-MBX-SIGNATURE': signature
            }
            
            # Faz requisição
            async with self._session.post(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    self.logger.error(f"Erro ao colocar ordem: {response.status}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Erro ao colocar ordem: {str(e)}")
            return None
    
    async def cancel_order(self, symbol: str, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Cancela ordem.
        
        Args:
            symbol: Símbolo.
            order_id: ID da ordem.
            
        Returns:
            Ordem.
        """
        try:
            # Prepara URL
            url = f"{self.config.exchange_url}/api/v3/order"
            
            # Prepara timestamp
            timestamp = int(time.time() * 1000)
            
            # Prepara parâmetros
            params = {
                'symbol': symbol,
                'orderId': order_id,
                'timestamp': timestamp
            }
            
            # Prepara query string
            query_string = '&'.join(f"{k}={v}" for k, v in params.items())
            
            # Prepara assinatura
            signature = hmac.new(
                self.config.api_secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            # Prepara headers
            headers = {
                'X-MBX-APIKEY': self.config.api_key,
                'X-MBX-SIGNATURE': signature
            }
            
            # Faz requisição
            async with self._session.delete(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    self.logger.error(f"Erro ao cancelar ordem: {response.status}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Erro ao cancelar ordem: {str(e)}")
            return None 