"""
Implementação da KuCoin
"""

import ccxt
import numpy as np
from typing import Dict, List, Optional
from .base_exchange import BaseExchange

class KuCoinExchange(BaseExchange):
    """Implementação da KuCoin"""
    
    async def initialize(self):
        """Inicializa conexão com KuCoin"""
        self.exchange = ccxt.kucoin({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'password': self.password,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'
            }
        })
        
    async def fetch_balance(self) -> Dict[str, Dict[str, float]]:
        """Obtém saldo da conta"""
        return await self.exchange.fetch_balance()
        
    async def fetch_ticker(self, symbol: str) -> Dict[str, float]:
        """Obtém ticker do par"""
        return await self.exchange.fetch_ticker(symbol)
        
    async def create_order(self, symbol: str, type: str, side: str, amount: float, price: Optional[float] = None) -> Dict:
        """Cria ordem"""
        return await self.exchange.create_order(symbol, type, side, amount, price)
        
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', since: Optional[int] = None, limit: Optional[int] = None) -> List[List[float]]:
        """Obtém dados OHLCV"""
        ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        return self.preprocess_ohlcv(ohlcv)
        
    async def cancel_order(self, id: str, symbol: str) -> Dict:
        """Cancela ordem"""
        return await self.exchange.cancel_order(id, symbol)
        
    async def fetch_order(self, id: str, symbol: str) -> Dict:
        """Obtém status da ordem"""
        return await self.exchange.fetch_order(id, symbol)
