"""
Implementação da Kraken
"""

import ccxt
import numpy as np
from typing import Dict, List, Optional
from .base_exchange import BaseExchange

class KrakenExchange(BaseExchange):
    """Implementação da Kraken"""
    
    async def initialize(self):
        """Inicializa conexão com Kraken"""
        self.exchange = ccxt.kraken({
            'apiKey': self.api_key,
            'secret': self.api_secret,
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
        # Kraken usa nomes de pares diferentes, então precisamos converter
        symbol = self._convert_symbol(symbol)
        ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        return self.preprocess_ohlcv(ohlcv)
        
    async def cancel_order(self, id: str, symbol: str) -> Dict:
        """Cancela ordem"""
        return await self.exchange.cancel_order(id, symbol)
        
    async def fetch_order(self, id: str, symbol: str) -> Dict:
        """Obtém status da ordem"""
        return await self.exchange.fetch_order(id, symbol)
        
    def _convert_symbol(self, symbol: str) -> str:
        """Converte símbolo para formato da Kraken"""
        # Kraken usa XBT ao invés de BTC
        return symbol.replace('BTC', 'XBT')
