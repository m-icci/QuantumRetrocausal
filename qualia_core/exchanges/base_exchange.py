"""
Exchange base para padronizar interface
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import ccxt
import numpy as np

class BaseExchange(ABC):
    """Exchange base com métodos padrão"""
    
    def __init__(self, api_key: str = "", api_secret: str = "", password: str = ""):
        """Inicializa exchange"""
        self.api_key = api_key
        self.api_secret = api_secret
        self.password = password
        self.exchange = None
        
    @abstractmethod
    async def initialize(self):
        """Inicializa conexão com exchange"""
        pass
        
    @abstractmethod
    async def fetch_balance(self) -> Dict[str, Dict[str, float]]:
        """Obtém saldo da conta"""
        pass
        
    @abstractmethod
    async def fetch_ticker(self, symbol: str) -> Dict[str, float]:
        """Obtém ticker do par"""
        pass
        
    @abstractmethod
    async def create_order(self, symbol: str, type: str, side: str, amount: float, price: Optional[float] = None) -> Dict:
        """Cria ordem"""
        pass
        
    @abstractmethod
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', since: Optional[int] = None, limit: Optional[int] = None) -> List[List[float]]:
        """Obtém dados OHLCV"""
        pass
        
    @abstractmethod
    async def cancel_order(self, id: str, symbol: str) -> Dict:
        """Cancela ordem"""
        pass
        
    @abstractmethod
    async def fetch_order(self, id: str, symbol: str) -> Dict:
        """Obtém status da ordem"""
        pass
        
    def preprocess_ohlcv(self, ohlcv: List[List[float]]) -> np.ndarray:
        """Preprocessa dados OHLCV para formato padrão"""
        data = np.array(ohlcv)
        return data
