"""
Factory para criar instâncias de exchanges
"""

import ccxt.async_support as ccxt
from typing import Dict, Optional
import logging

from config import KUCOIN_CONFIG, KRAKEN_CONFIG

logger = logging.getLogger(__name__)

class ExchangeFactory:
    """Factory para criar exchanges"""
    
    @staticmethod
    def create_exchange(name: str, config: Optional[Dict] = None) -> ccxt.Exchange:
        """
        Cria instância de exchange
        
        Args:
            name: Nome da exchange
            config: Configuração opcional
        """
        try:
            if name == 'kucoin':
                return ccxt.kucoin(config or KUCOIN_CONFIG)
            elif name == 'kraken':
                return ccxt.kraken(config or KRAKEN_CONFIG)
            else:
                raise ValueError(f"Exchange não suportada: {name}")
                
        except Exception as e:
            logger.error(f"Erro criando exchange {name}: {e}")
            raise
