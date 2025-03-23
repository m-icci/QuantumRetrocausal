"""
Interface unificada para dados de mercado da Kraken
"""

import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import asyncio
import logging

from config import KRAKEN_CONFIG

logger = logging.getLogger(__name__)

class MarketInterface:
    """Interface unificada para dados de mercado da Kraken"""

    def __init__(self):
        """Inicializa conexão com Kraken"""
        self.kraken = ccxt.kraken(KRAKEN_CONFIG)

        # Cache de dados
        self.cache = {}
        self.last_update = {}

    async def get_klines(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100
    ) -> List:
        """
        Busca dados OHLCV da Kraken

        Args:
            symbol: Par de trading
            timeframe: Intervalo
            limit: Quantidade de candles
        """
        try:
            # Usa cache se disponível e recente
            cache_key = f"kraken_{symbol}_{timeframe}"
            now = datetime.now().timestamp()

            if (
                cache_key in self.cache and
                cache_key in self.last_update and
                now - self.last_update[cache_key] < 60  # Cache de 1 minuto
            ):
                return self.cache[cache_key]

            # Busca dados
            klines = await self.kraken.fetch_ohlcv(
                symbol,
                timeframe,
                limit=limit
            )

            # Atualiza cache
            self.cache[cache_key] = klines
            self.last_update[cache_key] = now

            return klines

        except Exception as e:
            logger.error(f"Erro buscando dados: {e}")
            return []

    def get_latest_data(
        self,
        symbol: str,
        timeframe: str
    ) -> Optional[Dict]:
        """
        Retorna dados mais recentes

        Args:
            symbol: Par de trading
            timeframe: Intervalo
        """
        try:
            cache_key = f"kraken_{symbol}_{timeframe}"

            if cache_key in self.cache:
                klines = self.cache[cache_key]
                if klines:
                    last_candle = klines[-1]
                    return {
                        'timestamp': last_candle[0],
                        'open': last_candle[1],
                        'high': last_candle[2],
                        'low': last_candle[3],
                        'close': last_candle[4],
                        'volume': last_candle[5]
                    }
            return None

        except Exception as e:
            logger.error(f"Erro obtendo últimos dados: {e}")
            return None

    async def close(self):
        """Fecha conexão"""
        await self.kraken.close()