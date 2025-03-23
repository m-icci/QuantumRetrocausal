"""
Script para carregar dados históricos para backtesting.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import aiohttp
import pandas as pd
from dotenv import load_dotenv
import os

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_loader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataLoader:
    """Carregador de dados históricos."""
    
    def __init__(self, exchange: str, symbol: str, timeframe: str):
        """
        Inicializa o carregador.
        
        Args:
            exchange: Nome da exchange
            symbol: Par de trading
            timeframe: Intervalo de tempo
        """
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def _init_session(self):
        """Inicializa a sessão HTTP."""
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def _close_session(self):
        """Fecha a sessão HTTP."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def _fetch_candles(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """
        Busca dados de candlesticks.
        
        Args:
            start_time: Data inicial
            end_time: Data final
            
        Returns:
            Lista de candlesticks
        """
        try:
            await self._init_session()
            
            # Converter timeframe para milissegundos
            timeframe_ms = {
                '1m': 60 * 1000,
                '5m': 5 * 60 * 1000,
                '15m': 15 * 60 * 1000,
                '30m': 30 * 60 * 1000,
                '1h': 60 * 60 * 1000,
                '4h': 4 * 60 * 60 * 1000,
                '1d': 24 * 60 * 60 * 1000
            }.get(self.timeframe, 60 * 60 * 1000)  # Default 1h
            
            # Buscar dados da Binance
            if self.exchange.lower() == 'binance':
                url = 'https://api.binance.com/api/v3/klines'
                params = {
                    'symbol': self.symbol.replace('/', ''),
                    'interval': self.timeframe,
                    'startTime': int(start_time.timestamp() * 1000),
                    'endTime': int(end_time.timestamp() * 1000),
                    'limit': 1000
                }
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [
                            {
                                'timestamp': datetime.fromtimestamp(candle[0] / 1000),
                                'open': float(candle[1]),
                                'high': float(candle[2]),
                                'low': float(candle[3]),
                                'close': float(candle[4]),
                                'volume': float(candle[5]),
                                'close_time': datetime.fromtimestamp(candle[6] / 1000),
                                'quote_volume': float(candle[7]),
                                'trades': int(candle[8]),
                                'taker_buy_base': float(candle[9]),
                                'taker_buy_quote': float(candle[10])
                            }
                            for candle in data
                        ]
                    else:
                        logger.error(f"Erro ao buscar dados: {response.status}")
                        return []
            
            # Adicionar suporte para outras exchanges aqui
            
            return []
            
        except Exception as e:
            logger.error(f"Erro ao buscar candlesticks: {e}")
            return []
    
    async def _fetch_trades(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """
        Busca dados de trades.
        
        Args:
            start_time: Data inicial
            end_time: Data final
            
        Returns:
            Lista de trades
        """
        try:
            await self._init_session()
            
            # Buscar dados da Binance
            if self.exchange.lower() == 'binance':
                url = 'https://api.binance.com/api/v3/trades'
                params = {
                    'symbol': self.symbol.replace('/', ''),
                    'startTime': int(start_time.timestamp() * 1000),
                    'endTime': int(end_time.timestamp() * 1000),
                    'limit': 1000
                }
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [
                            {
                                'timestamp': datetime.fromtimestamp(trade['time'] / 1000),
                                'price': float(trade['price']),
                                'quantity': float(trade['qty']),
                                'quote_quantity': float(trade['quoteQty']),
                                'is_buyer_maker': trade['isBuyerMaker'],
                                'trade_id': trade['id']
                            }
                            for trade in data
                        ]
                    else:
                        logger.error(f"Erro ao buscar dados: {response.status}")
                        return []
            
            # Adicionar suporte para outras exchanges aqui
            
            return []
            
        except Exception as e:
            logger.error(f"Erro ao buscar trades: {e}")
            return []
    
    async def _fetch_orderbook(self, timestamp: datetime) -> Dict:
        """
        Busca dados do orderbook.
        
        Args:
            timestamp: Data/hora do snapshot
            
        Returns:
            Dados do orderbook
        """
        try:
            await self._init_session()
            
            # Buscar dados da Binance
            if self.exchange.lower() == 'binance':
                url = 'https://api.binance.com/api/v3/depth'
                params = {
                    'symbol': self.symbol.replace('/', ''),
                    'limit': 1000
                }
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'timestamp': timestamp,
                            'bids': [[float(price), float(qty)] for price, qty in data['bids']],
                            'asks': [[float(price), float(qty)] for price, qty in data['asks']]
                        }
                    else:
                        logger.error(f"Erro ao buscar dados: {response.status}")
                        return {}
            
            # Adicionar suporte para outras exchanges aqui
            
            return {}
            
        except Exception as e:
            logger.error(f"Erro ao buscar orderbook: {e}")
            return {}
    
    async def load_data(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Carrega todos os dados históricos.
        
        Args:
            start_time: Data inicial
            end_time: Data final
            
        Returns:
            DataFrame com dados históricos
        """
        try:
            # Buscar dados
            candles = await self._fetch_candles(start_time, end_time)
            trades = await self._fetch_trades(start_time, end_time)
            
            if not candles:
                logger.error("Não foi possível carregar dados")
                return pd.DataFrame()
            
            # Criar DataFrame
            df = pd.DataFrame(candles)
            
            # Adicionar indicadores técnicos
            df['rsi'] = self._calculate_rsi(df['close'])
            df['macd'], df['macd_signal'], df['macd_hist'] = self._calculate_macd(df['close'])
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])
            df['atr'] = self._calculate_atr(df['high'], df['low'], df['close'])
            
            # Adicionar dados de trades
            trades_df = pd.DataFrame(trades)
            if not trades_df.empty:
                trades_df.set_index('timestamp', inplace=True)
                trades_df = trades_df.resample(self.timeframe).agg({
                    'price': ['mean', 'std'],
                    'quantity': 'sum',
                    'quote_quantity': 'sum',
                    'trades': 'count'
                })
                trades_df.columns = ['trade_price_mean', 'trade_price_std', 'trade_volume', 'trade_count']
                df = df.join(trades_df)
            
            # Adicionar dados do orderbook
            orderbook = await self._fetch_orderbook(end_time)
            if orderbook:
                df['orderbook_bids'] = [orderbook['bids']] * len(df)
                df['orderbook_asks'] = [orderbook['asks']] * len(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {e}")
            return pd.DataFrame()
        
        finally:
            await self._close_session()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcula o RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Calcula o MACD."""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std: float = 2) -> tuple:
        """Calcula as Bollinger Bands."""
        middle = prices.rolling(window=period).mean()
        std_dev = prices.rolling(window=period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        return upper, middle, lower
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calcula o ATR."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

async def main():
    """Função principal."""
    try:
        # Carregar configuração
        load_dotenv()
        
        # Criar carregador
        loader = DataLoader(
            exchange=os.getenv('EXCHANGE', 'binance'),
            symbol=os.getenv('SYMBOL', 'BTC/USDT'),
            timeframe=os.getenv('TIMEFRAME', '1h')
        )
        
        # Definir período
        end_time = datetime.now()
        start_time = end_time - timedelta(days=30)
        
        # Carregar dados
        logger.info("Carregando dados históricos...")
        df = await loader.load_data(start_time, end_time)
        
        if not df.empty:
            # Salvar dados
            df.to_csv('historical_data.csv')
            logger.info(f"Dados salvos em historical_data.csv")
            logger.info(f"Shape: {df.shape}")
            logger.info(f"Colunas: {df.columns.tolist()}")
        else:
            logger.error("Não foi possível carregar dados")
        
    except Exception as e:
        logger.error(f"Erro fatal: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 