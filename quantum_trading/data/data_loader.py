"""
Carregador de dados
"""

import logging
import asyncio
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt

logger = logging.getLogger(__name__)

class DataLoader:
    """Carregador de dados"""
    
    def __init__(self, config: Dict):
        """
        Inicializa carregador.
        
        Args:
            config: Configuração.
        """
        self.config = config
        
        # Exchange
        exchange_id = config['exchange']['name']
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({
            'apiKey': config['exchange']['api_key'],
            'secret': config['exchange']['api_secret'],
            'enableRateLimit': True
        })
        
        # Cache
        self.cache = {
            'ticker': {},
            'orderbook': {},
            'trades': {},
            'candles': {},
            'balance': None
        }
        
        # Estado
        self.is_connected = False
        self.current_time = datetime.now()
        
    async def connect(self) -> None:
        """Conecta com exchange"""
        try:
            logger.info("Conectando com exchange")
            await self.exchange.load_markets()
            self.is_connected = True
            logger.info("Conectado com exchange")
            
        except Exception as e:
            logger.error(f"Erro ao conectar com exchange: {str(e)}")
            raise
            
    async def disconnect(self) -> None:
        """Desconecta da exchange"""
        try:
            logger.info("Desconectando da exchange")
            await self.exchange.close()
            self.is_connected = False
            logger.info("Desconectado da exchange")
            
        except Exception as e:
            logger.error(f"Erro ao desconectar da exchange: {str(e)}")
            raise
            
    async def update(self) -> None:
        """Atualiza dados"""
        try:
            if not self.is_connected:
                return
                
            # Atualiza tempo
            self.current_time = datetime.now()
            
            # Limpa cache antigo
            await self._clean_cache()
            
        except Exception as e:
            logger.error(f"Erro ao atualizar dados: {str(e)}")
            raise
            
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Obtém preço atual.
        
        Args:
            symbol: Símbolo.
            
        Returns:
            Preço atual.
        """
        try:
            # Verifica cache
            ticker = self.cache['ticker'].get(symbol)
            if ticker and datetime.now() - ticker['timestamp'] < timedelta(seconds=1):
                return ticker['last']
                
            # Obtém ticker
            if self.config['trading']['mode'] == 'simulated':
                ticker = await self._get_simulated_ticker(symbol)
            else:
                ticker = await self.exchange.fetch_ticker(symbol)
                
            if ticker is None:
                return None
                
            # Atualiza cache
            self.cache['ticker'][symbol] = {
                'last': ticker['last'],
                'timestamp': datetime.now()
            }
            
            return ticker['last']
            
        except Exception as e:
            logger.error(f"Erro ao obter preço atual: {str(e)}")
            return None
            
    async def get_order_book(self, symbol: str) -> Optional[Dict]:
        """
        Obtém order book.
        
        Args:
            symbol: Símbolo.
            
        Returns:
            Order book.
        """
        try:
            # Verifica cache
            orderbook = self.cache['orderbook'].get(symbol)
            if orderbook and datetime.now() - orderbook['timestamp'] < timedelta(seconds=1):
                return orderbook['data']
                
            # Obtém order book
            if self.config['trading']['mode'] == 'simulated':
                orderbook = await self._get_simulated_orderbook(symbol)
            else:
                orderbook = await self.exchange.fetch_order_book(symbol)
                
            if orderbook is None:
                return None
                
            # Atualiza cache
            self.cache['orderbook'][symbol] = {
                'data': orderbook,
                'timestamp': datetime.now()
            }
            
            return orderbook
            
        except Exception as e:
            logger.error(f"Erro ao obter order book: {str(e)}")
            return None
            
    async def get_trades(self, symbol: str) -> Optional[List[Dict]]:
        """
        Obtém trades recentes.
        
        Args:
            symbol: Símbolo.
            
        Returns:
            Lista de trades.
        """
        try:
            # Verifica cache
            trades = self.cache['trades'].get(symbol)
            if trades and datetime.now() - trades['timestamp'] < timedelta(seconds=1):
                return trades['data']
                
            # Obtém trades
            if self.config['trading']['mode'] == 'simulated':
                trades = await self._get_simulated_trades(symbol)
            else:
                trades = await self.exchange.fetch_trades(symbol)
                
            if trades is None:
                return None
                
            # Atualiza cache
            self.cache['trades'][symbol] = {
                'data': trades,
                'timestamp': datetime.now()
            }
            
            return trades
            
        except Exception as e:
            logger.error(f"Erro ao obter trades: {str(e)}")
            return None
            
    async def get_candles(self, symbol: str, timeframe: str) -> Optional[List[Dict]]:
        """
        Obtém candlesticks.
        
        Args:
            symbol: Símbolo.
            timeframe: Timeframe.
            
        Returns:
            Lista de candles.
        """
        try:
            # Verifica cache
            key = f"{symbol}_{timeframe}"
            candles = self.cache['candles'].get(key)
            if candles and datetime.now() - candles['timestamp'] < timedelta(minutes=1):
                return candles['data']
                
            # Obtém candles
            if self.config['trading']['mode'] == 'simulated':
                candles = await self._get_simulated_candles(symbol, timeframe)
            else:
                candles = await self.exchange.fetch_ohlcv(symbol, timeframe)
                
            if candles is None:
                return None
                
            # Atualiza cache
            self.cache['candles'][key] = {
                'data': candles,
                'timestamp': datetime.now()
            }
            
            return candles
            
        except Exception as e:
            logger.error(f"Erro ao obter candles: {str(e)}")
            return None
            
    async def get_balance(self) -> Optional[Dict]:
        """
        Obtém saldo.
        
        Returns:
            Saldo.
        """
        try:
            # Verifica cache
            if self.cache['balance'] and datetime.now() - self.cache['balance']['timestamp'] < timedelta(seconds=5):
                return self.cache['balance']['data']
                
            # Obtém saldo
            if self.config['trading']['mode'] == 'simulated':
                balance = {
                    'free': self.config['trading']['initial_balance'],
                    'used': 0,
                    'total': self.config['trading']['initial_balance']
                }
            else:
                balance = await self.exchange.fetch_balance()
                
            if balance is None:
                return None
                
            # Atualiza cache
            self.cache['balance'] = {
                'data': balance,
                'timestamp': datetime.now()
            }
            
            return balance
            
        except Exception as e:
            logger.error(f"Erro ao obter saldo: {str(e)}")
            return None
            
    async def get_trade_history(self) -> Optional[List[Dict]]:
        """
        Obtém histórico de trades.
        
        Returns:
            Lista de trades.
        """
        try:
            if self.config['trading']['mode'] == 'simulated':
                return []
                
            # Obtém histórico
            trades = await self.exchange.fetch_my_trades(self.config['trading']['symbol'])
            
            return trades
            
        except Exception as e:
            logger.error(f"Erro ao obter histórico de trades: {str(e)}")
            return None
            
    async def get_daily_history(self) -> Optional[List[Dict]]:
        """
        Obtém histórico diário.
        
        Returns:
            Lista de registros diários.
        """
        try:
            if self.config['trading']['mode'] == 'simulated':
                return []
                
            # Obtém histórico
            start_date = datetime.now() - timedelta(days=30)
            candles = await self.exchange.fetch_ohlcv(
                self.config['trading']['symbol'],
                '1d',
                since=int(start_date.timestamp() * 1000)
            )
            
            if candles is None:
                return None
                
            # Converte para registros diários
            daily = []
            for candle in candles:
                daily.append({
                    'date': datetime.fromtimestamp(candle[0] / 1000).date(),
                    'open': candle[1],
                    'high': candle[2],
                    'low': candle[3],
                    'close': candle[4],
                    'volume': candle[5]
                })
                
            return daily
            
        except Exception as e:
            logger.error(f"Erro ao obter histórico diário: {str(e)}")
            return None
            
    async def get_market_data(self, symbol: str = None) -> Optional[Dict]:
        """
        Obtém dados do mercado.
        
        Args:
            symbol: Símbolo do mercado (opcional). Se não fornecido, usa o símbolo da configuração.
            
        Returns:
            Dados do mercado.
        """
        try:
            # Usa o símbolo fornecido ou o da configuração
            symbol_to_use = symbol if symbol is not None else self.config['trading']['symbol']
            
            # Obtém dados
            candles = await self.get_candles(
                symbol_to_use,
                self.config['trading']['timeframe']
            )
            
            if candles is None:
                return None
                
            # Converte para DataFrame
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Calcula retornos
            df['returns'] = df['close'].pct_change()
            
            # Calcula retornos do mercado (BTC como proxy)
            market_candles = await self.get_candles('BTC/USDT', self.config['trading']['timeframe'])
            if market_candles is not None:
                market_df = pd.DataFrame(market_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                market_df['returns'] = market_df['close'].pct_change()
            else:
                market_df = df.copy()
                
            return {
                'symbol_returns': df['returns'].values,
                'market_returns': market_df['returns'].values,
                'df': df  # Retornando o DataFrame completo para permitir mais análises
            }
            
        except Exception as e:
            logger.error(f"Erro ao obter dados do mercado para {symbol}: {str(e)}")
            return None
            
    async def _clean_cache(self) -> None:
        """Limpa cache antigo"""
        try:
            current_time = datetime.now()
            
            # Limpa ticker (1 segundo)
            self.cache['ticker'] = {
                k: v for k, v in self.cache['ticker'].items()
                if current_time - v['timestamp'] < timedelta(seconds=1)
            }
            
            # Limpa order book (1 segundo)
            self.cache['orderbook'] = {
                k: v for k, v in self.cache['orderbook'].items()
                if current_time - v['timestamp'] < timedelta(seconds=1)
            }
            
            # Limpa trades (1 segundo)
            self.cache['trades'] = {
                k: v for k, v in self.cache['trades'].items()
                if current_time - v['timestamp'] < timedelta(seconds=1)
            }
            
            # Limpa candles (1 minuto)
            self.cache['candles'] = {
                k: v for k, v in self.cache['candles'].items()
                if current_time - v['timestamp'] < timedelta(minutes=1)
            }
            
            # Limpa saldo (5 segundos)
            if self.cache['balance'] and current_time - self.cache['balance']['timestamp'] >= timedelta(seconds=5):
                self.cache['balance'] = None
                
        except Exception as e:
            logger.error(f"Erro ao limpar cache: {str(e)}")
            
    async def _get_simulated_ticker(self, symbol: str) -> Optional[Dict]:
        """
        Gera ticker simulado.
        
        Args:
            symbol: Símbolo.
            
        Returns:
            Ticker simulado.
        """
        try:
            # Obtém último candle
            candles = await self.get_candles(symbol, '1m')
            if not candles:
                return None
                
            last_candle = candles[-1]
            
            # Gera preço simulado
            price = last_candle[4] * (1 + np.random.normal(0, 0.0001))
            
            return {
                'symbol': symbol,
                'timestamp': int(datetime.now().timestamp() * 1000),
                'datetime': datetime.now().isoformat(),
                'high': price * 1.001,
                'low': price * 0.999,
                'bid': price * 0.9995,
                'ask': price * 1.0005,
                'last': price,
                'close': price,
                'baseVolume': last_candle[5],
                'info': {}
            }
            
        except Exception as e:
            logger.error(f"Erro ao gerar ticker simulado: {str(e)}")
            return None
            
    async def _get_simulated_orderbook(self, symbol: str) -> Optional[Dict]:
        """
        Gera order book simulado.
        
        Args:
            symbol: Símbolo.
            
        Returns:
            Order book simulado.
        """
        try:
            # Obtém ticker
            ticker = await self.get_current_price(symbol)
            if ticker is None:
                return None
                
            # Gera ordens simuladas
            bids = []
            asks = []
            
            for i in range(20):
                bid_price = ticker * (1 - 0.0001 * (i + 1))
                ask_price = ticker * (1 + 0.0001 * (i + 1))
                
                volume = np.random.normal(1, 0.1)
                
                bids.append([bid_price, volume])
                asks.append([ask_price, volume])
                
            return {
                'bids': sorted(bids, key=lambda x: x[0], reverse=True),
                'asks': sorted(asks, key=lambda x: x[0]),
                'timestamp': int(datetime.now().timestamp() * 1000),
                'datetime': datetime.now().isoformat(),
                'nonce': None
            }
            
        except Exception as e:
            logger.error(f"Erro ao gerar order book simulado: {str(e)}")
            return None
            
    async def _get_simulated_trades(self, symbol: str) -> Optional[List[Dict]]:
        """
        Gera trades simulados.
        
        Args:
            symbol: Símbolo.
            
        Returns:
            Lista de trades simulados.
        """
        try:
            # Obtém ticker
            ticker = await self.get_current_price(symbol)
            if ticker is None:
                return None
                
            # Gera trades simulados
            trades = []
            
            for i in range(50):
                side = 'buy' if np.random.random() > 0.5 else 'sell'
                price = ticker * (1 + np.random.normal(0, 0.0001))
                amount = np.random.normal(1, 0.1)
                
                trades.append({
                    'id': str(int(datetime.now().timestamp() * 1000) + i),
                    'order': None,
                    'info': {},
                    'timestamp': int(datetime.now().timestamp() * 1000) - i * 1000,
                    'datetime': (datetime.now() - timedelta(seconds=i)).isoformat(),
                    'symbol': symbol,
                    'type': 'limit',
                    'side': side,
                    'price': price,
                    'amount': amount,
                    'cost': price * amount,
                    'fee': None
                })
                
            return trades
            
        except Exception as e:
            logger.error(f"Erro ao gerar trades simulados: {str(e)}")
            return None
            
    async def _get_simulated_candles(self, symbol: str, timeframe: str) -> Optional[List[List]]:
        """
        Gera candles simulados.
        
        Args:
            symbol: Símbolo.
            timeframe: Timeframe.
            
        Returns:
            Lista de candles simulados.
        """
        try:
            # Parâmetros de simulação
            num_candles = 100
            volatility = 0.001
            trend = 0.0001
            volume_mean = 1000
            volume_std = 100
            
            # Gera preços
            returns = np.random.normal(trend, volatility, num_candles)
            close_prices = np.exp(np.cumsum(returns)) * 100  # Preço base de 100
            
            # Gera volumes
            volumes = np.abs(np.random.normal(volume_mean, volume_std, num_candles))
            
            # Gera candles
            candles = []
            current_time = datetime.now()
            
            for i in range(num_candles):
                timestamp = int((current_time - timedelta(minutes=num_candles-i)).timestamp() * 1000)
                close = close_prices[i]
                
                # Gera preços OHLC
                open_price = close * (1 + np.random.normal(0, volatility))
                high = max(open_price, close) * (1 + abs(np.random.normal(0, volatility)))
                low = min(open_price, close) * (1 - abs(np.random.normal(0, volatility)))
                
                candles.append([
                    timestamp,
                    open_price,
                    high,
                    low,
                    close,
                    volumes[i]
                ])
                
            return candles
            
        except Exception as e:
            logger.error(f"Erro ao gerar candles simulados: {str(e)}")
            return None 