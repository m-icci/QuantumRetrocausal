"""
Análise de mercado.
"""

import logging
import asyncio
import pandas as pd
import numpy as np
# import talib  # Comentado temporariamente
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta

from .trading_config import TradingConfig
from .exchange_integration import ExchangeIntegration
from quantum_trading.data.data_loader import DataLoader

# Configuração do logger
logger = logging.getLogger(__name__)

# Funções temporárias para substituir o talib
def calculate_rsi(close, period=14):
    """Função simplificada para calcular RSI"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(close, fast_period=12, slow_period=26, signal_period=9):
    """Função simplificada para calcular MACD"""
    ema_fast = close.ewm(span=fast_period).mean()
    ema_slow = close.ewm(span=slow_period).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period).mean()
    hist = macd - signal
    return macd, signal, hist

def calculate_bbands(close, period=20, std=2):
    """Função simplificada para calcular Bollinger Bands"""
    middle = close.rolling(window=period).mean()
    upper = middle + std * close.rolling(window=period).std()
    lower = middle - std * close.rolling(window=period).std()
    return upper, middle, lower

def calculate_atr(high, low, close, period=14):
    """Função simplificada para calcular ATR"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

class MarketAnalysis:
    """Análise de mercado."""
    
    def __init__(self, config: Dict):
        """
        Inicializa análise de mercado.
        
        Args:
            config: Configuração.
        """
        self.config = config
        self.data_loader = DataLoader(config)
        
        # Cache de dados
        self.price_cache = {}
        self.volume_cache = {}
        self.order_book_cache = {}
        
    async def initialize(self) -> None:
        """Inicializa análise"""
        try:
            logger.info("Inicializando análise de mercado")
            await self.data_loader.connect()
            logger.info("Análise de mercado inicializada")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar análise de mercado: {str(e)}")
            raise
            
    async def update(self) -> None:
        """Atualiza análise"""
        try:
            # Atualiza dados
            await self.data_loader.update()
            
            # Atualiza cache
            await self._update_cache()
            
        except Exception as e:
            logger.error(f"Erro ao atualizar análise de mercado: {str(e)}")
            raise
            
    async def _update_cache(self) -> None:
        """Atualiza cache de dados"""
        try:
            symbol = self.config['trading']['symbol']
            
            # Atualiza preços
            current_price = await self.data_loader.get_current_price(symbol)
            if current_price is not None:
                self.price_cache[symbol] = current_price
                
            # Atualiza volume
            current_volume = await self.data_loader.get_current_volume(symbol)
            if current_volume is not None:
                self.volume_cache[symbol] = current_volume
                
            # Atualiza order book
            order_book = await self.data_loader.get_order_book(symbol)
            if order_book is not None:
                self.order_book_cache[symbol] = order_book
                
        except Exception as e:
            logger.error(f"Erro ao atualizar cache: {str(e)}")
            raise
            
    async def check_volume_profile(self, symbol: str) -> bool:
        """
        Verifica perfil de volume.
        
        Args:
            symbol: Símbolo.
            
        Returns:
            True se volume ok.
        """
        try:
            # Obtém volume atual
            current_volume = self.volume_cache.get(symbol)
            if current_volume is None:
                return False
                
            # Verifica volume mínimo
            min_volume = self.config['scalping']['min_volume_threshold']
            if current_volume < min_volume:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Erro ao verificar volume: {str(e)}")
            return False
            
    async def check_order_book(self, symbol: str) -> bool:
        """
        Verifica order book.
        
        Args:
            symbol: Símbolo.
            
        Returns:
            True se spread ok.
        """
        try:
            # Obtém order book
            order_book = self.order_book_cache.get(symbol)
            if order_book is None:
                return False
                
            # Calcula spread
            best_bid = max(order_book['bids'], key=lambda x: x[0])[0]
            best_ask = min(order_book['asks'], key=lambda x: x[0])[0]
            spread = (best_ask - best_bid) / best_bid
            
            # Verifica spread máximo
            max_spread = self.config['scalping']['max_spread_threshold']
            if spread > max_spread:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Erro ao verificar order book: {str(e)}")
            return False
            
    async def check_micro_movements(self, symbol: str) -> Optional[Tuple[float, str]]:
        """
        Analisa micro-movimentos.
        
        Args:
            symbol: Símbolo.
            
        Returns:
            Tuple com potencial de lucro e direção, ou None.
        """
        try:
            # Obtém dados recentes
            recent_data = await self.data_loader.get_recent_data(
                symbol,
                limit=10
            )
            
            if recent_data is None or len(recent_data) < 10:
                return None
                
            # Calcula médias
            sma_5 = recent_data['close'].rolling(5).mean()
            sma_10 = recent_data['close'].rolling(10).mean()
            
            # Calcula momentum
            momentum = recent_data['close'].diff()
            
            # Analisa padrão
            last_momentum = momentum.iloc[-1]
            last_sma_5 = sma_5.iloc[-1]
            last_sma_10 = sma_10.iloc[-1]
            
            # Identifica direção
            if last_sma_5 > last_sma_10 and last_momentum > 0:
                direction = 'long'
                profit_potential = abs(last_momentum) / recent_data['close'].iloc[-1]
                return profit_potential, direction
                
            elif last_sma_5 < last_sma_10 and last_momentum < 0:
                direction = 'short'
                profit_potential = abs(last_momentum) / recent_data['close'].iloc[-1]
                return profit_potential, direction
                
            return None
            
        except Exception as e:
            logger.error(f"Erro ao analisar micro-movimentos: {str(e)}")
            return None
    
    async def analyze_market(self) -> Dict[str, Any]:
        """
        Analisa mercado.
        
        Returns:
            Dados do mercado.
        """
        try:
            # Obtém dados
            market_data = await self._get_market_data()
            if not market_data:
                return {}
            
            # Calcula indicadores
            indicators = await self._calculate_indicators(self.config['trading']['symbol'])
            if not indicators:
                return {}
            
            # Identifica padrões
            patterns = await self._identify_patterns(self.config['trading']['symbol'])
            if not patterns:
                return {}
            
            # Gera sinais
            signals = await self._generate_signals(self.config['trading']['symbol'], indicators, patterns)
            
            # Combina dados
            data = {
                'market': market_data,
                'indicators': indicators,
                'patterns': patterns,
                'signals': signals
            }
            
            return data
            
        except Exception as e:
            logger.error(f"Erro ao analisar mercado: {str(e)}")
            return {}
    
    async def get_signals(self) -> List[Dict[str, Any]]:
        """
        Obtém sinais.
        
        Returns:
            Lista de sinais.
        """
        try:
            # Analisa mercado
            data = await self.analyze_market()
            
            return data.get('signals', [])
            
        except Exception as e:
            logger.error(f"Erro ao obter sinais: {str(e)}")
            return []
    
    async def _get_market_data(self) -> Optional[Dict[str, Any]]:
        """
        Obtém dados do mercado.
        
        Returns:
            Dados do mercado.
        """
        try:
            # Obtém dados
            orderbook = await self.data_loader.get_order_book(self.config['trading']['symbol'])
            trades = await self.data_loader.get_trades(self.config['trading']['symbol'])
            candles = await self.data_loader.get_candles(self.config['trading']['symbol'], self.config['trading']['timeframe'])
            ticker = await self.data_loader.get_ticker(self.config['trading']['symbol'])
            
            if not all([orderbook, trades, candles, ticker]):
                return None
            
            # Formata dados
            data = {
                'orderbook': orderbook,
                'trades': trades,
                'candles': candles,
                'ticker': ticker,
                'timestamp': int(time.time() * 1000)
            }
            
            return data
            
        except Exception as e:
            logger.error(f"Erro ao obter dados do mercado: {str(e)}")
            raise
    
    async def _calculate_indicators(self, symbol: str) -> Dict[str, Any]:
        """
        Calcula indicadores.
        
        Args:
            symbol: Símbolo.
            
        Returns:
            Indicadores.
        """
        try:
            # Verifica cache
            if symbol in self.price_cache:
                cache = self.price_cache[symbol]
                if datetime.now() - self.price_cache.get(f"{symbol}_timestamp", datetime.min) < timedelta(seconds=self.config['trading']['indicators_ttl']):
                    return cache
            
            # Obtém candlesticks
            candlesticks = await self.data_loader.get_candlesticks(symbol, self.config['trading']['timeframe'])
            
            # Converte para DataFrame
            df = pd.DataFrame(candlesticks)
            if df.empty:
                # Para demonstração, criar dados fictícios
                timestamps = [datetime.now() - timedelta(hours=i) for i in range(100, 0, -1)]
                close_prices = np.random.normal(100, 5, 100)
                high_prices = close_prices + np.abs(np.random.normal(0, 1, 100))
                low_prices = close_prices - np.abs(np.random.normal(0, 1, 100))
                open_prices = close_prices - np.random.normal(0, 2, 100)
                volumes = np.random.normal(1000, 200, 100)
                
                df = pd.DataFrame({
                    'timestamp': timestamps,
                    'open': open_prices,
                    'high': high_prices,
                    'low': low_prices,
                    'close': close_prices,
                    'volume': volumes
                })
            else:
                df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                df = df.astype(float)
            
            # Calcula RSI (usando substituto simplificado)
            rsi = calculate_rsi(df['close'], period=self.config['trading']['rsi_period'])
            
            # Calcula MACD (usando substituto simplificado)
            macd, signal, hist = calculate_macd(
                df['close'],
                fast_period=self.config['trading']['macd_fast'],
                slow_period=self.config['trading']['macd_slow'],
                signal_period=self.config['trading']['macd_signal']
            )
            
            # Calcula Bollinger Bands (usando substituto simplificado)
            upper, middle, lower = calculate_bbands(
                df['close'],
                period=self.config['trading']['bb_period'],
                std=self.config['trading']['bb_std']
            )
            
            # Calcula ATR (usando substituto simplificado)
            atr = calculate_atr(
                df['high'], 
                df['low'], 
                df['close'], 
                period=self.config['trading']['atr_period']
            )
            
            # Combina indicadores
            indicators = {
                'rsi': rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50,
                'macd': {
                    'macd': macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else 0,
                    'signal': signal.iloc[-1] if not pd.isna(signal.iloc[-1]) else 0,
                    'hist': hist.iloc[-1] if not pd.isna(hist.iloc[-1]) else 0
                },
                'bb': {
                    'upper': upper.iloc[-1] if not pd.isna(upper.iloc[-1]) else df['close'].iloc[-1] * 1.02,
                    'middle': middle.iloc[-1] if not pd.isna(middle.iloc[-1]) else df['close'].iloc[-1],
                    'lower': lower.iloc[-1] if not pd.isna(lower.iloc[-1]) else df['close'].iloc[-1] * 0.98
                },
                'atr': atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else df['close'].iloc[-1] * 0.01
            }
            
            # Atualiza cache
            self.price_cache[symbol] = indicators
            self.price_cache[f"{symbol}_timestamp"] = datetime.now()
            
            return indicators
            
        except Exception as e:
            logger.error(f"Erro ao calcular indicadores: {str(e)}")
            return {
                'rsi': 50,
                'macd': {'macd': 0, 'signal': 0, 'hist': 0},
                'bb': {'upper': 110, 'middle': 100, 'lower': 90},
                'atr': 1.0
            }
    
    async def _identify_patterns(self, symbol: str) -> List[str]:
        """
        Identifica padrões.
        
        Args:
            symbol: Símbolo.
            
        Returns:
            Lista de padrões.
        """
        try:
            # Verifica cache
            if symbol in self.price_cache:
                cache = self.price_cache[symbol]
                if datetime.now() - self.price_cache.get(f"{symbol}_timestamp", datetime.min) < timedelta(seconds=self.config['trading']['patterns_ttl']):
                    return cache
            
            # Obtém candlesticks
            candlesticks = await self.data_loader.get_candlesticks(symbol, self.config['trading']['timeframe'])
            
            # Converte para DataFrame
            df = pd.DataFrame(candlesticks)
            if df.empty:
                # Para demonstração, criar dados fictícios
                return ['doji']  # Retornar um padrão simulado para demonstração
            else:
                df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                df = df.astype(float)
            
            # Identifica padrões
            patterns = []
            
            # Doji
            if self._is_doji(df.iloc[-1]):
                patterns.append('doji')
            
            # Engulfing
            if len(df) >= 2:
                pattern = self._is_engulfing(df.iloc[-2:])
                if pattern == 'bullish':
                    patterns.append('bullish_engulfing')
                elif pattern == 'bearish':
                    patterns.append('bearish_engulfing')
            
            # Hammer
            if self._is_hammer(df.iloc[-1]):
                patterns.append('hammer')
                
            # Se nenhum padrão for identificado, adicionar um padrão aleatório para demonstração
            if not patterns:
                patterns = ['neutral']
                
            # Atualiza cache
            self.price_cache[symbol] = patterns
            self.price_cache[f"{symbol}_timestamp"] = datetime.now()
            
            return patterns
            
        except Exception as e:
            logger.error(f"Erro ao identificar padrões: {str(e)}")
            return ['neutral']  # Padrão para demonstração
    
    async def _generate_signals(self, symbol: str, indicators: Dict[str, Any], patterns: List[str]) -> List[Dict[str, Any]]:
        """
        Gera sinais.
        
        Args:
            symbol: Símbolo.
            indicators: Indicadores.
            patterns: Padrões.
            
        Returns:
            Lista de sinais.
        """
        try:
            # Verifica cache
            if symbol in self.price_cache:
                cache = self.price_cache[symbol]
                if datetime.now() - self.price_cache.get(f"{symbol}_timestamp", datetime.min) < timedelta(seconds=self.config['trading']['signals_ttl']):
                    return cache
            
            signals = []
            
            # Sinal RSI
            rsi = indicators.get('rsi', 50)
            if rsi < self.config['trading']['rsi_oversold']:
                signals.append({
                    'type': 'rsi',
                    'direction': 'buy',
                    'confidence': 0.7 + (self.config['trading']['rsi_oversold'] - rsi) / 100,
                    'price': None,
                    'timestamp': int(time.time() * 1000)
                })
            elif rsi > self.config['trading']['rsi_overbought']:
                signals.append({
                    'type': 'rsi',
                    'direction': 'sell',
                    'confidence': 0.7 + (rsi - self.config['trading']['rsi_overbought']) / 100,
                    'price': None,
                    'timestamp': int(time.time() * 1000)
                })
            
            # Sinal MACD
            macd = indicators.get('macd', {})
            macd_line = macd.get('macd', 0)
            signal_line = macd.get('signal', 0)
            hist = macd.get('hist', 0)
            
            if macd_line > signal_line and hist > 0:
                signals.append({
                    'type': 'macd',
                    'direction': 'buy',
                    'confidence': 0.7 + abs(hist) / 10,
                    'price': None,
                    'timestamp': int(time.time() * 1000)
                })
            elif macd_line < signal_line and hist < 0:
                signals.append({
                    'type': 'macd',
                    'direction': 'sell',
                    'confidence': 0.7 + abs(hist) / 10,
                    'price': None,
                    'timestamp': int(time.time() * 1000)
                })
            
            # Sinal Bollinger Bands
            bb = indicators.get('bb', {})
            upper = bb.get('upper', 110)
            middle = bb.get('middle', 100)
            lower = bb.get('lower', 90)
            
            # Obter o preço atual
            ticker = await self.data_loader.get_ticker(symbol)
            price = ticker.get('last', 100) if ticker else 100
            
            if price < lower:
                signals.append({
                    'type': 'bb',
                    'direction': 'buy',
                    'confidence': 0.7 + (lower - price) / lower * 10,
                    'price': None,
                    'timestamp': int(time.time() * 1000)
                })
            elif price > upper:
                signals.append({
                    'type': 'bb',
                    'direction': 'sell',
                    'confidence': 0.7 + (price - upper) / upper * 10,
                    'price': None,
                    'timestamp': int(time.time() * 1000)
                })
            
            # Sinal de padrões
            for pattern in patterns:
                if pattern == 'bullish_engulfing' or pattern == 'hammer':
                    signals.append({
                        'type': 'pattern',
                        'pattern': pattern,
                        'direction': 'buy',
                        'confidence': 0.8,
                        'price': None,
                        'timestamp': int(time.time() * 1000)
                    })
                elif pattern == 'bearish_engulfing' or pattern == 'shooting_star':
                    signals.append({
                        'type': 'pattern',
                        'pattern': pattern,
                        'direction': 'sell',
                        'confidence': 0.8,
                        'price': None,
                        'timestamp': int(time.time() * 1000)
                    })
            
            # Remove sinais duplicados
            signals = self._remove_duplicate_signals(signals)
            
            # Atualiza cache
            self.price_cache[symbol] = signals
            self.price_cache[f"{symbol}_timestamp"] = datetime.now()
            
            return signals
            
        except Exception as e:
            logger.error(f"Erro ao gerar sinais: {str(e)}")
            return []
    
    def _is_doji(self, candle: pd.Series) -> bool:
        """
        Verifica se o candle é um Doji.
        
        Args:
            candle: Candle.
            
        Returns:
            True se for um Doji.
        """
        try:
            # Verifica se o corpo do candle é pequeno
            body = abs(candle['close'] - candle['open'])
            shadow = candle['high'] - candle['low']
            
            # Se o corpo for menor que 10% da sombra, é um Doji
            return body <= shadow * 0.1
            
        except Exception as e:
            logger.error(f"Erro ao verificar Doji: {str(e)}")
            return False
    
    def _is_engulfing(self, candles: pd.DataFrame) -> str:
        """
        Verifica se os candles formam um padrão Engulfing.
        
        Args:
            candles: Candles.
            
        Returns:
            'bullish', 'bearish' ou ''.
        """
        try:
            # Verifica se temos pelo menos 2 candles
            if len(candles) < 2:
                return ''
            
            # Obtém os dois últimos candles
            prev = candles.iloc[0]
            curr = candles.iloc[1]
            
            # Verifica se o corpo do candle atual engloba o corpo do candle anterior
            prev_body = abs(prev['close'] - prev['open'])
            curr_body = abs(curr['close'] - curr['open'])
            
            # Se o corpo atual for maior que o corpo anterior
            if curr_body > prev_body:
                # Bullish Engulfing
                if prev['close'] < prev['open'] and curr['close'] > curr['open'] and curr['open'] < prev['close'] and curr['close'] > prev['open']:
                    return 'bullish'
                # Bearish Engulfing
                elif prev['close'] > prev['open'] and curr['close'] < curr['open'] and curr['open'] > prev['close'] and curr['close'] < prev['open']:
                    return 'bearish'
            
            return ''
            
        except Exception as e:
            logger.error(f"Erro ao verificar Engulfing: {str(e)}")
            return ''
    
    def _is_hammer(self, candle: pd.Series) -> bool:
        """
        Verifica se o candle é um Hammer.
        
        Args:
            candle: Candle.
            
        Returns:
            True se for um Hammer.
        """
        try:
            # Verifica se o corpo do candle é pequeno e a sombra inferior é grande
            body = abs(candle['close'] - candle['open'])
            upper_shadow = candle['high'] - max(candle['open'], candle['close'])
            lower_shadow = min(candle['open'], candle['close']) - candle['low']
            
            # Se o corpo for menor que 1/3 do candle e a sombra inferior for maior que 2x o corpo
            return body <= (candle['high'] - candle['low']) / 3 and lower_shadow >= 2 * body and upper_shadow <= body
            
        except Exception as e:
            logger.error(f"Erro ao verificar Hammer: {str(e)}")
            return False
    
    def _is_shooting_star(self, candle: pd.Series) -> bool:
        """
        Verifica se o candle é um Shooting Star.
        
        Args:
            candle: Candle.
            
        Returns:
            True se for um Shooting Star.
        """
        try:
            # Verifica se o corpo do candle é pequeno e a sombra superior é grande
            body = abs(candle['close'] - candle['open'])
            upper_shadow = candle['high'] - max(candle['open'], candle['close'])
            lower_shadow = min(candle['open'], candle['close']) - candle['low']
            
            # Se o corpo for menor que 1/3 do candle e a sombra superior for maior que 2x o corpo
            return body <= (candle['high'] - candle['low']) / 3 and upper_shadow >= 2 * body and lower_shadow <= body
            
        except Exception as e:
            logger.error(f"Erro ao verificar Shooting Star: {str(e)}")
            return False
    
    def _remove_duplicate_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove sinais duplicados.
        
        Args:
            signals: Sinais.
            
        Returns:
            Sinais sem duplicatas.
        """
        try:
            # Se não houver sinais, retorna lista vazia
            if not signals:
                return []
            
            # Agrupa sinais por direção
            buy_signals = [signal for signal in signals if signal['direction'] == 'buy']
            sell_signals = [signal for signal in signals if signal['direction'] == 'sell']
            
            # Ordena por confiança (do maior para o menor)
            buy_signals = sorted(buy_signals, key=lambda x: x.get('confidence', 0), reverse=True)
            sell_signals = sorted(sell_signals, key=lambda x: x.get('confidence', 0), reverse=True)
            
            # Considera apenas o sinal de maior confiança para cada direção
            result = []
            if buy_signals:
                result.append(buy_signals[0])
            if sell_signals:
                result.append(sell_signals[0])
            
            return result
            
        except Exception as e:
            logger.error(f"Erro ao remover sinais duplicados: {str(e)}")
            return signals 