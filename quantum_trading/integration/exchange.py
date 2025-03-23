"""
Exchange Module for QUALIA Trading System

Este módulo fornece a classe principal Exchange para interação com exchanges de criptomoedas.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
import os
import ccxt

# Configuração de logging
logger = logging.getLogger(__name__)

class Exchange:
    """
    Interface simplificada para exchanges de criptomoedas.
    Esta classe é usada pelo sistema QUALIA para interagir com diferentes exchanges.
    """

    def __init__(self, exchange_id: str, maker_fee: float = 0.001, taker_fee: float = 0.001, use_sandbox: bool = False):
        """
        Inicializa uma interface de exchange.
        
        Args:
            exchange_id: Identificador da exchange (ex: 'Kraken', 'Kucoin')
            maker_fee: Taxa para ordens maker (%)
            taker_fee: Taxa para ordens taker (%)
            use_sandbox: Se deve usar o modo sandbox para testes
        """
        self.exchange_id = exchange_id
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.use_sandbox = use_sandbox
        self.exchange = None
        
        # Inicializa a conexão com a exchange
        self._init_exchange()
        
        logger.info(f"Exchange {exchange_id} inicializada com taxas: maker={maker_fee}, taker={taker_fee}")
    
    def _init_exchange(self):
        """
        Inicializa a conexão com a exchange usando as credenciais do arquivo .env
        """
        try:
            # Configurar credenciais com base no ID da exchange
            if self.exchange_id.lower() == 'kucoin':
                if self.use_sandbox:
                    # Credenciais para sandbox
                    api_key = os.environ.get('KUCOIN_SANDBOX_API_KEY')
                    api_secret = os.environ.get('KUCOIN_SANDBOX_API_SECRET')
                    passphrase = os.environ.get('KUCOIN_SANDBOX_PASSPHRASE')
                    options = {'sandbox': True}
                else:
                    # Credenciais para produção
                    api_key = os.environ.get('KUCOIN_API_KEY')
                    api_secret = os.environ.get('KUCOIN_API_SECRET')
                    passphrase = os.environ.get('KUCOIN_PASSPHRASE')
                    options = {}
                
                # Cria instância da Kucoin
                self.exchange = ccxt.kucoin({
                    'apiKey': api_key,
                    'secret': api_secret,
                    'password': passphrase,
                    'enableRateLimit': True,
                    'options': options
                })
                
            elif self.exchange_id.lower() == 'kraken':
                # Credenciais Kraken (não tem sandbox)
                api_key = os.environ.get('KRAKEN_API_KEY')
                api_secret = os.environ.get('KRAKEN_API_SECRET')
                
                # Cria instância da Kraken
                self.exchange = ccxt.kraken({
                    'apiKey': api_key,
                    'secret': api_secret,
                    'enableRateLimit': True
                })
            
            logger.info(f"Conexão com {self.exchange_id} inicializada com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar conexão com {self.exchange_id}: {str(e)}")
            self.exchange = None
        
    def get_balance(self, asset: str = 'USDT') -> float:
        """
        Obtém o saldo de um determinado ativo na exchange.
        
        Args:
            asset: Símbolo do ativo (ex: 'USDT', 'BTC')
            
        Returns:
            Saldo disponível
        """
        try:
            if self.exchange is not None:
                # Tenta obter o saldo real da exchange
                balance = self.exchange.fetch_balance()
                if asset in balance['free']:
                    return float(balance['free'][asset])
                else:
                    logger.warning(f"Ativo {asset} não encontrado na exchange {self.exchange_id}")
                    return 0.0
            else:
                # Modo de simulação, retornamos valores simulados
                simulated_balances = {
                    'USDT': 10000.0,
                    'BTC': 0.5,
                    'ETH': 5.0,
                    'XMR': 50.0
                }
                
                balance = simulated_balances.get(asset, 0.0)
                logger.debug(f"Balance simulado para {asset} em {self.exchange_id}: {balance}")
                return balance
                
        except Exception as e:
            logger.error(f"Erro ao obter saldo para {asset}: {str(e)}")
            # Em caso de erro, retorna saldo simulado
            return {
                'USDT': 10000.0,
                'BTC': 0.5,
                'ETH': 5.0,
                'XMR': 50.0
            }.get(asset, 0.0)
        
    def get_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> List[List[Any]]:
        """
        Obtém dados OHLCV (Open, High, Low, Close, Volume) para um par de trading.
        
        Args:
            symbol: Par de trading (ex: 'BTC/USDT')
            timeframe: Intervalo de tempo (ex: '1m', '1h', '1d')
            limit: Número de candles a retornar
            
        Returns:
            Lista de candles [timestamp, open, high, low, close, volume]
        """
        try:
            if self.exchange is not None:
                # Tenta obter dados reais da exchange
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                if ohlcv and len(ohlcv) > 0:
                    logger.debug(f"Obtidos {len(ohlcv)} candles para {symbol} em {self.exchange_id}")
                    return ohlcv
            
            # Caso falhe ou não tenha exchange configurada, usa dados simulados
            import numpy as np
            import time
            
            # Dados simulados para teste
            current_time = int(time.time() * 1000)  # timestamp em milissegundos
            
            # Preço base para o símbolo (simulado)
            base_prices = {
                'BTC/USDT': 50000.0,
                'ETH/USDT': 3000.0,
                'XMR/USDT': 200.0
            }
            
            base_price = base_prices.get(symbol, 100.0)
            
            # Gera candles simulados
            candles = []
            for i in range(limit):
                timestamp = current_time - (limit - i) * 3600000  # 1 hora em milissegundos
                
                # Variação aleatória
                variation = np.random.normal(0, 0.01)
                
                price = base_price * (1 + variation)
                
                # Gera candle: [timestamp, open, high, low, close, volume]
                open_price = price
                close_price = price * (1 + np.random.normal(0, 0.005))
                high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.003)))
                low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.003)))
                volume = abs(np.random.normal(1000, 500))
                
                candle = [timestamp, open_price, high_price, low_price, close_price, volume]
                candles.append(candle)
                
            logger.debug(f"Gerados {len(candles)} candles simulados para {symbol} em {self.exchange_id}")
            return candles
        
        except Exception as e:
            logger.error(f"Erro ao obter dados OHLCV para {symbol}: {str(e)}")
            # Retorna lista vazia em caso de erro
            return []
            
    def create_order(self, symbol: str, type: str, side: str, amount: float, price: Optional[float] = None) -> Dict[str, Any]:
        """
        Cria uma ordem na exchange.
        
        Args:
            symbol: Par de trading (ex: 'BTC/USDT')
            type: Tipo de ordem ('limit' ou 'market')
            side: Lado da ordem ('buy' ou 'sell')
            amount: Quantidade a ser negociada
            price: Preço (apenas para ordens limit)
            
        Returns:
            Detalhes da ordem criada
        """
        try:
            if self.exchange is not None:
                # Tenta criar ordem real na exchange
                order = self.exchange.create_order(symbol, type, side, amount, price)
                logger.info(f"Ordem criada com sucesso: {order['id']} ({symbol} {side} {amount})")
                return order
            else:
                # Modo de simulação
                import time
                import uuid
                
                order_id = str(uuid.uuid4())
                current_time = int(time.time() * 1000)
                
                # Simula o preço atual
                current_price = price if price is not None else {
                    'BTC/USDT': 50000.0,
                    'ETH/USDT': 3000.0,
                    'XMR/USDT': 200.0
                }.get(symbol, 100.0)
                
                # Retorna ordem simulada
                simulated_order = {
                    'id': order_id,
                    'symbol': symbol,
                    'type': type,
                    'side': side,
                    'price': price,
                    'amount': amount,
                    'filled': amount,
                    'status': 'closed',
                    'timestamp': current_time,
                    'datetime': time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(current_time/1000)),
                    'fee': {
                        'cost': amount * current_price * (self.taker_fee if type == 'market' else self.maker_fee),
                        'currency': symbol.split('/')[1]
                    }
                }
                
                logger.info(f"Ordem simulada criada: {order_id} ({symbol} {side} {amount})")
                return simulated_order
                
        except Exception as e:
            logger.error(f"Erro ao criar ordem {symbol} {side} {amount}: {str(e)}")
            # Retorna erro
            return {
                'status': 'error',
                'error': str(e),
                'symbol': symbol,
                'type': type,
                'side': side,
                'amount': amount,
                'price': price
            } 