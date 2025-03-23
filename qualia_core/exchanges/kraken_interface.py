"""
Interface de integração com a Kraken
"""

from krakenex import API
import os
import time
from typing import Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime
from core.logging.quantum_logger import quantum_logger
from core.exchanges.rate_limit_manager import RateLimitManager

class KrakenInterface:
    def __init__(self):
        """Inicializa cliente Kraken com credenciais seguras"""
        self.api_key = os.environ.get('KRAKEN_API_KEY')
        self.api_secret = os.environ.get('KRAKEN_API_SECRET')

        if not all([self.api_key, self.api_secret]):
            raise ValueError("Credenciais da Kraken não encontradas")

        self.api = API(key=self.api_key, secret=self.api_secret)
        self.rate_limiter = RateLimitManager(max_retries=3, initial_delay=2.0)

        # Initialize nonce with current timestamp
        self._nonce = int(time.time() * 1000)
        quantum_logger.info(
            "Kraken Interface initialized",
            {"initial_nonce": self._nonce}
        )

    def _get_nonce(self) -> int:
        """
        Get a unique, incrementing nonce for API requests.
        Ensures the nonce is always increasing and unique.
        """
        current_time = int(time.time() * 1000)
        self._nonce = max(current_time, self._nonce + 1)
        return self._nonce

    def _format_pair(self, symbol: str) -> str:
        """
        Formata par de trading para o formato da Kraken
        Ex: SOL/USDT -> SOLUSDT
        """
        base, quote = symbol.split('/')

        # Mapeamento de moedas conforme documentação Kraken
        currency_map = {
            'BTC': 'XBT',
            'USDT': 'USDT',
            'SOL': 'SOL',
            'ETH': 'XETH',
        }

        base = currency_map.get(base, base)
        quote = currency_map.get(quote, quote)

        # Formata par final
        pair = f"{base}{quote}"

        return pair

    def check_balance(self, symbol: str, side: str, size: float) -> Tuple[bool, Dict[str, Any]]:
        """
        Verifica se há saldo suficiente para executar a ordem com retry e validação robusta
        """
        max_retries = 3
        retry_count = 0
        retry_delay = 2  # segundos

        # Inicializa current_price
        current_price = None

        while retry_count < max_retries:
            try:
                # Obtém saldo atual com delay para rate limit
                self.rate_limiter.wait_if_needed('private', 'Balance')
                response = self.api.query_private('Balance')

                if 'error' in response and response['error']:
                    quantum_logger.error(
                        "Erro obtendo saldo da API",
                        {"error": response['error'], "attempt": retry_count + 1}
                    )
                    retry_count += 1
                    time.sleep(retry_delay)
                    continue

                quantum_logger.info(
                    "Resposta bruta da API de saldo",
                    {
                        "status": "success" if 'result' in response else "error",
                        "raw_response": response,
                        "available_keys": list(response.get('result', {}).keys()) if 'result' in response else None
                    }
                )

                balances = response.get('result', {})
                base, quote = symbol.split('/')

                # Mapeamento de moedas conforme documentação Kraken
                currency_map = {
                    'USDT': 'USDT',  # USDT permanece USDT
                    'USD': 'ZUSD',   # USD usa prefixo Z
                    'SOL': 'SOL',    # SOL permanece SOL
                    'BTC': 'XXBT',   # Bitcoin usa XXBT
                    'ETH': 'XETH',   # Ethereum usa XETH
                }

                kraken_base = currency_map.get(base, f"X{base}")
                kraken_quote = currency_map.get(quote, f"X{quote}")

                quantum_logger.info(
                    "Mapeamento de moedas e saldos disponíveis",
                    {
                        "original_pair": f"{base}/{quote}",
                        "kraken_pair": f"{kraken_base}/{kraken_quote}",
                        "available_balances": balances,
                        "base_balance_found": kraken_base in balances,
                        "quote_balance_found": kraken_quote in balances
                    }
                )

                # Busca saldos usando os códigos da Kraken
                base_balance = float(balances.get(kraken_base, 0))
                quote_balance = float(balances.get(kraken_quote, 0))

                quantum_logger.info(
                    "Saldos encontrados",
                    {
                        f"{kraken_base}_balance": base_balance,
                        f"{kraken_quote}_balance": quote_balance
                    }
                )

                # Aplica margem de segurança mais conservadora de 25%
                safety_margin = 0.75  # Usa apenas 75% do saldo disponível
                required_amount = 0

                if side == 'buy':
                    # Para compra, verifica saldo em quote currency (ex: USDT)
                    current_price = self._get_current_price(symbol)
                    # Adiciona 2% para cobrir flutuações de preço e taxas
                    required_amount = size * current_price * 1.02
                    has_sufficient = quote_balance * safety_margin >= required_amount
                else:
                    # Para venda, verifica saldo em base currency (ex: SOL)
                    required_amount = size
                    has_sufficient = base_balance * safety_margin >= required_amount

                balance_detail = {
                    'base_currency': base,
                    'quote_currency': quote,
                    'base_balance': base_balance * safety_margin,
                    'quote_balance': quote_balance * safety_margin,
                    'required_amount': required_amount,
                    'side': side,
                    'has_sufficient': has_sufficient,
                    'safety_margin': safety_margin,
                    'current_price': current_price
                }

                quantum_logger.info(
                    "Resultado da verificação de saldo",
                    {
                        "has_sufficient": has_sufficient,
                        "side": side,
                        "required_amount": required_amount,
                        "available_amount": balance_detail['quote_balance'] if side == 'buy' else balance_detail['base_balance']
                    }
                )

                return has_sufficient, balance_detail

            except Exception as e:
                quantum_logger.error(
                    "Erro verificando saldo",
                    {
                        "error": str(e),
                        "attempt": retry_count + 1
                    }
                )
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(retry_delay)
                    continue
                raise

        raise Exception("Máximo de tentativas excedido ao verificar saldo")

    def _get_current_price(self, symbol: str) -> float:
        """Obtém preço atual do par"""
        try:
            kraken_pair = self._format_pair(symbol)
            response = self._make_request(
                'Ticker',
                is_private=False,
                pair=kraken_pair
            )

            if 'result' in response and kraken_pair in response['result']:
                return float(response['result'][kraken_pair]['c'][0])
            raise Exception(f"Preço não encontrado para {symbol}")

        except Exception as e:
            quantum_logger.error(
                "Erro obtendo preço atual",
                {"error": str(e)}
            )
            raise

    def _make_request(self, method: str, is_private: bool, **kwargs) -> Dict[str, Any]:
        """
        Faz requisição à API com gerenciamento de rate limit e nonce

        Args:
            method: Método da API
            is_private: Se é endpoint privado
            **kwargs: Argumentos adicionais

        Returns:
            Dict com resposta da API
        """
        endpoint_type = 'private' if is_private else 'public'
        max_attempts = 3
        attempt = 0
        last_nonce = None

        while attempt < max_attempts:
            try:
                # Aplica delay se necessário
                self.rate_limiter.wait_if_needed(endpoint_type, method)

                # Add nonce for private requests
                if is_private:
                    last_nonce = self._get_nonce()
                    kwargs['nonce'] = str(last_nonce)
                    quantum_logger.info(
                        "Making private API request",
                        {
                            "method": method,
                            "nonce": last_nonce,
                            "attempt": attempt + 1
                        }
                    )

                # Faz requisição
                if is_private:
                    response = self.api.query_private(method, kwargs)
                else:
                    response = self.api.query_public(method, kwargs)

                # Verifica erros
                if 'error' in response and response['error']:
                    error_msg = str(response['error'])
                    if 'Invalid nonce' in error_msg:
                        quantum_logger.warning(
                            "Invalid nonce error",
                            {
                                "nonce_used": last_nonce,
                                "error": error_msg,
                                "attempt": attempt + 1
                            }
                        )
                        # Force nonce to be higher on next attempt
                        self._nonce += 1000
                        attempt += 1
                        continue

                    if self.rate_limiter.handle_error(endpoint_type, error_msg):
                        attempt += 1
                        continue
                    raise Exception(f"Erro da API: {error_msg}")

                # Sucesso
                self.rate_limiter.reset(endpoint_type)
                return response

            except Exception as e:
                error_msg = str(e)
                if 'Invalid nonce' in error_msg:
                    quantum_logger.warning(
                        "Invalid nonce error in exception",
                        {
                            "nonce_used": last_nonce,
                            "error": error_msg,
                            "attempt": attempt + 1
                        }
                    )
                    # Force nonce to be higher on next attempt
                    self._nonce += 1000
                    attempt += 1
                    continue

                if self.rate_limiter.handle_error(endpoint_type, error_msg):
                    attempt += 1
                    continue
                raise

        raise Exception(f"Máximo de tentativas excedido para {method}")

    def _convert_timeframe(self, timeframe: str) -> int:
        """
        Convert human-readable timeframe to Kraken API format

        Args:
            timeframe: String like '1min', '5m', '1h', '1d'

        Returns:
            int: Minutes for the interval
        """
        # Remove 'min' suffix if present
        tf = timeframe.lower().replace('min', '')

        # Convert to minutes
        if tf.endswith('m'):
            return int(tf.replace('m', ''))
        elif tf.endswith('h'):
            return int(tf.replace('h', '')) * 60
        elif tf.endswith('d'):
            return int(tf.replace('d', '')) * 1440

        # Default to numeric value assuming minutes
        try:
            return int(tf)
        except ValueError:
            raise ValueError(f"Invalid timeframe format: {timeframe}")

    def get_balances(self) -> Dict[str, float]:
        """
        Obtém saldos da conta com validação robusta e retry

        Returns:
            Dict[str, float]: Dicionário com os saldos por moeda
        """
        try:
            # Usa método existente check_balance para consistência
            has_balance, balance_info = self.check_balance('SOL/USDT', 'buy', 0.0)

            return {
                'SOL': balance_info['base_balance'],
                'USDT': balance_info['quote_balance']
            }

        except Exception as e:
            quantum_logger.error(
                "Erro obtendo saldos da Kraken",
                {"error": str(e)}
            )
            return {'SOL': 0.0, 'USDT': 0.0}

    def get_market_price(self, symbol: str) -> float:
        """
        Obtém preço atual do mercado para um par específico

        Args:
            symbol: Par de trading (ex: SOL/USDT)

        Returns:
            float: Preço atual do mercado
        """
        try:
            return self._get_current_price(symbol)
        except Exception as e:
            quantum_logger.error(
                "Erro obtendo preço de mercado",
                {"error": str(e), "symbol": symbol}
            )
            raise

    def get_market_data(self, symbol: str, timeframe: str = '1min') -> Dict[str, Any]:
        """Obtém dados do mercado com validação robusta de parâmetros"""
        try:
            quantum_logger.info(
                "Iniciando chamada get_market_data",
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "function": "get_market_data"
                }
            )

            # Converte formato do par para Kraken
            kraken_pair = self._format_pair(symbol)

            # Converte timeframe para formato da API
            try:
                interval = self._convert_timeframe(timeframe)
                quantum_logger.info(
                    "Parâmetros convertidos",
                    {
                        "kraken_pair": kraken_pair,
                        "interval": interval
                    }
                )
            except ValueError as e:
                quantum_logger.error(
                    "Erro convertendo timeframe",
                    {
                        "error": str(e),
                        "timeframe": timeframe
                    }
                )
                raise

            # Obtém OHLCV com retry e backoff
            response = self._make_request(
                'OHLC',
                is_private=False,
                pair=kraken_pair,
                interval=interval
            )

            if 'error' in response and response['error']:
                quantum_logger.error(
                    "Erro na resposta da API",
                    {
                        "error": response['error'],
                        "pair": kraken_pair,
                        "interval": interval
                    }
                )
                raise Exception(f"Erro da API: {response['error']}")

            if 'result' not in response or kraken_pair not in response['result']:
                quantum_logger.error(
                    "Dados inválidos recebidos da API",
                    {
                        "response": response,
                        "pair": kraken_pair
                    }
                )
                raise Exception(f"Dados inválidos recebidos da API: {response}")

            quantum_logger.info(
                "Dados de mercado obtidos com sucesso",
                {
                    "pair": kraken_pair,
                    "candles_count": len(response['result'][kraken_pair])
                }
            )

            return {
                'timestamp': datetime.now(),
                'data': response['result'][kraken_pair],
                'symbol': symbol,
                'timeframe': timeframe
            }

        except Exception as e:
            quantum_logger.error(
                "Erro obtendo dados do mercado",
                {
                    "error": str(e),
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "location": "get_market_data"
                }
            )
            raise

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Obtém status da ordem"""
        try:
            response = self._make_request(
                'QueryOrders',
                is_private=True,
                txid=order_id
            )

            return response['result'][order_id]

        except Exception as e:
            quantum_logger.error(
                "Erro obtendo status da ordem",
                {"error": str(e)}
            )
            raise

    def get_trading_fees(self, pair: str) -> Dict[str, float]:
        """
        Obtém as taxas de trading para um par específico

        Args:
            pair: Par de trading (ex: SOL/USDT)

        Returns:
            Dict com taxas maker/taker
        """
        try:
            # Formata par
            kraken_pair = self._format_pair(pair)

            # Obtém info do par
            response = self._make_request(
                'AssetPairs',
                is_private=False,
                pair=kraken_pair
            )

            if 'result' not in response or kraken_pair not in response['result']:
                raise Exception(f"Informações não encontradas para {pair}")

            pair_info = response['result'][kraken_pair]

            # Taxas padrão da Kraken (primeiro tier)
            # A API retorna array de arrays: [[volume, fee], ...] 
            # Pegamos a taxa do primeiro tier [0][1]
            fees = {
                'maker': float(pair_info.get('fees_maker', [[0, 0.16]])[0][1]) / 100,  # 0.16%
                'taker': float(pair_info.get('fees', [[0, 0.26]])[0][1]) / 100  # 0.26%
            }

            quantum_logger.info(
                "Taxas de trading obtidas",
                {
                    "pair": pair,
                    "fees": fees
                }
            )

            return fees

        except Exception as e:
            quantum_logger.error(
                "Erro obtendo taxas de trading",
                {"error": str(e)}
            )
            raise

    def _adjust_order_size(self, symbol: str, side: str, size: float) -> Tuple[float, Dict[str, Any]]:
        """
        Ajusta tamanho da ordem baseado no saldo disponível

        Args:
            symbol: Par de trading
            side: buy/sell
            size: Tamanho original desejado

        Returns:
            Tuple[float, Dict]: (tamanho_ajustado, info_saldo)
        """
        try:
            # Verifica saldo atual
            has_balance, balance_info = self.check_balance(symbol, side, size)

            if has_balance:
                return size, balance_info

            # Calcula tamanho máximo possível
            base, quote = symbol.split('/')
            if side == 'buy':
                current_price = self._get_current_price(symbol)
                max_size = (balance_info['quote_balance'] * 0.95) / current_price  # 95% do saldo
            else:
                max_size = balance_info['base_balance'] * 0.95  # 95% do saldo

            # Ajusta para o mínimo entre o desejado e o possível
            adjusted_size = min(size, max_size)

            quantum_logger.info(
                "Tamanho da ordem ajustado",
                {
                    "original_size": size,
                    "adjusted_size": adjusted_size,
                    "available_balance": balance_info[f"{side=='buy' and 'quote' or 'base'}_balance"],
                    "side": side
                }
            )

            return adjusted_size, balance_info

        except Exception as e:
            quantum_logger.error(
                "Erro ajustando tamanho da ordem",
                {"error": str(e)}
            )
            raise

    def execute_trade(
        self,
        simulation_mode: bool = False,
        **trade_params
    ) -> Dict[str, Any]:
        """
        Executa ordem de trading com validação quântica e suporte a simulação
        """
        max_retries = 5  # Aumentado para 5 tentativas
        retry_count = 0
        base_delay = 2  # Delay base em segundos
        delay = 0  # Inicializa delay

        while retry_count < max_retries:
            try:
                symbol = trade_params.get('symbol')
                side = trade_params.get('side')
                size = trade_params.get('size')
                validation = trade_params.get('quantum_validation', {})

                if not all([symbol, side, size]):
                    raise ValueError("Parâmetros incompletos")

                # Reduz tamanho da ordem progressivamente em caso de erro
                if retry_count > 0:
                    # Redução exponencial: 50%, 25%, 12.5%, 6.25%
                    reduction_factor = 0.5 ** retry_count
                    size = float(size) * reduction_factor
                    trade_params['size'] = size

                    # Aplica delay exponencial
                    delay = base_delay * (2 ** retry_count)  # 2s, 4s, 8s, 16s, 32s
                    time.sleep(delay)

                # Ajusta tamanho da ordem
                adjusted_size, balance_info = self._adjust_order_size(symbol, side, size)

                # Prepara ordem
                pair = self._format_pair(symbol)
                order_type = 'market'

                order_params = {
                    'pair': pair,
                    'type': side,
                    'ordertype': order_type,
                    'volume': str(adjusted_size)
                }

                # Log detalhado pré-execução
                quantum_logger.info(
                    f"Tentativa {retry_count + 1} de {max_retries}",
                    {
                        'order_params': order_params,
                        'adjusted_size': adjusted_size,
                        'retry_count': retry_count,
                        'delay': delay
                    }
                )

                if simulation_mode:
                    return {
                        'success': True,
                        'simulation': True,
                        'order_id': f"sim_{datetime.now().timestamp()}",
                        'params': order_params,
                        'adjusted_size': adjusted_size
                    }

                # Executa ordem real
                response = self._make_request(
                    'AddOrder',
                    is_private=True,
                    **order_params
                )

                if 'error' in response and response['error']:
                    if 'Insufficient funds' in str(response['error']):
                        quantum_logger.warning(
                            "Saldo insuficiente, reduzindo tamanho da ordem",
                            {
                                'original_size': size,
                                'adjusted_size': adjusted_size,
                                'retry_count': retry_count,
                                'next_delay': base_delay * (2 ** (retry_count + 1))
                            }
                        )
                        retry_count += 1
                        continue

                    raise Exception(f"Erro da API: {response['error']}")

                order_id = response['result'].get('txid', [''])[0]

                quantum_logger.info(
                    "Ordem executada com sucesso",
                    {
                        'order_id': order_id,
                        'params': order_params,
                        'retry_count': retry_count
                    }
                )

                return {
                    'success': True,
                    'order_id': order_id,
                    'params': order_params,
                    'adjusted_size': adjusted_size
                }

            except Exception as e:
                quantum_logger.error(
                    f"Erro na tentativa {retry_count + 1}",
                    {
                        'error': str(e),
                        'retry_count': retry_count,
                        'next_delay': base_delay * (2 ** (retry_count + 1)) if retry_count < max_retries else None
                    }
                )
                retry_count += 1
                continue

        return {
            'success': False,
            'reason': 'max_retries_exceeded',
            'details': 'Máximo de tentativas atingido com backoff exponencial'
        }
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 100) -> list:
        """
        Obtém dados OHLCV do par

        Args:
            symbol: Par de trading (ex: SOL/USDT)
            timeframe: Intervalo (1m, 5m, 15m, 30m, 1h, 4h, 1d)
            limit: Número de candles

        Returns:
            List of OHLCV data [timestamp, open, high, low, close, volume]
        """
        try:
            # Converte timeframe para minutos
            interval_map = {
                '1m': 1,
                '5m': 5,
                '15m': 15,
                '30m': 30,
                '1h': 60,
                '4h': 240,
                '1d': 1440
            }
            interval = interval_map.get(timeframe, 1)

            # Obtém dados OHLCV
            pair = self._format_pair(symbol)
            response = self._make_request(
                'OHLC',
                is_private=False,
                pair=pair,
                interval=interval
            )

            if 'error' in response and response['error']:
                raise Exception(f"Erro obtendo OHLCV: {response['error']}")

            if 'result' not in response or pair not in response['result']:
                raise Exception("Dados OHLCV inválidos")

            # Formata dados
            ohlcv = []
            for candle in response['result'][pair][-limit:]:
                ohlcv.append([
                    int(candle[0]) * 1000,  # timestamp in ms
                    float(candle[1]),        # open
                    float(candle[2]),        # high
                    float(candle[3]),        # low 
                    float(candle[4]),        # close
                    float(candle[6])         # volume
                ])

            return ohlcv

        except Exception as e:
            quantum_logger.error(
                "Erro obtendo OHLCV",
                {"error": str(e), "symbol": symbol, "timeframe": timeframe}
            )
            raise