"""Exchange Interface for QUALIA Trading System"""
import logging
import ccxt
import numpy as np
import os
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Union
import json
import requests

logger = logging.getLogger(__name__)

@dataclass
class ExchangeStatus:
    """Exchange status information"""
    status: str
    message: str
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None

@dataclass
class ExchangeState:
    """Exchange state with quantum validation"""
    connected: bool = False
    quantum_validated: bool = False
    last_sync: datetime = datetime.now()
    error_rate: float = 0.0
    market_states: Dict[str, Any] = None
    is_sandbox: bool = True
    status: Optional[ExchangeStatus] = None

    def __post_init__(self):
        if self.market_states is None:
            self.market_states = {}
        if self.status is None:
            self.status = ExchangeStatus(
                status='INITIALIZING',
                message='Exchange state initialized',
                timestamp=datetime.now()
            )

class RateLimiter:
    """Simple rate limiter for API calls"""
    def __init__(self, max_requests: int = 15, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []

    def can_make_request(self) -> bool:
        now = time.time()
        self.requests = [req for req in self.requests if now - req < self.time_window]
        return len(self.requests) < self.max_requests

    def add_request(self):
        self.requests.append(time.time())

    def wait_if_needed(self):
        while not self.can_make_request():
            time.sleep(1)
        self.add_request()

class ExchangeInterface:
    """Quantum-enhanced exchange interface with advanced validation"""

    def __init__(
        self,
        simulation_mode: bool = True,
        quantum_dimension: int = 64,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        quantum_validation: bool = True,
        kraken_enabled: bool = False
    ):
        self.api_key = api_key or os.environ.get('KRAKEN_API_KEY')
        self.api_secret = api_secret or os.environ.get('KRAKEN_API_SECRET')
        self.quantum_dimension = quantum_dimension
        self.quantum_validation = quantum_validation
        self.simulation_mode = simulation_mode
        self.kraken_enabled = kraken_enabled
        self.trading_enabled = False
        self.exchange = None
        self.exchange_id = 'coinbase' if simulation_mode else 'kraken'
        self._state = ExchangeState()
        self.rate_limiter = RateLimiter()

        try:
            self._initialize_exchange()
            logger.info(f"Exchange interface initialized in {'simulation' if simulation_mode else 'live'} mode")
        except Exception as e:
            logger.error(f"Failed to initialize exchange interface: {str(e)}")
            self._fallback_to_simulation()
            if not simulation_mode:
                raise

    def _verify_api_credentials(self) -> bool:
        """Verify API credentials with enhanced checking"""
        if not all([self.api_key, self.api_secret]):
            logger.error("Missing API credentials")
            return False

        try:
            # Test API credentials with a simple public endpoint
            headers = {
                'API-Key': self.api_key,
                'API-Sign': self._generate_signature('/0/public/Time')
            }
            response = requests.get(
                'https://api.kraken.com/0/public/Time',
                headers=headers
            )

            if response.status_code == 403:
                logger.error("API authentication failed: Invalid credentials or permissions")
                return False

            if 'cloudflare' in response.headers.get('server', '').lower():
                logger.warning("Cloudflare protection detected")
                # Here you might implement additional Cloudflare handling if needed

            return response.status_code == 200

        except Exception as e:
            logger.error(f"API credential verification failed: {e}")
            return False

    def _initialize_exchange(self) -> None:
        """Initialize exchange with proper error handling"""
        try:
            exchange_config = {
                'enableRateLimit': True,
                'timeout': 30000,
                'rateLimit': 3000,  # 3 seconds between requests
                'headers': {
                    'User-Agent': 'QUALIA-Trading-Bot/1.0'
                }
            }

            if not self.simulation_mode and self.kraken_enabled:
                # Verify credentials before proceeding
                if not self._verify_api_credentials():
                    raise ValueError("Invalid API credentials")

                exchange_config.update({
                    'apiKey': self.api_key,
                    'secret': self.api_secret,
                    'passphrase': os.environ.get('KRAKEN_API_PASSPHRASE')
                })
                logger.info("Initializing Kraken in LIVE mode")
                self._state.is_sandbox = False
            else:
                exchange_config['sandbox'] = True
                logger.info("Initializing in SIMULATION mode")
                self._state.is_sandbox = True

            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class(exchange_config)

            # Test connection with rate limiting
            self.rate_limiter.wait_if_needed()
            self.exchange.load_markets()

            # Verify connection is working
            self.rate_limiter.wait_if_needed()
            self.exchange.fetch_ticker('BTC/USD')

        except ccxt.AuthenticationError as e:
            logger.error(f"Authentication failed: {str(e)}")
            self._fallback_to_simulation()
            if not self.simulation_mode:
                raise

        except ccxt.DDoSProtection as e:
            logger.error(f"Rate limit or DDoS protection triggered: {str(e)}")
            self._fallback_to_simulation()
            if not self.simulation_mode:
                raise

        except Exception as e:
            logger.error(f"Failed to initialize {self.exchange_id}: {str(e)}")
            self._fallback_to_simulation()
            if not self.simulation_mode:
                raise

    def _generate_signature(self, endpoint: str) -> str:
        """Generate API request signature"""
        try:
            import hmac
            import hashlib
            import base64

            nonce = str(int(time.time() * 1000))
            message = nonce + endpoint
            signature = hmac.new(
                base64.b64decode(self.api_secret),
                message.encode('utf-8'),
                hashlib.sha256
            ).digest()

            return base64.b64encode(signature).decode('utf-8')

        except Exception as e:
            logger.error(f"Failed to generate API signature: {e}")
            return ''

    @property  # Add property for state access
    def state(self) -> ExchangeState:
        return self._state

    def _update_status(self, status: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Update exchange status"""
        self._state.status = ExchangeStatus(
            status=status,
            message=message,
            timestamp=datetime.now(),
            details=details
        )
        logger.info(f"Exchange status updated: {status} - {message}")

    def _fallback_to_simulation(self) -> None:
        """Fallback to simulation mode"""
        logger.warning("Falling back to Coinbase simulation mode")
        self.simulation_mode = True
        self.kraken_enabled = False
        self.trading_enabled = False
        self.exchange_id = 'coinbase'

        try:
            self.exchange = ccxt.coinbase({
                'enableRateLimit': True,
                'timeout': 30000,
                'sandbox': True
            })
            self.exchange.load_markets()
            self._update_status('READY', 'Switched to simulation mode')
        except Exception as e:
            logger.error(f"Failed to initialize fallback exchange: {str(e)}")
            self._update_status('ERROR', f"Failed to initialize fallback exchange: {str(e)}")
            self.exchange = None

    def create_market_buy_order(self, symbol: str, amount: float) -> Dict[str, Any]:
        """Execute market buy order with enhanced error handling"""
        if not isinstance(amount, (int, float)) or amount <= 0:
            logger.error(f"Invalid amount specified: {amount}")
            return {
                'status': 'error',
                'message': 'Invalid amount specified',
                'symbol': symbol,
                'amount': amount,
                'type': 'market',
                'side': 'buy'
            }

        # Handle simulation mode first with clear status
        if self.simulation_mode or not self.kraken_enabled:
            logger.info("Executing simulated market buy order")
            return {
                'status': 'simulated',
                'message': 'Order simulated successfully',
                'symbol': symbol,
                'amount': amount,
                'type': 'market',
                'side': 'buy',
                'timestamp': datetime.now().timestamp(),
                'price': 0.0  # Simulated price
            }

        try:
            if not self.exchange:
                logger.error("Exchange not initialized")
                return {
                    'status': 'error',
                    'message': 'Exchange not initialized',
                    'symbol': symbol,
                    'amount': amount,
                    'type': 'market',
                    'side': 'buy'
                }

            # Execute the order
            self.rate_limiter.wait_if_needed() #Added rate limiting here.
            logger.info(f"Executing market buy order: {symbol}, amount: {amount}")
            response = self.exchange.create_market_buy_order(symbol, amount)

            # Handle null response
            if response is None:
                logger.error("Received null response from exchange API")
                return {
                    'status': 'error',
                    'message': 'Order execution failed - null response',
                    'symbol': symbol,
                    'amount': amount,
                    'type': 'market',
                    'side': 'buy'
                }

            # Log the raw response for debugging
            logger.debug(f"Raw exchange response: {response}")

            # Extract and normalize status
            status = str(response.get('status', '')).lower()
            logger.debug(f"Extracted status: {status}")

            # Check for success statuses
            if status in ['closed', 'done', 'filled', 'complete', 'success']:
                logger.info(f"Order executed successfully with status: {status}")
                return {
                    'status': 'executed',
                    'message': 'Order executed successfully',
                    'symbol': symbol,
                    'amount': amount,
                    'type': 'market',
                    'side': 'buy',
                    'order_details': response,
                    'price': response.get('price', 0.0)
                }

            # If status is not recognized as success, log and return error
            logger.warning(f"Unexpected order status received: {status}")
            return {
                'status': 'error',
                'message': f'Unexpected order status: {status}',
                'symbol': symbol,
                'amount': amount,
                'type': 'market',
                'side': 'buy',
                'raw_response': response
            }

        except Exception as e:
            error_msg = f"Market buy order failed: {str(e)}"
            logger.error(error_msg)
            return {
                'status': 'error',
                'message': error_msg,
                'symbol': symbol,
                'amount': amount,
                'type': 'market',
                'side': 'buy'
            }

    def create_market_sell_order(self, symbol: str, amount: float) -> Dict[str, Any]:
        """Execute market sell order with enhanced error handling"""
        if self.simulation_mode:
            return {
                'status': 'simulated',
                'message': 'Order simulated in sandbox mode',
                'symbol': symbol,
                'amount': amount,
                'type': 'market',
                'side': 'sell'
            }

        try:
            if not self.exchange:
                raise ValueError("Exchange not initialized")

            self.rate_limiter.wait_if_needed() #Added rate limiting here.
            response = self.exchange.create_market_sell_order(symbol, amount)
            return self._process_kraken_response(response)

        except Exception as e:
            error_msg = f"Market sell order failed: {str(e)}"
            logger.error(error_msg)
            return {
                'status': 'error',
                'message': error_msg,
                'symbol': symbol,
                'amount': amount,
                'type': 'market',
                'side': 'sell'
            }

    def _process_kraken_response(self, response: Union[str, dict]) -> Dict[str, Any]:
        """Process Kraken API response with proper error handling"""
        try:
            if response is None:
                logger.error("Received null response from Kraken API")
                return {
                    'status': 'error',
                    'message': 'Received null response from API',
                    'raw_response': None
                }

            # Convert string response to dict if needed
            if isinstance(response, str):
                try:
                    response = json.loads(response)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode response as JSON: {response}")
                    return {
                        'status': 'error',
                        'message': f'Invalid response format: {str(e)}',
                        'raw_response': response
                    }

            # Process dictionary response
            if isinstance(response, dict):
                # Check for explicit error field
                if 'error' in response and response['error']:
                    error_msg = str(response['error'])
                    logger.error(f"Kraken API error: {error_msg}")
                    return {
                        'status': 'error',
                        'message': error_msg,
                        'raw_response': response
                    }

                # Extract and normalize status
                status = str(response.get('status', '')).lower()
                if status in ['closed', 'done', 'filled', 'complete', 'success']:
                    status = 'executed'

                # Return normalized response
                return {
                    'status': status,
                    'orderId': response.get('id'),
                    'symbol': response.get('symbol'),
                    'type': response.get('type'),
                    'side': response.get('side'),
                    'price': response.get('price'),
                    'amount': response.get('amount'),
                    'raw_response': response
                }

            # Handle unexpected response type
            logger.error(f"Invalid response type: {type(response)}")
            return {
                'status': 'error',
                'message': f'Invalid response type: {type(response)}',
                'raw_response': response
            }

        except Exception as e:
            logger.error(f"Failed to process Kraken response: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'raw_response': response
            }

    def fetch_market_data(self, symbol: str) -> Optional[np.ndarray]:
        """Fetch market data with proper error handling"""
        try:
            if not self.exchange:
                logger.error("Exchange not initialized")
                return None

            self.rate_limiter.wait_if_needed() #Added rate limiting here.
            ohlcv = self.exchange.fetch_ohlcv(symbol)
            if not ohlcv:
                logger.warning(f"No data returned for {symbol}")
                return None

            return np.array(ohlcv)

        except Exception as e:
            logger.error(f"Failed to fetch market data for {symbol}: {str(e)}")
            return None

    def get_market_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get market information"""
        try:
            if not self.exchange:
                raise ValueError("Exchange not initialized")
            self.rate_limiter.wait_if_needed() #Added rate limiting here.
            return self.exchange.market(symbol)
        except Exception as e:
            logger.error(f"Failed to get market info for {symbol}: {str(e)}")
            return None

    def get_balance(self) -> Optional[Dict[str, Any]]:
        """Get account balance"""
        try:
            if not self.exchange:
                raise ValueError("Exchange not initialized")

            if self.simulation_mode:
                return {
                    'total': {'USDT': 10000.0},
                    'free': {'USDT': 10000.0},
                    'used': {'USDT': 0.0}
                }

            self.rate_limiter.wait_if_needed() #Added rate limiting here.
            return self.exchange.fetch_balance()
        except Exception as e:
            logger.error(f"Failed to fetch balance: {str(e)}")
            return None

    def enable_kraken(self) -> bool:
        """Enable Kraken exchange connection"""
        try:
            if not self.kraken_enabled:
                if not all([
                    os.environ.get('KRAKEN_API_KEY'),
                    os.environ.get('KRAKEN_API_SECRET')
                ]):
                    logger.error("Missing Kraken API credentials")
                    return False

                try:
                    test_exchange = ccxt.kraken({
                        'apiKey': os.environ.get('KRAKEN_API_KEY'),
                        'secret': os.environ.get('KRAKEN_API_SECRET'),
                        'enableRateLimit': True
                    })
                    test_exchange.fetch_balance()
                except Exception as e:
                    logger.error(f"Kraken validation failed: {str(e)}")
                    return False

                self.kraken_enabled = True
                self.trading_enabled = True
                self.exchange_id = 'kraken'
                self.simulation_mode = False

                try:
                    self._initialize_exchange()
                    logger.info("Kraken exchange enabled - Switched to LIVE trading mode")
                    return True
                except Exception as e:
                    logger.error(f"Failed to initialize Kraken exchange: {str(e)}")
                    self._fallback_to_simulation()
                    return False

            return self.kraken_enabled

        except Exception as e:
            logger.error(f"Failed to enable Kraken: {str(e)}")
            self._fallback_to_simulation()
            return False

    def disable_kraken(self) -> bool:
        """Disable Kraken and switch to simulation mode"""
        try:
            if self.kraken_enabled:
                self._fallback_to_simulation()
                logger.info("Kraken exchange disabled - Switched to SIMULATION mode with Coinbase data")
            return True
        except Exception as e:
            logger.error(f"Failed to disable Kraken: {str(e)}")
            return False

    def verify_connection(self) -> ExchangeStatus:
        """Verify exchange connection status"""
        try:
            if not self.exchange:
                return ExchangeStatus(
                    status='ERROR',
                    message='Exchange not initialized',
                    timestamp=datetime.now()
                )

            try:
                self.rate_limiter.wait_if_needed() #Added rate limiting here.
                self.exchange.fetch_time()
                return ExchangeStatus(
                    status='READY',
                    message=f"{self.exchange_id} connection operational",
                    timestamp=datetime.now(),
                    details={
                        'mode': 'LIVE' if not self.simulation_mode else 'SIMULATION',
                        'kraken_enabled': self.kraken_enabled,
                        'trading_enabled': self.trading_enabled
                    }
                )
            except Exception as e:
                return ExchangeStatus(
                    status='ERROR',
                    message=f"Connection verification failed: {str(e)}",
                    timestamp=datetime.now(),
                    details={'error': str(e)}
                )

        except Exception as e:
            return ExchangeStatus(
                status='ERROR',
                message=f"Connection verification failed: {str(e)}",
                timestamp=datetime.now(),
                details={'error': str(e)}
            )

    def get_order_status(self, order_id: str, symbol: Optional[str] = None) -> Optional[Dict]:
        """Get order status - only available in live mode"""
        if self.simulation_mode:
            return {
                'status': 'simulated',
                'message': 'Order status check simulated in sandbox mode',
                'order_id': order_id
            }

        try:
            if not self.exchange:
                raise ValueError("Exchange not initialized")

            self.rate_limiter.wait_if_needed() #Added rate limiting here.
            logger.info(f"Checking order status: {order_id}")
            return self.exchange.fetch_order(order_id, symbol)

        except Exception as e:
            error_msg = f"Order status check failed: {str(e)}"
            logger.error(error_msg)
            return None

    def _validate_credentials(self) -> bool:
        """Validate API credentials"""
        if not all([self.api_key, self.api_secret]):
            logger.warning("Missing Kraken API credentials")
            return False

        if not all(isinstance(cred, str) and cred.strip()
                  for cred in [self.api_key, self.api_secret]):
            logger.warning("Invalid Kraken API credentials format")
            return False

        return True
    
    def fetch_balance(self):
        """
        Fetch the account balance from the exchange.
        Returns a dictionary with the following structure:
        {
            'total': {'BTC': 0.1, 'USDT': 100.0, ...},
            'free': {'BTC': 0.1, 'USDT': 100.0, ...},
            'used': {'BTC': 0.0, 'USDT': 0.0, ...},
            'error': None or error message
        }
        """
        try:
            self.rate_limiter.wait_if_needed()
            
            # Verificar se a exchange está configurada
            if not self.exchange:
                logger.error("Exchange não configurada")
                return self._empty_balance_result("Exchange não configurada")
            
            # Obter saldos
            raw_balance = self.exchange.fetch_balance()
            if not isinstance(raw_balance, dict):
                logger.error(f"Formato de saldo inesperado: {type(raw_balance)}")
                return self._empty_balance_result(f"Formato de saldo inesperado: {type(raw_balance)}")

            processed_balance = {
                'total': {},
                'free': {},
                'used': {},
                'error': None
            }

            # Processar saldos por moeda
            for currency in raw_balance.keys():
                if currency in ['total', 'free', 'used', 'info']:
                    continue

                try:
                    currency_data = raw_balance[currency]
                    if isinstance(currency_data, dict):
                        # Converter valores para float, tratando None como 0
                        free = float(currency_data.get('free', 0) or 0)
                        used = float(currency_data.get('used', 0) or 0)
                        total = float(currency_data.get('total', 0) or 0)

                        # Só incluir se houver algum saldo
                        if total > 0 or free > 0 or used > 0:
                            # Normalizar nomes de moedas
                            clean_currency = self._normalize_currency_name(currency)
                            processed_balance['total'][clean_currency] = total
                            processed_balance['free'][clean_currency] = free
                            processed_balance['used'][clean_currency] = used
                            
                            logger.debug(f"Saldo processado para {clean_currency}: total={total}, free={free}, used={used}")
                except Exception as e:
                    logger.warning(f"Erro ao processar saldo para {currency}: {str(e)}")
                    continue

            logger.info(f"Saldos obtidos com sucesso: {len(processed_balance['total'])} moedas encontradas")
            return processed_balance

        except Exception as e:
            error_msg = f"Erro ao obter saldos: {str(e)}"
            logger.error(error_msg)
            return self._empty_balance_result(error_msg)

    def _empty_balance_result(self, error_message):
        """Retorna um resultado vazio com mensagem de erro"""
        return {
            'total': {},
            'free': {},
            'used': {},
            'error': error_message
        }

    def _normalize_currency_name(self, currency):
        """Normaliza nomes de moedas para um formato padrão"""
        # Remover prefixos da Kraken
        if currency.startswith('X'):
            if currency == 'XXBT':
                return 'BTC'
            elif currency == 'XETH':
                return 'ETH'
            elif currency == 'XDG':
                return 'DOGE'
            return currency[1:]
        elif currency.startswith('Z'):
            return currency[1:]
        
        # Mapeamentos específicos
        currency_map = {
            'XBT': 'BTC',
        }
        return currency_map.get(currency, currency)

    def fetch_my_trades(self, symbol):
        """Fetch the trade history for the specified trading pair."""
        if not self.kraken_enabled:
            logger.error("Kraken is not enabled.")
            return None
        try:
            self.rate_limiter.wait_if_needed() #Added rate limiting here.
            return self.exchange.fetch_my_trades(symbol)
        except Exception as e:
            logger.error(f"Failed to fetch trades: {str(e)}")
            return None
    
    def place_order(self, symbol: str, order_type: str, side: str, amount: float, price: Optional[float] = None, params: Dict = None) -> Optional[Dict]:
        """Trading operation - only available in live mode with proper validation"""
        if self.simulation_mode or not self.kraken_enabled:
            logger.warning("Trading disabled - running in SIMULATION mode")
            return {
                'status': 'simulated',
                'message': 'Order simulated in sandbox mode',
                'symbol': symbol,
                'type': order_type,
                'side': side,
                'amount': amount,
                'price': price
            }

        try:
            if not self.exchange:
                raise ValueError("Exchange not initialized")

            if not all([self.api_key, self.api_secret]):
                raise ValueError("Missing API credentials for live trading")

            self.rate_limiter.wait_if_needed() #Added rate limiting here.
            logger.info(f"Placing {order_type} {side} order: {symbol}, amount: {amount}, price: {price}")
            return self.exchange.create_order(symbol, order_type, side, amount, price, params)

        except Exception as e:
            error_msg = f"Order placement failed: {str(e)}"
            logger.error(error_msg)
            return {
                'status': 'error',
                'error': error_msg,
                'symbol': symbol,
                'type': order_type,
                'side': side,
                'amount': amount,
                'price': price
            }

    def execute_market_order(self, symbol: str, side: str, amount: float) -> Optional[Dict[str, Any]]:
        """Execute market order with proper error handling"""
        try:
            if not self.exchange:
                raise ValueError("Exchange not initialized")

            if self.simulation_mode:
                return {
                    'status': 'simulated',
                    'symbol': symbol,
                    'side': side,
                    'amount': amount,
                    'timestamp': datetime.now().timestamp()
                }

            self.rate_limiter.wait_if_needed() #Added rate limiting here.
            if side == 'buy':
                return self.exchange.create_market_buy_order(symbol, amount)
            else:
                return self.exchange.create_market_sell_order(symbol, amount)

        except Exception as e:
            logger.error(f"Failed to execute market order: {str(e)}")
            return None


    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get currently open trading positions

        Args:
            symbol: Optional trading pair to filter positions

        Returns:
            List of open position dictionaries with keys:
            symbol, side, amount, entry_price, current_price, pnl
        """
        try:
            if self.simulation_mode:
                return []

            if not self.exchange:
                logger.warning("Exchange not initialized")
                return []

            self.rate_limiter.wait_if_needed()
            positions = self.exchange.fetch_open_positions(symbol) if symbol else self.exchange.fetch_positions()

            # Format positions
            return [{
                'symbol': pos['symbol'],
                'side': pos['side'],
                'amount': float(pos['contracts'] or pos['amount']),
                'entry_price': float(pos['entryPrice']),
                'current_price': float(pos['markPrice']),
                'pnl': float(pos['unrealizedPnl'])
            } for pos in positions]

        except Exception as e:
            logger.error(f"Error fetching positions: {str(e)}")
            return []

    def fetch_positions(self) -> List[Dict[str, Any]]:
        """Alias for get_open_positions to maintain compatibility"""
        return self.get_open_positions()

class ExchangeInitializationError(Exception):
    pass

class GeoRestrictionError(Exception):
    pass