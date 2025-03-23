"""Market data API integration module"""
import requests
import pandas as pd
import hmac
import hashlib
import base64
import time
import urllib.parse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import os
import logging
import json
from pathlib import Path

class MarketAPI:
    """Kraken market data API handler"""
    def __init__(self):
        # Load credentials from environment with better logging
        self._api_key: Optional[str] = None
        self._api_secret: Optional[str] = None
        self.base_url = "https://api.kraken.com"
        self.max_retries = 3
        self.retry_delay = 1  # seconds

        # Configure logging with more detailed format
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('KrakenAPI')

        # Initialize from environment if available
        env_key = os.getenv('KRAKEN_API_KEY')
        env_secret = os.getenv('KRAKEN_API_SECRET')
        if env_key:
            self.api_key = env_key
        if env_secret:
            self.api_secret = env_secret

        # Log credential status (without exposing values)
        self.logger.info(f"API Key status: {'Present' if self._api_key else 'Missing'}")
        self.logger.info(f"API Secret status: {'Present' if self._api_secret else 'Missing'}")

        # Initialize nonce tracking with microsecond precision
        self._nonce_file = Path('.last_nonce')
        self._initialize_nonce()

        # Validate credentials on initialization
        if self._api_key and self._api_secret:
            try:
                self.validate_credentials()
                self.logger.info("✅ API credentials validated successfully")
            except Exception as e:
                self.logger.error(f"❌ API credential validation failed: {str(e)}")

    def _initialize_nonce(self):
        """Initialize nonce tracking with microsecond precision and file persistence"""
        try:
            current_time_nonce = int(time.time() * 1000000)

            if self._nonce_file.exists():
                try:
                    last_nonce = int(self._nonce_file.read_text().strip())
                    # Add a large buffer to avoid conflicts
                    self._last_nonce = max(last_nonce + 10000, current_time_nonce)
                except (ValueError, IOError) as e:
                    self.logger.warning(f"Could not read nonce file: {e}")
                    self._last_nonce = current_time_nonce
            else:
                self._last_nonce = current_time_nonce

            # Save initial nonce
            self._nonce_file.write_text(str(self._last_nonce))
            self.logger.info(f"Initialized nonce tracking at: {self._last_nonce}")

        except Exception as e:
            self.logger.error(f"Error initializing nonce: {str(e)}")
            self._last_nonce = int(time.time() * 1000000)

    def _get_nonce(self) -> str:
        """Get unique nonce with improved synchronization and microsecond precision"""
        try:
            # Get current time in microseconds with added buffer
            current_nonce = int(time.time() * 1000000) + 100000

            # Ensure nonce is always increasing with a significant buffer
            if current_nonce <= self._last_nonce:
                current_nonce = self._last_nonce + 100000  # Larger buffer for safety

            self._last_nonce = current_nonce

            # Persist nonce to file for recovery
            try:
                self._nonce_file.write_text(str(current_nonce))
            except IOError as e:
                self.logger.warning(f"Could not persist nonce: {e}")

            return str(current_nonce)

        except Exception as e:
            self.logger.error(f"Error generating nonce: {str(e)}")
            # Fallback with added buffer
            return str(int(time.time() * 1000000) + 200000)

    @property
    def api_key(self) -> Optional[str]:
        return self._api_key

    @api_key.setter
    def api_key(self, value: str):
        if value and not isinstance(value, str):
            raise ValueError("API key must be a string")
        self._api_key = value

    @property
    def api_secret(self) -> Optional[str]:
        return self._api_secret

    @api_secret.setter
    def api_secret(self, value: str):
        if value:
            if not isinstance(value, str):
                raise ValueError("API secret must be a string")
            try:
                # Verify the secret is valid base64
                base64.b64decode(value)
                self._api_secret = value
            except Exception:
                raise ValueError("API secret must be base64 encoded")

    def validate_credentials(self) -> bool:
        """Validate API credentials with improved error handling"""
        if not self.api_key or not self.api_secret:
            self.logger.error("Missing API credentials")
            return False

        try:
            # Verify API secret format
            try:
                base64.b64decode(self.api_secret)
                self.logger.info("API secret format validated")
            except Exception as e:
                self.logger.error(f"Invalid API secret format: {str(e)}")
                return False

            # Test authentication with a simple balance query
            try:
                result = self._kraken_request('/0/private/Balance', {}, auth=True)
                if result.get('error'):
                    error_msg = ', '.join(result['error'])
                    self.logger.error(f"API credentials validation failed: {error_msg}")
                    return False

                self.logger.info("Successfully authenticated with Kraken API")
                return True

            except Exception as e:
                self.logger.error(f"Failed to validate credentials with API: {str(e)}")
                return False

        except Exception as e:
            self.logger.error(f"Unexpected error during credential validation: {str(e)}")
            return False

    def _get_kraken_signature(self, urlpath: str, data: Dict) -> str:
        """Create authenticated signature for Kraken API"""
        try:
            if not self.api_secret:
                raise ValueError("API secret is required for authenticated endpoints")

            postdata = urllib.parse.urlencode(data)
            encoded = (str(data['nonce']) + postdata).encode('utf-8')
            message = urlpath.encode('utf-8') + hashlib.sha256(encoded).digest()

            mac = hmac.new(base64.b64decode(self.api_secret), message, hashlib.sha512)
            sigdigest = base64.b64encode(mac.digest())
            return sigdigest.decode('utf-8')
        except Exception as e:
            self.logger.error(f"Error generating API signature: {str(e)}")
            raise

    def _kraken_request(self, uri_path: str, data: Dict, auth: bool = False) -> Dict[str,Any]:
        """Make request to Kraken API with retries and proper error handling"""
        if auth and (not self.api_key or not self.api_secret):
            raise ValueError("API key and secret required for authenticated endpoints")

        headers = {
            'User-Agent': 'QUALIA Trading System',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        retry_count = 0

        while retry_count < self.max_retries:
            try:
                if auth:
                    nonce = self._get_nonce()
                    data['nonce'] = nonce
                    headers['API-Key'] = self.api_key
                    headers['API-Sign'] = self._get_kraken_signature(uri_path, data)

                url = self.base_url + uri_path
                self.logger.debug(f"Making request to {url}")
                response = requests.post(
                    url,
                    headers=headers,
                    data=data,
                    timeout=30
                )
                response.raise_for_status()

                result = response.json()
                if result.get('error'):
                    error_msg = ', '.join(result['error'])
                    if 'EAPI:Invalid nonce' in error_msg and retry_count < self.max_retries - 1:
                        retry_count += 1
                        self.logger.warning(f"Invalid nonce error, retrying ({retry_count}/{self.max_retries})")
                        time.sleep(self.retry_delay * retry_count)
                        continue
                    self.logger.error(f"Kraken API error: {error_msg}")
                    raise Exception(f"Kraken API error: {error_msg}")

                return result

            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request failed (attempt {retry_count + 1}/{self.max_retries}): {str(e)}")
                if hasattr(e, 'response') and e.response:
                    self.logger.error(f"Response status: {e.response.status_code}")
                    self.logger.error(f"Response text: {e.response.text}")
                retry_count += 1
                if retry_count < self.max_retries:
                    time.sleep(self.retry_delay * retry_count)
                else:
                    raise Exception(f"Failed after {self.max_retries} attempts: {str(e)}")

        return {}  # Fallback empty response if all retries fail

    def get_balance(self) -> Dict[str, Any]:
        """Get account balance with improved error handling"""
        try:
            self.logger.info("Fetching account balance")
            result = self._kraken_request('/0/private/Balance', {}, auth=True)
            self.logger.debug(f"Balance result: {result}")

            if not result or not isinstance(result.get('result'), dict):
                self.logger.error("Invalid balance response format")
                self.logger.debug(f"Full response: {result}")
                raise ValueError("Invalid balance response format")

            # Convert all balances to float and handle asset names
            balances = {}
            for asset, amount in result['result'].items():
                try:
                    float_amount = float(amount)
                    if float_amount > 0:  # Only include non-zero balances
                        # Convert X-prefixed assets to standard names
                        if asset.startswith('X'):
                            if asset == 'XXBT':
                                asset_name = 'BTC'
                            elif asset == 'XETH':
                                asset_name = 'ETH'
                            else:
                                asset_name = asset[1:]  # Remove X prefix
                        # Convert Z-prefixed fiat to standard names
                        elif asset.startswith('Z'):
                            if asset == 'ZUSD':
                                asset_name = 'USD'
                            else:
                                asset_name = asset[1:]  # Remove Z prefix
                        else:
                            asset_name = asset
                        balances[asset_name] = float_amount
                except (ValueError, TypeError):
                    self.logger.error(f"Invalid balance format for {asset}: {amount}")
                    continue

            self.logger.info(f"Successfully fetched balance for {len(balances)} assets")
            return {'result': balances}

        except Exception as e:
            self.logger.error(f"Error fetching balance: {str(e)}")
            raise

    def get_all_balances(self) -> Dict[str, Any]:
        """Get all account balances with improved error handling, including zero balances"""
        try:
            self.logger.info("Fetching all account balances")
            result = self._kraken_request('/0/private/Balance', {}, auth=True)
            self.logger.debug(f"Balance result: {result}")

            if not result or not isinstance(result.get('result'), dict):
                self.logger.error("Invalid balance response format")
                self.logger.debug(f"Full response: {result}")
                raise ValueError("Invalid balance response format")

            # Convert all balances to float, including zero balances
            balances = {}
            for asset, amount in result['result'].items():
                try:
                    float_amount = float(amount)
                    # Convert X-prefixed assets to standard names
                    if asset.startswith('X'):
                        if asset == 'XXBT':
                            asset_name = 'BTC'
                        elif asset == 'XETH':
                            asset_name = 'ETH'
                        else:
                            asset_name = asset[1:]  # Remove X prefix
                    # Convert Z-prefixed fiat to standard names
                    elif asset.startswith('Z'):
                        if asset == 'ZUSD':
                            asset_name = 'USD'
                        else:
                            asset_name = asset[1:]  # Remove Z prefix
                    else:
                        asset_name = asset
                    balances[asset_name] = float_amount
                except (ValueError, TypeError):
                    self.logger.error(f"Invalid balance format for {asset}: {amount}")
                    continue

            self.logger.info(f"Successfully fetched all balances for {len(balances)} assets")
            return {'result': balances}

        except Exception as e:
            self.logger.error(f"Error fetching all balances: {str(e)}")
            raise

    def create_order(self, symbol: str, side: str, size: float, price: Optional[float] = None) -> Dict[str, Any]:
        """Create new order with minimum size validation"""
        try:
            # Get minimum order size
            min_size = self.get_minimum_order_size(symbol)

            # Adjust size if below minimum
            if size < min_size:
                self.logger.warning(f"Order size {size} below minimum {min_size}, adjusting to minimum")
                size = min_size

            data = {
                'pair': self._format_symbol_for_api(symbol),
                'type': side,
                'ordertype': 'market' if price is None else 'limit',
                'volume': str(size)
            }

            if price is not None:
                data['price'] = str(price)

            return self._kraken_request('/0/private/AddOrder', data, auth=True)

        except Exception as e:
            self.logger.error(f"Error creating order: {str(e)}")
            raise

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get open positions"""
        result = self._kraken_request('/0/private/OpenPositions', {}, auth=True)
        return list(result.get('result', {}).values())

    def get_klines(self, symbol: str, interval: str = "1m", limit: int = 720) -> pd.DataFrame:
        """Get OHLCV data from Kraken"""
        endpoint = f"{self.base_url}/0/public/OHLC"
        kraken_symbol = self._format_symbol_for_api(symbol)

        params = {
            "pair": kraken_symbol,
            "interval": self.timeframes[interval]
        }

        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get('error'):
                raise Exception(f"Kraken API error: {data['error']}")

            # Extract data for the specific pair
            pair_data = list(data['result'].values())[0]

            # Convert to DataFrame
            df = pd.DataFrame(pair_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 
                'vwap', 'volume', 'count'
            ])

            # Clean and format data
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            for col in ['open', 'high', 'low', 'close', 'volume', 'vwap']:
                df[col] = df[col].astype(float)

            # Calculate volatility
            df['volatility'] = (df['high'] - df['low']) / df['low'] * 100

            return df

        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch market data: {str(e)}")

    def get_available_pairs(self) -> List[str]:
        """Get available trading pairs from Kraken"""
        endpoint = f"{self.base_url}/0/public/AssetPairs"

        try:
            response = requests.get(endpoint)
            response.raise_for_status()
            data = response.json()

            if data.get('error'):
                raise Exception(f"Kraken API error: {data['error']}")

            available_pairs = set()

            for pair_name, pair_info in data['result'].items():
                # Skip dark pool pairs and inactive pairs
                if pair_name.startswith('.') or not pair_info.get('wsname'):
                    continue

                # Get the display name
                display_name = self._format_symbol_for_display(pair_info['wsname'])

                # Only add pairs that are in our priority list
                if display_name in self.priority_pairs:
                    available_pairs.add(display_name)

            # Sort pairs with priority pairs order
            sorted_pairs = [pair for pair in self.priority_pairs if pair in available_pairs]

            return sorted_pairs

        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch pairs: {str(e)}")

    def _format_symbol_for_api(self, symbol: str) -> str:
        """Convert display symbol to Kraken API format with improved validation"""
        try:
            if not symbol or '/' not in symbol:
                self.logger.error(f"Invalid symbol format: {symbol}")
                return symbol

            # Remove any futures contract suffix before processing
            symbol = symbol.split('.')[0]  # Remove .F if present

            quote, base = symbol.split('/')  # USD/CRYPTO format

            # Convert common names to Kraken format
            if base == 'BTC':
                base = 'XBT'
            elif base == 'DOGE':
                base = 'XDG'

            if quote == 'BTC':
                quote = 'XBT'
            elif quote == 'DOGE':
                quote = 'XDG'

            # For Kraken, format as BASEUSD for USD pairs
            if quote == 'USD':
                formatted_symbol = f"X{base}Z{quote}"
            else:
                formatted_symbol = f"{base}{quote}"

            self.logger.info(f"Formatted {symbol} to {formatted_symbol}")
            return formatted_symbol

        except Exception as e:
            self.logger.error(f"Error formatting symbol {symbol}: {str(e)}")
            return symbol

    def _format_symbol_for_display(self, wsname: str) -> str:
        """Convert Kraken websocket name to display format"""
        # Remove any .F suffix if present
        wsname = wsname.split('.')[0]

        try:
            base, quote = wsname.split('/')

            # Convert Kraken format to common names
            if base == 'XBT':
                base = 'BTC'
            elif base == 'XDG':
                base = 'DOGE'

            if quote == 'XBT':
                quote = 'BTC'
            elif quote == 'XDG':
                quote = 'DOGE'

            # Retorna no formato USD/CRYPTO
            return f"{quote}/{base}"
        except:
            return wsname

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker data from Kraken"""
        try:
            if not symbol or '/' not in symbol:
                raise ValueError(f"Invalid symbol format: {symbol}")

            kraken_symbol = self._format_symbol_for_api(symbol)
            self.logger.debug(f"Getting ticker for {symbol} (Kraken: {kraken_symbol})")

            response = requests.get(
                f"{self.base_url}/0/public/Ticker",
                params={"pair": kraken_symbol},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            if data.get('error') and len(data['error']) > 0:
                error_msg = ', '.join(data['error'])
                self.logger.error(f"Kraken API error for {symbol}: {error_msg}")
                raise Exception(f"Kraken API error: {error_msg}")

            return list(data['result'].values())[0]

        except Exception as e:
            self.logger.error(f"Failed to fetch ticker data for {symbol}: {str(e)}")
            raise

    @property
    def timeframes(self):
        return {
            "1m": 1,
            "5m": 5,
            "15m": 15
        }

    @property
    def priority_pairs(self):
        """List of primary trading pairs focused on USD pairs
        Only high-liquidity USD pairs to maximize trading efficiency"""
        return [
            "USD/BTC",  # Bitcoin: Most liquid pair
            "USD/ETH",  # Ethereum: Second largest by cap
            "USD/SOL"   # Solana: High volume on Kraken
        ]

    def get_minimum_order_size(self, symbol: str) -> float:
        """Get minimum order size for a symbol"""
        try:
            # Kraken's official minimum order sizes (updated for USD/CRYPTO format)
            default_minimums = {
                'USD/BTC': 0.0001,  # 0.0001 BTC (aprox. $4)
                'USD/ETH': 0.02,    # 0.02 ETH (aprox. $5)
                'USD/SOL': 1.0,     # 1.0 SOL (aprox. $5)
                'USD/USDT': 0.0001,
                'USD/DOGE': 0.02
            }
            min_size = default_minimums.get(symbol)

            if min_size is None:
                self.logger.warning(f"No predefined minimum size for {symbol}, using default 0.02")
                min_size = 0.02  # Conservative default

            self.logger.info(f"Minimum order size for {symbol}: {min_size}")
            return min_size

        except Exception as e:
            self.logger.error(f"Error getting minimum order size for {symbol}: {str(e)}")
            return 0.02  # Conservative default