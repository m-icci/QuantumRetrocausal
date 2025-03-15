import os
import sys
import time
import logging
import hmac
import hashlib
import base64
import json
from typing import Dict, List, Any, Optional
import requests
from datetime import datetime
import urllib.parse

logger = logging.getLogger(__name__)

class KrakenAPI:
    def __init__(self):
        self.base_url = "https://api.kraken.com"
        self.api_key = os.environ.get("KRAKEN_API_KEY")
        self.api_secret = os.environ.get("KRAKEN_API_SECRET")

        if not all([self.api_key, self.api_secret]):
            raise ValueError("Kraken API credentials not configured")

        logger.info("Initializing Kraken API connection...")
        self._test_connection()

    def _test_connection(self):
        """Verify API connection and credentials"""
        try:
            # Test public API first
            response = self._make_request("GET", "/0/public/Time", None)
            if not response or 'result' not in response:
                raise ValueError("Unable to connect to Kraken API")

            # Test private API with minimal request
            response = self._make_request("POST", "/0/private/Balance", {})
            if not response or 'result' not in response:
                raise ValueError("Invalid API credentials")

            logger.info("Successfully connected to Kraken API")
        except Exception as e:
            logger.error(f"Failed to connect to Kraken API: {str(e)}")
            raise

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current market price"""
        try:
            # Convert symbol format (e.g., BTC-USD -> XBTUSD)
            kraken_symbol = self._convert_symbol(symbol)
            logger.debug(f"Getting ticker for {symbol} (Kraken: {kraken_symbol})")

            response = self._make_request("GET", f"/0/public/Ticker?pair={kraken_symbol}", None)
            if not response or 'result' not in response:
                raise ValueError(f"Failed to get ticker for {symbol}")

            ticker_data = response['result'][kraken_symbol]
            return {
                "symbol": symbol,
                "price": float(ticker_data['c'][0]),  # Last trade closed price
                "time": int(time.time() * 1000)
            }
        except Exception as e:
            logger.error(f"Error getting ticker for {symbol}: {str(e)}")
            raise

    def get_balance(self) -> Dict[str, Any]:
        """Get real account balance"""
        try:
            response = self._make_request("POST", "/0/private/Balance", {})
            if not response or 'result' not in response:
                raise ValueError("Failed to get account balance")

            balances = response['result']
            holdings = {}
            total_value = 0.0

            for asset, balance in balances.items():
                amount = float(balance)
                if amount > 0:
                    std_asset = self._standardize_asset(asset)
                    holdings[std_asset] = amount

                    if std_asset != "USD":
                        try:
                            ticker = self.get_ticker(f"{std_asset}-USD")
                            value = amount * ticker["price"]
                        except Exception as e:
                            logger.warning(f"Failed to get price for {std_asset}: {str(e)}")
                            value = 0
                    else:
                        value = amount
                    total_value += value

            return {
                "total_value": total_value,
                "holdings": holdings,
                "timestamp": int(time.time() * 1000)
            }
        except Exception as e:
            logger.error(f"Error getting balance: {str(e)}")
            raise

    def _make_request(self, method: str, endpoint: str, data: Optional[Dict]) -> Dict[str, Any]:
        """Make an authenticated API request"""
        try:
            url = f"{self.base_url}{endpoint}"
            logger.debug(f"Making {method} request to {endpoint}")

            headers = {}
            params = {}

            if endpoint.startswith("/0/private/"):
                if data is None:
                    data = {}

                # Add nonce to data
                data['nonce'] = str(int(time.time() * 1000000))  # Microsecond nonce

                # URL-encode the data
                encoded_data = urllib.parse.urlencode(data)

                # Create signature
                api_path = endpoint
                api_sha256 = hashlib.sha256(encoded_data.encode('utf-8')).digest()
                api_hash = api_path.encode('utf-8') + api_sha256
                api_secret = base64.b64decode(self.api_secret)
                api_signature = base64.b64encode(
                    hmac.new(api_secret, api_hash, hashlib.sha512).digest()
                ).decode('utf-8')

                headers.update({
                    'API-Key': self.api_key,
                    'API-Sign': api_signature,
                    'Content-Type': 'application/x-www-form-urlencoded'
                })

            # Make the request
            if method == "GET":
                response = requests.get(url, headers=headers, params=params)
            else:  # POST
                response = requests.post(url, headers=headers, data=data)

            if response.status_code != 200:
                error_msg = f"Request failed with status {response.status_code}: {response.text}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            result = response.json()
            if result.get('error'):
                error_msg = f"Kraken API error: {result['error']}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            return result

        except Exception as e:
            logger.error(f"API request error: {str(e)}")
            raise

    def _convert_symbol(self, symbol: str) -> str:
        """Convert standard symbol format to Kraken format"""
        try:
            base, quote = symbol.split("-")

            # Kraken uses XBT for Bitcoin
            conversions = {
                "BTC": "XBT",
                "USDT": "USD"  # Kraken uses USD
            }

            base = conversions.get(base, base)
            quote = conversions.get(quote, quote)

            # Add X/Z prefix for certain currencies
            if base in ["XBT", "ETH"]:
                base = "X" + base
            if quote == "USD":
                quote = "Z" + quote

            return base + quote

        except Exception as e:
            logger.error(f"Error converting symbol {symbol}: {str(e)}")
            raise

    def _standardize_asset(self, asset: str) -> str:
        """Convert Kraken asset code to standard format"""
        conversions = {
            "XXBT": "BTC",
            "XETH": "ETH",
            "ZUSD": "USD"
        }
        return conversions.get(asset, asset.lstrip("X").lstrip("Z"))

class TradingAPI:
    def __init__(self):
        logger.info("Initializing Trading API...")
        self.kraken = KrakenAPI()
        logger.info("Trading API initialized successfully")

    def get_market_data(self) -> Dict[str, Any]:
        """Get current market data"""
        try:
            market_data = {}
            for symbol in ["BTC-USD", "ETH-USD"]:
                ticker = self.kraken.get_ticker(symbol)
                market_data[symbol] = {
                    "price": ticker["price"],
                    "exchange": "kraken",
                    "time": ticker["time"]
                }
            return market_data
        except Exception as e:
            logger.error(f"Error getting market data: {str(e)}")
            raise

    def get_portfolio(self) -> Dict[str, Any]:
        """Get current portfolio status"""
        return self.kraken.get_balance()

    def get_active_trades(self) -> List[Dict[str, Any]]:
        """Get all active trades"""
        return []  # To be implemented with Kraken open orders functionality

    def close_trade(self, trade_id: str) -> Dict[str, Any]:
        """Close an active trade"""
        return {"success": False, "message": "Trading functionality coming soon"}