import os
import logging
import hmac
import hashlib
import base64
import time
import json
from typing import Dict, List, Any, Optional
import requests
from datetime import datetime
import urllib.parse

logger = logging.getLogger(__name__)

class MarketAPI:
    def __init__(self, exchange: str = "kraken"):
        self.exchange = exchange
        self.logger = logging.getLogger("market_api")

        # Configure exchange API settings
        if self.exchange == "kraken":
            self.base_url = "https://api.kraken.com"
            self.api_key = os.environ.get("KRAKEN_API_KEY")
            self.api_secret = os.environ.get("KRAKEN_API_SECRET")

            if not all([self.api_key, self.api_secret]):
                raise ValueError("Missing Kraken API credentials")

            # Test API connection
            self._test_api_connection()

    def _test_api_connection(self):
        """Test API connection and verify credentials"""
        try:
            # Test public API first
            response = self._make_request("GET", "/0/public/Time")
            if not response or 'result' not in response:
                raise ValueError("Invalid API response")

            self.logger.info("Successfully connected to Kraken API")

        except Exception as e:
            self.logger.error(f"API connection test failed: {str(e)}")
            raise ValueError(f"Failed to connect to Kraken API: {str(e)}")

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker data for a symbol"""
        try:
            # Convert symbol format for Kraken (e.g., BTC-USDT -> XBTUSDT)
            kraken_symbol = self._convert_to_kraken_symbol(symbol)
            endpoint = f"/0/public/Ticker?pair={kraken_symbol}"
            response = self._make_request("GET", endpoint)

            if response and "result" in response:
                ticker_data = response["result"][kraken_symbol]
                return {
                    "symbol": symbol,
                    "price": float(ticker_data["c"][0]),  # Last trade closed price
                    "time": int(time.time() * 1000)
                }

            raise ValueError(f"Invalid response from Kraken API: {response}")

        except Exception as e:
            self.logger.error(f"Error getting ticker for {symbol}: {str(e)}")
            raise

    def get_portfolio(self) -> Dict[str, Any]:
        """Get current portfolio balances"""
        try:
            # Get account balance
            endpoint = "/0/private/Balance"
            response = self._make_request("POST", endpoint)

            if response and "result" in response:
                holdings = {}
                total_value = 0.0
                balances = response["result"]

                for currency, balance in balances.items():
                    amount = float(balance)
                    if amount > 0:
                        # Convert Kraken asset codes to standard format
                        std_currency = self._convert_from_kraken_asset(currency)
                        holdings[std_currency] = amount

                        # Calculate total value in USDT
                        if std_currency != "USDT":
                            try:
                                ticker = self.get_ticker(f"{std_currency}-USDT")
                                value = amount * ticker["price"]
                            except:
                                value = 0
                        else:
                            value = amount
                        total_value += value

                return {
                    "total_value": total_value,
                    "holdings": holdings,
                    "timestamp": int(time.time() * 1000)
                }

            raise ValueError(f"Invalid response from Kraken API: {response}")

        except Exception as e:
            self.logger.error(f"Error getting portfolio: {str(e)}")
            raise

    def _make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict[str, Any]:
        """Make an authenticated request to the Kraken API"""
        try:
            url = f"{self.base_url}{endpoint}"

            if endpoint.startswith("/0/private/"):
                if data is None:
                    data = {}

                nonce = str(int(time.time() * 1000))
                data["nonce"] = nonce

                # Create signature
                postdata = urllib.parse.urlencode(data)
                encoded = (str(data["nonce"]) + postdata).encode()
                message = endpoint.encode() + hashlib.sha256(encoded).digest()

                signature = hmac.new(
                    base64.b64decode(self.api_secret),
                    message,
                    hashlib.sha512
                ).hexdigest()

                headers = {
                    "API-Key": self.api_key,
                    "API-Sign": signature,
                    "Content-Type": "application/x-www-form-urlencoded"
                }

                response = requests.post(url, headers=headers, data=data)
            else:
                response = requests.get(url)

            if response.status_code == 200:
                result = response.json()
                if "error" in result and result["error"]:
                    raise ValueError(f"Kraken API error: {result['error']}")
                return result
            else:
                raise ValueError(f"API request failed: {response.text}")

        except Exception as e:
            self.logger.error(f"API request error: {str(e)}")
            raise

    def _convert_to_kraken_symbol(self, symbol: str) -> str:
        """Convert standard symbol format to Kraken format"""
        # Example: BTC-USDT -> XBTUSDT
        base, quote = symbol.split("-")
        conversions = {
            "BTC": "XBT",
            "USDT": "USD"  # Kraken uses USD for USDT
        }
        base = conversions.get(base, base)
        quote = conversions.get(quote, quote)
        return f"{base}{quote}"

    def _convert_from_kraken_asset(self, asset: str) -> str:
        """Convert Kraken asset code to standard format"""
        # Example: XXBT -> BTC
        conversions = {
            "XXBT": "BTC",
            "XETH": "ETH",
            "ZUSD": "USDT"  # Kraken uses USD, we convert to USDT
        }
        return conversions.get(asset, asset.lstrip("X").lstrip("Z"))

class MultiExchangeAPI:
    def __init__(self):
        try:
            self.kraken = MarketAPI(exchange="kraken")
            self.exchanges = {"kraken": self.kraken}
            logger.info("MultiExchangeAPI initialized with Kraken exchange")
        except Exception as e:
            logger.error(f"Error initializing Kraken exchange: {str(e)}")
            raise

    def get_market_data(self) -> Dict[str, Any]:
        """Get market data from Kraken"""
        market_data = {}
        for symbol in ["BTC-USDT", "ETH-USDT"]:
            try:
                ticker = self.kraken.get_ticker(symbol)
                market_data[symbol] = {
                    "price": ticker["price"],
                    "exchange": "kraken",
                    "time": ticker["time"]
                }
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                raise

        return market_data

    def get_portfolio(self) -> Dict[str, Any]:
        """Get portfolio data from Kraken"""
        try:
            return self.kraken.get_portfolio()
        except Exception as e:
            logger.error(f"Error getting portfolio: {str(e)}")
            raise

    def get_active_trades(self) -> List[Dict[str, Any]]:
        try:
            # Get open orders
            endpoint = "/0/private/OpenOrders"
            response = self.kraken._make_request("POST", endpoint)

            active_trades = []
            if response and "result" in response and "open" in response["result"]:
                for order_id, order in response["result"]["open"].items():
                    current_price = self.kraken.get_ticker(order["descr"]["pair"])["price"]

                    active_trades.append({
                        "id": order_id,
                        "symbol": order["descr"]["pair"],
                        "type": order["descr"]["type"],
                        "price": float(order["descr"]["price"]),
                        "volume": float(order["vol"]),
                        "current_price": current_price,
                        "status": order["status"]
                    })

            return active_trades

        except Exception as e:
            logger.error(f"Error getting active trades: {str(e)}")
            raise

    def close_trade(self, trade_id: str) -> Dict[str, Any]:
        try:
            # Cancel order
            endpoint = "/0/private/CancelOrder"
            response = self.kraken._make_request("POST", endpoint, {"txid": trade_id})

            if response and "result" in response:
                return {
                    "success": True,
                    "trade_id": trade_id,
                    "status": "cancelled"
                }
            else:
                raise ValueError(f"Failed to cancel trade: {response}")

        except Exception as e:
            logger.error(f"Error closing trade: {str(e)}")
            raise