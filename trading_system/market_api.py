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
    def __init__(self, exchange: str = "kraken", real_mode: bool = False):
        self.exchange = exchange
        self.real_mode = real_mode
        self.logger = logging.getLogger("market_api")
        self.simulation_mode_active = False
        self.last_error = None
        self.status_message = "Initializing..."

        # Configure exchange API settings
        if self.exchange == "kraken":
            self.base_url = "https://api.kraken.com"
            self.api_key = os.environ.get("KRAKEN_API_KEY")
            self.api_secret = os.environ.get("KRAKEN_API_SECRET")

            if real_mode and not all([self.api_key, self.api_secret]):
                self.simulation_mode_active = True
                self.last_error = "Missing API credentials"
                self.status_message = "Missing Kraken API credentials"
                self.logger.warning(self.status_message)
            else:
                # Test API connection
                self._test_api_connection()

    def _test_api_connection(self):
        """Test API connection and set appropriate status"""
        try:
            response = self._make_request("GET", "/0/public/Time")
            if response and 'result' in response:
                self.status_message = "Connected to Kraken API"
                self.simulation_mode_active = False
                self.last_error = None
            else:
                raise ValueError("Invalid API response")
        except Exception as e:
            self.simulation_mode_active = True
            self.last_error = str(e)
            self.status_message = "Failed to connect to Kraken API"
            self.logger.error(f"API connection test failed: {str(e)}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get detailed system status"""
        return {
            "trading_mode": "simulation" if self.simulation_mode_active else "real",
            "status_message": self.status_message,
            "error_details": self.last_error,
            "exchange": self.exchange,
            "timestamp": int(time.time() * 1000)
        }

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker data for a symbol"""
        if not self.real_mode or self.simulation_mode_active:
            # Simulation mode - return dummy data
            current_time = int(time.time() * 1000)
            base_prices = {
                "BTC-USDT": 50000.0,
                "ETH-USDT": 3000.0
            }
            price = base_prices.get(symbol, 1000.0)
            return {
                "symbol": symbol,
                "price": price,
                "time": current_time,
                "simulation": True
            }

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

            # If we get an error response, switch to simulation mode
            self.simulation_mode_active = True
            self.logger.warning(f"Switching to simulation mode due to API error: {response}")
            return self.get_ticker(symbol)

        except Exception as e:
            self.logger.error(f"Error getting ticker for {symbol}: {str(e)}")
            self.simulation_mode_active = True
            return self.get_ticker(symbol)

    def get_portfolio(self) -> Dict[str, Any]:
        """Get current portfolio balances"""
        if not self.real_mode or self.simulation_mode_active:
            return {
                "total_value": 10000.0,
                "holdings": {
                    "USDT": 8000.0,
                    "BTC": 0.02,
                    "ETH": 0.5
                },
                "timestamp": int(time.time() * 1000),
                "simulation": True
            }

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

            self.simulation_mode_active = True
            self.logger.warning(f"Switching to simulation mode due to API error: {response}")
            return self.get_portfolio()

        except Exception as e:
            self.logger.error(f"Error getting portfolio: {str(e)}")
            self.simulation_mode_active = True
            return self.get_portfolio()

    def _make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict[str, Any]:
        """Make an authenticated request to the Kraken API"""
        if not self.real_mode or self.simulation_mode_active:
            return {"result": {}}

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
                    "API-Sign": signature
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
                self.logger.error(f"API request failed: {response.text}")
                self.simulation_mode_active = True
                raise ValueError(f"API request failed: {response.text}")

        except Exception as e:
            self.logger.error(f"API request error: {str(e)}")
            self.simulation_mode_active = True
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
    def __init__(self, real_mode: bool = False):
        self.real_mode = real_mode
        try:
            self.kraken = MarketAPI(exchange="kraken", real_mode=real_mode)
            self.exchanges = {"kraken": self.kraken}
            logger.info(f"MultiExchangeAPI initialized in {'real' if real_mode else 'simulation'} mode")
        except Exception as e:
            logger.error(f"Error initializing exchanges: {str(e)}")
            self.exchanges = {}

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status from all exchanges"""
        if not self.exchanges:
            return {
                "trading_mode": "simulation",
                "status_message": "No exchanges initialized",
                "error_details": "Failed to initialize exchange connections",
                "timestamp": int(time.time() * 1000)
            }

        # Use Kraken status
        if "kraken" in self.exchanges:
            return self.kraken.get_system_status()

        return {
            "trading_mode": "simulation",
            "status_message": "No active exchanges",
            "error_details": None,
            "timestamp": int(time.time() * 1000)
        }

    def get_market_data(self) -> Dict[str, Any]:
        """Get market data from all configured exchanges"""
        market_data = {}
        for symbol in ["BTC-USDT", "ETH-USDT"]:
            try:
                # Try each exchange until we get valid data
                for name, exchange in self.exchanges.items():
                    try:
                        ticker = exchange.get_ticker(symbol)
                        if ticker and "price" in ticker:
                            market_data[symbol] = {
                                "price": ticker["price"],
                                "exchange": name,
                                "time": ticker["time"]
                            }
                            break
                    except Exception as e:
                        logger.warning(f"Error getting {symbol} data from {name}: {str(e)}")
                        continue

                if symbol not in market_data:
                    logger.warning(f"No valid data available for {symbol}")
                    # Use simulation data as fallback
                    market_data[symbol] = {
                        "price": 50000.0 if symbol == "BTC-USDT" else 3000.0,
                        "exchange": "simulation",
                        "time": int(time.time() * 1000)
                    }

            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                continue

        return market_data

    def get_portfolio(self) -> Dict[str, Any]:
        """Get portfolio data from all exchanges"""
        try:
            # Just use Kraken as primary exchange
            if "kraken" in self.exchanges:
                return self.kraken.get_portfolio()
            else:
                return {
                    "total_value": 10000.0,
                    "holdings": {"USDT": 10000.0},
                    "timestamp": int(time.time() * 1000)
                }
        except Exception as e:
            logger.error(f"Error getting portfolio: {str(e)}")
            return {
                "total_value": 0.0,
                "holdings": {},
                "timestamp": int(time.time() * 1000),
                "error": str(e)
            }

    def get_active_trades(self) -> List[Dict[str, Any]]:
        # Placeholder for future implementation
        return []

    def close_trade(self, trade_id: str) -> Dict[str, Any]:
        # Placeholder for future implementation
        return {"success": False, "message": "close_trade not implemented"}