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

logger = logging.getLogger(__name__)

class MarketAPI:
    def __init__(self, exchange: str = "kucoin", real_mode: bool = False):
        self.exchange = exchange
        self.real_mode = real_mode
        self.logger = logging.getLogger("market_api")

        # Configure exchange API settings
        if self.exchange == "kucoin":
            self.base_url = "https://api.kucoin.com"
            self.api_key = os.environ.get("KUCOIN_API_KEY")
            self.api_secret = os.environ.get("KUCOIN_API_SECRET")
            self.api_passphrase = os.environ.get("KUCOIN_API_PASSPHRASE")

            if real_mode and not all([self.api_key, self.api_secret, self.api_passphrase]):
                raise ValueError("Missing KuCoin API credentials for real mode")

            self.logger.info(f"Initialized KuCoin API in {'real' if real_mode else 'simulation'} mode")

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker data for a symbol"""
        if not self.real_mode:
            # Simulation mode - return dummy data
            current_time = int(time.time() * 1000)
            return {
                "symbol": symbol,
                "price": 50000.0,  # Dummy price
                "time": current_time
            }

        try:
            if self.exchange == "kucoin":
                endpoint = f"/api/v1/market/orderbook/level1?symbol={symbol}"
                response = self._make_request("GET", endpoint)

                if response and "data" in response:
                    data = response["data"]
                    return {
                        "symbol": symbol,
                        "price": float(data.get("price", 0)),
                        "time": data.get("time", int(time.time() * 1000))
                    }

                raise ValueError(f"Invalid response from KuCoin API: {response}")

        except Exception as e:
            self.logger.error(f"Error getting ticker for {symbol}: {str(e)}")
            # Return dummy data in case of error
            return {
                "symbol": symbol,
                "price": 50000.0,
                "time": int(time.time() * 1000),
                "error": str(e)
            }

    def get_portfolio(self) -> Dict[str, Any]:
        """Get current portfolio balances"""
        if not self.real_mode:
            return {
                "total_value": 10000.0,
                "holdings": {"USDT": 10000.0},
                "timestamp": int(time.time() * 1000)
            }

        try:
            if self.exchange == "kucoin":
                endpoint = "/api/v1/accounts"
                response = self._make_request("GET", endpoint)

                if response and "data" in response:
                    total_value = 0.0
                    holdings = {}

                    for account in response["data"]:
                        currency = account.get("currency", "")
                        balance = float(account.get("balance", 0))

                        if balance > 0:
                            holdings[currency] = balance
                            if currency == "USDT":
                                total_value += balance
                            else:
                                # Get current price for non-USDT assets
                                ticker = self.get_ticker(f"{currency}-USDT")
                                total_value += balance * ticker["price"]

                    return {
                        "total_value": total_value,
                        "holdings": holdings,
                        "timestamp": int(time.time() * 1000)
                    }

                raise ValueError(f"Invalid response from KuCoin API: {response}")

        except Exception as e:
            self.logger.error(f"Error getting portfolio: {str(e)}")
            return {
                "total_value": 0.0,
                "holdings": {},
                "timestamp": int(time.time() * 1000),
                "error": str(e)
            }

    def _make_request(self, method: str, endpoint: str, params: Dict = None) -> Dict[str, Any]:
        """Make an authenticated request to the exchange API"""
        if not self.real_mode:
            return {"status": "ok", "data": {}}

        try:
            url = f"{self.base_url}{endpoint}"
            timestamp = str(int(time.time() * 1000))

            # Build signature for KuCoin
            str_to_sign = f"{timestamp}{method}{endpoint}"
            if params:
                str_to_sign += json.dumps(params)

            signature = base64.b64encode(
                hmac.new(
                    self.api_secret.encode('utf-8'),
                    str_to_sign.encode('utf-8'),
                    hashlib.sha256
                ).digest()
            ).decode('utf-8')

            # KC-API-PASSPHRASE is also signed in v2
            passphrase = base64.b64encode(
                hmac.new(
                    self.api_secret.encode('utf-8'),
                    self.api_passphrase.encode('utf-8'),
                    hashlib.sha256
                ).digest()
            ).decode('utf-8')

            headers = {
                "KC-API-KEY": self.api_key,
                "KC-API-SIGN": signature,
                "KC-API-TIMESTAMP": timestamp,
                "KC-API-PASSPHRASE": passphrase,
                "KC-API-KEY-VERSION": "2",
                "Content-Type": "application/json"
            }

            if method == "GET":
                response = requests.get(url, headers=headers)
            else:
                response = requests.post(url, headers=headers, json=params or {})

            if response.status_code == 200:
                return response.json()
            else:
                raise ValueError(f"API request failed: {response.text}")

        except Exception as e:
            self.logger.error(f"API request error: {str(e)}")
            raise

class MultiExchangeAPI:
    def __init__(self, real_mode: bool = False):
        self.real_mode = real_mode
        try:
            self.kucoin = MarketAPI(exchange="kucoin", real_mode=real_mode)
            self.exchanges = {"kucoin": self.kucoin}
            logger.info(f"MultiExchangeAPI initialized in {'real' if real_mode else 'simulation'} mode")
        except Exception as e:
            logger.error(f"Error initializing exchanges: {str(e)}")
            self.exchanges = {}

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
            # For now, just use KuCoin as primary exchange
            if "kucoin" in self.exchanges:
                return self.kucoin.get_portfolio()
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
        #  This method requires significant adaptation.  The original relies on exchange.get_open_orders(),
        # which is not present in the new MarketAPI class.  Placeholder for future implementation.
        return []

    def close_trade(self, trade_id: str) -> Dict[str, Any]:
        # This method also requires significant adaptation to work with the new MarketAPI class.
        # It depends on get_active_trades and exchange.cancel_order(), neither of which are directly
        # compatible with the revised structure.  Placeholder for future implementation.
        return {"success": False, "message": "close_trade not implemented"}