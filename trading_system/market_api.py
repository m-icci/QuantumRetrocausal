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
import numpy as np

logger = logging.getLogger(__name__)

class ExchangeAPI:
    def __init__(self, exchange: str):
        self.exchange = exchange
        self.api_key = os.environ.get(f"{exchange.upper()}_API_KEY")
        self.api_secret = os.environ.get(f"{exchange.upper()}_API_SECRET")
        self.api_passphrase = os.environ.get(f"{exchange.upper()}_API_PASSPHRASE")
        
        if not all([self.api_key, self.api_secret]):
            raise ValueError(f"Missing API credentials for {exchange}")
            
        self.base_urls = {
            "kucoin": "https://api.kucoin.com",
            "kraken": "https://api.kraken.com"
        }
        self.base_url = self.base_urls[exchange]
        
    def _sign_request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        timestamp = str(int(time.time() * 1000))
        
        if self.exchange == "kucoin":
            data_str = "" if not data else json.dumps(data)
            sign_str = f"{timestamp}{method}{endpoint}{data_str}"
            signature = base64.b64encode(
                hmac.new(
                    self.api_secret.encode('utf-8'),
                    sign_str.encode('utf-8'),
                    hashlib.sha256
                ).digest()
            ).decode()
            
            passphrase = base64.b64encode(
                hmac.new(
                    self.api_secret.encode('utf-8'),
                    self.api_passphrase.encode('utf-8'),
                    hashlib.sha256
                ).digest()
            ).decode()
            
            headers = {
                "KC-API-SIGN": signature,
                "KC-API-TIMESTAMP": timestamp,
                "KC-API-KEY": self.api_key,
                "KC-API-PASSPHRASE": passphrase,
                "KC-API-KEY-VERSION": "2"
            }
        else:
            # Kraken signing
            sign_str = f"{timestamp}{method}{endpoint}"
            signature = hmac.new(
                base64.b64decode(self.api_secret),
                sign_str.encode(),
                hashlib.sha256
            ).hexdigest()
            
            headers = {
                "API-Key": self.api_key,
                "API-Sign": signature
            }
            
        return headers

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        try:
            if self.exchange == "kucoin":
                endpoint = f"/api/v1/market/orderbook/level1?symbol={symbol}"
                headers = self._sign_request("GET", endpoint)
                response = requests.get(f"{self.base_url}{endpoint}", headers=headers)
            else:
                endpoint = "/0/public/Ticker"
                headers = self._sign_request("GET", endpoint)
                response = requests.get(f"{self.base_url}{endpoint}?pair={symbol}", headers=headers)
                
            response.raise_for_status()
            data = response.json()
            
            if self.exchange == "kucoin":
                return {
                    "symbol": symbol,
                    "price": float(data["data"]["price"]),
                    "time": data["data"]["time"]
                }
            else:
                # Parse Kraken response
                kraken_symbol = symbol.replace("-", "")
                ticker = data["result"][kraken_symbol]
                return {
                    "symbol": symbol,
                    "price": float(ticker["c"][0]),
                    "time": int(time.time() * 1000)
                }
                
        except Exception as e:
            logger.error(f"Error getting ticker for {symbol} on {self.exchange}: {str(e)}")
            raise

class MultiExchangeAPI:
    def __init__(self):
        self.exchanges = {
            "kucoin": ExchangeAPI("kucoin"),
            "kraken": ExchangeAPI("kraken")
        }
        self.pairs = ["BTC-USDT", "ETH-USDT"]
        
    def get_market_data(self) -> Dict[str, Any]:
        market_data = {}
        for pair in self.pairs:
            best_price = None
            best_exchange = None
            
            for name, exchange in self.exchanges.items():
                try:
                    ticker = exchange.get_ticker(pair)
                    if best_price is None or ticker["price"] < best_price:
                        best_price = ticker["price"]
                        best_exchange = name
                except Exception as e:
                    logger.warning(f"Failed to get {pair} price from {name}: {str(e)}")
                    
            if best_price is not None:
                market_data[pair] = {
                    "price": best_price,
                    "exchange": best_exchange,
                    "time": int(time.time() * 1000)
                }
                
        return market_data
        
    def get_portfolio(self) -> Dict[str, Any]:
        total_value = 0
        holdings = {}
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                # Get account balances
                balances = exchange.get_balances()
                
                for currency, amount in balances.items():
                    if currency not in holdings:
                        holdings[currency] = 0
                    holdings[currency] += amount
                    
                    # Convert to USDT value
                    if currency != "USDT":
                        ticker = exchange.get_ticker(f"{currency}-USDT")
                        value = amount * ticker["price"]
                    else:
                        value = amount
                        
                    total_value += value
                    
            except Exception as e:
                logger.error(f"Error getting portfolio from {exchange_name}: {str(e)}")
                
        return {
            "total_value": total_value,
            "holdings": holdings,
            "timestamp": int(time.time() * 1000)
        }
        
    def get_active_trades(self) -> List[Dict[str, Any]]:
        active_trades = []
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                trades = exchange.get_open_orders()
                
                for trade in trades:
                    # Calculate P&L
                    current_price = exchange.get_ticker(trade["symbol"])["price"]
                    entry_price = float(trade["price"])
                    
                    if trade["side"] == "buy":
                        pnl = (current_price - entry_price) / entry_price * 100
                    else:
                        pnl = (entry_price - current_price) / entry_price * 100
                        
                    active_trades.append({
                        "id": trade["id"],
                        "symbol": trade["symbol"],
                        "type": trade["side"],
                        "entry_price": entry_price,
                        "current_price": current_price,
                        "pnl": pnl,
                        "quantum_score": np.random.random(),  # This should come from quantum analysis
                        "exchange": exchange_name
                    })
                    
            except Exception as e:
                logger.error(f"Error getting active trades from {exchange_name}: {str(e)}")
                
        return active_trades
        
    def close_trade(self, trade_id: str) -> Dict[str, Any]:
        # Find trade in active trades
        active_trades = self.get_active_trades()
        trade = next((t for t in active_trades if t["id"] == trade_id), None)
        
        if not trade:
            raise ValueError(f"Trade {trade_id} not found")
            
        exchange = self.exchanges[trade["exchange"]]
        
        try:
            result = exchange.cancel_order(trade_id)
            logger.info(f"Closed trade {trade_id} on {trade['exchange']}")
            return {
                "success": True,
                "trade_id": trade_id,
                "status": "closed"
            }
        except Exception as e:
            logger.error(f"Error closing trade {trade_id}: {str(e)}")
            raise
