"""
Trade History Management Module
Handles the persistence and analysis of trading history
"""
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime
import json
import os

class TradeHistory:
    """Manages trade history with persistence capabilities"""

    def __init__(self, history_file: str = 'trade_history.json'):
        """Initialize trade history"""
        self.trades: List[Dict[str, Any]] = []
        self.history_file = history_file
        self.load_history()

    def add_trade(self, trade: Dict[str, Any]) -> None:
        """Add a trade to history"""
        trade['timestamp'] = datetime.now().isoformat()
        self.trades.append(trade)
        self.save_history()

    def get_trades(self, limit: int = None) -> List[Dict[str, Any]]:
        """Return the last N trades"""
        if limit and isinstance(limit, int):
            return self.trades[-limit:]
        return self.trades

    def get_trades_df(self) -> pd.DataFrame:
        """Return trades as DataFrame"""
        if not self.trades:
            return pd.DataFrame()
        df = pd.DataFrame(self.trades)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    def save_history(self) -> None:
        """Save history to file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.trades, f)
        except Exception as e:
            print(f"Error saving history: {e}")

    def load_history(self) -> None:
        """Load history from file"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    self.trades = json.load(f)
        except Exception as e:
            print(f"Error loading history: {e}")

    def clear_history(self) -> None:
        """Clear trade history"""
        self.trades = []
        if os.path.exists(self.history_file):
            os.remove(self.history_file)

    def get_summary(self) -> Dict[str, Any]:
        """Return trade history summary"""
        if not self.trades:
            return {
                'total_trades': 0,
                'profitable_trades': 0,
                'total_profit': 0.0,
                'win_rate': 0.0
            }

        df = self.get_trades_df()
        profitable_trades = len(df[df['profit_usdt'] > 0])

        return {
            'total_trades': len(df),
            'profitable_trades': profitable_trades,
            'total_profit': df['profit_usdt'].sum(),
            'win_rate': (profitable_trades / len(df)) * 100
        }