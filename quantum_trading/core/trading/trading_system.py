"""
Trading System Module
"""

import logging
import asyncio
from typing import Dict, Optional, List, Any
from datetime import datetime

from ...analysis import MarketAnalysis
from ...risk import RiskManager

logger = logging.getLogger(__name__)

class TradingSystem:
    """Trading system component."""
    
    def __init__(self, config: Dict):
        """Initialize trading system."""
        self.config = config
        self.market_analysis = None
        self.risk_manager = None
        self.active_trades = {}
        self.running = False
        
    async def initialize(self) -> None:
        """Initialize trading system."""
        self.market_analysis = MarketAnalysis(self.config)
        self.risk_manager = RiskManager(self.config)
        await self.market_analysis.initialize()
        
    async def start(self) -> None:
        """Start trading system."""
        self.running = True
        
    async def stop(self) -> None:
        """Stop trading system."""
        self.running = False
        
    async def update(self) -> None:
        """Update trading system state."""
        await self.market_analysis.update()
        
    async def check_entry(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Check for entry opportunities.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Entry signal dict or None
        """
        # Check volume profile
        volume_ok = await self.market_analysis.check_volume_profile(symbol)
        if not volume_ok:
            return None
            
        # Check order book
        orderbook_ok = await self.market_analysis.check_order_book(symbol)
        if not orderbook_ok:
            return None
            
        # Check micro movements
        micro_result = await self.market_analysis.check_micro_movements(symbol)
        if not micro_result:
            return None
            
        movement, direction = micro_result
        
        return {
            'symbol': symbol,
            'direction': direction,
            'movement': movement,
            'timestamp': datetime.now()
        }
        
    async def check_exit(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Check for exit opportunities.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Exit signal dict or None
        """
        # Check if we have an active trade
        if symbol not in self.active_trades:
            return None
            
        trade = self.active_trades[symbol]
        
        # Check micro movements for reversal
        micro_result = await self.market_analysis.check_micro_movements(symbol)
        if not micro_result:
            return None
            
        movement, direction = micro_result
        
        # Check if direction is opposite to our trade
        if (trade['direction'] == 'long' and direction == 'short') or \
           (trade['direction'] == 'short' and direction == 'long'):
            return {
                'symbol': symbol,
                'direction': direction,
                'movement': movement,
                'timestamp': datetime.now(),
                'trade_id': trade['id']
            }
            
        return None
        
    def register_trade(self, trade: Dict[str, Any]) -> None:
        """Register active trade."""
        self.active_trades[trade['symbol']] = trade
        
    def remove_trade(self, symbol: str) -> None:
        """Remove active trade."""
        if symbol in self.active_trades:
            del self.active_trades[symbol] 