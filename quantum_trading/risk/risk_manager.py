"""
Risk Manager Module
"""

import logging
from typing import Dict, Optional, List, Union, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class RiskManager:
    """Risk management component."""
    
    def __init__(self, config: Dict):
        """Initialize risk manager."""
        self.config = config
        self.max_position_size = config.get('max_position_size', 1.0)
        self.max_risk_per_trade = config.get('max_risk_per_trade', 0.02)
        self.max_daily_risk = config.get('max_daily_risk', 0.05)
        self.daily_loss = 0.0
        
    def check_risk_limits(self, symbol: str, side: str, amount: float, price: float) -> bool:
        """
        Check if trade is within risk limits.
        
        Args:
            symbol: Trading symbol
            side: Trade side (buy/sell)
            amount: Trade amount
            price: Trade price
            
        Returns:
            True if trade is within risk limits, False otherwise
        """
        # Calculate trade value
        trade_value = amount * price
        
        # Calculate position risk
        position_risk = trade_value * self.max_risk_per_trade
        
        # Check position size limit
        if trade_value > self.max_position_size:
            logger.warning(f"Trade exceeds max position size: {trade_value} > {self.max_position_size}")
            return False
            
        # Check daily risk limit
        if self.daily_loss + position_risk > self.max_daily_risk:
            logger.warning(f"Trade exceeds daily risk limit: {self.daily_loss + position_risk} > {self.max_daily_risk}")
            return False
            
        return True
        
    def calculate_position_size(self, symbol: str, price: float, stop_loss: float) -> float:
        """
        Calculate optimal position size based on risk parameters.
        
        Args:
            symbol: Trading symbol
            price: Current price
            stop_loss: Stop loss price
            
        Returns:
            Optimal position size
        """
        # Calculate risk per unit
        if price > stop_loss:  # Long position
            risk_per_unit = price - stop_loss
        else:  # Short position
            risk_per_unit = stop_loss - price
            
        # Calculate risk percentage
        risk_percentage = risk_per_unit / price
        
        # Calculate position size
        position_size = self.max_risk_per_trade / risk_percentage
        
        # Ensure position size is within limits
        return min(position_size, self.max_position_size)
        
    def update_daily_loss(self, loss: float) -> None:
        """Update daily loss counter."""
        self.daily_loss += loss
        
    def reset_daily_metrics(self) -> None:
        """Reset daily metrics."""
        self.daily_loss = 0.0 