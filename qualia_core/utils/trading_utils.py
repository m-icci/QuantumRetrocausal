"""
Trading utility functions
"""
from typing import Dict

def format_trading_pair(symbol: str) -> str:
    """Format trading pair string"""
    return symbol.replace('/', '').replace('-', '').upper()

def calculate_fee_rate(volume: float) -> float:
    """Calculate dynamic fee rate based on volume"""
    if volume < 50000:
        return 0.0026
    elif volume < 100000:
        return 0.0024
    else:
        return 0.0022
