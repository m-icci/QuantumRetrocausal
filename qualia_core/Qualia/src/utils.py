# src/utils.py
import logging
import os
from datetime import datetime

def setup_logging(log_level=logging.INFO):
    """Configure logging for the miner"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_file = os.path.join(log_dir, f"miner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def calculate_hashrate(hashes: int, time_elapsed: float) -> float:
    """Calculate hashrate in H/s"""
    return hashes / time_elapsed if time_elapsed > 0 else 0

def format_hashrate(hashrate: float) -> str:
    """Format hashrate with appropriate unit"""
    units = ['H/s', 'KH/s', 'MH/s', 'GH/s', 'TH/s']
    unit_index = 0
    
    while hashrate >= 1000 and unit_index < len(units) - 1:
        hashrate /= 1000
        unit_index += 1
        
    return f"{hashrate:.2f} {units[unit_index]}"

def hex_to_bin(hex_str: str) -> str:
    """Convert hexadecimal string to binary string"""
    return bin(int(hex_str, 16))[2:].zfill(len(hex_str) * 4)

def bin_to_hex(bin_str: str) -> str:
    """Convert binary string to hexadecimal string"""
    return hex(int(bin_str, 2))[2:].zfill(len(bin_str) // 4)