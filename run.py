#!/usr/bin/env python3
"""
QUALIA - Quantum Trading System
Main execution file for running the trading system.
"""

import os
import json
import logging
import asyncio
import argparse
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path

from quantum_trading.scalping import ScalpingSystem
from quantum_trading.core.trading.trading_system import TradingSystem
from quantum_trading.data.data_loader import DataLoader
from quantum_trading.core.trading.market_analysis import MarketAnalysis
from quantum_trading.core.trading.order_executor import OrderExecutor
from quantum_trading.core.trading.risk_manager import RiskManager

def setup_logging(config):
    """Setup logging configuration."""
    log_level = getattr(logging, config['logging']['level'])
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(config['logging']['file']),
            logging.StreamHandler() if config['logging']['console'] else None
        ]
    )
    
    return logging.getLogger('QUALIA')

def load_config():
    """Load configuration from config.json file."""
    config_path = Path('config.json')
    if not config_path.exists():
        raise FileNotFoundError("config.json not found")
        
    with open(config_path) as f:
        return json.load(f)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='QUALIA - Quantum Trading System')
    parser.add_argument('--mode', choices=['real', 'simulated'], 
                      help='Trading mode (real/simulated)')
    parser.add_argument('--symbol', help='Trading symbol (e.g. BTC/USDT)')
    parser.add_argument('--config', help='Path to config file')
    parser.add_argument('--optimize', action='store_true', 
                      help='Run parameter optimization')
    return parser.parse_args()

async def main():
    """Main execution function."""
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config()
    
    # Override config with command line arguments
    if args.mode:
        config['trading']['mode'] = args.mode
    if args.symbol:
        config['trading']['symbol'] = args.symbol
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("Starting QUALIA Trading System...")
    
    try:
        # Initialize components
        data_loader = DataLoader(config)
        market_analysis = MarketAnalysis(config)
        order_executor = OrderExecutor(config)
        risk_manager = RiskManager(config)
        
        # Initialize trading system
        if config['scalping']['enabled']:
            trading_system = ScalpingSystem(
                config=config,
                data_loader=data_loader,
                market_analysis=market_analysis,
                order_executor=order_executor,
                risk_manager=risk_manager
            )
        else:
            trading_system = TradingSystem(
                config=config,
                data_loader=data_loader,
                market_analysis=market_analysis,
                order_executor=order_executor,
                risk_manager=risk_manager
            )
        
        # Connect to exchange
        await data_loader.connect()
        await order_executor.connect()
        
        # Run optimization if requested
        if args.optimize and config['optimization']['enabled']:
            logger.info("Starting parameter optimization...")
            await trading_system.optimize()
        
        # Start trading system
        logger.info(f"Starting trading in {config['trading']['mode']} mode...")
        await trading_system.run()
        
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
        await trading_system.stop()
        await data_loader.disconnect()
        await order_executor.disconnect()
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        raise
    
    finally:
        # Cleanup
        await asyncio.gather(
            data_loader.disconnect(),
            order_executor.disconnect()
        )
        logger.info("System shutdown complete")

if __name__ == "__main__":
    asyncio.run(main()) 