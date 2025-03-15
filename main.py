import os
import sys
import time
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, Any
from web_interface.app import app  # Import the Flask app directly
from trading_system.market_api import MultiExchangeAPI
from trading_system.quantum_analysis import QuantumAnalyzer
from trading_system.cgr_analysis import CGRAnalyzer
from trading_system.risk_manager import RiskManager

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs('logs', exist_ok=True)
os.makedirs(os.path.join('web_interface', 'static'), exist_ok=True)
os.makedirs(os.path.join('web_interface', 'templates'), exist_ok=True)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Quantum Trading System")
    parser.add_argument('--mode', choices=['real', 'simulation'], default='simulation',
                       help='Trading mode (real or simulation)')
    parser.add_argument('--duration', type=int, default=60,
                       help='Duration in minutes')
    parser.add_argument('--pairs', nargs='+', default=['BTC-USDT', 'ETH-USDT'],
                       help='Trading pairs')
    parser.add_argument('--web', action='store_true',
                       help='Start web interface')
    return parser.parse_args()

class QuantumTradingSystem:
    """Main trading system coordinator"""
    def __init__(self, mode: str = 'simulation', trading_pairs: list = None):
        self.mode = mode
        self.trading_pairs = trading_pairs or ['BTC-USDT', 'ETH-USDT']

        # Initialize components
        self.market_api = MultiExchangeAPI()
        self.quantum_analyzer = QuantumAnalyzer()
        self.cgr_analyzer = CGRAnalyzer()
        self.risk_manager = RiskManager()

        self.running = False
        self.last_analysis_time = datetime.now()

        logger.info(f"Trading system initialized in {mode} mode")
        logger.info(f"Trading pairs: {self.trading_pairs}")

    def analyze_market(self) -> Dict[str, Any]:
        """Perform comprehensive market analysis"""
        try:
            # Get market data
            market_data = self.market_api.get_market_data()
            if not market_data:
                raise ValueError("No market data available")
                
            # Quantum analysis
            quantum_metrics = self.quantum_analyzer.analyze(market_data)
            
            # CGR analysis
            cgr_metrics = self.cgr_analyzer.analyze(market_data)
            
            # Risk metrics
            risk_metrics = self.risk_manager.get_risk_metrics()
            
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'market_data': market_data,
                'quantum_metrics': quantum_metrics,
                'cgr_metrics': cgr_metrics,
                'risk_metrics': risk_metrics
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in market analysis: {str(e)}")
            return {}
            
    def execute_trades(self, analysis: Dict[str, Any]) -> None:
        """Execute trades based on analysis"""
        if not analysis:
            return
            
        try:
            portfolio = self.market_api.get_portfolio()
            portfolio_value = portfolio['total_value']
            
            for symbol in self.trading_pairs:
                if symbol not in analysis['market_data']:
                    continue
                    
                current_price = analysis['market_data'][symbol]['price']
                quantum_metrics = analysis['quantum_metrics']
                cgr_metrics = analysis['cgr_metrics']
                
                # Calculate position size
                position_size = self.risk_manager.calculate_position_size(
                    symbol, portfolio_value, quantum_metrics, cgr_metrics
                )
                
                if position_size > 0:
                    # Check quantum signal
                    if quantum_metrics['trend_signal'] > 0.5:
                        logger.info(f"Opening long position for {symbol}")
                        if self.mode == 'real':
                            # Execute real trade
                            self.market_api.execute_trade(symbol, 'buy', position_size)
                            # Add position to risk manager
                            self.risk_manager.add_position(
                                symbol, current_price, position_size, quantum_metrics
                            )
                        else:
                            logger.info("[Simulation] Would open long position")
                            
        except Exception as e:
            logger.error(f"Error executing trades: {str(e)}")
            
    def monitor_positions(self) -> None:
        """Monitor and update open positions"""
        try:
            market_data = self.market_api.get_market_data()
            quantum_metrics = self.quantum_analyzer.analyze(market_data)
            
            for symbol in self.trading_pairs:
                if symbol not in market_data:
                    continue
                    
                current_price = market_data[symbol]['price']
                update = self.risk_manager.update_position(symbol, current_price, quantum_metrics)
                
                if update.get('status') == 'closed':
                    logger.info(f"Position closed for {symbol}: {update}")
                    if self.mode == 'real':
                        self.market_api.close_position(symbol)
                        
        except Exception as e:
            logger.error(f"Error monitoring positions: {str(e)}")
            
    def run(self, duration_minutes: int) -> None:
        """Run the trading system for specified duration"""
        self.running = True
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        logger.info(f"Starting trading run until {end_time}")
        
        try:
            while self.running and datetime.now() < end_time:
                # Perform analysis
                analysis = self.analyze_market()
                
                # Execute trades based on analysis
                self.execute_trades(analysis)
                
                # Monitor existing positions
                self.monitor_positions()
                
                # Log current status
                portfolio = self.market_api.get_portfolio()
                logger.info(f"Portfolio value: {portfolio['total_value']:.2f} USDT")
                
                # Wait for next cycle (5 seconds)
                time.sleep(5)
                
        except KeyboardInterrupt:
            logger.info("Trading interrupted by user")
        except Exception as e:
            logger.error(f"Error in trading loop: {str(e)}")
        finally:
            self.running = False
            logger.info("Trading run completed")

def main():
    """Main entry point"""
    args = parse_arguments()

    try:
        # Initialize trading system
        trading_system = QuantumTradingSystem(
            mode=args.mode,
            trading_pairs=args.pairs
        )

        # Start web interface if requested
        if args.web:
            logger.info("Web interface enabled, starting Flask application...")
            app.run(host='0.0.0.0', port=5000, debug=True) #Simplified startup


        # Run trading system
        trading_system.run(args.duration)

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()