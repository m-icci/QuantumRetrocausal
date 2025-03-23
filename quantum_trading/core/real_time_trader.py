"""
QUALIA Trading System - Real Trading Launcher
This script launches the QUALIA trading system for a time-limited real trading session 
on KuCoin and Kraken exchanges, integrating CGR analysis and retrocausal methods.
"""
import os
import logging
import time
import argparse
from dotenv import load_dotenv
from datetime import datetime
import traceback
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import json

from ..strategies import BaseStrategy, StrategyFactory
from ..exceptions import ExecutionError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"trading_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger("real_trading")

# Load environment variables
load_dotenv()

@dataclass
class TradeSignal:
    """Classe para representar sinais de negociação."""
    timestamp: datetime
    symbol: str
    side: str  # "buy" ou "sell"
    amount: float
    price: float
    confidence: float
    strategy: str
    metadata: Dict[str, Any] = None

class RealTimeTrader:
    """Classe para execução de negociação em tempo real."""
    
    def __init__(
        self,
        strategy: BaseStrategy,
        config: Dict[str, Any],
        exchange_config: Dict[str, Any]
    ):
        """
        Inicializa o trader em tempo real.
        
        Args:
            strategy: Estratégia de negociação.
            config: Configuração geral.
            exchange_config: Configuração da exchange.
        """
        self.strategy = strategy
        self.config = config
        self.exchange_config = exchange_config
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.trades: List[Dict[str, Any]] = []
        self.last_update = None
        self.is_running = False
    
    def start(self) -> None:
        """Inicia o trader."""
        self.is_running = True
        logger.info("Iniciando trader em tempo real")
        
        try:
            while self.is_running:
                self.update()
                time.sleep(self.config.get("update_interval", 1))
        except Exception as e:
            logger.error(f"Erro durante execução: {str(e)}")
            self.stop()
    
    def stop(self) -> None:
        """Para o trader."""
        self.is_running = False
        logger.info("Parando trader")
    
    def update(self) -> None:
        """
        Atualiza o estado do trader.
        
        Este método deve ser implementado para:
        1. Obter dados de mercado atualizados
        2. Gerar sinais usando a estratégia
        3. Executar ordens baseadas nos sinais
        4. Atualizar posições e trades
        """
        raise NotImplementedError("Método update() deve ser implementado")
    
    def process_signal(self, signal: TradeSignal) -> None:
        """
        Processa um sinal de negociação.
        
        Args:
            signal: Sinal de negociação.
        """
        try:
            # Verifica se já existe posição
            position = self.positions.get(signal.symbol)
            
            if signal.side == "buy" and not position:
                # Abre nova posição
                self._execute_buy(signal)
            elif signal.side == "sell" and position:
                # Fecha posição existente
                self._execute_sell(signal)
            
            # Atualiza timestamp da última atualização
            self.last_update = signal.timestamp
            
        except Exception as e:
            logger.error(f"Erro ao processar sinal: {str(e)}")
            raise ExecutionError(f"Falha ao processar sinal: {str(e)}")
    
    def _execute_buy(self, signal: TradeSignal) -> None:
        """
        Executa ordem de compra.
        
        Args:
            signal: Sinal de compra.
        """
        # Implementa lógica de execução de compra
        # Este é um placeholder - a implementação real dependerá da exchange
        order = {
            "symbol": signal.symbol,
            "side": "buy",
            "amount": signal.amount,
            "price": signal.price,
            "timestamp": signal.timestamp,
            "status": "filled"
        }
        
        # Registra ordem
        self.trades.append(order)
        
        # Atualiza posição
        self.positions[signal.symbol] = {
            "amount": signal.amount,
            "entry_price": signal.price,
            "timestamp": signal.timestamp
        }
        
        logger.info(f"Ordem de compra executada: {order}")
    
    def _execute_sell(self, signal: TradeSignal) -> None:
        """
        Executa ordem de venda.
        
        Args:
            signal: Sinal de venda.
        """
        # Implementa lógica de execução de venda
        # Este é um placeholder - a implementação real dependerá da exchange
        position = self.positions[signal.symbol]
        
        order = {
            "symbol": signal.symbol,
            "side": "sell",
            "amount": position["amount"],
            "price": signal.price,
            "timestamp": signal.timestamp,
            "status": "filled"
        }
        
        # Registra ordem
        self.trades.append(order)
        
        # Remove posição
        del self.positions[signal.symbol]
        
        logger.info(f"Ordem de venda executada: {order}")
    
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Retorna as posições atuais.
        
        Returns:
            Dicionário com posições.
        """
        return self.positions
    
    def get_trades(self) -> List[Dict[str, Any]]:
        """
        Retorna o histórico de trades.
        
        Returns:
            Lista de trades.
        """
        return self.trades
    
    def get_performance(self) -> Dict[str, Any]:
        """
        Calcula métricas de desempenho.
        
        Returns:
            Dicionário com métricas.
        """
        if not self.trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "total_pnl": 0
            }
        
        # Calcula PnL por trade
        pnls = []
        for i in range(1, len(self.trades), 2):
            if i >= len(self.trades):
                break
            
            buy = self.trades[i-1]
            sell = self.trades[i]
            
            pnl = (sell["price"] - buy["price"]) * buy["amount"]
            pnls.append(pnl)
        
        # Calcula métricas
        total_trades = len(pnls)
        winning_trades = sum(1 for pnl in pnls if pnl > 0)
        total_pnl = sum(pnls)
        
        profits = sum(pnl for pnl in pnls if pnl > 0)
        losses = abs(sum(pnl for pnl in pnls if pnl < 0))
        
        return {
            "total_trades": total_trades,
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
            "profit_factor": profits / losses if losses > 0 else float("inf"),
            "total_pnl": total_pnl
        }

def check_credentials():
    """Check if all necessary API credentials are available"""
    required_vars = {
        'KUCOIN_API_KEY': os.getenv('KUCOIN_API_KEY'),
        'KUCOIN_API_SECRET': os.getenv('KUCOIN_API_SECRET'),
        'KUCOIN_API_PASSPHRASE': os.getenv('KUCOIN_API_PASSPHRASE'),
        'KRAKEN_API_KEY': os.getenv('KRAKEN_API_KEY'),
        'KRAKEN_API_SECRET': os.getenv('KRAKEN_API_SECRET')
    }
    
    missing = [key for key, value in required_vars.items() if not value]
    
    if missing:
        logger.error(f"Missing required API credentials: {', '.join(missing)}")
        logger.error("Please add them to your .env file or set them as environment variables")
        return False
    
    logger.info("API credentials validation successful")
    return True

def main():
    """Main function to launch the real trading session"""
    parser = argparse.ArgumentParser(description='QUALIA Real Trading Launcher')
    parser.add_argument('--duration', type=int, default=60,
                        help='Trading session duration in minutes (default: 60)')
    parser.add_argument('--pairs', type=str, default='BTC/USDT,ETH/USDT',
                        help='Comma-separated list of trading pairs (default: BTC/USDT,ETH/USDT)')
    parser.add_argument('--max-drawdown', type=float, default=2.0,
                        help='Maximum allowed drawdown percentage (default: 2.0%%)')
    parser.add_argument('--safe-mode', action='store_true',
                        help='Enable additional safety measures for risk mitigation')
    parser.add_argument('--test-mode', action='store_true',
                        help='Run in test mode with simulated trading (no real trades)')
    
    args = parser.parse_args()
    
    # Process arguments
    duration_minutes = args.duration
    trading_pairs = [pair.strip() for pair in args.pairs.split(',')]
    max_drawdown = args.max_drawdown
    test_mode = args.test_mode
    
    logger.info(f"Starting QUALIA Real Trading session with the following parameters:")
    logger.info(f"- Duration: {duration_minutes} minutes")
    logger.info(f"- Trading pairs: {', '.join(trading_pairs)}")
    logger.info(f"- Max drawdown: {max_drawdown}%")
    logger.info(f"- Safe mode: {'Enabled' if args.safe_mode else 'Disabled'}")
    logger.info(f"- Test mode: {'Enabled' if test_mode else 'Disabled'}")
    
    # Validate API credentials if not in test mode
    if not test_mode and not check_credentials():
        logger.warning("Running in test mode with simulated data due to missing credentials")
        test_mode = True
    
    try:
        # Import real-time trader
        from quantum_trading.real_time_trader import RealTimeTrader, TestModeTrader
        
        # Create the real-time trader
        if test_mode:
            logger.info("Initializing test mode trader with simulated data")
            trader = TestModeTrader(
                duration_minutes=duration_minutes,
                max_drawdown_pct=max_drawdown,
                trading_pairs=trading_pairs,
                exchanges=['kucoin', 'kraken']
            )
        else:
            trader = RealTimeTrader(
                duration_minutes=duration_minutes,
                max_drawdown_pct=max_drawdown,
                trading_pairs=trading_pairs,
                exchanges=['kucoin', 'kraken']
            )
        
        # Start the trading session
        logger.info("Initializing trading session...")
        trader.start_trading_session()
        
        # Main loop - display status periodically while the session is active
        logger.info("Trading session started. Press Ctrl+C to terminate early (will gracefully close positions)")
        
        try:
            while trader.is_trading_active:
                # Display current status
                status = trader.get_session_summary()
                
                logger.info(f"Session status: Active for {status['duration_minutes']:.1f} minutes")
                logger.info(f"Trades executed: {status['trades_executed']} "
                           f"(Success: {status['successful_trades']}, Failed: {status['failed_trades']})")
                logger.info(f"Current P/L: {status['profit_loss_pct']:.2f}%")
                logger.info(f"Max drawdown: {status['max_drawdown_pct']:.2f}%")
                logger.info("=" * 50)
                
                # Wait before next status update
                time.sleep(30)
                
        except KeyboardInterrupt:
            logger.info("User requested termination. Stopping trading session...")
            trader.stop_trading_session()
        
        # Display final summary
        final_status = trader.get_session_summary()
        
        logger.info("\n" + "=" * 30 + " TRADING SESSION COMPLETED " + "=" * 30)
        logger.info(f"Session duration: {final_status['duration_minutes']:.1f} minutes")
        logger.info(f"Trades executed: {final_status['trades_executed']}")
        logger.info(f"Successful trades: {final_status['successful_trades']}")
        logger.info(f"Failed trades: {final_status['failed_trades']}")
        logger.info(f"Final P/L: {final_status['profit_loss_pct']:.2f}%")
        logger.info(f"Maximum drawdown: {final_status['max_drawdown_pct']:.2f}%")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in trading session: {str(e)}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
