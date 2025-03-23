"""
Script para backtesting do sistema de trading.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from quantum_trading.core.trading import TradingSystem, TradingConfig
from quantum_trading.data_loader import DataLoader

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BacktestRunner:
    """Runner para backtesting do sistema de trading."""
    
    def __init__(self, start_date: datetime, end_date: datetime):
        """
        Inicializa o runner de backtesting.
        
        Args:
            start_date: Data inicial
            end_date: Data final
        """
        self.start_date = start_date
        self.end_date = end_date
        self.trading_system: Optional[TradingSystem] = None
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []
        self.initial_balance = 10000  # Balance inicial para backtesting
    
    async def _load_historical_data(self, config: TradingConfig) -> pd.DataFrame:
        """
        Carrega dados históricos para backtesting.
        
        Args:
            config: Configuração do trading
            
        Returns:
            DataFrame com dados históricos
        """
        try:
            # Criar carregador
            loader = DataLoader(
                exchange=config.exchange,
                symbol=config.symbol,
                timeframe=config.timeframe
            )
            
            # Carregar dados
            df = await loader.load_data(self.start_date, self.end_date)
            
            if df.empty:
                logger.error("Não foi possível carregar dados históricos")
                return pd.DataFrame()
            
            return df
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados históricos: {e}")
            return pd.DataFrame()
    
    def _calculate_metrics(self) -> Dict:
        """
        Calcula métricas de performance.
        
        Returns:
            Dict com métricas
        """
        try:
            if not self.trades:
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'profit_factor': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'total_return': 0,
                    'annual_return': 0
                }
            
            # Calcular métricas básicas
            total_trades = len(self.trades)
            winning_trades = len([t for t in self.trades if t['pnl'] > 0])
            win_rate = winning_trades / total_trades
            
            # Calcular retornos
            total_profit = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
            total_loss = abs(sum(t['pnl'] for t in self.trades if t['pnl'] < 0))
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # Calcular retorno total e anual
            total_return = (self.equity_curve[-1] - self.initial_balance) / self.initial_balance
            days = (self.end_date - self.start_date).days
            annual_return = (1 + total_return) ** (365 / days) - 1
            
            # Calcular Sharpe Ratio
            returns = pd.Series(self.equity_curve).pct_change().dropna()
            sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
            
            # Calcular drawdown máximo
            equity_series = pd.Series(self.equity_curve)
            rolling_max = equity_series.expanding().max()
            drawdowns = (equity_series - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_return': total_return,
                'annual_return': annual_return
            }
            
        except Exception as e:
            logger.error(f"Erro ao calcular métricas: {e}")
            return {}
    
    async def run(self):
        """Executa o backtesting."""
        try:
            # Carregar configuração
            load_dotenv()
            config = TradingConfig.from_dict({
                'exchange': os.getenv('EXCHANGE'),
                'api_key': os.getenv('API_KEY'),
                'api_secret': os.getenv('API_SECRET'),
                'symbol': os.getenv('SYMBOL'),
                'timeframe': os.getenv('TIMEFRAME'),
                'leverage': float(os.getenv('LEVERAGE', '1')),
                'max_positions': int(os.getenv('MAX_POSITIONS', '3')),
                'daily_trades_limit': int(os.getenv('DAILY_TRADES_LIMIT', '10')),
                'daily_loss_limit': float(os.getenv('DAILY_LOSS_LIMIT', '100')),
                'min_confidence': float(os.getenv('MIN_CONFIDENCE', '0.7')),
                'position_size': float(os.getenv('POSITION_SIZE', '100')),
                'min_position_size': float(os.getenv('MIN_POSITION_SIZE', '10')),
                'max_position_size': float(os.getenv('MAX_POSITION_SIZE', '1000')),
                'stop_loss': float(os.getenv('STOP_LOSS', '0.02')),
                'take_profit': float(os.getenv('TAKE_PROFIT', '0.04')),
                'risk_per_trade': float(os.getenv('RISK_PER_TRADE', '0.01')),
                'rsi_period': int(os.getenv('RSI_PERIOD', '14')),
                'rsi_overbought': float(os.getenv('RSI_OVERBOUGHT', '70')),
                'rsi_oversold': float(os.getenv('RSI_OVERSOLD', '30')),
                'macd_fast': int(os.getenv('MACD_FAST', '12')),
                'macd_slow': int(os.getenv('MACD_SLOW', '26')),
                'macd_signal': int(os.getenv('MACD_SIGNAL', '9')),
                'bb_period': int(os.getenv('BB_PERIOD', '20')),
                'bb_std': float(os.getenv('BB_STD', '2')),
                'atr_period': int(os.getenv('ATR_PERIOD', '14')),
                'atr_multiplier': float(os.getenv('ATR_MULTIPLIER', '2')),
                'log_level': os.getenv('LOG_LEVEL', 'INFO')
            })
            
            # Carregar dados históricos
            historical_data = await self._load_historical_data(config)
            if historical_data.empty:
                logger.error("Não foi possível carregar dados históricos")
                return
            
            # Inicializar sistema
            self.trading_system = TradingSystem(config)
            current_balance = self.initial_balance
            self.equity_curve.append(current_balance)
            
            # Executar backtesting
            logger.info("Iniciando backtesting...")
            for index, row in historical_data.iterrows():
                # Atualizar estado do sistema
                await self.trading_system._update_quantum_state(row)
                
                # Executar trades
                trades = await self.trading_system._execute_trades()
                
                # Atualizar trades e curva de equity
                for trade in trades:
                    self.trades.append(trade)
                    current_balance += trade['pnl']
                    self.equity_curve.append(current_balance)
            
            # Calcular métricas
            metrics = self._calculate_metrics()
            
            # Log de resultados
            logger.info("\n=== Resultados do Backtesting ===")
            logger.info(f"Período: {self.start_date} - {self.end_date}")
            logger.info(f"Balance Inicial: ${self.initial_balance:.2f}")
            logger.info(f"Balance Final: ${current_balance:.2f}")
            logger.info(f"Retorno Total: {metrics['total_return']:.2%}")
            logger.info(f"Retorno Anual: {metrics['annual_return']:.2%}")
            logger.info(f"Total de Trades: {metrics['total_trades']}")
            logger.info(f"Win Rate: {metrics['win_rate']:.2%}")
            logger.info(f"Profit Factor: {metrics['profit_factor']:.2f}")
            logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
            
            # Salvar resultados
            results = {
                'trades': self.trades,
                'equity_curve': self.equity_curve,
                'metrics': metrics
            }
            pd.DataFrame(results).to_csv('backtest_results.csv')
            
            logger.info("\nBacktesting concluído")
            
        except Exception as e:
            logger.error(f"Erro ao executar backtesting: {e}")

def main():
    """Função principal."""
    try:
        # Definir período de backtesting
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # 30 dias de dados
        
        # Executar backtesting
        runner = BacktestRunner(start_date, end_date)
        asyncio.run(runner.run())
        
    except KeyboardInterrupt:
        logger.info("Encerrando por solicitação do usuário")
    except Exception as e:
        logger.error(f"Erro fatal: {e}")

if __name__ == "__main__":
    main() 