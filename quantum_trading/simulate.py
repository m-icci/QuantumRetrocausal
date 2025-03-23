"""
Script para simular o sistema de trading.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from quantum_trading.core.trading import TradingSystem, TradingConfig
from quantum_trading.data_loader import DataLoader

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulate.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimulationRunner:
    """Runner para simulação do sistema de trading."""
    
    def __init__(self, start_date: datetime, end_date: datetime, initial_balance: float = 10000):
        """
        Inicializa o runner de simulação.
        
        Args:
            start_date: Data inicial
            end_date: Data final
            initial_balance: Balance inicial
        """
        self.start_date = start_date
        self.end_date = end_date
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.trading_system: Optional[TradingSystem] = None
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []
        self.positions: Dict[str, Dict] = {}
        self.running = False
    
    async def _load_historical_data(self) -> pd.DataFrame:
        """
        Carrega dados históricos.
        
        Returns:
            DataFrame com dados históricos
        """
        try:
            # Criar carregador
            loader = DataLoader(
                exchange=self.trading_system.config.exchange,
                symbol=self.trading_system.config.symbol,
                timeframe=self.trading_system.config.timeframe
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
                    'daily_return': 0
                }
            
            # Calcular métricas básicas
            total_trades = len(self.trades)
            winning_trades = len([t for t in self.trades if t['pnl'] > 0])
            win_rate = winning_trades / total_trades
            
            # Calcular retornos
            total_profit = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
            total_loss = abs(sum(t['pnl'] for t in self.trades if t['pnl'] < 0))
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # Calcular retorno total e diário
            total_return = (self.current_balance - self.initial_balance) / self.initial_balance
            daily_return = (self.current_balance - self.equity_curve[-2]) / self.equity_curve[-2] if len(self.equity_curve) > 1 else 0
            
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
                'daily_return': daily_return
            }
            
        except Exception as e:
            logger.error(f"Erro ao calcular métricas: {e}")
            return {}
    
    async def _execute_simulated_trade(self, trade: Dict):
        """
        Executa um trade em modo simulação.
        
        Args:
            trade: Dados do trade
        """
        try:
            symbol = trade['symbol']
            side = trade['side']
            price = trade['price']
            size = trade['size']
            
            # Calcular P&L
            if side == 'buy':
                # Abrir posição long
                self.positions[symbol] = {
                    'side': 'long',
                    'entry_price': price,
                    'size': size,
                    'entry_time': datetime.now()
                }
            else:
                # Abrir posição short
                self.positions[symbol] = {
                    'side': 'short',
                    'entry_price': price,
                    'size': size,
                    'entry_time': datetime.now()
                }
            
            # Registrar trade
            self.trades.append({
                'symbol': symbol,
                'side': side,
                'price': price,
                'size': size,
                'time': datetime.now(),
                'pnl': 0  # P&L será calculado no fechamento
            })
            
            logger.info(f"Trade simulado executado: {side.upper()} {size} {symbol} @ {price}")
            
        except Exception as e:
            logger.error(f"Erro ao executar trade simulado: {e}")
    
    async def _close_simulated_position(self, symbol: str, price: float):
        """
        Fecha uma posição em modo simulação.
        
        Args:
            symbol: Símbolo da posição
            price: Preço de fechamento
        """
        try:
            if symbol not in self.positions:
                return
            
            position = self.positions[symbol]
            entry_price = position['entry_price']
            size = position['size']
            
            # Calcular P&L
            if position['side'] == 'long':
                pnl = (price - entry_price) * size
            else:
                pnl = (entry_price - price) * size
            
            # Atualizar balance
            self.current_balance += pnl
            
            # Atualizar trade
            for trade in self.trades:
                if trade['symbol'] == symbol and trade['pnl'] == 0:
                    trade['pnl'] = pnl
                    trade['close_price'] = price
                    trade['close_time'] = datetime.now()
                    break
            
            # Remover posição
            del self.positions[symbol]
            
            logger.info(f"Posição fechada: {symbol} @ {price} (P&L: {pnl:.2f})")
            
        except Exception as e:
            logger.error(f"Erro ao fechar posição: {e}")
    
    async def _monitor_positions(self, current_price: float):
        """
        Monitora posições abertas.
        
        Args:
            current_price: Preço atual
        """
        try:
            for symbol, position in list(self.positions.items()):
                entry_price = position['entry_price']
                
                # Verificar stop loss
                if position['side'] == 'long':
                    if current_price <= entry_price * (1 - self.trading_system.config.stop_loss):
                        await self._close_simulated_position(symbol, current_price)
                else:
                    if current_price >= entry_price * (1 + self.trading_system.config.stop_loss):
                        await self._close_simulated_position(symbol, current_price)
                
                # Verificar take profit
                if position['side'] == 'long':
                    if current_price >= entry_price * (1 + self.trading_system.config.take_profit):
                        await self._close_simulated_position(symbol, current_price)
                else:
                    if current_price <= entry_price * (1 - self.trading_system.config.take_profit):
                        await self._close_simulated_position(symbol, current_price)
            
        except Exception as e:
            logger.error(f"Erro ao monitorar posições: {e}")
    
    async def _update_loop(self, historical_data: pd.DataFrame):
        """
        Loop principal de atualização.
        
        Args:
            historical_data: Dados históricos
        """
        try:
            # Inicializar curva de equity
            self.equity_curve.append(self.initial_balance)
            
            # Iterar sobre dados históricos
            for i in range(len(historical_data)):
                if not self.running:
                    break
                
                # Obter dados do candle atual
                current_data = historical_data.iloc[i]
                
                # Atualizar estado do sistema
                await self.trading_system._update_quantum_state(current_data)
                
                # Executar trades
                trades = await self.trading_system._execute_trades()
                for trade in trades:
                    await self._execute_simulated_trade(trade)
                
                # Monitorar posições
                current_price = current_data['close']
                await self._monitor_positions(current_price)
                
                # Atualizar curva de equity
                self.equity_curve.append(self.current_balance)
                
                # Calcular e logar métricas
                metrics = self._calculate_metrics()
                logger.info("\n=== Métricas de Simulação ===")
                logger.info(f"Balance Atual: ${self.current_balance:.2f}")
                logger.info(f"Retorno Total: {metrics['total_return']:.2%}")
                logger.info(f"Retorno Diário: {metrics['daily_return']:.2%}")
                logger.info(f"Total de Trades: {metrics['total_trades']}")
                logger.info(f"Win Rate: {metrics['win_rate']:.2%}")
                logger.info(f"Profit Factor: {metrics['profit_factor']:.2f}")
                logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
                
                # Aguardar próximo ciclo
                await asyncio.sleep(0.1)  # 100ms entre candles
                
        except Exception as e:
            logger.error(f"Erro no loop de atualização: {e}")
    
    async def run(self):
        """Executa a simulação."""
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
            
            # Inicializar sistema
            self.trading_system = TradingSystem(config)
            
            # Carregar dados históricos
            historical_data = await self._load_historical_data()
            if historical_data.empty:
                logger.error("Não foi possível carregar dados históricos")
                return
            
            # Iniciar loop de atualização
            self.running = True
            logger.info("Iniciando simulação...")
            await self._update_loop(historical_data)
            
        except Exception as e:
            logger.error(f"Erro ao executar simulação: {e}")
        finally:
            self.running = False

def main():
    """Função principal."""
    try:
        # Definir período
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Últimos 30 dias
        
        # Executar simulação
        runner = SimulationRunner(start_date, end_date)
        asyncio.run(runner.run())
        
    except KeyboardInterrupt:
        logger.info("Encerrando por solicitação do usuário")
    except Exception as e:
        logger.error(f"Erro fatal: {e}")

if __name__ == "__main__":
    main() 