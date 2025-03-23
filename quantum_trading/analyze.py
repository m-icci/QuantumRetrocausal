"""
Script para análise do sistema de trading.
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
        logging.FileHandler('analyze.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingAnalyzer:
    """Analisador do sistema de trading."""
    
    def __init__(self, trading_system: TradingSystem):
        """
        Inicializa o analisador.
        
        Args:
            trading_system: Sistema de trading
        """
        self.trading_system = trading_system
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []
        self.metrics_history: List[Dict] = []
    
    async def _load_trading_data(self) -> bool:
        """
        Carrega dados de trading.
        
        Returns:
            True se carregado com sucesso
        """
        try:
            # Carregar trades
            self.trades = await self.trading_system.get_trade_history()
            
            # Carregar curva de equity
            self.equity_curve = await self.trading_system.get_equity_curve()
            
            # Carregar histórico de métricas
            self.metrics_history = await self.trading_system.get_metrics_history()
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados de trading: {e}")
            return False
    
    def _analyze_performance(self) -> Dict:
        """
        Analisa performance do sistema.
        
        Returns:
            Dict com análise
        """
        try:
            if not self.trades:
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'profit_factor': 0,
                    'total_return': 0,
                    'avg_daily_return': 0,
                    'daily_volatility': 0,
                    'annual_volatility': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'avg_position_time': 0
                }
            
            # Calcular métricas básicas
            total_trades = len(self.trades)
            winning_trades = len([t for t in self.trades if t['pnl'] > 0])
            win_rate = winning_trades / total_trades
            
            # Calcular retornos
            total_profit = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
            total_loss = abs(sum(t['pnl'] for t in self.trades if t['pnl'] < 0))
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # Calcular retorno total
            total_return = (self.equity_curve[-1] - self.equity_curve[0]) / self.equity_curve[0]
            
            # Calcular retorno diário médio
            daily_returns = pd.Series(self.equity_curve).pct_change().dropna()
            avg_daily_return = daily_returns.mean()
            
            # Calcular volatilidade
            daily_volatility = daily_returns.std()
            annual_volatility = daily_volatility * np.sqrt(252)
            
            # Calcular Sharpe Ratio
            risk_free_rate = 0.02  # 2% ao ano
            sharpe_ratio = (avg_daily_return * 252 - risk_free_rate) / (daily_volatility * np.sqrt(252))
            
            # Calcular drawdown máximo
            equity_series = pd.Series(self.equity_curve)
            rolling_max = equity_series.expanding().max()
            drawdowns = (equity_series - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            # Calcular tempo médio de posição
            hold_times = []
            for trade in self.trades:
                if 'close_time' in trade and 'time' in trade:
                    hold_time = (trade['close_time'] - trade['time']).total_seconds() / 3600  # em horas
                    hold_times.append(hold_time)
            avg_position_time = sum(hold_times) / len(hold_times) if hold_times else 0
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_return': total_return,
                'avg_daily_return': avg_daily_return,
                'daily_volatility': daily_volatility,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'avg_position_time': avg_position_time
            }
            
        except Exception as e:
            logger.error(f"Erro ao analisar performance: {e}")
            return {}
    
    def _analyze_risk(self) -> Dict:
        """
        Analisa risco do sistema.
        
        Returns:
            Dict com análise
        """
        try:
            if not self.trades:
                return {
                    'max_drawdown': 0,
                    'var_95': 0,
                    'var_99': 0,
                    'es_95': 0,
                    'es_99': 0,
                    'beta': 0,
                    'correlation': 0
                }
            
            # Calcular drawdown máximo
            equity_series = pd.Series(self.equity_curve)
            rolling_max = equity_series.expanding().max()
            drawdowns = (equity_series - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            # Calcular Value at Risk (VaR)
            returns = pd.Series(self.equity_curve).pct_change().dropna()
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            # Calcular Expected Shortfall (ES)
            es_95 = returns[returns <= var_95].mean()
            es_99 = returns[returns <= var_99].mean()
            
            # Calcular Beta e Correlação
            market_returns = pd.Series(self.metrics_history).pct_change().dropna()
            beta = returns.cov(market_returns) / market_returns.var()
            correlation = returns.corr(market_returns)
            
            return {
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'var_99': var_99,
                'es_95': es_95,
                'es_99': es_99,
                'beta': beta,
                'correlation': correlation
            }
            
        except Exception as e:
            logger.error(f"Erro ao analisar risco: {e}")
            return {}
    
    def _analyze_trades(self) -> Dict:
        """
        Analisa trades do sistema.
        
        Returns:
            Dict com análise
        """
        try:
            if not self.trades:
                return {
                    'by_direction': {},
                    'by_outcome': {},
                    'by_size': {}
                }
            
            # Análise por direção
            long_trades = [t for t in self.trades if t['side'] == 'buy']
            short_trades = [t for t in self.trades if t['side'] == 'sell']
            
            by_direction = {
                'long': {
                    'count': len(long_trades),
                    'win_rate': len([t for t in long_trades if t['pnl'] > 0]) / len(long_trades) if long_trades else 0,
                    'avg_pnl': sum(t['pnl'] for t in long_trades) / len(long_trades) if long_trades else 0,
                    'total_pnl': sum(t['pnl'] for t in long_trades)
                },
                'short': {
                    'count': len(short_trades),
                    'win_rate': len([t for t in short_trades if t['pnl'] > 0]) / len(short_trades) if short_trades else 0,
                    'avg_pnl': sum(t['pnl'] for t in short_trades) / len(short_trades) if short_trades else 0,
                    'total_pnl': sum(t['pnl'] for t in short_trades)
                }
            }
            
            # Análise por resultado
            winning_trades = [t for t in self.trades if t['pnl'] > 0]
            losing_trades = [t for t in self.trades if t['pnl'] < 0]
            
            by_outcome = {
                'winning': {
                    'count': len(winning_trades),
                    'avg_pnl': sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0,
                    'total_pnl': sum(t['pnl'] for t in winning_trades)
                },
                'losing': {
                    'count': len(losing_trades),
                    'avg_pnl': sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0,
                    'total_pnl': sum(t['pnl'] for t in losing_trades)
                }
            }
            
            # Análise por tamanho
            small_trades = [t for t in self.trades if t['size'] < 100]
            medium_trades = [t for t in self.trades if 100 <= t['size'] < 500]
            large_trades = [t for t in self.trades if t['size'] >= 500]
            
            by_size = {
                'small': {
                    'count': len(small_trades),
                    'win_rate': len([t for t in small_trades if t['pnl'] > 0]) / len(small_trades) if small_trades else 0,
                    'avg_pnl': sum(t['pnl'] for t in small_trades) / len(small_trades) if small_trades else 0,
                    'total_pnl': sum(t['pnl'] for t in small_trades)
                },
                'medium': {
                    'count': len(medium_trades),
                    'win_rate': len([t for t in medium_trades if t['pnl'] > 0]) / len(medium_trades) if medium_trades else 0,
                    'avg_pnl': sum(t['pnl'] for t in medium_trades) / len(medium_trades) if medium_trades else 0,
                    'total_pnl': sum(t['pnl'] for t in medium_trades)
                },
                'large': {
                    'count': len(large_trades),
                    'win_rate': len([t for t in large_trades if t['pnl'] > 0]) / len(large_trades) if large_trades else 0,
                    'avg_pnl': sum(t['pnl'] for t in large_trades) / len(large_trades) if large_trades else 0,
                    'total_pnl': sum(t['pnl'] for t in large_trades)
                }
            }
            
            return {
                'by_direction': by_direction,
                'by_outcome': by_outcome,
                'by_size': by_size
            }
            
        except Exception as e:
            logger.error(f"Erro ao analisar trades: {e}")
            return {}
    
    def _generate_analysis(self) -> str:
        """
        Gera análise completa.
        
        Returns:
            String com análise
        """
        try:
            # Calcular análises
            performance_analysis = self._analyze_performance()
            risk_analysis = self._analyze_risk()
            trade_analysis = self._analyze_trades()
            
            # Gerar análise
            analysis = "=== Análise do Sistema de Trading ===\n\n"
            
            # Análise de Performance
            analysis += "Análise de Performance:\n"
            analysis += f"Total de Trades: {performance_analysis['total_trades']}\n"
            analysis += f"Win Rate: {performance_analysis['win_rate']:.2%}\n"
            analysis += f"Profit Factor: {performance_analysis['profit_factor']:.2f}\n"
            analysis += f"Retorno Total: {performance_analysis['total_return']:.2%}\n"
            analysis += f"Retorno Diário Médio: {performance_analysis['avg_daily_return']:.2%}\n"
            analysis += f"Volatilidade Diária: {performance_analysis['daily_volatility']:.2%}\n"
            analysis += f"Volatilidade Anual: {performance_analysis['annual_volatility']:.2%}\n"
            analysis += f"Sharpe Ratio: {performance_analysis['sharpe_ratio']:.2f}\n"
            analysis += f"Max Drawdown: {performance_analysis['max_drawdown']:.2%}\n"
            analysis += f"Tempo Médio de Posição: {performance_analysis['avg_position_time']:.2f}h\n\n"
            
            # Análise de Risco
            analysis += "Análise de Risco:\n"
            analysis += f"VaR (95%): {risk_analysis['var_95']:.2%}\n"
            analysis += f"VaR (99%): {risk_analysis['var_99']:.2%}\n"
            analysis += f"ES (95%): {risk_analysis['es_95']:.2%}\n"
            analysis += f"ES (99%): {risk_analysis['es_99']:.2%}\n"
            analysis += f"Beta: {risk_analysis['beta']:.2f}\n"
            analysis += f"Correlação com Mercado: {risk_analysis['correlation']:.2f}\n\n"
            
            # Análise de Trades
            analysis += "Análise de Trades:\n\n"
            
            # Por Direção
            analysis += "Por Direção:\n"
            for direction, stats in trade_analysis['by_direction'].items():
                analysis += f"\n{direction.upper()}:\n"
                analysis += f"Quantidade: {stats['count']}\n"
                analysis += f"Win Rate: {stats['win_rate']:.2%}\n"
                analysis += f"P&L Médio: ${stats['avg_pnl']:.2f}\n"
                analysis += f"P&L Total: ${stats['total_pnl']:.2f}\n"
            
            # Por Resultado
            analysis += "\nPor Resultado:\n"
            for outcome, stats in trade_analysis['by_outcome'].items():
                analysis += f"\n{outcome.upper()}:\n"
                analysis += f"Quantidade: {stats['count']}\n"
                analysis += f"P&L Médio: ${stats['avg_pnl']:.2f}\n"
                analysis += f"P&L Total: ${stats['total_pnl']:.2f}\n"
            
            # Por Tamanho
            analysis += "\nPor Tamanho:\n"
            for size, stats in trade_analysis['by_size'].items():
                analysis += f"\n{size.upper()}:\n"
                analysis += f"Quantidade: {stats['count']}\n"
                analysis += f"Win Rate: {stats['win_rate']:.2%}\n"
                analysis += f"P&L Médio: ${stats['avg_pnl']:.2f}\n"
                analysis += f"P&L Total: ${stats['total_pnl']:.2f}\n"
            
            return analysis
            
        except Exception as e:
            logger.error(f"Erro ao gerar análise: {e}")
            return ""
    
    async def run(self):
        """Executa a análise."""
        try:
            # Carregar dados
            if not await self._load_trading_data():
                logger.error("Não foi possível carregar dados de trading")
                return
            
            # Gerar análise
            analysis = self._generate_analysis()
            if not analysis:
                logger.error("Não foi possível gerar análise")
                return
            
            # Salvar análise
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trading_analysis_{timestamp}.txt"
            
            with open(filename, 'w') as f:
                f.write(analysis)
            
            logger.info(f"Análise salva em {filename}")
            
        except Exception as e:
            logger.error(f"Erro ao executar análise: {e}")

def main():
    """Função principal."""
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
        trading_system = TradingSystem(config)
        
        # Executar análise
        analyzer = TradingAnalyzer(trading_system)
        asyncio.run(analyzer.run())
        
    except KeyboardInterrupt:
        logger.info("Encerrando por solicitação do usuário")
    except Exception as e:
        logger.error(f"Erro fatal: {e}")

if __name__ == "__main__":
    main() 