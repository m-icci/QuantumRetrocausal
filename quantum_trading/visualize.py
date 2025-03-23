"""
Script para visualização do sistema de trading.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from quantum_trading.core.trading import TradingSystem, TradingConfig
from quantum_trading.data_loader import DataLoader

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('visualize.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingVisualizer:
    """Visualizador do sistema de trading."""
    
    def __init__(self, trading_system: TradingSystem):
        """
        Inicializa o visualizador.
        
        Args:
            trading_system: Sistema de trading
        """
        self.trading_system = trading_system
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []
        self.metrics_history: List[Dict] = []
        
        # Configurar estilo dos gráficos
        plt.style.use('seaborn')
        sns.set_palette("husl")
    
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
    
    def _plot_equity_curve(self):
        """Plota curva de equity."""
        try:
            # Criar figura
            plt.figure(figsize=(12, 6))
            
            # Plotar curva de equity
            plt.plot(self.equity_curve, label='Equity')
            
            # Configurar título e labels
            plt.title('Curva de Equity')
            plt.xlabel('Tempo')
            plt.ylabel('Equity ($)')
            
            # Adicionar grid e legenda
            plt.grid(True)
            plt.legend()
            
            # Salvar figura
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"equity_curve_{timestamp}.png"
            plt.savefig(filename)
            plt.close()
            
            logger.info(f"Gráfico de equity salvo em {filename}")
            
        except Exception as e:
            logger.error(f"Erro ao plotar curva de equity: {e}")
    
    def _plot_drawdown(self):
        """Plota drawdown."""
        try:
            # Criar figura
            plt.figure(figsize=(12, 6))
            
            # Calcular drawdown
            equity_series = pd.Series(self.equity_curve)
            rolling_max = equity_series.expanding().max()
            drawdowns = (equity_series - rolling_max) / rolling_max
            
            # Plotar drawdown
            plt.plot(drawdowns, label='Drawdown')
            
            # Configurar título e labels
            plt.title('Drawdown')
            plt.xlabel('Tempo')
            plt.ylabel('Drawdown (%)')
            
            # Adicionar grid e legenda
            plt.grid(True)
            plt.legend()
            
            # Salvar figura
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"drawdown_{timestamp}.png"
            plt.savefig(filename)
            plt.close()
            
            logger.info(f"Gráfico de drawdown salvo em {filename}")
            
        except Exception as e:
            logger.error(f"Erro ao plotar drawdown: {e}")
    
    def _plot_returns_distribution(self):
        """Plota distribuição de retornos."""
        try:
            # Criar figura
            plt.figure(figsize=(12, 6))
            
            # Calcular retornos
            returns = pd.Series(self.equity_curve).pct_change().dropna()
            
            # Plotar distribuição
            sns.histplot(returns, bins=50, kde=True)
            
            # Configurar título e labels
            plt.title('Distribuição de Retornos')
            plt.xlabel('Retorno (%)')
            plt.ylabel('Frequência')
            
            # Adicionar grid
            plt.grid(True)
            
            # Salvar figura
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"returns_distribution_{timestamp}.png"
            plt.savefig(filename)
            plt.close()
            
            logger.info(f"Gráfico de distribuição de retornos salvo em {filename}")
            
        except Exception as e:
            logger.error(f"Erro ao plotar distribuição de retornos: {e}")
    
    def _plot_trade_analysis(self):
        """Plota análise de trades."""
        try:
            # Criar figura com subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Análise por direção
            long_trades = [t for t in self.trades if t['side'] == 'buy']
            short_trades = [t for t in self.trades if t['side'] == 'sell']
            
            directions = ['Long', 'Short']
            counts = [len(long_trades), len(short_trades)]
            win_rates = [
                len([t for t in long_trades if t['pnl'] > 0]) / len(long_trades) if long_trades else 0,
                len([t for t in short_trades if t['pnl'] > 0]) / len(short_trades) if short_trades else 0
            ]
            
            # Plotar quantidade por direção
            axes[0, 0].bar(directions, counts)
            axes[0, 0].set_title('Quantidade de Trades por Direção')
            axes[0, 0].set_ylabel('Quantidade')
            
            # Plotar win rate por direção
            axes[0, 1].bar(directions, win_rates)
            axes[0, 1].set_title('Win Rate por Direção')
            axes[0, 1].set_ylabel('Win Rate (%)')
            
            # Análise por resultado
            winning_trades = [t for t in self.trades if t['pnl'] > 0]
            losing_trades = [t for t in self.trades if t['pnl'] < 0]
            
            outcomes = ['Ganhos', 'Perdas']
            counts = [len(winning_trades), len(losing_trades)]
            avg_pnls = [
                sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0,
                sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
            ]
            
            # Plotar quantidade por resultado
            axes[1, 0].bar(outcomes, counts)
            axes[1, 0].set_title('Quantidade de Trades por Resultado')
            axes[1, 0].set_ylabel('Quantidade')
            
            # Plotar P&L médio por resultado
            axes[1, 1].bar(outcomes, avg_pnls)
            axes[1, 1].set_title('P&L Médio por Resultado')
            axes[1, 1].set_ylabel('P&L Médio ($)')
            
            # Ajustar layout
            plt.tight_layout()
            
            # Salvar figura
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trade_analysis_{timestamp}.png"
            plt.savefig(filename)
            plt.close()
            
            logger.info(f"Gráfico de análise de trades salvo em {filename}")
            
        except Exception as e:
            logger.error(f"Erro ao plotar análise de trades: {e}")
    
    def _plot_risk_metrics(self):
        """Plota métricas de risco."""
        try:
            # Criar figura
            plt.figure(figsize=(12, 6))
            
            # Calcular retornos
            returns = pd.Series(self.equity_curve).pct_change().dropna()
            
            # Calcular VaR e ES
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            es_95 = returns[returns <= var_95].mean()
            es_99 = returns[returns <= var_99].mean()
            
            # Plotar distribuição com VaR e ES
            sns.histplot(returns, bins=50, kde=True)
            plt.axvline(x=var_95, color='r', linestyle='--', label=f'VaR 95%: {var_95:.2%}')
            plt.axvline(x=var_99, color='g', linestyle='--', label=f'VaR 99%: {var_99:.2%}')
            plt.axvline(x=es_95, color='y', linestyle='--', label=f'ES 95%: {es_95:.2%}')
            plt.axvline(x=es_99, color='m', linestyle='--', label=f'ES 99%: {es_99:.2%}')
            
            # Configurar título e labels
            plt.title('Métricas de Risco')
            plt.xlabel('Retorno (%)')
            plt.ylabel('Frequência')
            
            # Adicionar grid e legenda
            plt.grid(True)
            plt.legend()
            
            # Salvar figura
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"risk_metrics_{timestamp}.png"
            plt.savefig(filename)
            plt.close()
            
            logger.info(f"Gráfico de métricas de risco salvo em {filename}")
            
        except Exception as e:
            logger.error(f"Erro ao plotar métricas de risco: {e}")
    
    async def run(self):
        """Executa a visualização."""
        try:
            # Carregar dados
            if not await self._load_trading_data():
                logger.error("Não foi possível carregar dados de trading")
                return
            
            # Gerar visualizações
            self._plot_equity_curve()
            self._plot_drawdown()
            self._plot_returns_distribution()
            self._plot_trade_analysis()
            self._plot_risk_metrics()
            
            logger.info("Visualizações geradas com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao executar visualização: {e}")

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
        
        # Executar visualização
        visualizer = TradingVisualizer(trading_system)
        asyncio.run(visualizer.run())
        
    except KeyboardInterrupt:
        logger.info("Encerrando por solicitação do usuário")
    except Exception as e:
        logger.error(f"Erro fatal: {e}")

if __name__ == "__main__":
    main() 