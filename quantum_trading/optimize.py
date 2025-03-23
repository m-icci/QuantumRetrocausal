"""
Script para otimização do sistema de trading.
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
        logging.FileHandler('optimize.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingOptimizer:
    """Otimizador do sistema de trading."""
    
    def __init__(self, trading_system: TradingSystem):
        """
        Inicializa o otimizador.
        
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
    
    def _generate_parameter_combinations(self) -> List[Dict]:
        """
        Gera combinações de parâmetros para teste.
        
        Returns:
            Lista de combinações
        """
        try:
            combinations = []
            
            # RSI
            rsi_periods = np.arange(10, 30, 2)
            rsi_overboughts = np.arange(60, 80, 5)
            rsi_oversolds = np.arange(20, 40, 5)
            
            # MACD
            macd_fasts = np.arange(8, 16, 2)
            macd_slows = np.arange(20, 32, 2)
            macd_signals = np.arange(6, 12, 2)
            
            # Bollinger Bands
            bb_periods = np.arange(15, 25, 2)
            bb_stds = np.arange(1.5, 3.0, 0.2)
            
            # ATR
            atr_periods = np.arange(10, 20, 2)
            atr_multipliers = np.arange(1.5, 3.0, 0.2)
            
            # Stop Loss e Take Profit
            stop_losses = np.arange(0.01, 0.05, 0.01)
            take_profits = np.arange(0.02, 0.08, 0.01)
            
            # Tamanho da Posição
            position_sizes = np.arange(50, 500, 50)
            
            # Limites
            max_positions = np.arange(2, 5)
            daily_trades_limits = np.arange(5, 15)
            
            # Gerar combinações
            for rsi_period in rsi_periods:
                for rsi_overbought in rsi_overboughts:
                    for rsi_oversold in rsi_oversolds:
                        for macd_fast in macd_fasts:
                            for macd_slow in macd_slows:
                                for macd_signal in macd_signals:
                                    for bb_period in bb_periods:
                                        for bb_std in bb_stds:
                                            for atr_period in atr_periods:
                                                for atr_multiplier in atr_multipliers:
                                                    for stop_loss in stop_losses:
                                                        for take_profit in take_profits:
                                                            for position_size in position_sizes:
                                                                for max_position in max_positions:
                                                                    for daily_trades_limit in daily_trades_limits:
                                                                        # Validar combinação
                                                                        if self._validate_combination({
                                                                            'rsi_period': rsi_period,
                                                                            'rsi_overbought': rsi_overbought,
                                                                            'rsi_oversold': rsi_oversold,
                                                                            'macd_fast': macd_fast,
                                                                            'macd_slow': macd_slow,
                                                                            'macd_signal': macd_signal,
                                                                            'bb_period': bb_period,
                                                                            'bb_std': bb_std,
                                                                            'atr_period': atr_period,
                                                                            'atr_multiplier': atr_multiplier,
                                                                            'stop_loss': stop_loss,
                                                                            'take_profit': take_profit,
                                                                            'position_size': position_size,
                                                                            'max_positions': max_position,
                                                                            'daily_trades_limit': daily_trades_limit
                                                                        }):
                                                                            combinations.append({
                                                                                'rsi_period': rsi_period,
                                                                                'rsi_overbought': rsi_overbought,
                                                                                'rsi_oversold': rsi_oversold,
                                                                                'macd_fast': macd_fast,
                                                                                'macd_slow': macd_slow,
                                                                                'macd_signal': macd_signal,
                                                                                'bb_period': bb_period,
                                                                                'bb_std': bb_std,
                                                                                'atr_period': atr_period,
                                                                                'atr_multiplier': atr_multiplier,
                                                                                'stop_loss': stop_loss,
                                                                                'take_profit': take_profit,
                                                                                'position_size': position_size,
                                                                                'max_positions': max_position,
                                                                                'daily_trades_limit': daily_trades_limit
                                                                            })
            
            return combinations
            
        except Exception as e:
            logger.error(f"Erro ao gerar combinações de parâmetros: {e}")
            return []
    
    def _validate_combination(self, combination: Dict) -> bool:
        """
        Valida uma combinação de parâmetros.
        
        Args:
            combination: Combinação de parâmetros
            
        Returns:
            True se válida
        """
        try:
            # Validar RSI
            if combination['rsi_period'] >= combination['rsi_overbought'] or \
               combination['rsi_period'] >= combination['rsi_oversold'] or \
               combination['rsi_oversold'] >= combination['rsi_overbought']:
                return False
            
            # Validar MACD
            if combination['macd_fast'] >= combination['macd_slow'] or \
               combination['macd_signal'] >= combination['macd_slow']:
                return False
            
            # Validar Stop Loss e Take Profit
            if combination['stop_loss'] >= combination['take_profit']:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao validar combinação: {e}")
            return False
    
    async def _run_backtest(self, combination: Dict) -> Dict:
        """
        Executa backtest com uma combinação de parâmetros.
        
        Args:
            combination: Combinação de parâmetros
            
        Returns:
            Dict com resultados
        """
        try:
            # Atualizar configuração
            config = self.trading_system.config.copy()
            config.update(combination)
            
            # Executar backtest
            results = await self.trading_system.run_backtest(config)
            
            # Calcular métricas
            total_trades = len(results['trades'])
            winning_trades = len([t for t in results['trades'] if t['pnl'] > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            total_profit = sum(t['pnl'] for t in results['trades'] if t['pnl'] > 0)
            total_loss = abs(sum(t['pnl'] for t in results['trades'] if t['pnl'] < 0))
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            returns = pd.Series(results['equity_curve']).pct_change().dropna()
            sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
            
            equity_series = pd.Series(results['equity_curve'])
            rolling_max = equity_series.expanding().max()
            drawdowns = (equity_series - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            return {
                'parameters': combination,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_profit': total_profit,
                'total_loss': total_loss
            }
            
        except Exception as e:
            logger.error(f"Erro ao executar backtest: {e}")
            return {}
    
    def _analyze_results(self, results: List[Dict]):
        """
        Analisa resultados da otimização.
        
        Args:
            results: Lista de resultados
        """
        try:
            # Converter para DataFrame
            df = pd.DataFrame(results)
            
            # Ordenar por Sharpe Ratio
            df = df.sort_values('sharpe_ratio', ascending=False)
            
            # Logar top 10 resultados
            logger.info("\nTop 10 Resultados:")
            for i, row in df.head(10).iterrows():
                logger.info(f"\nPosição {i+1}:")
                logger.info(f"Sharpe Ratio: {row['sharpe_ratio']:.2f}")
                logger.info(f"Win Rate: {row['win_rate']:.2%}")
                logger.info(f"Profit Factor: {row['profit_factor']:.2f}")
                logger.info(f"Max Drawdown: {row['max_drawdown']:.2%}")
                logger.info(f"Total Profit: ${row['total_profit']:.2f}")
                logger.info(f"Total Loss: ${row['total_loss']:.2f}")
                logger.info("\nParâmetros:")
                for param, value in row['parameters'].items():
                    logger.info(f"{param}: {value}")
            
            # Salvar resultados
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_results_{timestamp}.csv"
            df.to_csv(filename, index=False)
            
            logger.info(f"\nResultados salvos em {filename}")
            
            # Gerar relatório
            self._generate_report(df)
            
        except Exception as e:
            logger.error(f"Erro ao analisar resultados: {e}")
    
    def _generate_report(self, df: pd.DataFrame):
        """
        Gera relatório de otimização.
        
        Args:
            df: DataFrame com resultados
        """
        try:
            # Criar relatório
            report = "=== Relatório de Otimização ===\n\n"
            
            # Análise de Parâmetros
            report += "Análise de Parâmetros:\n\n"
            
            # Correlação com Sharpe Ratio
            correlations = df.corr()['sharpe_ratio'].sort_values(ascending=False)
            report += "Correlação com Sharpe Ratio:\n"
            for param, corr in correlations.items():
                if param != 'sharpe_ratio':
                    report += f"{param}: {corr:.3f}\n"
            
            # Recomendações
            report += "\nRecomendações:\n"
            
            # RSI
            rsi_corr = correlations['rsi_period']
            if rsi_corr > 0.1:
                report += "- Aumentar período do RSI\n"
            elif rsi_corr < -0.1:
                report += "- Diminuir período do RSI\n"
            
            # MACD
            macd_fast_corr = correlations['macd_fast']
            if macd_fast_corr > 0.1:
                report += "- Aumentar período rápido do MACD\n"
            elif macd_fast_corr < -0.1:
                report += "- Diminuir período rápido do MACD\n"
            
            # Bollinger Bands
            bb_std_corr = correlations['bb_std']
            if bb_std_corr > 0.1:
                report += "- Aumentar desvio padrão das Bandas de Bollinger\n"
            elif bb_std_corr < -0.1:
                report += "- Diminuir desvio padrão das Bandas de Bollinger\n"
            
            # Stop Loss e Take Profit
            stop_loss_corr = correlations['stop_loss']
            if stop_loss_corr > 0.1:
                report += "- Aumentar Stop Loss\n"
            elif stop_loss_corr < -0.1:
                report += "- Diminuir Stop Loss\n"
            
            take_profit_corr = correlations['take_profit']
            if take_profit_corr > 0.1:
                report += "- Aumentar Take Profit\n"
            elif take_profit_corr < -0.1:
                report += "- Diminuir Take Profit\n"
            
            # Tamanho da Posição
            position_size_corr = correlations['position_size']
            if position_size_corr > 0.1:
                report += "- Aumentar tamanho da posição\n"
            elif position_size_corr < -0.1:
                report += "- Diminuir tamanho da posição\n"
            
            # Limites
            max_positions_corr = correlations['max_positions']
            if max_positions_corr > 0.1:
                report += "- Aumentar número máximo de posições\n"
            elif max_positions_corr < -0.1:
                report += "- Diminuir número máximo de posições\n"
            
            daily_trades_limit_corr = correlations['daily_trades_limit']
            if daily_trades_limit_corr > 0.1:
                report += "- Aumentar limite diário de trades\n"
            elif daily_trades_limit_corr < -0.1:
                report += "- Diminuir limite diário de trades\n"
            
            # Estatísticas Gerais
            report += "\nEstatísticas Gerais:\n"
            report += f"Total de Combinações Testadas: {len(df)}\n"
            report += f"Melhor Sharpe Ratio: {df['sharpe_ratio'].max():.2f}\n"
            report += f"Pior Sharpe Ratio: {df['sharpe_ratio'].min():.2f}\n"
            report += f"Sharpe Ratio Médio: {df['sharpe_ratio'].mean():.2f}\n"
            report += f"Desvio Padrão do Sharpe Ratio: {df['sharpe_ratio'].std():.2f}\n"
            
            # Salvar relatório
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_report_{timestamp}.txt"
            
            with open(filename, 'w') as f:
                f.write(report)
            
            logger.info(f"Relatório salvo em {filename}")
            
        except Exception as e:
            logger.error(f"Erro ao gerar relatório: {e}")
    
    async def run(self):
        """Executa a otimização."""
        try:
            # Carregar dados
            if not await self._load_trading_data():
                logger.error("Não foi possível carregar dados de trading")
                return
            
            # Gerar combinações
            combinations = self._generate_parameter_combinations()
            if not combinations:
                logger.error("Não foi possível gerar combinações de parâmetros")
                return
            
            logger.info(f"Geradas {len(combinations)} combinações de parâmetros")
            
            # Executar backtests
            results = []
            for i, combination in enumerate(combinations, 1):
                logger.info(f"Executando backtest {i}/{len(combinations)}")
                result = await self._run_backtest(combination)
                if result:
                    results.append(result)
            
            # Analisar resultados
            if results:
                self._analyze_results(results)
            else:
                logger.error("Nenhum resultado válido obtido")
            
        except Exception as e:
            logger.error(f"Erro ao executar otimização: {e}")

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
        
        # Executar otimização
        optimizer = TradingOptimizer(trading_system)
        asyncio.run(optimizer.run())
        
    except KeyboardInterrupt:
        logger.info("Encerrando por solicitação do usuário")
    except Exception as e:
        logger.error(f"Erro fatal: {e}")

if __name__ == "__main__":
    main() 