"""
Script para monitoramento do sistema de trading.
"""

import asyncio
import logging
import os
import psutil
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
        logging.FileHandler('monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingMonitor:
    """Monitor do sistema de trading."""
    
    def __init__(self, trading_system: TradingSystem):
        """
        Inicializa o monitor.
        
        Args:
            trading_system: Sistema de trading
        """
        self.trading_system = trading_system
        self.running = False
        self.metrics_history: List[Dict] = []
        self.alerts: List[Dict] = []
        self.last_check = datetime.now()
    
    async def _update_market_data(self) -> pd.DataFrame:
        """
        Atualiza dados de mercado.
        
        Returns:
            DataFrame com dados de mercado
        """
        try:
            # Criar carregador
            loader = DataLoader(
                exchange=self.trading_system.config.exchange,
                symbol=self.trading_system.config.symbol,
                timeframe=self.trading_system.config.timeframe
            )
            
            # Carregar dados do último dia
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1)
            df = await loader.load_data(start_date, end_date)
            
            if df.empty:
                logger.error("Não foi possível carregar dados de mercado")
                return pd.DataFrame()
            
            return df
            
        except Exception as e:
            logger.error(f"Erro ao atualizar dados de mercado: {e}")
            return pd.DataFrame()
    
    def _check_risk_metrics(self, metrics: Dict):
        """
        Verifica métricas de risco.
        
        Args:
            metrics: Métricas atuais
        """
        try:
            # Verificar drawdown
            if metrics['max_drawdown'] < -0.1:  # 10% de drawdown
                self.alerts.append({
                    'type': 'risk',
                    'level': 'high',
                    'message': f"Drawdown alto: {metrics['max_drawdown']:.2%}",
                    'time': datetime.now()
                })
            
            # Verificar exposição
            if metrics['total_positions'] > self.trading_system.config.max_positions:
                self.alerts.append({
                    'type': 'risk',
                    'level': 'medium',
                    'message': f"Exposição alta: {metrics['total_positions']} posições",
                    'time': datetime.now()
                })
            
            # Verificar concentração
            if metrics['largest_position'] > metrics['total_equity'] * 0.3:  # 30% em uma posição
                self.alerts.append({
                    'type': 'risk',
                    'level': 'high',
                    'message': f"Concentração alta: {metrics['largest_position']:.2%} em uma posição",
                    'time': datetime.now()
                })
            
        except Exception as e:
            logger.error(f"Erro ao verificar métricas de risco: {e}")
    
    def _check_performance_metrics(self, metrics: Dict):
        """
        Verifica métricas de performance.
        
        Args:
            metrics: Métricas atuais
        """
        try:
            # Verificar win rate
            if metrics['win_rate'] < 0.4:  # 40% de win rate
                self.alerts.append({
                    'type': 'performance',
                    'level': 'medium',
                    'message': f"Win rate baixo: {metrics['win_rate']:.2%}",
                    'time': datetime.now()
                })
            
            # Verificar profit factor
            if metrics['profit_factor'] < 1.5:  # 1.5 de profit factor
                self.alerts.append({
                    'type': 'performance',
                    'level': 'medium',
                    'message': f"Profit factor baixo: {metrics['profit_factor']:.2f}",
                    'time': datetime.now()
                })
            
            # Verificar Sharpe ratio
            if metrics['sharpe_ratio'] < 1.0:  # 1.0 de Sharpe ratio
                self.alerts.append({
                    'type': 'performance',
                    'level': 'medium',
                    'message': f"Sharpe ratio baixo: {metrics['sharpe_ratio']:.2f}",
                    'time': datetime.now()
                })
            
        except Exception as e:
            logger.error(f"Erro ao verificar métricas de performance: {e}")
    
    def _check_system_health(self):
        """Verifica saúde do sistema."""
        try:
            # Verificar uso de CPU
            cpu_percent = psutil.cpu_percent()
            if cpu_percent > 80:  # 80% de uso
                self.alerts.append({
                    'type': 'system',
                    'level': 'high',
                    'message': f"Uso de CPU alto: {cpu_percent}%",
                    'time': datetime.now()
                })
            
            # Verificar uso de memória
            memory = psutil.virtual_memory()
            if memory.percent > 80:  # 80% de uso
                self.alerts.append({
                    'type': 'system',
                    'level': 'high',
                    'message': f"Uso de memória alto: {memory.percent}%",
                    'time': datetime.now()
                })
            
            # Verificar uso de disco
            disk = psutil.disk_usage('/')
            if disk.percent > 80:  # 80% de uso
                self.alerts.append({
                    'type': 'system',
                    'level': 'medium',
                    'message': f"Uso de disco alto: {disk.percent}%",
                    'time': datetime.now()
                })
            
        except Exception as e:
            logger.error(f"Erro ao verificar saúde do sistema: {e}")
    
    def _update_metrics(self, metrics: Dict):
        """
        Atualiza histórico de métricas.
        
        Args:
            metrics: Métricas atuais
        """
        try:
            # Adicionar timestamp
            metrics['timestamp'] = datetime.now()
            
            # Adicionar ao histórico
            self.metrics_history.append(metrics)
            
            # Manter apenas últimas 24h
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.metrics_history = [
                m for m in self.metrics_history
                if m['timestamp'] > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Erro ao atualizar métricas: {e}")
    
    async def _update_loop(self):
        """Loop principal de atualização."""
        try:
            while self.running:
                # Atualizar dados de mercado
                market_data = await self._update_market_data()
                if market_data.empty:
                    await asyncio.sleep(60)  # 1 minuto
                    continue
                
                # Atualizar estado do sistema
                await self.trading_system._update_quantum_state(market_data.iloc[-1])
                
                # Coletar métricas
                metrics = self.trading_system._calculate_metrics()
                
                # Verificar alertas
                self._check_risk_metrics(metrics)
                self._check_performance_metrics(metrics)
                self._check_system_health()
                
                # Atualizar histórico
                self._update_metrics(metrics)
                
                # Logar status
                logger.info("\n=== Status do Sistema ===")
                logger.info(f"Balance: ${metrics['total_equity']:.2f}")
                logger.info(f"Posições: {metrics['total_positions']}")
                logger.info(f"Win Rate: {metrics['win_rate']:.2%}")
                logger.info(f"Profit Factor: {metrics['profit_factor']:.2f}")
                logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
                
                # Logar alertas
                if self.alerts:
                    logger.info("\n=== Alertas ===")
                    for alert in self.alerts:
                        logger.info(f"[{alert['level'].upper()}] {alert['message']}")
                    self.alerts = []  # Limpar alertas
                
                # Aguardar próximo ciclo
                await asyncio.sleep(60)  # 1 minuto
                
        except Exception as e:
            logger.error(f"Erro no loop de atualização: {e}")
    
    async def run(self):
        """Executa o monitoramento."""
        try:
            # Iniciar loop
            self.running = True
            logger.info("Iniciando monitoramento...")
            await self._update_loop()
            
        except Exception as e:
            logger.error(f"Erro ao executar monitoramento: {e}")
        finally:
            self.running = False

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
        
        # Executar monitoramento
        monitor = TradingMonitor(trading_system)
        asyncio.run(monitor.run())
        
    except KeyboardInterrupt:
        logger.info("Encerrando por solicitação do usuário")
    except Exception as e:
        logger.error(f"Erro fatal: {e}")

if __name__ == "__main__":
    main() 