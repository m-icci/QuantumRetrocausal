"""
Script principal para executar o sistema de trading.
"""

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from quantum_trading.core.trading import TradingSystem, TradingConfig

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TradingRunner:
    """Runner para o sistema de trading."""
    
    def __init__(self):
        """Inicializa o runner."""
        self.trading_system: Optional[TradingSystem] = None
        self.running = False
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Configura handlers para sinais do sistema."""
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
    
    def _handle_shutdown(self, signum, frame):
        """Manipula sinais de shutdown."""
        logger.info(f"Recebido sinal {signum}. Iniciando shutdown...")
        self.running = False
    
    async def _validate_config(self, config: TradingConfig) -> bool:
        """
        Valida a configuração antes de iniciar.
        
        Args:
            config: Configuração do trading
            
        Returns:
            bool: True se válida, False caso contrário
        """
        try:
            # Validar configurações básicas
            if not config.exchange or not config.api_key or not config.api_secret:
                logger.error("Configurações de exchange não fornecidas")
                return False
            
            if not config.symbol:
                logger.error("Símbolo não configurado")
                return False
            
            # Validar limites de risco
            if config.daily_loss_limit <= 0:
                logger.error("Limite de perda diária deve ser positivo")
                return False
            
            if config.risk_per_trade <= 0 or config.risk_per_trade > 1:
                logger.error("Risco por trade deve estar entre 0 e 1")
                return False
            
            # Validar parâmetros técnicos
            if config.rsi_period <= 0:
                logger.error("Período do RSI deve ser positivo")
                return False
            
            if config.macd_fast <= 0 or config.macd_slow <= 0:
                logger.error("Períodos do MACD devem ser positivos")
                return False
            
            if config.bb_period <= 0:
                logger.error("Período das Bollinger Bands deve ser positivo")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao validar configuração: {e}")
            return False
    
    async def _check_balance(self, config: TradingConfig) -> bool:
        """
        Verifica saldo disponível.
        
        Args:
            config: Configuração do trading
            
        Returns:
            bool: True se saldo suficiente, False caso contrário
        """
        try:
            # TODO: Implementar verificação de saldo
            return True
            
        except Exception as e:
            logger.error(f"Erro ao verificar saldo: {e}")
            return False
    
    async def _check_api_permissions(self, config: TradingConfig) -> bool:
        """
        Verifica permissões da API.
        
        Args:
            config: Configuração do trading
            
        Returns:
            bool: True se permissões ok, False caso contrário
        """
        try:
            # TODO: Implementar verificação de permissões
            return True
            
        except Exception as e:
            logger.error(f"Erro ao verificar permissões da API: {e}")
            return False
    
    async def _monitor_trading(self):
        """Monitora o sistema de trading."""
        while self.running:
            try:
                if self.trading_system:
                    status = self.trading_system.get_status()
                    
                    # Log de métricas
                    logger.info("=== Métricas de Trading ===")
                    logger.info(f"Win Rate: {status['metrics']['win_rate']:.2%}")
                    logger.info(f"Profit Factor: {status['metrics']['profit_factor']:.2f}")
                    logger.info(f"Sharpe Ratio: {status['metrics']['sharpe_ratio']:.2f}")
                    logger.info(f"Max Drawdown: {status['metrics']['max_drawdown']:.2%}")
                    
                    # Log de estatísticas diárias
                    logger.info("\n=== Estatísticas Diárias ===")
                    logger.info(f"Trades Hoje: {status['daily_stats']['trades']}")
                    logger.info(f"P&L Hoje: {status['daily_stats']['pnl']:.2f}")
                    
                    # Log de posições ativas
                    if status['active_trades']:
                        logger.info("\n=== Posições Ativas ===")
                        for trade in status['active_trades']:
                            logger.info(
                                f"Symbol: {trade['symbol']}, "
                                f"Side: {trade['side']}, "
                                f"Entry: {trade['entry_price']:.2f}, "
                                f"Current: {trade['current_price']:.2f}, "
                                f"P&L: {trade['pnl']:.2f}"
                            )
                    
                    # Log de histórico
                    if status['trade_history']:
                        logger.info("\n=== Últimos Trades ===")
                        for trade in status['trade_history'][-5:]:
                            logger.info(
                                f"Symbol: {trade['symbol']}, "
                                f"Side: {trade['side']}, "
                                f"Entry: {trade['entry_price']:.2f}, "
                                f"Exit: {trade['exit_price']:.2f}, "
                                f"P&L: {trade['pnl']:.2f}"
                            )
                    
                    logger.info("\n" + "="*50)
                
                await asyncio.sleep(60)  # Atualiza a cada minuto
                
            except Exception as e:
                logger.error(f"Erro no monitoramento: {e}")
                await asyncio.sleep(5)
    
    async def run(self):
        """Executa o sistema de trading."""
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
            
            # Validar configuração
            if not await self._validate_config(config):
                logger.error("Configuração inválida")
                return
            
            # Verificar saldo
            if not await self._check_balance(config):
                logger.error("Saldo insuficiente")
                return
            
            # Verificar permissões da API
            if not await self._check_api_permissions(config):
                logger.error("Permissões da API insuficientes")
                return
            
            # Inicializar sistema
            self.trading_system = TradingSystem(config)
            self.running = True
            
            # Iniciar monitoramento
            monitor_task = asyncio.create_task(self._monitor_trading())
            
            # Iniciar sistema
            logger.info("Iniciando sistema de trading...")
            await self.trading_system.start()
            
            # Aguardar shutdown
            while self.running:
                await asyncio.sleep(1)
            
            # Parar sistema
            logger.info("Parando sistema de trading...")
            await self.trading_system.stop()
            
            # Cancelar monitoramento
            monitor_task.cancel()
            
            logger.info("Sistema de trading encerrado")
            
        except Exception as e:
            logger.error(f"Erro ao executar sistema: {e}")
            if self.trading_system:
                await self.trading_system.stop()

def main():
    """Função principal."""
    try:
        runner = TradingRunner()
        asyncio.run(runner.run())
    except KeyboardInterrupt:
        logger.info("Encerrando por solicitação do usuário")
    except Exception as e:
        logger.error(f"Erro fatal: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 