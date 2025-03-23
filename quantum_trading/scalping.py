"""
Sistema de Scalping
"""

import logging
import asyncio
from typing import Dict, Optional
from datetime import datetime, timedelta

from .core.trading.trading_system import TradingSystem
from .core.trading.market_analysis import MarketAnalysis
from .core.trading.order_executor import OrderExecutor
from .core.trading.risk_manager import RiskManager

logger = logging.getLogger(__name__)

class ScalpingSystem:
    """Sistema de Scalping"""
    
    def __init__(self, config: Dict):
        """
        Inicializa sistema.
        
        Args:
            config: Configuração.
        """
        self.config = config
        self.trading_system = TradingSystem(config)
        self.market_analysis = MarketAnalysis(config)
        self.order_executor = OrderExecutor(config)
        self.risk_manager = RiskManager(config)
        
        # Estado
        self.is_running = False
        self.current_position = None
        self.last_trade_time = None
        
    async def run(self) -> None:
        """Executa sistema"""
        try:
            logger.info("Iniciando sistema de scalping")
            self.is_running = True
            
            # Inicializa componentes
            await self.trading_system.start()
            await self.market_analysis.initialize()
            await self.order_executor.connect()
            await self.risk_manager.initialize()
            
            # Loop principal
            while self.is_running:
                try:
                    # Atualiza estado
                    await self._update()
                    
                    # Verifica oportunidades
                    if not self.current_position:
                        await self._check_entry()
                    else:
                        await self._check_exit()
                        
                    # Aguarda próximo ciclo
                    await asyncio.sleep(1)  # 1 segundo
                    
                except Exception as e:
                    logger.error(f"Erro no ciclo principal: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Erro ao executar sistema: {str(e)}")
            raise
            
        finally:
            # Para sistema
            await self.stop()
            
    async def stop(self) -> None:
        """Para sistema"""
        try:
            logger.info("Parando sistema de scalping")
            self.is_running = False
            
            # Fecha posição aberta
            if self.current_position:
                await self._close_position("stop")
                
            # Para componentes
            await self.trading_system.stop()
            await self.order_executor.disconnect()
            
            logger.info("Sistema de scalping parado")
            
        except Exception as e:
            logger.error(f"Erro ao parar sistema: {str(e)}")
            raise
            
    async def _update(self) -> None:
        """Atualiza estado"""
        try:
            # Atualiza componentes
            await self.trading_system.update()
            await self.market_analysis.update()
            await self.risk_manager.update()
            
            # Verifica limites de risco
            if not self.risk_manager.check_risk_limits():
                logger.warning("Limites de risco atingidos")
                await self.stop()
                return
                
            # Atualiza posição atual
            if self.current_position:
                await self._update_position()
                
        except Exception as e:
            logger.error(f"Erro ao atualizar estado: {str(e)}")
            raise
            
    async def _check_entry(self) -> None:
        """Verifica entrada"""
        try:
            symbol = self.config['trading']['symbol']
            
            # Verifica volume
            if not await self.market_analysis.check_volume_profile(symbol):
                return
                
            # Verifica spread
            if not await self.market_analysis.check_order_book(symbol):
                return
                
            # Analisa micro-movimentos
            signal = await self.market_analysis.check_micro_movements(symbol)
            if not signal:
                return
                
            profit_potential, direction = signal
            
            # Verifica potencial mínimo
            min_profit = self.config['scalping']['min_profit_threshold']
            if profit_potential < min_profit:
                return
                
            # Calcula tamanho
            size = self.risk_manager.calculate_position_size(symbol, direction)
            if size <= 0:
                return
                
            # Cria ordem
            order = {
                'id': f"scalp_{datetime.now().timestamp()}",
                'symbol': symbol,
                'type': 'market',
                'side': 'buy' if direction == 'long' else 'sell',
                'size': size
            }
            
            # Executa ordem
            success = await self.order_executor.execute_order(order)
            
            if success:
                # Registra posição
                self.current_position = {
                    'id': order['id'],
                    'symbol': symbol,
                    'direction': direction,
                    'size': size,
                    'entry_time': datetime.now(),
                    'entry_price': await self.trading_system.data_loader.get_current_price(symbol),
                    'costs': order.get('costs', 0)
                }
                
                self.last_trade_time = datetime.now()
                
                logger.info(f"Posição aberta: {order['id']}")
                
        except Exception as e:
            logger.error(f"Erro ao verificar entrada: {str(e)}")
            
    async def _check_exit(self) -> None:
        """Verifica saída"""
        try:
            if not self.current_position:
                return
                
            # Verifica tempo máximo
            max_time = timedelta(seconds=self.config['scalping']['max_position_time'])
            if datetime.now() - self.current_position['entry_time'] >= max_time:
                await self._close_position("timeout")
                return
                
            # Obtém preço atual
            current_price = await self.trading_system.data_loader.get_current_price(
                self.current_position['symbol']
            )
            
            if current_price is None:
                return
                
            # Calcula P&L
            pnl = self.risk_manager.calculate_pnl(
                self.current_position,
                current_price
            )
            
            # Verifica stop loss
            max_loss = self.config['scalping']['max_loss_threshold']
            if pnl <= -max_loss:
                await self._close_position("stop_loss")
                return
                
            # Verifica take profit
            min_profit = self.config['scalping']['min_profit_threshold']
            if pnl >= min_profit:
                await self._close_position("take_profit")
                return
                
        except Exception as e:
            logger.error(f"Erro ao verificar saída: {str(e)}")
            
    async def _update_position(self) -> None:
        """Atualiza posição"""
        try:
            if not self.current_position:
                return
                
            # Obtém preço atual
            current_price = await self.trading_system.data_loader.get_current_price(
                self.current_position['symbol']
            )
            
            if current_price is None:
                return
                
            # Atualiza P&L
            pnl = self.risk_manager.calculate_pnl(
                self.current_position,
                current_price
            )
            
            # Atualiza métricas
            self.current_position['current_price'] = current_price
            self.current_position['pnl'] = pnl
            self.current_position['duration'] = (
                datetime.now() - self.current_position['entry_time']
            ).total_seconds()
            
        except Exception as e:
            logger.error(f"Erro ao atualizar posição: {str(e)}")
            
    async def _close_position(self, reason: str) -> None:
        """
        Fecha posição.
        
        Args:
            reason: Motivo do fechamento.
        """
        try:
            if not self.current_position:
                return
                
            # Fecha posição
            success = await self.order_executor.close_position(
                self.current_position
            )
            
            if success:
                # Registra trade
                self.trading_system.trade_history.append({
                    'entry_time': self.current_position['entry_time'],
                    'exit_time': datetime.now(),
                    'symbol': self.current_position['symbol'],
                    'direction': self.current_position['direction'],
                    'entry_price': self.current_position['entry_price'],
                    'exit_price': self.current_position['current_price'],
                    'size': self.current_position['size'],
                    'pnl': self.current_position['pnl'],
                    'duration': self.current_position['duration'],
                    'reason': reason
                })
                
                # Limpa posição
                self.current_position = None
                
                logger.info(f"Posição fechada: {reason}")
                
        except Exception as e:
            logger.error(f"Erro ao fechar posição: {str(e)}") 