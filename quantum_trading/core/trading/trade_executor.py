"""
Execução de trades.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

from .trading_config import TradingConfig
from .exchange_integration import ExchangeIntegration
from .market_analysis import MarketAnalysis
from .risk_manager import RiskManager

class TradeExecutor:
    """Executor de trades."""
    
    def __init__(
        self,
        config: TradingConfig,
        exchange: ExchangeIntegration,
        analysis: MarketAnalysis,
        risk_manager: RiskManager
    ):
        """
        Inicializa executor.
        
        Args:
            config: Configuração.
            exchange: Integração com exchange.
            analysis: Análise de mercado.
            risk_manager: Gerenciador de risco.
        """
        self.logger = logging.getLogger('TradeExecutor')
        
        # Componentes
        self.config = config
        self.exchange = exchange
        self.analysis = analysis
        self.risk_manager = risk_manager
        
        # Estado
        self._active_trades: Dict[str, Dict[str, Any]] = {}
        self._pending_orders: Dict[str, Dict[str, Any]] = {}
        self._trade_history: List[Dict[str, Any]] = []
        
        # Controle
        self._running = False
        self._update_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Inicia executor."""
        try:
            self.logger.info("Iniciando executor")
            
            # Inicia análise
            await self.analysis.start()
            
            # Inicia loop
            self._running = True
            self._update_task = asyncio.create_task(self._update_loop())
            
        except Exception as e:
            self.logger.error(f"Erro ao iniciar executor: {str(e)}")
            raise
    
    async def stop(self) -> None:
        """Para executor."""
        try:
            self.logger.info("Parando executor")
            
            # Para loop
            self._running = False
            if self._update_task:
                await self._update_task
            
            # Para análise
            await self.analysis.stop()
            
            # Fecha posições
            await self._close_all_positions()
            
        except Exception as e:
            self.logger.error(f"Erro ao parar executor: {str(e)}")
            raise
    
    async def _update_loop(self) -> None:
        """Loop de atualização."""
        try:
            while self._running:
                # Atualiza posições
                await self._update_positions()
                
                # Atualiza ordens
                await self._update_orders()
                
                # Executa trades
                await self._execute_trades()
                
                # Aguarda próximo ciclo
                await asyncio.sleep(1)
                
        except Exception as e:
            self.logger.error(f"Erro no loop de atualização: {str(e)}")
            raise
    
    async def _update_positions(self) -> None:
        """Atualiza posições."""
        try:
            # Obtém posições ativas
            positions = self.risk_manager.get_positions()
            
            # Atualiza cada posição
            for position_id, position in positions.items():
                # Obtém preço atual
                ticker = self.exchange.get_ticker()
                if not ticker:
                    continue
                
                current_price = ticker['last']
                
                # Verifica stop loss
                if position['side'] == 'buy' and current_price <= position['stop_loss']:
                    await self._close_position(position_id, current_price)
                    continue
                
                if position['side'] == 'sell' and current_price >= position['stop_loss']:
                    await self._close_position(position_id, current_price)
                    continue
                
                # Verifica take profit
                if position['side'] == 'buy' and current_price >= position['take_profit']:
                    await self._close_position(position_id, current_price)
                    continue
                
                if position['side'] == 'sell' and current_price <= position['take_profit']:
                    await self._close_position(position_id, current_price)
                    continue
                
        except Exception as e:
            self.logger.error(f"Erro ao atualizar posições: {str(e)}")
    
    async def _update_orders(self) -> None:
        """Atualiza ordens."""
        try:
            # Obtém ordens pendentes
            orders = self.exchange.get_orders()
            if not orders:
                return
            
            # Atualiza cada ordem
            for order in orders:
                order_id = order['id']
                
                # Verifica se é ordem pendente
                if order['status'] != 'open':
                    if order_id in self._pending_orders:
                        del self._pending_orders[order_id]
                    continue
                
                # Atualiza ordem
                self._pending_orders[order_id] = order
                
        except Exception as e:
            self.logger.error(f"Erro ao atualizar ordens: {str(e)}")
    
    async def _execute_trades(self) -> None:
        """Executa trades."""
        try:
            # Obtém sinais
            signals = self.analysis.get_signals()
            if not signals:
                return
            
            # Filtra sinais
            filtered_signals = [
                signal for signal in signals
                if self.risk_manager.validate_trade(signal)
            ]
            
            # Executa cada sinal
            for signal in filtered_signals:
                # Prepara ordem
                order = self._prepare_order(signal)
                if not order:
                    continue
                
                # Executa ordem
                executed_order = await self._execute_order(order)
                if not executed_order:
                    continue
                
                # Atualiza posição
                position = self.risk_manager.update_position(signal)
                if not position:
                    continue
                
                # Adiciona trade ativo
                self._active_trades[position['id']] = {
                    'position': position,
                    'order': executed_order
                }
                
                # Coloca stop loss e take profit
                await self._place_stop_loss(position)
                await self._place_take_profit(position)
                
        except Exception as e:
            self.logger.error(f"Erro ao executar trades: {str(e)}")
    
    def _prepare_order(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Prepara ordem.
        
        Args:
            signal: Sinal.
            
        Returns:
            Ordem.
        """
        try:
            # Calcula tamanho
            size = self.risk_manager._calculate_position_size(signal['confidence'])
            if size <= 0:
                return None
            
            # Prepara ordem
            order = {
                'symbol': self.config.symbol,
                'side': signal['type'],
                'type': 'market',
                'size': size
            }
            
            return order
            
        except Exception as e:
            self.logger.error(f"Erro ao preparar ordem: {str(e)}")
            return None
    
    async def _execute_order(self, order: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Executa ordem.
        
        Args:
            order: Ordem.
            
        Returns:
            Ordem executada.
        """
        try:
            # Executa ordem
            executed_order = await self.exchange.place_order(order)
            if not executed_order:
                return None
            
            return executed_order
            
        except Exception as e:
            self.logger.error(f"Erro ao executar ordem: {str(e)}")
            return None
    
    async def _place_stop_loss(self, position: Dict[str, Any]) -> None:
        """
        Coloca stop loss.
        
        Args:
            position: Posição.
        """
        try:
            # Prepara ordem
            order = {
                'symbol': self.config.symbol,
                'side': 'sell' if position['side'] == 'buy' else 'buy',
                'type': 'stop',
                'size': position['size'],
                'price': position['stop_loss']
            }
            
            # Executa ordem
            await self._execute_order(order)
            
        except Exception as e:
            self.logger.error(f"Erro ao colocar stop loss: {str(e)}")
    
    async def _place_take_profit(self, position: Dict[str, Any]) -> None:
        """
        Coloca take profit.
        
        Args:
            position: Posição.
        """
        try:
            # Prepara ordem
            order = {
                'symbol': self.config.symbol,
                'side': 'sell' if position['side'] == 'buy' else 'buy',
                'type': 'limit',
                'size': position['size'],
                'price': position['take_profit']
            }
            
            # Executa ordem
            await self._execute_order(order)
            
        except Exception as e:
            self.logger.error(f"Erro ao colocar take profit: {str(e)}")
    
    async def _close_position(self, position_id: str, price: float) -> None:
        """
        Fecha posição.
        
        Args:
            position_id: ID da posição.
            price: Preço.
        """
        try:
            # Verifica trade ativo
            if position_id not in self._active_trades:
                return
            
            # Obtém trade
            trade = self._active_trades[position_id]
            
            # Cancela ordens pendentes
            await self._cancel_pending_orders(position_id)
            
            # Prepara ordem de fechamento
            order = {
                'symbol': self.config.symbol,
                'side': 'sell' if trade['position']['side'] == 'buy' else 'buy',
                'type': 'market',
                'size': trade['position']['size']
            }
            
            # Executa ordem
            executed_order = await self._execute_order(order)
            if not executed_order:
                return
            
            # Fecha posição
            closed_position = self.risk_manager.close_position(position_id, price)
            if not closed_position:
                return
            
            # Adiciona ao histórico
            self._trade_history.append({
                'position': closed_position,
                'order': executed_order
            })
            
            # Remove trade ativo
            del self._active_trades[position_id]
            
        except Exception as e:
            self.logger.error(f"Erro ao fechar posição: {str(e)}")
    
    async def _cancel_pending_orders(self, position_id: str) -> None:
        """
        Cancela ordens pendentes.
        
        Args:
            position_id: ID da posição.
        """
        try:
            # Cancela cada ordem pendente
            for order_id, order in self._pending_orders.items():
                if order['symbol'] == self.config.symbol:
                    await self.exchange.cancel_order(order_id)
            
        except Exception as e:
            self.logger.error(f"Erro ao cancelar ordens pendentes: {str(e)}")
    
    async def _close_all_positions(self) -> None:
        """Fecha todas as posições."""
        try:
            # Obtém preço atual
            ticker = self.exchange.get_ticker()
            if not ticker:
                return
            
            current_price = ticker['last']
            
            # Fecha cada posição
            for position_id in list(self._active_trades.keys()):
                await self._close_position(position_id, current_price)
            
        except Exception as e:
            self.logger.error(f"Erro ao fechar todas as posições: {str(e)}")
    
    def get_active_trades(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtém trades ativos.
        
        Returns:
            Trades ativos.
        """
        return self._active_trades
    
    def get_trade_history(self) -> List[Dict[str, Any]]:
        """
        Obtém histórico de trades.
        
        Returns:
            Histórico de trades.
        """
        return self._trade_history 