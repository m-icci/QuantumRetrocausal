"""
PWA Integration for Quantum Trading.

This module provides the integration layer between the PWA frontend
and the quantum trading backend services.
"""

from typing import Dict, List, Optional, Any
import asyncio
import logging
from datetime import datetime
import json

from .quantum_trading_service import QuantumTradingService
from utils.types.trading_types import MarketQuantumState

logger = logging.getLogger(__name__)

class QuantumTradingPWA:
    """PWA integration for quantum trading functionality."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize PWA integration.
        
        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
        self.trading_service = QuantumTradingService(config)
        self.websocket_clients: List = []
        
    async def handle_websocket_connection(self, websocket) -> None:
        """
        Handle new WebSocket connection.
        
        Args:
            websocket: WebSocket connection object
        """
        self.websocket_clients.append(websocket)
        try:
            async for message in websocket:
                await self._process_websocket_message(websocket, message)
        finally:
            self.websocket_clients.remove(websocket)
            
    async def _process_websocket_message(
        self,
        websocket,
        message: str
    ) -> None:
        """
        Process incoming WebSocket message.
        
        Args:
            websocket: WebSocket connection
            message: Message content
        """
        try:
            data = json.loads(message)
            command = data.get('command')
            
            if command == 'analyze_market':
                result = await self.trading_service.analyze_market(
                    data['market_data'],
                    data.get('strategy_params')
                )
                await self._send_analysis_update(websocket, result)
                
            elif command == 'execute_trade':
                result = await self.trading_service.execute_trade(
                    data['symbol'],
                    data['side'],
                    data['amount'],
                    data.get('params')
                )
                await self._send_trade_update(websocket, result)
                
            elif command == 'monitor_trade':
                result = await self.trading_service.monitor_trade(
                    data['trade_id']
                )
                await self._send_monitoring_update(websocket, result)
                
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            await websocket.send(json.dumps({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }))
            
    async def _send_analysis_update(
        self,
        websocket,
        analysis_result: Dict[str, Any]
    ) -> None:
        """
        Send market analysis update to client.
        
        Args:
            websocket: WebSocket connection
            analysis_result: Analysis results to send
        """
        quantum_state = await self.trading_service.get_quantum_state()
        message = {
            'type': 'analysis_update',
            'data': {
                'consciousness_metrics': analysis_result['consciousness_metrics'],
                'market_awareness': analysis_result['market_awareness'],
                'trading_signals': analysis_result['trading_signals'],
                'quantum_state': {
                    'vector': quantum_state.to_dict(),
                    'coherence': quantum_state.get_coherence(),
                    'entanglement': quantum_state.get_entanglement()
                },
                'timestamp': datetime.now().isoformat()
            }
        }
        await websocket.send(json.dumps(message))
        
    async def _send_trade_update(
        self,
        websocket,
        trade_result: Dict[str, Any]
    ) -> None:
        """
        Send trade execution update to client.
        
        Args:
            websocket: WebSocket connection
            trade_result: Trade execution results
        """
        quantum_state = await self.trading_service.get_trade_quantum_state(
            trade_result['trade_id']
        )
        message = {
            'type': 'trade_update',
            'data': {
                'trade_id': trade_result['trade_id'],
                'execution_result': trade_result['execution_result'],
                'consciousness_metrics': trade_result['consciousness_metrics'],
                'quantum_state': {
                    'vector': quantum_state.to_dict(),
                    'coherence': quantum_state.get_coherence(),
                    'entanglement': quantum_state.get_entanglement()
                },
                'timestamp': datetime.now().isoformat()
            }
        }
        await websocket.send(json.dumps(message))
        
    async def _send_monitoring_update(
        self,
        websocket,
        monitoring_result: Dict[str, Any]
    ) -> None:
        """
        Send trade monitoring update to client.
        
        Args:
            websocket: WebSocket connection
            monitoring_result: Monitoring results
        """
        quantum_state = await self.trading_service.get_trade_quantum_state(
            monitoring_result['trade_id']
        )
        message = {
            'type': 'monitoring_update',
            'data': {
                'trade_id': monitoring_result['trade_id'],
                'state_evolution': monitoring_result['state_evolution'],
                'signals': monitoring_result['signals'],
                'quantum_state': {
                    'vector': quantum_state.to_dict(),
                    'coherence': quantum_state.get_coherence(),
                    'entanglement': quantum_state.get_entanglement()
                },
                'timestamp': datetime.now().isoformat()
            }
        }
        await websocket.send(json.dumps(message))
        
    async def broadcast_market_update(
        self,
        market_data: Dict[str, Any]
    ) -> None:
        """
        Broadcast market update to all connected clients.
        
        Args:
            market_data: Market data to broadcast
        """
        analysis_result = await self.trading_service.analyze_market(market_data)
        message = {
            'type': 'market_update',
            'data': {
                'market_data': market_data,
                'analysis': {
                    'consciousness_metrics': analysis_result['consciousness_metrics'],
                    'trading_signals': analysis_result['trading_signals']
                },
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Broadcast to all connected clients
        for client in self.websocket_clients:
            try:
                await client.send(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to client: {str(e)}")
                
    async def start_market_monitoring(
        self,
        symbols: List[str],
        interval: float = 1.0
    ) -> None:
        """
        Start continuous market monitoring.
        
        Args:
            symbols: List of symbols to monitor
            interval: Update interval in seconds
        """
        while True:
            for symbol in symbols:
                try:
                    market_data = await self.trading_service._fetch_market_data(symbol)
                    await self.broadcast_market_update(market_data)
                except Exception as e:
                    logger.error(f"Error monitoring {symbol}: {str(e)}")
                    
            await asyncio.sleep(interval)
