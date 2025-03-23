"""
Real-time market data feed using Kraken's websocket API
"""
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, Callable, Optional
import websockets
from core.logging.quantum_logger import quantum_logger

class KrakenWebsocket:
    def __init__(self):
        self.ws_url = "wss://ws.kraken.com"
        self.subscribers: Dict[str, Callable] = {}
        self.running = False
        self.last_update: Optional[datetime] = None
        
    async def connect(self):
        """Estabelece conexão websocket"""
        try:
            async with websockets.connect(self.ws_url) as websocket:
                self.running = True
                quantum_logger.info("Conexão websocket estabelecida")
                
                # Subscribe to ticker
                subscribe_message = {
                    "event": "subscribe",
                    "pair": ["XBT/USD"],
                    "subscription": {"name": "ticker"}
                }
                await websocket.send(json.dumps(subscribe_message))
                
                while self.running:
                    try:
                        message = await websocket.recv()
                        data = json.loads(message)
                        
                        # Processa apenas mensagens de ticker
                        if isinstance(data, list) and len(data) > 2:
                            self.last_update = datetime.now()
                            ticker_data = {
                                'pair': data[3],
                                'price': float(data[1]['c'][0]),
                                'volume': float(data[1]['v'][0]),
                                'timestamp': self.last_update
                            }
                            
                            # Notifica subscribers
                            for callback in self.subscribers.values():
                                callback(ticker_data)
                                
                    except Exception as e:
                        quantum_logger.error(
                            "Erro processando mensagem websocket",
                            {"error": str(e)}
                        )
                        
        except Exception as e:
            quantum_logger.error(
                "Erro na conexão websocket",
                {"error": str(e)}
            )
            self.running = False
            
    def subscribe(self, name: str, callback: Callable):
        """Adiciona subscriber para receber atualizações"""
        self.subscribers[name] = callback
        quantum_logger.debug(f"Novo subscriber adicionado: {name}")
        
    def unsubscribe(self, name: str):
        """Remove subscriber"""
        if name in self.subscribers:
            del self.subscribers[name]
            quantum_logger.debug(f"Subscriber removido: {name}")
            
    def stop(self):
        """Para o websocket"""
        self.running = False
        quantum_logger.info("Websocket desconectado")
