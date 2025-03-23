"""
Rede P2P Quântica
Implementa comunicação entre nós Qualia via entrelaçamento quântico
"""

import numpy as np
import asyncio
import websockets
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from ..metaspace.quantum_dance import QuantumDanceNetwork
from ..bitwise.qualia_bitwise import GeometricConstants
from datetime import datetime
import logging

@dataclass
class QuantumMessage:
    """Mensagem quântica entre nós"""
    state: np.ndarray  # Estado quântico
    timestamp: float   # Tempo do envio
    source_id: str    # ID do nó origem
    target_id: str    # ID do nó destino
    retrocausal: bool # Se é mensagem retrocausal

    def to_json(self) -> str:
        """Converte para JSON."""
        data = {
            'node_id': self.source_id,
            'message_type': 'QUANTUM_STATE',
            'content': {
                'state': self.state.tolist(),
                'timestamp': self.timestamp,
                'source_id': self.source_id,
                'target_id': self.target_id,
                'retrocausal': self.retrocausal
            },
            'timestamp': self.timestamp.isoformat()
        }
        return json.dumps(data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'QuantumMessage':
        """Cria uma mensagem a partir de JSON."""
        data = json.loads(json_str)
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(
            state=np.array(data['content']['state']),
            timestamp=data['content']['timestamp'],
            source_id=data['content']['source_id'],
            target_id=data['content']['target_id'],
            retrocausal=data['content']['retrocausal']
        )

class QuantumNode:
    """Nó quântico na rede P2P"""
    
    def __init__(
        self,
        node_id: str,
        host: str = "localhost",
        port: int = 8765,
        dimensions: int = 64
    ):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.dimensions = dimensions
        
        # Rede quântica local
        self.quantum_network = QuantumDanceNetwork(
            dimensions=dimensions,
            realities=4,
            layers=3,
            memory_limit=1000
        )
        
        # Estado de entrelaçamento
        self.entangled_state = np.random.random(dimensions)
        self.geometry = GeometricConstants()
        
        # Conexões com outros nós
        self.peers: Dict[str, websockets.WebSocketServerProtocol] = {}
        
        # Histórico de mensagens retrocausais
        self.retrocausal_history: List[QuantumMessage] = []
        
        self.logger = logging.getLogger(__name__)
        self.messages: List[QuantumMessage] = []
        self.connected_nodes: Dict[str, Tuple[str, int]] = {}
        
        self.logger.info(f"QuantumNode {node_id} initialized at {host}:{port}")
    
    async def start(self):
        """Inicia servidor websocket"""
        self.logger.info(f"Starting node {self.node_id}")
        await self._start_server()
    
    async def _start_server(self) -> None:
        """Inicia o servidor."""
        server = await websockets.serve(
            self.handle_connection,
            self.host,
            self.port
        )
        await server.wait_closed()
    
    async def connect_to_peer(self, peer_host: str, peer_port: int):
        """Conecta a outro nó"""
        uri = f"ws://{peer_host}:{peer_port}"
        async with websockets.connect(uri) as websocket:
            # Envia ID
            await websocket.send(json.dumps({
                "type": "handshake",
                "node_id": self.node_id
            }))
            
            # Recebe resposta
            response = await websocket.recv()
            data = json.loads(response)
            peer_id = data["node_id"]
            
            # Armazena conexão
            self.peers[peer_id] = websocket
            
            # Inicia entrelaçamento
            await self._entangle_with_peer(peer_id)
    
    async def _entangle_with_peer(self, peer_id: str):
        """Estabelece entrelaçamento com par"""
        # Gera estado de Bell
        bell_state = np.zeros((2, self.dimensions))
        bell_state[0] = self.entangled_state
        bell_state[1] = np.roll(self.entangled_state, 1)
        
        # Normaliza
        bell_state = bell_state / np.sqrt(2)
        
        # Envia parte do estado
        await self.peers[peer_id].send(json.dumps({
            "type": "entangle",
            "state": bell_state[1].tolist()
        }))
    
    async def send_quantum_message(
        self,
        target_id: str,
        message: np.ndarray,
        retrocausal: bool = False
    ):
        """Envia mensagem quântica para par"""
        # Aplica entrelaçamento
        entangled_msg = self.quantum_network.qualia._entangle(
            message,
            self.entangled_state
        )
        
        # Cria mensagem
        quantum_msg = QuantumMessage(
            state=entangled_msg,
            timestamp=asyncio.get_event_loop().time(),
            source_id=self.node_id,
            target_id=target_id,
            retrocausal=retrocausal
        )
        
        # Se for retrocausal, armazena no histórico
        if retrocausal:
            self.retrocausal_history.append(quantum_msg)
        
        # Envia
        await self.peers[target_id].send(quantum_msg.to_json())
    
    def predict_future_state(self, future_time: float) -> Optional[np.ndarray]:
        """Prediz estado futuro via retrocausalidade"""
        # Filtra mensagens do futuro
        future_msgs = [
            msg for msg in self.retrocausal_history
            if msg.timestamp > future_time
        ]
        
        if not future_msgs:
            return None
        
        # Usa média ponderada pelo tempo
        weights = [1 / (msg.timestamp - future_time) for msg in future_msgs]
        total_weight = sum(weights)
        
        predicted_state = np.zeros(self.dimensions)
        for msg, weight in zip(future_msgs, weights):
            predicted_state += msg.state * (weight / total_weight)
        
        return predicted_state
    
    async def handle_connection(
        self,
        websocket: websockets.WebSocketServerProtocol,
        path: str
    ):
        """Manipula conexões websocket"""
        try:
            async for message in websocket:
                data = json.loads(message)
                
                if data["type"] == "handshake":
                    # Responde handshake
                    peer_id = data["node_id"]
                    self.peers[peer_id] = websocket
                    await websocket.send(json.dumps({
                        "type": "handshake",
                        "node_id": self.node_id
                    }))
                
                elif data["type"] == "entangle":
                    # Recebe estado de Bell
                    received_state = np.array(data["state"])
                    self.entangled_state = received_state
                
                elif data["type"] == "quantum_message":
                    # Recebe mensagem quântica
                    msg_data = data["message"]
                    quantum_msg = QuantumMessage(
                        state=np.array(msg_data["state"]),
                        timestamp=msg_data["timestamp"],
                        source_id=msg_data["source_id"],
                        target_id=msg_data["target_id"],
                        retrocausal=msg_data["retrocausal"]
                    )
                    
                    # Se for retrocausal, armazena
                    if quantum_msg.retrocausal:
                        self.retrocausal_history.append(quantum_msg)
                    
                    # Processa mensagem
                    await self._process_quantum_message(quantum_msg)
        
        except websockets.exceptions.ConnectionClosed:
            # Remove peer desconectado
            disconnected_peer = None
            for peer_id, peer_ws in self.peers.items():
                if peer_ws == websocket:
                    disconnected_peer = peer_id
                    break
            
            if disconnected_peer:
                del self.peers[disconnected_peer]
    
    async def _process_quantum_message(self, message: QuantumMessage):
        """Processa mensagem quântica recebida"""
        # Desentrelaça mensagem
        desentangled = self.quantum_network.qualia._entangle(
            message.state,
            self.entangled_state
        )
        
        # Aplica ao estado local
        self.quantum_network.qualia.state = desentangled
        
        # Evolui rede local
        self.quantum_network.dance()

    def process_message(self, message: QuantumMessage) -> None:
        """Processa mensagem recebida."""
        self.messages.append(message)
        self.logger.info(f"Message received from {message.source_id}: {message.message_type}")
        
        # Processa dependendo do tipo
        if message.message_type == "QUANTUM_STATE":
            self._process_state_message(message)
    
    def _process_state_message(self, message: QuantumMessage) -> None:
        """Processa mensagem de estado quântico."""
        # Desentrelaça mensagem
        desentangled = self.quantum_network.qualia._entangle(
            message.state,
            self.entangled_state
        )
        
        # Aplica ao estado local
        self.quantum_network.qualia.state = desentangled
        
        # Evolui rede local
        self.quantum_network.dance()

    def get_network_state(self) -> Dict[str, Any]:
        """Obtém estado da rede."""
        return {
            "node_id": self.node_id,
            "connected_nodes": list(self.connected_nodes.keys()),
            "messages_count": len(self.messages),
            "network_metrics": self.quantum_network._calculate_metrics().__dict__
        }
    
    def broadcast(self, message_type: str, content: Dict[str, Any]) -> None:
        """Envia mensagem para todos os nós conectados."""
        for node_id in self.connected_nodes:
            asyncio.create_task(
                self.send_quantum_message(node_id, content["state"], content["retrocausal"])
            )
        
        self.logger.info(f"Broadcast {message_type} to {len(self.connected_nodes)} nodes")
