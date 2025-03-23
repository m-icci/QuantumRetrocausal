"""
Testes para módulo de comunicação
=====================

Testes unitários para o sistema de comunicação do QUALIA.
"""

import pytest
import json
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from quantum_trading.communication import (
    MessageBroker,
    WebSocketClient,
    RESTClient,
    EventEmitter,
    CommunicationManager
)
from quantum_trading.exceptions import CommunicationError

@pytest.fixture
def communication_config():
    """Fixture com configuração de comunicação."""
    return {
        "websocket": {
            "url": "wss://api.exchange.com/ws",
            "ping_interval": 30,
            "reconnect_delay": 5
        },
        "rest": {
            "base_url": "https://api.exchange.com",
            "timeout": 30,
            "retry_attempts": 3
        },
        "broker": {
            "host": "localhost",
            "port": 5672,
            "username": "guest",
            "password": "guest"
        },
        "events": {
            "max_listeners": 10,
            "buffer_size": 1000
        }
    }

@pytest.fixture
def sample_message():
    """Fixture com mensagem de exemplo."""
    return {
        "type": "trade",
        "symbol": "BTC/USDT",
        "price": 50000.0,
        "amount": 0.1,
        "side": "buy",
        "timestamp": datetime.now().isoformat()
    }

@pytest.mark.asyncio
async def test_websocket_client(communication_config):
    """Testa cliente WebSocket."""
    client = WebSocketClient(config=communication_config)
    
    # Mock para conexão WebSocket
    mock_ws = Mock()
    mock_ws.send = Mock()
    mock_ws.recv = Mock(return_value=json.dumps({"type": "ping"}))
    
    with patch("websockets.connect", return_value=mock_ws):
        # Conecta
        await client.connect()
        
        # Verifica conexão
        assert client.is_connected()
        
        # Envia mensagem
        await client.send({
            "type": "subscribe",
            "channel": "trades",
            "symbol": "BTC/USDT"
        })
        
        # Recebe mensagem
        message = await client.receive()
        
        # Verifica mensagem
        assert message["type"] == "ping"
        
        # Desconecta
        await client.disconnect()
        assert not client.is_connected()

def test_rest_client(communication_config):
    """Testa cliente REST."""
    client = RESTClient(config=communication_config)
    
    # Mock para requisições HTTP
    mock_response = Mock()
    mock_response.json = Mock(return_value={"status": "success"})
    mock_response.status_code = 200
    
    with patch("requests.get", return_value=mock_response):
        # Faz requisição GET
        response = client.get("/api/v1/trades")
        
        # Verifica resposta
        assert response["status"] == "success"
    
    with patch("requests.post", return_value=mock_response):
        # Faz requisição POST
        response = client.post("/api/v1/orders", {
            "symbol": "BTC/USDT",
            "side": "buy",
            "amount": 0.1
        })
        
        # Verifica resposta
        assert response["status"] == "success"

def test_message_broker(communication_config):
    """Testa broker de mensagens."""
    broker = MessageBroker(config=communication_config)
    
    # Conecta ao broker
    broker.connect()
    
    # Cria canal e fila
    channel = broker.create_channel("trades")
    queue = broker.create_queue("BTC_trades")
    
    # Publica mensagem
    message = {
        "type": "trade",
        "symbol": "BTC/USDT",
        "price": 50000.0
    }
    
    broker.publish(channel, message)
    
    # Consome mensagem
    received = broker.consume(queue)
    
    # Verifica mensagem
    assert received["type"] == "trade"
    assert received["symbol"] == "BTC/USDT"
    
    # Desconecta
    broker.disconnect()

def test_event_emitter(communication_config):
    """Testa emissor de eventos."""
    emitter = EventEmitter(config=communication_config)
    
    # Mock para handler
    mock_handler = Mock()
    
    # Registra handler
    emitter.on("trade", mock_handler)
    
    # Emite evento
    event_data = {
        "symbol": "BTC/USDT",
        "price": 50000.0
    }
    
    emitter.emit("trade", event_data)
    
    # Verifica handler
    mock_handler.assert_called_once_with(event_data)
    
    # Remove handler
    emitter.off("trade", mock_handler)
    
    # Emite novamente
    emitter.emit("trade", event_data)
    
    # Verifica que não foi chamado novamente
    assert mock_handler.call_count == 1

def test_communication_manager(communication_config, sample_message):
    """Testa gerenciador de comunicação."""
    manager = CommunicationManager(config=communication_config)
    
    # Inicia gerenciador
    manager.start()
    
    # Registra handlers
    trade_handler = Mock()
    manager.register_handler("trade", trade_handler)
    
    # Processa mensagem
    manager.process_message(sample_message)
    
    # Verifica handler
    trade_handler.assert_called_once_with(sample_message)
    
    # Para gerenciador
    manager.stop()

@pytest.mark.asyncio
async def test_websocket_reconnection(communication_config):
    """Testa reconexão WebSocket."""
    client = WebSocketClient(config=communication_config)
    
    # Mock para conexão com erro
    mock_ws = Mock()
    mock_ws.connect = Mock(side_effect=[Exception(), None])
    
    with patch("websockets.connect", mock_ws):
        # Tenta conectar
        await client.connect()
        
        # Verifica reconexão
        assert mock_ws.connect.call_count == 2
        assert client.is_connected()

def test_message_validation(communication_config):
    """Testa validação de mensagens."""
    manager = CommunicationManager(config=communication_config)
    
    # Mensagem válida
    valid_message = {
        "type": "trade",
        "symbol": "BTC/USDT",
        "price": 50000.0,
        "amount": 0.1
    }
    
    assert manager.validate_message(valid_message)
    
    # Mensagem inválida
    invalid_message = {
        "type": "trade",
        "symbol": "BTC/USDT"
        # Faltando campos obrigatórios
    }
    
    assert not manager.validate_message(invalid_message)

def test_message_routing(communication_config):
    """Testa roteamento de mensagens."""
    manager = CommunicationManager(config=communication_config)
    
    # Handlers para diferentes tipos
    trade_handler = Mock()
    order_handler = Mock()
    
    manager.register_handler("trade", trade_handler)
    manager.register_handler("order", order_handler)
    
    # Mensagens
    trade_message = {"type": "trade", "data": "trade_data"}
    order_message = {"type": "order", "data": "order_data"}
    
    # Processa mensagens
    manager.process_message(trade_message)
    manager.process_message(order_message)
    
    # Verifica roteamento
    trade_handler.assert_called_once_with(trade_message)
    order_handler.assert_called_once_with(order_message)

def test_message_persistence(communication_config, tmp_path):
    """Testa persistência de mensagens."""
    manager = CommunicationManager(config=communication_config)
    
    # Mensagens para salvar
    messages = [
        {"type": "trade", "id": 1},
        {"type": "order", "id": 2},
        {"type": "trade", "id": 3}
    ]
    
    # Salva mensagens
    save_path = tmp_path / "messages.json"
    manager.save_messages(messages, save_path)
    
    # Carrega mensagens
    loaded = manager.load_messages(save_path)
    
    # Verifica dados
    assert len(loaded) == len(messages)
    assert all(
        loaded[i]["type"] == messages[i]["type"]
        for i in range(len(messages))
    )

def test_rate_limiting(communication_config):
    """Testa limitação de taxa."""
    client = RESTClient(config=communication_config)
    
    # Configura rate limit
    client.set_rate_limit(
        requests_per_second=2,
        burst_size=5
    )
    
    # Mock para requisições
    mock_response = Mock()
    mock_response.json = Mock(return_value={"status": "success"})
    mock_response.status_code = 200
    
    with patch("requests.get", return_value=mock_response):
        # Faz várias requisições
        start_time = datetime.now()
        
        for _ in range(5):
            client.get("/api/v1/trades")
        
        end_time = datetime.now()
        
        # Verifica que levou pelo menos 2 segundos
        assert (end_time - start_time).total_seconds() >= 2 