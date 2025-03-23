"""
Testes para módulo de integração
=====================

Testes unitários para o sistema de integração do QUALIA.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from quantum_trading.integration import (
    ExchangeConnector,
    APIConnector,
    DatabaseConnector,
    MessageBroker,
    IntegrationManager
)
from quantum_trading.exceptions import IntegrationError

@pytest.fixture
def exchange_config():
    """Fixture com configuração de exchange."""
    return {
        "name": "binance",
        "testnet": True,
        "api_key": "test_key",
        "api_secret": "test_secret",
        "timeout": 30000,
        "enableRateLimit": True
    }

@pytest.fixture
def api_config():
    """Fixture com configuração de API."""
    return {
        "base_url": "https://api.test.com",
        "version": "v1",
        "timeout": 5000,
        "retry_attempts": 3,
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer test_token"
        }
    }

@pytest.fixture
def database_config():
    """Fixture com configuração de banco de dados."""
    return {
        "host": "localhost",
        "port": 5432,
        "database": "qualia_test",
        "user": "test_user",
        "password": "test_pass",
        "pool_size": 5
    }

def test_exchange_connector(exchange_config):
    """Testa conector de exchange."""
    connector = ExchangeConnector()
    
    # Configura conector
    connector.configure(exchange_config)
    
    # Conecta exchange
    connector.connect()
    assert connector.is_connected()
    
    # Busca mercados
    markets = connector.fetch_markets()
    assert isinstance(markets, dict)
    assert "BTC/USDT" in markets
    
    # Busca saldo
    balance = connector.fetch_balance()
    assert isinstance(balance, dict)
    assert "total" in balance
    
    # Cria ordem
    order = connector.create_order(
        symbol="BTC/USDT",
        type="limit",
        side="buy",
        amount=0.1,
        price=45000
    )
    assert "id" in order
    
    # Cancela ordem
    canceled = connector.cancel_order(order["id"], "BTC/USDT")
    assert canceled["id"] == order["id"]

def test_api_connector(api_config):
    """Testa conector de API."""
    connector = APIConnector()
    
    # Configura conector
    connector.configure(api_config)
    
    # Faz requisição GET
    response = connector.get("/data")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
    
    # Faz requisição POST
    data = {"key": "value"}
    response = connector.post("/data", data=data)
    assert response.status_code == 201
    
    # Testa retry
    with patch("requests.get") as mock_get:
        mock_get.side_effect = [
            Exception("Connection error"),
            Exception("Timeout"),
            {"data": "success"}
        ]
        
        response = connector.get("/data")
        assert response["data"] == "success"
        assert mock_get.call_count == 3

def test_database_connector(database_config):
    """Testa conector de banco de dados."""
    connector = DatabaseConnector()
    
    # Configura conector
    connector.configure(database_config)
    
    # Conecta banco
    connector.connect()
    assert connector.is_connected()
    
    # Cria tabela
    connector.execute("""
        CREATE TABLE IF NOT EXISTS test_table (
            id SERIAL PRIMARY KEY,
            name VARCHAR(50),
            value FLOAT
        )
    """)
    
    # Insere dados
    connector.execute(
        "INSERT INTO test_table (name, value) VALUES (%s, %s)",
        ["test", 123.45]
    )
    
    # Busca dados
    results = connector.query(
        "SELECT * FROM test_table WHERE name = %s",
        ["test"]
    )
    assert len(results) > 0
    assert results[0]["value"] == 123.45
    
    # Limpa tabela
    connector.execute("DROP TABLE test_table")

def test_message_broker():
    """Testa broker de mensagens."""
    broker = MessageBroker()
    
    # Configura broker
    broker.configure(
        host="localhost",
        port=5672,
        username="guest",
        password="guest"
    )
    
    # Conecta broker
    broker.connect()
    assert broker.is_connected()
    
    # Define callback
    messages = []
    def callback(message):
        messages.append(message)
    
    # Subscreve tópico
    broker.subscribe("test_topic", callback)
    
    # Publica mensagem
    broker.publish("test_topic", {"data": "test"})
    
    # Verifica mensagem
    assert len(messages) > 0
    assert messages[0]["data"] == "test"
    
    # Cancela subscrição
    broker.unsubscribe("test_topic")

def test_integration_manager(exchange_config, api_config, database_config):
    """Testa gerenciador de integração."""
    manager = IntegrationManager()
    
    # Registra conectores
    manager.register_connector("exchange", ExchangeConnector())
    manager.register_connector("api", APIConnector())
    manager.register_connector("database", DatabaseConnector())
    
    # Configura conectores
    manager.configure_connector("exchange", exchange_config)
    manager.configure_connector("api", api_config)
    manager.configure_connector("database", database_config)
    
    # Conecta todos
    manager.connect_all()
    assert manager.all_connected()
    
    # Desconecta todos
    manager.disconnect_all()
    assert not manager.all_connected()

def test_exchange_integration(exchange_config):
    """Testa integração com exchange."""
    connector = ExchangeConnector()
    connector.configure(exchange_config)
    
    # Busca dados OHLCV
    ohlcv = connector.fetch_ohlcv(
        symbol="BTC/USDT",
        timeframe="1h",
        limit=100
    )
    assert isinstance(ohlcv, list)
    assert len(ohlcv) == 100
    
    # Busca orderbook
    orderbook = connector.fetch_order_book(
        symbol="BTC/USDT",
        limit=20
    )
    assert "bids" in orderbook
    assert "asks" in orderbook
    
    # Busca trades
    trades = connector.fetch_trades(
        symbol="BTC/USDT",
        limit=50
    )
    assert isinstance(trades, list)
    assert len(trades) == 50

def test_api_integration(api_config):
    """Testa integração com API."""
    connector = APIConnector()
    connector.configure(api_config)
    
    # Testa autenticação
    auth = connector.authenticate()
    assert auth["authenticated"]
    
    # Testa rate limiting
    for _ in range(10):
        response = connector.get("/data")
        assert response.status_code != 429
    
    # Testa websocket
    messages = []
    def on_message(msg):
        messages.append(msg)
    
    connector.ws_connect("/stream", on_message)
    assert connector.ws_connected()
    
    connector.ws_disconnect()
    assert not connector.ws_connected()

def test_database_integration(database_config):
    """Testa integração com banco de dados."""
    connector = DatabaseConnector()
    connector.configure(database_config)
    
    # Testa transação
    with connector.transaction():
        connector.execute(
            "INSERT INTO test_table (name, value) VALUES (%s, %s)",
            ["test1", 1]
        )
        connector.execute(
            "INSERT INTO test_table (name, value) VALUES (%s, %s)",
            ["test2", 2]
        )
    
    # Testa pool de conexões
    results = []
    def worker():
        result = connector.query(
            "SELECT COUNT(*) FROM test_table"
        )
        results.append(result[0]["count"])
    
    threads = [Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    assert len(results) == 5
    assert all(r == results[0] for r in results)

def test_message_broker_integration():
    """Testa integração com message broker."""
    broker = MessageBroker()
    
    # Testa diferentes exchanges
    broker.declare_exchange("events", type="topic")
    broker.declare_exchange("data", type="fanout")
    
    # Testa filas
    broker.declare_queue("events_queue", exchange="events")
    broker.declare_queue("data_queue", exchange="data")
    
    # Testa routing
    messages = []
    def callback(message):
        messages.append(message)
    
    broker.bind_queue("events_queue", "events", "trade.#")
    broker.consume("events_queue", callback)
    
    broker.publish("events", "trade.new", {"symbol": "BTC/USDT"})
    assert len(messages) > 0

def test_integration_error_handling():
    """Testa tratamento de erros de integração."""
    manager = IntegrationManager()
    
    # Testa erro de conexão
    with pytest.raises(IntegrationError):
        manager.connect_connector("invalid")
    
    # Testa erro de configuração
    with pytest.raises(IntegrationError):
        manager.configure_connector("invalid", {})
    
    # Testa erro de operação
    connector = Mock()
    connector.operation = Mock(side_effect=Exception("Test error"))
    
    manager.register_connector("test", connector)
    
    with pytest.raises(IntegrationError):
        manager.execute_operation("test", "operation") 