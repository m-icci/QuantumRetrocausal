"""
Testes para o gerenciador de pools.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
import json
import os

from quantum_trading.core.mining.pool_manager import PoolManager

@pytest.fixture
def pool_manager():
    """Cria um gerenciador de pools para testes."""
    return PoolManager()

@pytest.fixture
def mock_pool_config():
    """Cria uma configuração de pools mock para testes."""
    return {
        "pools": [
            {
                "name": "TestPool1",
                "url": "test1.pool.com",
                "port": 3333,
                "region": "test",
                "min_payout": 0.1,
                "fee": 0.001
            },
            {
                "name": "TestPool2",
                "url": "test2.pool.com",
                "port": 3333,
                "region": "test",
                "min_payout": 0.1,
                "fee": 0.001
            }
        ]
    }

def test_init(pool_manager):
    """Testa inicialização do gerenciador."""
    assert pool_manager.pools is not None
    assert pool_manager.metrics is not None
    assert len(pool_manager.pools) > 0

def test_load_default_pools(pool_manager):
    """Testa carregamento das pools padrão."""
    # Limpa pools existentes
    pool_manager.pools = []
    
    # Carrega pools padrão
    pool_manager._load_default_pools()
    
    # Verifica se pools foram carregadas
    assert len(pool_manager.pools) > 0
    
    # Verifica estrutura das pools
    for pool in pool_manager.pools:
        assert "name" in pool
        assert "url" in pool
        assert "port" in pool
        assert "region" in pool
        assert "min_payout" in pool
        assert "fee" in pool

@pytest.mark.asyncio
async def test_get_pool_status(pool_manager):
    """Testa obtenção do status de uma pool."""
    # Mock da resposta da pool
    mock_response = {
        "status": "online",
        "hashrate": 1000.0,
        "workers": 10,
        "last_block": 123456
    }
    
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_get.return_value.__aenter__.return_value.json = Mock(return_value=mock_response)
        
        # Obtém status
        status = await pool_manager.get_pool_status(pool_manager.pools[0])
        
        # Verifica resultado
        assert status["status"] == "online"
        assert status["hashrate"] == 1000.0
        assert status["workers"] == 10
        assert status["last_block"] == 123456
        assert "latency" in status

@pytest.mark.asyncio
async def test_select_best_pool(pool_manager):
    """Testa seleção da melhor pool."""
    # Mock das respostas das pools
    mock_responses = [
        {"status": "online", "latency": 0.1},
        {"status": "online", "latency": 0.2},
        {"status": "offline", "latency": 1.0}
    ]
    
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_get.return_value.__aenter__.return_value.json = Mock(side_effect=mock_responses)
        
        # Seleciona melhor pool
        best_pool = await pool_manager.select_best_pool()
        
        # Verifica resultado
        assert best_pool is not None
        assert best_pool["status"] == "online"
        assert best_pool["latency"] == 0.1

def test_get_pool_metrics(pool_manager):
    """Testa obtenção das métricas das pools."""
    # Adiciona métricas mock
    pool_manager.metrics = {
        "pool1": {"status": "online", "latency": 0.1},
        "pool2": {"status": "offline", "latency": 1.0}
    }
    
    # Obtém métricas
    metrics = pool_manager.get_pool_metrics()
    
    # Verifica resultado
    assert metrics == pool_manager.metrics

def test_add_pool(pool_manager):
    """Testa adição de uma nova pool."""
    # Nova pool
    new_pool = {
        "name": "NewPool",
        "url": "new.pool.com",
        "port": 3333,
        "region": "test",
        "min_payout": 0.1,
        "fee": 0.001
    }
    
    # Adiciona pool
    pool_manager.add_pool(new_pool)
    
    # Verifica se pool foi adicionada
    assert len(pool_manager.pools) == len(pool_manager.pools) + 1
    assert new_pool in pool_manager.pools

def test_remove_pool(pool_manager):
    """Testa remoção de uma pool."""
    # Pool para remover
    pool_to_remove = pool_manager.pools[0]
    
    # Remove pool
    pool_manager.remove_pool(pool_to_remove["name"])
    
    # Verifica se pool foi removida
    assert pool_to_remove not in pool_manager.pools

def test_get_pool_config(pool_manager):
    """Testa obtenção da configuração de uma pool."""
    # Pool existente
    existing_pool = pool_manager.pools[0]
    
    # Obtém configuração
    config = pool_manager.get_pool_config(existing_pool["name"])
    
    # Verifica resultado
    assert config == existing_pool

def test_save_pools(pool_manager, tmp_path):
    """Testa salvamento das pools em arquivo."""
    # Caminho do arquivo
    filepath = tmp_path / "test_pools.json"
    
    # Salva pools
    pool_manager.save_pools(str(filepath))
    
    # Verifica se arquivo foi criado
    assert filepath.exists()
    
    # Verifica conteúdo
    with open(filepath) as f:
        saved_data = json.load(f)
        assert "pools" in saved_data
        assert len(saved_data["pools"]) == len(pool_manager.pools)

def test_load_pools(pool_manager, mock_pool_config, tmp_path):
    """Testa carregamento das pools de arquivo."""
    # Caminho do arquivo
    filepath = tmp_path / "test_pools.json"
    
    # Salva configuração mock
    with open(filepath, "w") as f:
        json.dump(mock_pool_config, f)
    
    # Carrega pools
    pool_manager.load_pools(str(filepath))
    
    # Verifica se pools foram carregadas
    assert len(pool_manager.pools) == len(mock_pool_config["pools"])
    
    # Verifica conteúdo
    for pool in pool_manager.pools:
        assert pool["name"] in [p["name"] for p in mock_pool_config["pools"]]

def test_load_pools_error(pool_manager, tmp_path):
    """Testa carregamento de pools com erro."""
    # Caminho do arquivo inválido
    filepath = tmp_path / "invalid.json"
    
    # Tenta carregar pools
    pool_manager.load_pools(str(filepath))
    
    # Verifica se pools padrão foram carregadas
    assert len(pool_manager.pools) > 0 