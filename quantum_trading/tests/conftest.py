"""
Configurações adicionais do pytest.
"""

import pytest
import os
import json
from pathlib import Path

@pytest.fixture(scope="session")
def test_data_dir():
    """Retorna o diretório de dados de teste."""
    return Path(__file__).parent / "data"

@pytest.fixture(scope="session")
def test_config_dir():
    """Retorna o diretório de configurações de teste."""
    return Path(__file__).parent / "config"

@pytest.fixture(scope="session")
def test_pool_config():
    """Retorna a configuração de pools de teste."""
    config = {
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
    return config

@pytest.fixture(scope="session")
def test_mining_config():
    """Retorna a configuração de mineração de teste."""
    config = {
        "wallet_address": "test-wallet-address",
        "miner_type": "xmr-stak",
        "threads": 4,
        "intensity": 100,
        "priority": 2,
        "cpu_threads": 4,
        "gpu_threads": 2,
        "gpu_platform": 0,
        "gpu_device": 0,
        "auto_optimize": True,
        "optimization_interval": 300.0,
        "min_hashrate": 0.0,
        "max_power": 100.0,
        "max_temperature": 80.0,
        "pool_config_file": "pools.json",
        "min_pool_latency": 1.0,
        "pool_switch_interval": 300.0
    }
    return config

@pytest.fixture(scope="session")
def test_pool_config_file(test_config_dir, test_pool_config):
    """Cria um arquivo de configuração de pools para testes."""
    # Cria diretório se não existir
    test_config_dir.mkdir(parents=True, exist_ok=True)
    
    # Caminho do arquivo
    filepath = test_config_dir / "pools.json"
    
    # Salva configuração
    with open(filepath, "w") as f:
        json.dump(test_pool_config, f, indent=4)
    
    return filepath

@pytest.fixture(scope="session")
def test_mining_config_file(test_config_dir, test_mining_config):
    """Cria um arquivo de configuração de mineração para testes."""
    # Cria diretório se não existir
    test_config_dir.mkdir(parents=True, exist_ok=True)
    
    # Caminho do arquivo
    filepath = test_config_dir / "mining_config.json"
    
    # Salva configuração
    with open(filepath, "w") as f:
        json.dump(test_mining_config, f, indent=4)
    
    return filepath

@pytest.fixture(scope="session")
def test_logs_dir():
    """Retorna o diretório de logs de teste."""
    return Path(__file__).parent / "logs"

@pytest.fixture(scope="session")
def test_metrics_dir():
    """Retorna o diretório de métricas de teste."""
    return Path(__file__).parent / "metrics"

@pytest.fixture(scope="session")
def test_temp_dir():
    """Retorna o diretório temporário para testes."""
    return Path(__file__).parent / "temp"

@pytest.fixture(scope="session", autouse=True)
def setup_test_dirs(test_data_dir, test_config_dir, test_logs_dir, test_metrics_dir, test_temp_dir):
    """Configura diretórios de teste."""
    # Cria diretórios
    for dir_path in [test_data_dir, test_config_dir, test_logs_dir, test_metrics_dir, test_temp_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    yield
    
    # Limpa diretórios temporários
    for dir_path in [test_logs_dir, test_metrics_dir, test_temp_dir]:
        for file_path in dir_path.glob("*"):
            try:
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir():
                    import shutil
                    shutil.rmtree(file_path)
            except Exception:
                pass 