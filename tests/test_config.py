"""
Testes para módulo de configuração
==============================

Testes unitários para o sistema de configuração do QUALIA.
"""

import os
import pytest
import tempfile
import json
from pathlib import Path

from quantum_trading.config import Config
from quantum_trading.exceptions import ConfigError

@pytest.fixture
def sample_config():
    """Fixture com configuração de exemplo."""
    return {
        "exchange": {
            "name": "binance",
            "api_key": "test_key",
            "api_secret": "test_secret"
        },
        "trading": {
            "pairs": ["BTC/USDT", "ETH/USDT"],
            "timeframe": "1m",
            "max_position_size": 0.1,
            "stop_loss": 0.02,
            "take_profit": 0.05
        },
        "analysis": {
            "window_size": 100,
            "consciousness_threshold": 0.7,
            "field_strength": 0.8,
            "memory_size": 1000
        },
        "logging": {
            "level": "INFO",
            "file": "qualia.log"
        }
    }

@pytest.fixture
def config_file(sample_config):
    """Fixture que cria arquivo temporário de configuração."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        json.dump(sample_config, f)
        return f.name

def test_config_initialization(config_file):
    """Testa inicialização da configuração."""
    config = Config(config_file)
    
    assert config.exchange["name"] == "binance"
    assert config.trading["pairs"] == ["BTC/USDT", "ETH/USDT"]
    assert config.analysis["window_size"] == 100
    assert config.logging["level"] == "INFO"

def test_config_validation(sample_config):
    """Testa validação da configuração."""
    # Testa configuração válida
    config = Config.validate(sample_config)
    assert config == sample_config
    
    # Testa configuração inválida - exchange
    invalid_config = sample_config.copy()
    del invalid_config["exchange"]
    with pytest.raises(ConfigError):
        Config.validate(invalid_config)
    
    # Testa configuração inválida - trading
    invalid_config = sample_config.copy()
    invalid_config["trading"]["max_position_size"] = -1
    with pytest.raises(ConfigError):
        Config.validate(invalid_config)
    
    # Testa configuração inválida - analysis
    invalid_config = sample_config.copy()
    invalid_config["analysis"]["consciousness_threshold"] = 2.0
    with pytest.raises(ConfigError):
        Config.validate(invalid_config)

def test_config_save(sample_config):
    """Testa salvamento da configuração."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config.json"
        
        # Salva configuração
        config = Config(sample_config)
        config.save(config_path)
        
        # Verifica se arquivo foi criado
        assert config_path.exists()
        
        # Carrega e verifica conteúdo
        with open(config_path) as f:
            loaded_config = json.load(f)
        assert loaded_config == sample_config

def test_config_update(sample_config):
    """Testa atualização da configuração."""
    config = Config(sample_config)
    
    # Atualiza valores
    updates = {
        "trading": {
            "max_position_size": 0.2,
            "stop_loss": 0.03
        },
        "analysis": {
            "window_size": 200
        }
    }
    
    config.update(updates)
    
    # Verifica atualizações
    assert config.trading["max_position_size"] == 0.2
    assert config.trading["stop_loss"] == 0.03
    assert config.analysis["window_size"] == 200
    
    # Verifica que outros valores permanecem inalterados
    assert config.exchange["name"] == "binance"
    assert config.trading["pairs"] == ["BTC/USDT", "ETH/USDT"]

def test_config_env_override(sample_config):
    """Testa override de configuração por variáveis de ambiente."""
    # Define variáveis de ambiente
    os.environ["QUALIA_EXCHANGE_API_KEY"] = "env_key"
    os.environ["QUALIA_TRADING_MAX_POSITION_SIZE"] = "0.5"
    os.environ["QUALIA_ANALYSIS_WINDOW_SIZE"] = "150"
    
    config = Config(sample_config)
    config.load_env_overrides()
    
    # Verifica overrides
    assert config.exchange["api_key"] == "env_key"
    assert config.trading["max_position_size"] == 0.5
    assert config.analysis["window_size"] == 150
    
    # Limpa variáveis
    del os.environ["QUALIA_EXCHANGE_API_KEY"]
    del os.environ["QUALIA_TRADING_MAX_POSITION_SIZE"]
    del os.environ["QUALIA_ANALYSIS_WINDOW_SIZE"]

def test_config_defaults():
    """Testa valores padrão da configuração."""
    minimal_config = {
        "exchange": {
            "name": "binance",
            "api_key": "test_key",
            "api_secret": "test_secret"
        }
    }
    
    config = Config(minimal_config)
    
    # Verifica valores padrão
    assert "trading" in config
    assert "analysis" in config
    assert "logging" in config
    assert config.trading["timeframe"] == "1m"  # valor padrão
    assert config.logging["level"] == "INFO"    # valor padrão

def test_config_clone(sample_config):
    """Testa clonagem da configuração."""
    config = Config(sample_config)
    clone = config.clone()
    
    # Verifica que é uma cópia profunda
    assert config == clone
    assert config is not clone
    assert config.trading is not clone.trading
    
    # Modifica clone e verifica que original não muda
    clone.trading["max_position_size"] = 0.5
    assert config.trading["max_position_size"] == 0.1 