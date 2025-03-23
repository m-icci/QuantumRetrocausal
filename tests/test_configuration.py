"""
Testes para módulo de configuração
=====================

Testes unitários para o sistema de configuração do QUALIA.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import os
import json
import yaml

from quantum_trading.configuration import (
    ConfigManager,
    ConfigValidator,
    ConfigLoader,
    ConfigParser,
    EnvironmentManager
)
from quantum_trading.exceptions import ConfigurationError

@pytest.fixture
def config_data():
    """Fixture com dados de configuração."""
    return {
        "system": {
            "name": "qualia",
            "version": "1.0.0",
            "mode": "production",
            "debug": False
        },
        "trading": {
            "pairs": ["BTC/USDT", "ETH/USDT"],
            "timeframes": ["1m", "5m", "15m", "1h"],
            "max_positions": 10,
            "risk_limits": {
                "max_drawdown": 0.1,
                "max_leverage": 3,
                "stop_loss": 0.05
            }
        },
        "analysis": {
            "indicators": ["sma", "ema", "rsi", "macd"],
            "window_sizes": [20, 50, 100],
            "optimization": {
                "method": "quantum",
                "iterations": 1000,
                "population": 100
            }
        },
        "execution": {
            "exchange": {
                "name": "binance",
                "testnet": True,
                "api_key": "${BINANCE_API_KEY}",
                "api_secret": "${BINANCE_API_SECRET}"
            },
            "order_types": ["market", "limit"],
            "max_slippage": 0.001
        },
        "monitoring": {
            "metrics": ["performance", "risk", "execution"],
            "alerts": {
                "email": "alerts@qualia.ai",
                "telegram": "@qualia_alerts"
            },
            "logging": {
                "level": "INFO",
                "handlers": ["console", "file"]
            }
        }
    }

@pytest.fixture
def env_vars():
    """Fixture com variáveis de ambiente."""
    return {
        "BINANCE_API_KEY": "test_key",
        "BINANCE_API_SECRET": "test_secret",
        "QUALIA_MODE": "production",
        "QUALIA_DEBUG": "false"
    }

def test_config_manager(config_data):
    """Testa gerenciador de configuração."""
    manager = ConfigManager()
    
    # Carrega configuração
    manager.load_config(config_data)
    
    # Obtém configuração
    config = manager.get_config()
    assert config == config_data
    
    # Obtém valor específico
    trading_pairs = manager.get("trading.pairs")
    assert isinstance(trading_pairs, list)
    assert "BTC/USDT" in trading_pairs
    
    # Atualiza configuração
    manager.set("trading.max_positions", 20)
    assert manager.get("trading.max_positions") == 20
    
    # Valida configuração
    assert manager.validate_config()

def test_config_validator(config_data):
    """Testa validador de configuração."""
    validator = ConfigValidator()
    
    # Valida configuração
    assert validator.validate(config_data)
    
    # Testa configuração inválida
    invalid_config = config_data.copy()
    del invalid_config["system"]["name"]
    
    with pytest.raises(ConfigurationError):
        validator.validate(invalid_config)
    
    # Valida tipos
    assert validator.validate_type(config_data["trading"]["pairs"], list)
    assert validator.validate_type(config_data["trading"]["max_positions"], int)
    assert validator.validate_type(config_data["trading"]["risk_limits"], dict)

def test_config_loader():
    """Testa carregador de configuração."""
    loader = ConfigLoader()
    
    # Salva configuração em arquivo
    config_file = "test_config.yaml"
    loader.save_config(config_data, config_file)
    
    # Carrega configuração
    loaded_config = loader.load_config(config_file)
    assert loaded_config == config_data
    
    # Testa diferentes formatos
    loader.save_config(config_data, "test_config.json")
    json_config = loader.load_config("test_config.json")
    assert json_config == config_data
    
    # Limpa arquivos
    os.remove("test_config.yaml")
    os.remove("test_config.json")

def test_config_parser(config_data, env_vars):
    """Testa parser de configuração."""
    parser = ConfigParser()
    
    # Configura variáveis de ambiente
    with patch.dict(os.environ, env_vars):
        # Parse configuração
        parsed = parser.parse_config(config_data)
        
        # Verifica substituição de variáveis
        assert parsed["execution"]["exchange"]["api_key"] == "test_key"
        assert parsed["execution"]["exchange"]["api_secret"] == "test_secret"
        
        # Verifica valores padrão
        assert parsed["system"]["mode"] == "production"
        assert not parsed["system"]["debug"]

def test_environment_manager(env_vars):
    """Testa gerenciador de ambiente."""
    manager = EnvironmentManager()
    
    # Carrega variáveis
    with patch.dict(os.environ, env_vars):
        manager.load_environment()
        
        # Verifica variáveis
        assert manager.get("BINANCE_API_KEY") == "test_key"
        assert manager.get("QUALIA_MODE") == "production"
        
        # Testa valor padrão
        assert manager.get("UNKNOWN_VAR", default="default") == "default"
        
        # Valida variáveis
        assert manager.validate_required_vars(["BINANCE_API_KEY", "BINANCE_API_SECRET"])

def test_config_inheritance(config_data):
    """Testa herança de configuração."""
    manager = ConfigManager()
    
    # Configuração base
    base_config = {
        "system": {
            "name": "qualia_base",
            "version": "1.0.0"
        },
        "trading": {
            "pairs": ["BTC/USDT"],
            "timeframes": ["1m", "5m"]
        }
    }
    
    # Herda configuração
    merged = manager.merge_configs(base_config, config_data)
    
    # Verifica herança
    assert merged["system"]["name"] == "qualia"  # Sobrescrito
    assert merged["trading"]["pairs"] == ["BTC/USDT", "ETH/USDT"]  # Estendido
    assert "analysis" in merged  # Adicionado

def test_config_validation_rules(config_data):
    """Testa regras de validação."""
    validator = ConfigValidator()
    
    # Adiciona regras
    validator.add_rule(
        "trading.max_positions",
        lambda x: isinstance(x, int) and x > 0
    )
    
    validator.add_rule(
        "trading.risk_limits.max_drawdown",
        lambda x: isinstance(x, float) and 0 < x < 1
    )
    
    # Valida regras
    assert validator.validate_rules(config_data)
    
    # Testa violação de regra
    invalid_config = config_data.copy()
    invalid_config["trading"]["max_positions"] = -1
    
    with pytest.raises(ConfigurationError):
        validator.validate_rules(invalid_config)

def test_config_persistence(config_data):
    """Testa persistência de configuração."""
    manager = ConfigManager()
    
    # Salva configuração
    manager.save_config(config_data, "test_config")
    
    # Carrega configuração
    loaded = manager.load_config("test_config")
    
    # Verifica configuração
    assert loaded == config_data
    
    # Remove arquivo
    manager.remove_config("test_config")
    assert not os.path.exists("test_config")

def test_config_encryption(config_data):
    """Testa encriptação de configuração."""
    manager = ConfigManager()
    
    # Encripta configuração
    encrypted = manager.encrypt_config(
        config_data,
        password="test_password"
    )
    
    # Decripta configuração
    decrypted = manager.decrypt_config(
        encrypted,
        password="test_password"
    )
    
    # Verifica configuração
    assert decrypted == config_data
    
    # Testa senha inválida
    with pytest.raises(ConfigurationError):
        manager.decrypt_config(encrypted, password="wrong_password")

def test_config_versioning(config_data):
    """Testa versionamento de configuração."""
    manager = ConfigManager()
    
    # Salva versão
    manager.save_version(config_data, "v1.0")
    
    # Modifica configuração
    new_config = config_data.copy()
    new_config["system"]["version"] = "1.1.0"
    manager.save_version(new_config, "v1.1")
    
    # Carrega versões
    v1_0 = manager.load_version("v1.0")
    v1_1 = manager.load_version("v1.1")
    
    # Verifica versões
    assert v1_0["system"]["version"] == "1.0.0"
    assert v1_1["system"]["version"] == "1.1.0" 