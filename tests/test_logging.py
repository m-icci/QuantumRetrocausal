"""
Testes para módulo de logging
=====================

Testes unitários para o sistema de logging do QUALIA.
"""

import pytest
import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from quantum_trading.logging import (
    QuantumLogger,
    LogFormatter,
    LogHandler,
    LogAnalyzer,
    LoggingManager
)
from quantum_trading.exceptions import LoggingError

@pytest.fixture
def logging_config():
    """Fixture com configuração de logging."""
    return {
        "logger": {
            "name": "qualia",
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "handlers": ["console", "file", "database"]
        },
        "handlers": {
            "console": {
                "level": "INFO",
                "format": "simple"
            },
            "file": {
                "level": "DEBUG",
                "format": "detailed",
                "filename": "qualia.log",
                "max_bytes": 10485760,  # 10MB
                "backup_count": 5
            },
            "database": {
                "level": "WARNING",
                "format": "json",
                "connection": "sqlite:///logs.db",
                "table": "system_logs"
            }
        },
        "formatters": {
            "simple": {
                "format": "%(levelname)s - %(message)s"
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(context)s"
            },
            "json": {
                "format": "json",
                "fields": ["timestamp", "level", "message", "context"]
            }
        }
    }

@pytest.fixture
def sample_logs():
    """Fixture com logs de exemplo."""
    np.random.seed(42)
    dates = pd.date_range(
        start=datetime.now() - timedelta(hours=1),
        end=datetime.now(),
        freq="1min"
    )
    
    levels = ["INFO", "WARNING", "ERROR", "CRITICAL"]
    components = ["system", "trading", "analysis", "execution"]
    
    return pd.DataFrame({
        "timestamp": dates,
        "level": np.random.choice(levels, len(dates)),
        "component": np.random.choice(components, len(dates)),
        "message": [f"Log message {i}" for i in range(len(dates))],
        "context": [{"data": f"context_{i}"} for i in range(len(dates))]
    })

def test_quantum_logger(logging_config):
    """Testa logger quântico."""
    logger = QuantumLogger(config=logging_config)
    
    # Configura logger
    logger.setup()
    
    # Testa níveis de log
    logger.info("Test info message")
    logger.warning("Test warning message")
    logger.error("Test error message")
    
    # Verifica handlers
    assert len(logger.handlers) == len(logging_config["logger"]["handlers"])
    assert all(isinstance(h, logging.Handler) for h in logger.handlers)
    
    # Testa contexto
    context = {"operation": "test", "status": "success"}
    logger.info("Test with context", extra={"context": context})

def test_log_formatter(logging_config):
    """Testa formatador de logs."""
    formatter = LogFormatter(config=logging_config)
    
    # Cria record de log
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="Test message",
        args=(),
        exc_info=None
    )
    
    # Formata log
    formatted = formatter.format(record)
    
    # Verifica formato
    assert isinstance(formatted, str)
    assert "Test message" in formatted
    
    # Testa formato JSON
    json_formatter = LogFormatter(config=logging_config, format_type="json")
    json_formatted = json_formatter.format(record)
    
    # Verifica JSON
    parsed = json.loads(json_formatted)
    assert "message" in parsed
    assert "level" in parsed
    assert "timestamp" in parsed

def test_log_handler(logging_config, tmp_path):
    """Testa handler de logs."""
    handler = LogHandler(config=logging_config)
    
    # Configura handler de arquivo
    log_file = tmp_path / "test.log"
    handler.setup_file_handler(str(log_file))
    
    # Envia logs
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="Test message",
        args=(),
        exc_info=None
    )
    
    handler.emit(record)
    
    # Verifica arquivo
    assert os.path.exists(log_file)
    with open(log_file) as f:
        content = f.read()
        assert "Test message" in content

def test_log_analyzer(logging_config, sample_logs):
    """Testa analisador de logs."""
    analyzer = LogAnalyzer(config=logging_config)
    
    # Analisa logs
    analysis = analyzer.analyze_logs(sample_logs)
    
    # Verifica análise
    assert "error_rate" in analysis
    assert "component_stats" in analysis
    assert "level_distribution" in analysis
    
    # Testa filtros
    errors = analyzer.filter_logs(
        sample_logs,
        level="ERROR",
        component="system"
    )
    
    assert all(log["level"] == "ERROR" for log in errors)
    assert all(log["component"] == "system" for log in errors)

def test_logging_manager(logging_config):
    """Testa gerenciador de logging."""
    manager = LoggingManager(config=logging_config)
    
    # Registra handlers
    handlers = {
        "console": Mock(),
        "file": Mock(),
        "database": Mock()
    }
    
    for name, handler in handlers.items():
        manager.register_handler(name, handler)
    
    # Envia log
    manager.log(
        level="INFO",
        message="Test message",
        component="test",
        context={"data": "test"}
    )
    
    # Verifica handlers
    assert all(h.handle.called for h in handlers.values())

def test_log_rotation(logging_config, tmp_path):
    """Testa rotação de logs."""
    handler = LogHandler(config=logging_config)
    
    # Configura rotação
    log_file = tmp_path / "rotating.log"
    handler.setup_rotating_handler(
        filename=str(log_file),
        max_bytes=1024,
        backup_count=3
    )
    
    # Gera logs até rotação
    for i in range(1000):
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg=f"Test message {i}",
            args=(),
            exc_info=None
        )
        handler.emit(record)
    
    # Verifica arquivos de backup
    assert os.path.exists(log_file)
    assert os.path.exists(str(log_file) + ".1")

def test_log_filtering(logging_config, sample_logs):
    """Testa filtragem de logs."""
    analyzer = LogAnalyzer(config=logging_config)
    
    # Define filtros
    filters = {
        "level": ["ERROR", "CRITICAL"],
        "component": "system",
        "start_time": datetime.now() - timedelta(minutes=30),
        "end_time": datetime.now()
    }
    
    # Aplica filtros
    filtered = analyzer.apply_filters(sample_logs, filters)
    
    # Verifica filtros
    assert all(log["level"] in filters["level"] for log in filtered)
    assert all(log["component"] == filters["component"] for log in filtered)
    assert all(filters["start_time"] <= log["timestamp"] <= filters["end_time"] for log in filtered)

def test_log_persistence(logging_config, sample_logs):
    """Testa persistência de logs."""
    manager = LoggingManager(config=logging_config)
    
    # Salva logs
    manager.save_logs(sample_logs, "test_logs")
    
    # Carrega logs
    loaded = manager.load_logs("test_logs")
    
    # Verifica logs
    assert isinstance(loaded, pd.DataFrame)
    assert all(loaded.equals(sample_logs))

def test_log_aggregation(logging_config, sample_logs):
    """Testa agregação de logs."""
    analyzer = LogAnalyzer(config=logging_config)
    
    # Agrega logs
    aggregated = analyzer.aggregate_logs(
        sample_logs,
        group_by=["level", "component"],
        metrics=["count", "first_occurrence", "last_occurrence"]
    )
    
    # Verifica agregação
    assert isinstance(aggregated, pd.DataFrame)
    assert "count" in aggregated.columns
    assert "first_occurrence" in aggregated.columns
    assert "last_occurrence" in aggregated.columns

def test_logging_validation(logging_config):
    """Testa validação de logging."""
    manager = LoggingManager(config=logging_config)
    
    # Configuração válida
    assert manager.validate_config(logging_config)
    
    # Configuração inválida
    invalid_config = logging_config.copy()
    del invalid_config["logger"]["handlers"]
    
    with pytest.raises(LoggingError):
        manager.validate_config(invalid_config)
