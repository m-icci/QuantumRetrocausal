"""
Testes para módulo de dados
=====================

Testes unitários para o sistema de dados do QUALIA.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from quantum_trading.data import (
    DataLoader,
    DataProcessor,
    DataValidator,
    DataTransformer,
    DataPipeline
)
from quantum_trading.exceptions import DataError

@pytest.fixture
def market_data():
    """Fixture com dados de mercado."""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="1H")
    np.random.seed(42)
    
    return pd.DataFrame({
        "timestamp": dates,
        "open": np.random.normal(100, 10, len(dates)),
        "high": np.random.normal(105, 10, len(dates)),
        "low": np.random.normal(95, 10, len(dates)),
        "close": np.random.normal(100, 10, len(dates)),
        "volume": np.random.normal(1000, 100, len(dates))
    }).set_index("timestamp")

@pytest.fixture
def trade_data():
    """Fixture com dados de trades."""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="1min")
    np.random.seed(42)
    
    return pd.DataFrame({
        "timestamp": dates,
        "price": np.random.normal(100, 5, len(dates)),
        "amount": np.random.exponential(1, len(dates)),
        "side": np.random.choice(["buy", "sell"], len(dates)),
        "type": np.random.choice(["market", "limit"], len(dates))
    }).set_index("timestamp")

def test_data_loader():
    """Testa carregador de dados."""
    loader = DataLoader()
    
    # Configura loader
    loader.configure(
        exchange="binance",
        symbols=["BTC/USDT", "ETH/USDT"],
        timeframes=["1m", "5m", "1h"]
    )
    
    # Carrega dados
    data = loader.load_data(
        symbol="BTC/USDT",
        timeframe="1h",
        start="2023-01-01",
        end="2023-12-31"
    )
    
    # Verifica dados
    assert isinstance(data, pd.DataFrame)
    assert "open" in data.columns
    assert "high" in data.columns
    assert "low" in data.columns
    assert "close" in data.columns
    assert "volume" in data.columns
    
    # Carrega trades
    trades = loader.load_trades(
        symbol="BTC/USDT",
        start="2023-01-01",
        end="2023-01-02"
    )
    
    # Verifica trades
    assert isinstance(trades, pd.DataFrame)
    assert "price" in trades.columns
    assert "amount" in trades.columns

def test_data_processor(market_data):
    """Testa processador de dados."""
    processor = DataProcessor()
    
    # Remove outliers
    cleaned = processor.remove_outliers(
        market_data,
        columns=["close", "volume"],
        method="zscore",
        threshold=3
    )
    assert len(cleaned) <= len(market_data)
    
    # Preenche missing
    filled = processor.fill_missing(
        market_data,
        method="ffill"
    )
    assert not filled.isnull().any().any()
    
    # Normaliza dados
    normalized = processor.normalize_data(
        market_data,
        columns=["close", "volume"],
        method="minmax"
    )
    assert (normalized <= 1).all().all()
    assert (normalized >= 0).all().all()

def test_data_validator(market_data, trade_data):
    """Testa validador de dados."""
    validator = DataValidator()
    
    # Valida estrutura
    assert validator.validate_structure(market_data)
    assert validator.validate_structure(trade_data)
    
    # Valida tipos
    assert validator.validate_types(market_data)
    assert validator.validate_types(trade_data)
    
    # Valida valores
    assert validator.validate_values(market_data)
    assert validator.validate_values(trade_data)
    
    # Testa dados inválidos
    invalid_data = market_data.copy()
    invalid_data.loc[invalid_data.index[0], "close"] = "invalid"
    
    with pytest.raises(DataError):
        validator.validate_types(invalid_data)

def test_data_transformer(market_data):
    """Testa transformador de dados."""
    transformer = DataTransformer()
    
    # Calcula retornos
    returns = transformer.calculate_returns(
        market_data["close"],
        method="log"
    )
    assert len(returns) == len(market_data)
    
    # Calcula volatilidade
    volatility = transformer.calculate_volatility(
        market_data["close"],
        window=20
    )
    assert len(volatility) == len(market_data)
    
    # Calcula momentum
    momentum = transformer.calculate_momentum(
        market_data["close"],
        window=10
    )
    assert len(momentum) == len(market_data)

def test_data_pipeline(market_data):
    """Testa pipeline de dados."""
    pipeline = DataPipeline()
    
    # Adiciona etapas
    pipeline.add_step("validate", DataValidator())
    pipeline.add_step("process", DataProcessor())
    pipeline.add_step("transform", DataTransformer())
    
    # Configura pipeline
    pipeline.configure(
        input_columns=["close", "volume"],
        output_columns=["returns", "volatility", "momentum"]
    )
    
    # Executa pipeline
    result = pipeline.run(market_data)
    
    # Verifica resultado
    assert "returns" in result.columns
    assert "volatility" in result.columns
    assert "momentum" in result.columns

def test_data_aggregation(trade_data):
    """Testa agregação de dados."""
    processor = DataProcessor()
    
    # Agrega por minuto
    minute_data = processor.aggregate_trades(
        trade_data,
        freq="1min",
        price_column="price",
        volume_column="amount"
    )
    
    # Verifica agregação
    assert isinstance(minute_data, pd.DataFrame)
    assert "open" in minute_data.columns
    assert "high" in minute_data.columns
    assert "low" in minute_data.columns
    assert "close" in minute_data.columns
    assert "volume" in minute_data.columns

def test_data_sampling(market_data):
    """Testa amostragem de dados."""
    processor = DataProcessor()
    
    # Reamostra dados
    resampled = processor.resample_data(
        market_data,
        freq="4H",
        method="ohlcv"
    )
    
    # Verifica amostragem
    assert len(resampled) < len(market_data)
    assert "open" in resampled.columns
    assert "high" in resampled.columns
    assert "low" in resampled.columns
    assert "close" in resampled.columns
    assert "volume" in resampled.columns

def test_data_filtering(market_data):
    """Testa filtragem de dados."""
    processor = DataProcessor()
    
    # Filtra por data
    filtered = processor.filter_by_date(
        market_data,
        start="2023-06-01",
        end="2023-06-30"
    )
    
    # Verifica filtragem
    assert len(filtered) < len(market_data)
    assert filtered.index.min().strftime("%Y-%m-%d") >= "2023-06-01"
    assert filtered.index.max().strftime("%Y-%m-%d") <= "2023-06-30"

def test_data_persistence(market_data):
    """Testa persistência de dados."""
    loader = DataLoader()
    
    # Salva dados
    loader.save_data(
        market_data,
        filename="test_data.parquet",
        format="parquet"
    )
    
    # Carrega dados
    loaded = loader.load_from_file("test_data.parquet")
    
    # Verifica dados
    assert loaded.equals(market_data)
    
    # Remove arquivo
    loader.remove_file("test_data.parquet")

def test_data_streaming():
    """Testa streaming de dados."""
    loader = DataLoader()
    
    # Configura stream
    loader.configure_stream(
        exchange="binance",
        symbols=["BTC/USDT"],
        channels=["trades", "ticker"]
    )
    
    # Inicia stream
    stream = loader.start_stream()
    
    # Processa alguns dados
    data = []
    for i, msg in enumerate(stream):
        data.append(msg)
        if i >= 10:
            break
    
    # Para stream
    loader.stop_stream()
    
    # Verifica dados
    assert len(data) > 0
    assert all(isinstance(d, dict) for d in data) 