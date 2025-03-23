"""
Testes da estratégia quântica.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from ..trading_config import TradingConfig
from ..exchange_integration import ExchangeIntegration
from ..market_analysis import MarketAnalysis
from ..risk_manager import RiskManager
from ..trade_executor import TradeExecutor
from ..quantum_strategy import QuantumStrategy

@pytest.fixture
def config():
    """Cria configuração."""
    return TradingConfig(
        exchange='binance',
        api_key='test_key',
        api_secret='test_secret',
        symbol='BTC/USDT',
        timeframe='1h',
        leverage=1,
        max_positions=3,
        daily_trades_limit=10,
        daily_loss_limit=0.02,
        min_confidence=0.7,
        position_size=0.1,
        min_position_size=0.01,
        max_position_size=0.5,
        stop_loss=0.02,
        take_profit=0.04,
        risk_per_trade=0.01,
        rsi_period=14,
        rsi_oversold=30,
        rsi_overbought=70,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        bb_period=20,
        bb_std=2,
        atr_period=14
    )

@pytest.fixture
def exchange(config):
    """Cria integração com exchange."""
    return Mock(spec=ExchangeIntegration)

@pytest.fixture
def analysis(config, exchange):
    """Cria análise de mercado."""
    return Mock(spec=MarketAnalysis)

@pytest.fixture
def risk_manager(config, exchange):
    """Cria gerenciador de risco."""
    return Mock(spec=RiskManager)

@pytest.fixture
def executor(config, exchange, analysis, risk_manager):
    """Cria executor de trades."""
    return Mock(spec=TradeExecutor)

@pytest.fixture
def strategy(config, exchange, analysis, risk_manager, executor):
    """Cria estratégia quântica."""
    return QuantumStrategy(config, exchange, analysis, risk_manager, executor)

@pytest.mark.asyncio
async def test_start(strategy):
    """Testa inicialização da estratégia."""
    # Inicia estratégia
    await strategy.start()
    
    # Verifica estado
    assert strategy._running
    assert strategy._update_task is not None

@pytest.mark.asyncio
async def test_stop(strategy):
    """Testa parada da estratégia."""
    # Inicia estratégia
    strategy._running = True
    strategy._update_task = Mock()
    
    # Para estratégia
    await strategy.stop()
    
    # Verifica estado
    assert not strategy._running

@pytest.mark.asyncio
async def test_update_quantum_state(strategy, exchange):
    """Testa atualização do estado quântico."""
    # Configura mocks
    candles = [
        {'timestamp': 0, 'open': 1.0, 'high': 2.0, 'low': 0.5, 'close': 1.5, 'volume': 1.0},
        {'timestamp': 1, 'open': 1.5, 'high': 2.5, 'low': 1.0, 'close': 2.0, 'volume': 2.0}
    ]
    exchange.get_candles = Mock(return_value=candles)
    
    # Atualiza estado quântico
    await strategy._update_quantum_state()
    
    # Verifica estado
    assert strategy._quantum_state is not None
    assert strategy._morphic_field is not None
    assert strategy._consciousness_level is not None
    assert strategy._entanglement_score is not None

def test_normalize_price(strategy):
    """Testa normalização de preço."""
    # Cria dados de teste
    prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Normaliza preço
    result = strategy._normalize_price(prices)
    
    # Verifica resultado
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0

def test_normalize_volume(strategy):
    """Testa normalização de volume."""
    # Cria dados de teste
    volumes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Normaliza volume
    result = strategy._normalize_volume(volumes)
    
    # Verifica resultado
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0

def test_calculate_momentum(strategy):
    """Testa cálculo do momentum."""
    # Cria dados de teste
    prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Calcula momentum
    result = strategy._calculate_momentum(prices)
    
    # Verifica resultado
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0

def test_calculate_orderbook_pressure(strategy, exchange):
    """Testa cálculo da pressão do livro de ordens."""
    # Configura mocks
    orderbook = {'bids': [[1.0, 1.0]], 'asks': [[2.0, 1.0]]}
    exchange.get_orderbook = Mock(return_value=orderbook)
    
    # Calcula pressão do livro de ordens
    result = strategy._calculate_orderbook_pressure(orderbook)
    
    # Verifica resultado
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0

def test_calculate_quantum_state(strategy):
    """Testa cálculo do estado quântico."""
    # Cria dados de teste
    price = 0.5
    volume = 0.5
    momentum = 0.5
    orderbook = 0.5
    
    # Calcula estado quântico
    result = strategy._calculate_quantum_state(price, volume, momentum, orderbook)
    
    # Verifica resultado
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0

def test_calculate_morphic_field(strategy):
    """Testa cálculo do campo mórfico."""
    # Cria dados de teste
    quantum_state = 0.5
    
    # Calcula campo mórfico
    result = strategy._calculate_morphic_field(quantum_state)
    
    # Verifica resultado
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0

def test_calculate_consciousness_level(strategy):
    """Testa cálculo do nível de consciência."""
    # Cria dados de teste
    quantum_state = 0.5
    morphic_field = 0.5
    
    # Calcula nível de consciência
    result = strategy._calculate_consciousness_level(quantum_state, morphic_field)
    
    # Verifica resultado
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0

def test_calculate_entanglement_score(strategy):
    """Testa cálculo do score de emaranhamento."""
    # Cria dados de teste
    quantum_state = 0.5
    morphic_field = 0.5
    consciousness_level = 0.5
    
    # Calcula score de emaranhamento
    result = strategy._calculate_entanglement_score(quantum_state, morphic_field, consciousness_level)
    
    # Verifica resultado
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0 