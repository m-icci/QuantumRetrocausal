"""
Testes do sistema de trading.
"""

import pytest
from unittest.mock import Mock, patch

from ..trading_config import TradingConfig
from ..exchange_integration import ExchangeIntegration
from ..market_analysis import MarketAnalysis
from ..risk_manager import RiskManager
from ..trade_executor import TradeExecutor
from ..quantum_strategy import QuantumStrategy
from ..trading_system import TradingSystem

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
    return Mock(spec=QuantumStrategy)

@pytest.fixture
def system(config, exchange, analysis, risk_manager, executor, strategy):
    """Cria sistema de trading."""
    return TradingSystem(config, exchange, analysis, risk_manager, executor, strategy)

@pytest.mark.asyncio
async def test_start(system):
    """Testa inicialização do sistema."""
    # Configura mocks
    system.exchange.start = Mock()
    system.executor.start = Mock()
    system.strategy.start = Mock()
    
    # Inicia sistema
    await system.start()
    
    # Verifica chamadas
    system.exchange.start.assert_called_once()
    system.executor.start.assert_called_once()
    system.strategy.start.assert_called_once()
    
    # Verifica estado
    assert system._running
    assert system._start_time is not None
    assert system._update_task is not None

@pytest.mark.asyncio
async def test_stop(system):
    """Testa parada do sistema."""
    # Configura mocks
    system.exchange.stop = Mock()
    system.executor.stop = Mock()
    system.strategy.stop = Mock()
    
    # Inicia sistema
    system._running = True
    system._update_task = Mock()
    
    # Para sistema
    await system.stop()
    
    # Verifica chamadas
    system.exchange.stop.assert_called_once()
    system.executor.stop.assert_called_once()
    system.strategy.stop.assert_called_once()
    
    # Verifica estado
    assert not system._running

@pytest.mark.asyncio
async def test_update_loop(system):
    """Testa loop de atualização."""
    # Configura mocks
    system.strategy._update_quantum_state = Mock()
    system._update_metrics = Mock()
    
    # Inicia loop
    system._running = True
    await system._update_loop()
    system._running = False
    
    # Verifica chamadas
    system.strategy._update_quantum_state.assert_called()
    system._update_metrics.assert_called()

def test_update_metrics(system):
    """Testa atualização de métricas."""
    # Configura mocks
    system.risk_manager.get_metrics = Mock(return_value={'test': 1.0})
    system.risk_manager.get_daily_stats = Mock(return_value={'test': 1.0})
    system.executor.get_active_trades = Mock(return_value={})
    system.executor.get_trade_history = Mock(return_value=[])
    
    # Atualiza métricas
    system._update_metrics()
    
    # Verifica chamadas
    system.risk_manager.get_metrics.assert_called_once()
    system.risk_manager.get_daily_stats.assert_called_once()
    system.executor.get_active_trades.assert_called_once()
    system.executor.get_trade_history.assert_called_once()

def test_get_status(system):
    """Testa obtenção de status."""
    # Configura mocks
    system._running = True
    system._start_time = Mock()
    system.risk_manager.get_metrics = Mock(return_value={'test': 1.0})
    system.risk_manager.get_daily_stats = Mock(return_value={'test': 1.0})
    system.executor.get_active_trades = Mock(return_value={})
    system.executor.get_trade_history = Mock(return_value=[])
    
    # Obtém status
    status = system.get_status()
    
    # Verifica status
    assert status['running']
    assert status['start_time'] is not None
    assert status['metrics'] == {'test': 1.0}
    assert status['daily_stats'] == {'test': 1.0}
    assert status['active_trades'] == 0
    assert status['trade_history'] == 0 