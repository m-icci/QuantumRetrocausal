"""
Testes do executor de trades.
"""

import pytest
from unittest.mock import Mock, patch

from ..trading_config import TradingConfig
from ..exchange_integration import ExchangeIntegration
from ..market_analysis import MarketAnalysis
from ..risk_manager import RiskManager
from ..trade_executor import TradeExecutor

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
    return TradeExecutor(config, exchange, analysis, risk_manager)

@pytest.mark.asyncio
async def test_start(executor):
    """Testa inicialização do executor."""
    # Inicia executor
    await executor.start()
    
    # Verifica estado
    assert executor._running
    assert executor._update_task is not None

@pytest.mark.asyncio
async def test_stop(executor):
    """Testa parada do executor."""
    # Inicia executor
    executor._running = True
    executor._update_task = Mock()
    
    # Para executor
    await executor.stop()
    
    # Verifica estado
    assert not executor._running

@pytest.mark.asyncio
async def test_execute_trades(executor, exchange, analysis, risk_manager):
    """Testa execução de trades."""
    # Configura mocks
    analysis.get_signals = Mock(return_value={'buy': True, 'sell': False})
    exchange.get_ticker = Mock(return_value={'bid': 1.0, 'ask': 2.0, 'last': 1.5})
    risk_manager.validate_trade = Mock(return_value=True)
    risk_manager.calculate_position_size = Mock(return_value=1.0)
    exchange.place_order = Mock(return_value={'id': '1'})
    
    # Executa trades
    await executor.execute_trades()
    
    # Verifica chamadas
    analysis.get_signals.assert_called_once()
    exchange.get_ticker.assert_called_once()
    risk_manager.validate_trade.assert_called_once()
    risk_manager.calculate_position_size.assert_called_once()
    exchange.place_order.assert_called_once()

@pytest.mark.asyncio
async def test_prepare_order(executor, exchange, risk_manager):
    """Testa preparação de ordem."""
    # Configura mocks
    exchange.get_ticker = Mock(return_value={'bid': 1.0, 'ask': 2.0, 'last': 1.5})
    risk_manager.validate_trade = Mock(return_value=True)
    risk_manager.calculate_position_size = Mock(return_value=1.0)
    
    # Prepara ordem
    result = await executor._prepare_order('buy')
    
    # Verifica resultado
    assert result is not None
    assert result['side'] == 'buy'
    assert result['price'] == 2.0
    assert result['amount'] == 1.0

@pytest.mark.asyncio
async def test_execute_order(executor, exchange):
    """Testa execução de ordem."""
    # Configura mocks
    order = {'side': 'buy', 'price': 1.0, 'amount': 1.0}
    exchange.place_order = Mock(return_value={'id': '1'})
    
    # Executa ordem
    result = await executor._execute_order(order)
    
    # Verifica resultado
    assert result == {'id': '1'}

@pytest.mark.asyncio
async def test_place_stop_loss(executor, exchange):
    """Testa colocação de stop loss."""
    # Configura mocks
    exchange.place_order = Mock(return_value={'id': '1'})
    
    # Coloca stop loss
    result = await executor._place_stop_loss('buy', 1.0, 1.0)
    
    # Verifica resultado
    assert result == {'id': '1'}

@pytest.mark.asyncio
async def test_place_take_profit(executor, exchange):
    """Testa colocação de take profit."""
    # Configura mocks
    exchange.place_order = Mock(return_value={'id': '1'})
    
    # Coloca take profit
    result = await executor._place_take_profit('buy', 1.0, 1.0)
    
    # Verifica resultado
    assert result == {'id': '1'}

@pytest.mark.asyncio
async def test_close_position(executor, exchange):
    """Testa fechamento de posição."""
    # Configura mocks
    exchange.get_ticker = Mock(return_value={'bid': 1.0, 'ask': 2.0, 'last': 1.5})
    exchange.place_order = Mock(return_value={'id': '1'})
    
    # Fecha posição
    await executor._close_position('1', 'buy', 1.0)
    
    # Verifica chamadas
    exchange.get_ticker.assert_called_once()
    exchange.place_order.assert_called_once()

@pytest.mark.asyncio
async def test_cancel_order(executor, exchange):
    """Testa cancelamento de ordem."""
    # Configura mocks
    exchange.cancel_order = Mock(return_value={'id': '1'})
    
    # Cancela ordem
    await executor._cancel_order('1')
    
    # Verifica chamadas
    exchange.cancel_order.assert_called_once_with('1')

def test_get_active_trades(executor):
    """Testa obtenção de trades ativos."""
    # Configura estado
    executor._active_trades = {'1': {'side': 'buy', 'price': 1.0, 'amount': 1.0}}
    
    # Obtém trades ativos
    result = executor.get_active_trades()
    
    # Verifica resultado
    assert result == executor._active_trades

def test_get_trade_history(executor):
    """Testa obtenção do histórico de trades."""
    # Configura estado
    executor._trade_history = [{'id': '1', 'side': 'buy', 'price': 1.0, 'amount': 1.0}]
    
    # Obtém histórico de trades
    result = executor.get_trade_history()
    
    # Verifica resultado
    assert result == executor._trade_history 