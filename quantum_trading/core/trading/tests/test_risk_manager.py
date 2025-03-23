"""
Testes do gerenciador de risco.
"""

import pytest
from unittest.mock import Mock, patch

from ..trading_config import TradingConfig
from ..exchange_integration import ExchangeIntegration
from ..risk_manager import RiskManager

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
def risk_manager(config, exchange):
    """Cria gerenciador de risco."""
    return RiskManager(config, exchange)

@pytest.mark.asyncio
async def test_validate_trade(risk_manager, exchange):
    """Testa validação de trade."""
    # Configura mocks
    exchange.get_balance = Mock(return_value={'USDT': {'free': 1000.0, 'used': 0.0, 'total': 1000.0}})
    risk_manager._positions = {}
    risk_manager._daily_trades = 0
    risk_manager._daily_loss = 0.0
    
    # Valida trade
    result = await risk_manager.validate_trade('buy', 1.0, 1.0)
    
    # Verifica resultado
    assert result

@pytest.mark.asyncio
async def test_calculate_position_size(risk_manager, exchange):
    """Testa cálculo do tamanho da posição."""
    # Configura mocks
    exchange.get_balance = Mock(return_value={'USDT': {'free': 1000.0, 'used': 0.0, 'total': 1000.0}})
    
    # Calcula tamanho da posição
    result = await risk_manager.calculate_position_size(1.0)
    
    # Verifica resultado
    assert result > 0.0

@pytest.mark.asyncio
async def test_update_position(risk_manager):
    """Testa atualização de posição."""
    # Atualiza posição
    await risk_manager.update_position('1', 'buy', 1.0, 1.0)
    
    # Verifica estado
    assert '1' in risk_manager._positions
    assert risk_manager._positions['1']['side'] == 'buy'
    assert risk_manager._positions['1']['price'] == 1.0
    assert risk_manager._positions['1']['amount'] == 1.0

@pytest.mark.asyncio
async def test_close_position(risk_manager):
    """Testa fechamento de posição."""
    # Configura estado
    risk_manager._positions = {'1': {'side': 'buy', 'price': 1.0, 'amount': 1.0}}
    
    # Fecha posição
    await risk_manager.close_position('1', 2.0)
    
    # Verifica estado
    assert '1' not in risk_manager._positions
    assert risk_manager._metrics['total_trades'] == 1
    assert risk_manager._metrics['winning_trades'] == 1
    assert risk_manager._metrics['total_profit'] == 1.0

def test_update_metrics(risk_manager):
    """Testa atualização de métricas."""
    # Configura estado
    risk_manager._metrics = {
        'total_trades': 10,
        'winning_trades': 7,
        'total_profit': 10.0,
        'total_loss': 3.0
    }
    
    # Atualiza métricas
    risk_manager._update_metrics(1.0)
    
    # Verifica estado
    assert risk_manager._metrics['win_rate'] == 0.7
    assert risk_manager._metrics['profit_factor'] == 10.0 / 3.0
    assert risk_manager._metrics['average_trade'] == 0.7

def test_reset_daily_stats(risk_manager):
    """Testa reset das estatísticas diárias."""
    # Configura estado
    risk_manager._daily_trades = 5
    risk_manager._daily_loss = 0.01
    
    # Reseta estatísticas
    risk_manager.reset_daily_stats()
    
    # Verifica estado
    assert risk_manager._daily_trades == 0
    assert risk_manager._daily_loss == 0.0

def test_get_metrics(risk_manager):
    """Testa obtenção de métricas."""
    # Configura estado
    risk_manager._metrics = {
        'total_trades': 10,
        'winning_trades': 7,
        'total_profit': 10.0,
        'total_loss': 3.0,
        'win_rate': 0.7,
        'profit_factor': 10.0 / 3.0,
        'average_trade': 0.7
    }
    
    # Obtém métricas
    result = risk_manager.get_metrics()
    
    # Verifica resultado
    assert result == risk_manager._metrics

def test_get_daily_stats(risk_manager):
    """Testa obtenção das estatísticas diárias."""
    # Configura estado
    risk_manager._daily_trades = 5
    risk_manager._daily_loss = 0.01
    
    # Obtém estatísticas
    result = risk_manager.get_daily_stats()
    
    # Verifica resultado
    assert result['daily_trades'] == 5
    assert result['daily_loss'] == 0.01 