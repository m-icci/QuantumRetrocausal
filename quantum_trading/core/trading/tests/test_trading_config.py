"""
Testes da configuração do sistema de trading.
"""

import pytest
import json
import os
from unittest.mock import mock_open, patch

from ..trading_config import TradingConfig

def test_init():
    """Testa inicialização da configuração."""
    config = TradingConfig(
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
    
    assert config.exchange == 'binance'
    assert config.api_key == 'test_key'
    assert config.api_secret == 'test_secret'
    assert config.symbol == 'BTC/USDT'
    assert config.timeframe == '1h'
    assert config.leverage == 1
    assert config.max_positions == 3
    assert config.daily_trades_limit == 10
    assert config.daily_loss_limit == 0.02
    assert config.min_confidence == 0.7
    assert config.position_size == 0.1
    assert config.min_position_size == 0.01
    assert config.max_position_size == 0.5
    assert config.stop_loss == 0.02
    assert config.take_profit == 0.04
    assert config.risk_per_trade == 0.01
    assert config.rsi_period == 14
    assert config.rsi_oversold == 30
    assert config.rsi_overbought == 70
    assert config.macd_fast == 12
    assert config.macd_slow == 26
    assert config.macd_signal == 9
    assert config.bb_period == 20
    assert config.bb_std == 2
    assert config.atr_period == 14

def test_validate():
    """Testa validação da configuração."""
    with pytest.raises(ValueError):
        TradingConfig(
            exchange='invalid',
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

def test_from_dict():
    """Testa criação da configuração a partir de dicionário."""
    config_dict = {
        'exchange': 'binance',
        'api_key': 'test_key',
        'api_secret': 'test_secret',
        'symbol': 'BTC/USDT',
        'timeframe': '1h',
        'leverage': 1,
        'max_positions': 3,
        'daily_trades_limit': 10,
        'daily_loss_limit': 0.02,
        'min_confidence': 0.7,
        'position_size': 0.1,
        'min_position_size': 0.01,
        'max_position_size': 0.5,
        'stop_loss': 0.02,
        'take_profit': 0.04,
        'risk_per_trade': 0.01,
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'bb_period': 20,
        'bb_std': 2,
        'atr_period': 14
    }
    
    config = TradingConfig.from_dict(config_dict)
    
    assert config.exchange == 'binance'
    assert config.api_key == 'test_key'
    assert config.api_secret == 'test_secret'
    assert config.symbol == 'BTC/USDT'
    assert config.timeframe == '1h'
    assert config.leverage == 1
    assert config.max_positions == 3
    assert config.daily_trades_limit == 10
    assert config.daily_loss_limit == 0.02
    assert config.min_confidence == 0.7
    assert config.position_size == 0.1
    assert config.min_position_size == 0.01
    assert config.max_position_size == 0.5
    assert config.stop_loss == 0.02
    assert config.take_profit == 0.04
    assert config.risk_per_trade == 0.01
    assert config.rsi_period == 14
    assert config.rsi_oversold == 30
    assert config.rsi_overbought == 70
    assert config.macd_fast == 12
    assert config.macd_slow == 26
    assert config.macd_signal == 9
    assert config.bb_period == 20
    assert config.bb_std == 2
    assert config.atr_period == 14

def test_from_file():
    """Testa criação da configuração a partir de arquivo."""
    config_dict = {
        'exchange': 'binance',
        'api_key': 'test_key',
        'api_secret': 'test_secret',
        'symbol': 'BTC/USDT',
        'timeframe': '1h',
        'leverage': 1,
        'max_positions': 3,
        'daily_trades_limit': 10,
        'daily_loss_limit': 0.02,
        'min_confidence': 0.7,
        'position_size': 0.1,
        'min_position_size': 0.01,
        'max_position_size': 0.5,
        'stop_loss': 0.02,
        'take_profit': 0.04,
        'risk_per_trade': 0.01,
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'bb_period': 20,
        'bb_std': 2,
        'atr_period': 14
    }
    
    with patch('builtins.open', mock_open(read_data=json.dumps(config_dict))):
        config = TradingConfig.from_file('config.json')
    
    assert config.exchange == 'binance'
    assert config.api_key == 'test_key'
    assert config.api_secret == 'test_secret'
    assert config.symbol == 'BTC/USDT'
    assert config.timeframe == '1h'
    assert config.leverage == 1
    assert config.max_positions == 3
    assert config.daily_trades_limit == 10
    assert config.daily_loss_limit == 0.02
    assert config.min_confidence == 0.7
    assert config.position_size == 0.1
    assert config.min_position_size == 0.01
    assert config.max_position_size == 0.5
    assert config.stop_loss == 0.02
    assert config.take_profit == 0.04
    assert config.risk_per_trade == 0.01
    assert config.rsi_period == 14
    assert config.rsi_oversold == 30
    assert config.rsi_overbought == 70
    assert config.macd_fast == 12
    assert config.macd_slow == 26
    assert config.macd_signal == 9
    assert config.bb_period == 20
    assert config.bb_std == 2
    assert config.atr_period == 14

def test_to_dict():
    """Testa conversão da configuração para dicionário."""
    config = TradingConfig(
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
    
    config_dict = config.to_dict()
    
    assert config_dict['exchange'] == 'binance'
    assert config_dict['api_key'] == 'test_key'
    assert config_dict['api_secret'] == 'test_secret'
    assert config_dict['symbol'] == 'BTC/USDT'
    assert config_dict['timeframe'] == '1h'
    assert config_dict['leverage'] == 1
    assert config_dict['max_positions'] == 3
    assert config_dict['daily_trades_limit'] == 10
    assert config_dict['daily_loss_limit'] == 0.02
    assert config_dict['min_confidence'] == 0.7
    assert config_dict['position_size'] == 0.1
    assert config_dict['min_position_size'] == 0.01
    assert config_dict['max_position_size'] == 0.5
    assert config_dict['stop_loss'] == 0.02
    assert config_dict['take_profit'] == 0.04
    assert config_dict['risk_per_trade'] == 0.01
    assert config_dict['rsi_period'] == 14
    assert config_dict['rsi_oversold'] == 30
    assert config_dict['rsi_overbought'] == 70
    assert config_dict['macd_fast'] == 12
    assert config_dict['macd_slow'] == 26
    assert config_dict['macd_signal'] == 9
    assert config_dict['bb_period'] == 20
    assert config_dict['bb_std'] == 2
    assert config_dict['atr_period'] == 14

def test_save():
    """Testa salvamento da configuração em arquivo."""
    config = TradingConfig(
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
    
    with patch('builtins.open', mock_open()) as mock_file:
        config.save('config.json')
        
    mock_file.assert_called_once_with('config.json', 'w')
    mock_file().write.assert_called_once() 