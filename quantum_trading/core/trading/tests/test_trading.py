"""
Testes do módulo de trading.
"""

import pytest
from unittest.mock import Mock, patch

from .. import (
    TradingConfig,
    ExchangeIntegration,
    MarketAnalysis,
    RiskManager,
    TradeExecutor,
    QuantumStrategy,
    TradingSystem
)

def test_imports():
    """Testa importações do módulo."""
    assert TradingConfig is not None
    assert ExchangeIntegration is not None
    assert MarketAnalysis is not None
    assert RiskManager is not None
    assert TradeExecutor is not None
    assert QuantumStrategy is not None
    assert TradingSystem is not None

def test_trading_config():
    """Testa configuração de trading."""
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
    
    assert isinstance(config, TradingConfig)

def test_exchange_integration():
    """Testa integração com exchange."""
    config = Mock(spec=TradingConfig)
    exchange = ExchangeIntegration(config)
    
    assert isinstance(exchange, ExchangeIntegration)

def test_market_analysis():
    """Testa análise de mercado."""
    config = Mock(spec=TradingConfig)
    exchange = Mock(spec=ExchangeIntegration)
    analysis = MarketAnalysis(config, exchange)
    
    assert isinstance(analysis, MarketAnalysis)

def test_risk_manager():
    """Testa gerenciador de risco."""
    config = Mock(spec=TradingConfig)
    exchange = Mock(spec=ExchangeIntegration)
    risk_manager = RiskManager(config, exchange)
    
    assert isinstance(risk_manager, RiskManager)

def test_trade_executor():
    """Testa executor de trades."""
    config = Mock(spec=TradingConfig)
    exchange = Mock(spec=ExchangeIntegration)
    analysis = Mock(spec=MarketAnalysis)
    risk_manager = Mock(spec=RiskManager)
    executor = TradeExecutor(config, exchange, analysis, risk_manager)
    
    assert isinstance(executor, TradeExecutor)

def test_quantum_strategy():
    """Testa estratégia quântica."""
    config = Mock(spec=TradingConfig)
    exchange = Mock(spec=ExchangeIntegration)
    analysis = Mock(spec=MarketAnalysis)
    risk_manager = Mock(spec=RiskManager)
    executor = Mock(spec=TradeExecutor)
    strategy = QuantumStrategy(config, exchange, analysis, risk_manager, executor)
    
    assert isinstance(strategy, QuantumStrategy)

def test_trading_system():
    """Testa sistema de trading."""
    config = Mock(spec=TradingConfig)
    exchange = Mock(spec=ExchangeIntegration)
    analysis = Mock(spec=MarketAnalysis)
    risk_manager = Mock(spec=RiskManager)
    executor = Mock(spec=TradeExecutor)
    strategy = Mock(spec=QuantumStrategy)
    system = TradingSystem(config, exchange, analysis, risk_manager, executor, strategy)
    
    assert isinstance(system, TradingSystem) 