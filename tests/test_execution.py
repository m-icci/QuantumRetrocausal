"""
Testes para módulo de execução
=====================

Testes unitários para o sistema de execução do QUALIA.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from quantum_trading.execution import (
    OrderExecutor,
    PositionManager,
    RiskManager,
    ExecutionOptimizer,
    ExecutionEngine
)
from quantum_trading.exceptions import ExecutionError

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
def order_data():
    """Fixture com dados de ordem."""
    return {
        "symbol": "BTC/USDT",
        "type": "limit",
        "side": "buy",
        "amount": 1.5,
        "price": 45000,
        "timestamp": datetime.now(),
        "params": {
            "post_only": True,
            "reduce_only": False
        }
    }

@pytest.fixture
def position_data():
    """Fixture com dados de posição."""
    return {
        "BTC/USDT": {
            "amount": 1.5,
            "entry_price": 45000,
            "current_price": 47000,
            "unrealized_pnl": 3000,
            "realized_pnl": 1000,
            "side": "long",
            "leverage": 3
        },
        "ETH/USDT": {
            "amount": 10,
            "entry_price": 3000,
            "current_price": 3200,
            "unrealized_pnl": 2000,
            "realized_pnl": 500,
            "side": "long",
            "leverage": 2
        }
    }

def test_order_executor(order_data):
    """Testa executor de ordens."""
    executor = OrderExecutor()
    
    # Configura executor
    executor.configure(
        exchange="binance",
        testnet=True,
        rate_limit=10
    )
    
    # Executa ordem
    result = executor.execute_order(order_data)
    
    # Verifica resultado
    assert "order_id" in result
    assert "status" in result
    assert "filled_amount" in result
    assert "average_price" in result
    
    # Verifica status
    status = executor.get_order_status(result["order_id"])
    assert status in ["open", "closed", "canceled"]
    
    # Cancela ordem
    if status == "open":
        cancel = executor.cancel_order(result["order_id"])
        assert cancel["success"]

def test_position_manager(position_data):
    """Testa gerenciador de posições."""
    manager = PositionManager()
    
    # Abre posição
    position = manager.open_position(
        symbol="BTC/USDT",
        side="long",
        amount=1.5,
        price=45000
    )
    
    # Verifica posição
    assert position["symbol"] == "BTC/USDT"
    assert position["side"] == "long"
    assert position["amount"] == 1.5
    
    # Atualiza posição
    updated = manager.update_position(
        symbol="BTC/USDT",
        current_price=47000
    )
    
    # Verifica PnL
    assert updated["unrealized_pnl"] > 0
    
    # Fecha posição
    closed = manager.close_position(
        symbol="BTC/USDT",
        price=47000
    )
    
    # Verifica fechamento
    assert closed["realized_pnl"] > 0
    assert not manager.has_position("BTC/USDT")

def test_risk_manager(position_data):
    """Testa gerenciador de risco."""
    manager = RiskManager()
    
    # Configura limites
    manager.set_limits({
        "max_position_size": 2.0,
        "max_leverage": 5,
        "max_drawdown": 0.1
    })
    
    # Verifica ordem
    order = {
        "symbol": "BTC/USDT",
        "amount": 1.5,
        "leverage": 3
    }
    
    assert manager.validate_order(order)
    
    # Verifica posição
    assert manager.validate_position(position_data["BTC/USDT"])
    
    # Calcula risco
    risk = manager.calculate_risk(position_data)
    assert "value_at_risk" in risk
    assert "position_risk" in risk
    assert "portfolio_risk" in risk

def test_execution_optimizer(market_data, order_data):
    """Testa otimizador de execução."""
    optimizer = ExecutionOptimizer()
    
    # Configura otimizador
    optimizer.configure(
        window_size=20,
        min_trade_size=0.1,
        max_slippage=0.001
    )
    
    # Otimiza ordem
    optimized = optimizer.optimize_order(
        order=order_data,
        market_data=market_data
    )
    
    # Verifica otimização
    assert "optimal_size" in optimized
    assert "optimal_price" in optimized
    assert "expected_impact" in optimized
    
    # Valida limites
    assert optimized["optimal_size"] >= 0.1
    assert abs(optimized["expected_impact"]) <= 0.001

def test_execution_engine(market_data, order_data, position_data):
    """Testa motor de execução."""
    engine = ExecutionEngine()
    
    # Configura componentes
    engine.configure(
        executor=OrderExecutor(),
        position_manager=PositionManager(),
        risk_manager=RiskManager(),
        optimizer=ExecutionOptimizer()
    )
    
    # Executa ordem
    result = engine.execute(
        order=order_data,
        market_data=market_data,
        positions=position_data
    )
    
    # Verifica execução
    assert "success" in result
    assert "order_id" in result
    assert "execution_info" in result
    
    # Verifica estado
    state = engine.get_state()
    assert "active_orders" in state
    assert "positions" in state
    assert "risk_metrics" in state

def test_smart_order_routing(order_data):
    """Testa roteamento inteligente de ordens."""
    engine = ExecutionEngine()
    
    # Configura venues
    venues = [
        {"name": "binance", "fee": 0.001, "liquidity": 1000},
        {"name": "kraken", "fee": 0.002, "liquidity": 800},
        {"name": "coinbase", "fee": 0.0015, "liquidity": 900}
    ]
    
    # Roteia ordem
    route = engine.route_order(order_data, venues)
    
    # Verifica roteamento
    assert "venue" in route
    assert "expected_cost" in route
    assert "execution_strategy" in route
    
    # Valida seleção
    assert route["venue"] in [v["name"] for v in venues]

def test_execution_monitoring(order_data):
    """Testa monitoramento de execução."""
    engine = ExecutionEngine()
    
    # Inicia monitoramento
    engine.start_monitoring()
    
    # Executa ordem
    engine.execute(order_data)
    
    # Obtém métricas
    metrics = engine.get_metrics()
    
    # Verifica métricas
    assert "fill_rate" in metrics
    assert "slippage" in metrics
    assert "execution_time" in metrics
    
    # Para monitoramento
    engine.stop_monitoring()

def test_execution_algorithms(market_data, order_data):
    """Testa algoritmos de execução."""
    engine = ExecutionEngine()
    
    # Testa VWAP
    vwap_result = engine.execute_vwap(
        order_data,
        market_data,
        target_percentage=0.1
    )
    assert "vwap_price" in vwap_result
    
    # Testa TWAP
    twap_result = engine.execute_twap(
        order_data,
        market_data,
        interval_minutes=5
    )
    assert "twap_price" in twap_result
    
    # Testa Iceberg
    iceberg_result = engine.execute_iceberg(
        order_data,
        visible_size=0.1
    )
    assert "visible_orders" in iceberg_result

def test_execution_constraints(order_data):
    """Testa restrições de execução."""
    engine = ExecutionEngine()
    
    # Configura restrições
    constraints = {
        "min_trade_size": 0.1,
        "max_trade_size": 2.0,
        "price_deviation": 0.01,
        "min_volume": 1000
    }
    
    # Valida ordem
    validation = engine.validate_constraints(
        order_data,
        constraints
    )
    
    # Verifica validação
    assert validation["valid"]
    assert "checks" in validation
    assert all(validation["checks"].values())

def test_execution_persistence():
    """Testa persistência de execução."""
    engine = ExecutionEngine()
    
    # Salva estado
    state = {
        "orders": {"order1": {"status": "filled"}},
        "positions": {"BTC/USDT": {"amount": 1.5}},
        "metrics": {"fill_rate": 0.95}
    }
    
    engine.save_state(state, "test_state")
    
    # Carrega estado
    loaded = engine.load_state("test_state")
    
    # Verifica estado
    assert loaded == state
    
    # Remove arquivo
    engine.remove_state("test_state") 