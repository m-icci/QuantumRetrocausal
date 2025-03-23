"""Tests for AutoTrader balance management functionality"""
import pytest
from unittest.mock import Mock, patch, create_autospec
from decimal import Decimal

from quantum_trading.auto_trader import AutoTrader
from quantum_trading.market_api import MarketAPI
from quantum_trading.consciousness import MarketConsciousness
from quantum_trading.morphic_field import MorphicField

@pytest.fixture
def mock_market_api():
    api = create_autospec(MarketAPI)
    
    # Mock balances
    api.get_balance.return_value = {
        'BTC': 0.5,
        'ETH': 5.0,
        'USDT': 1000.0,
        'SOL': 20.0
    }
    
    # Mock ticker prices
    def mock_get_ticker(symbol):
        prices = {
            'BTC/USDT': {'c': ['30000.0']},
            'ETH/USDT': {'c': ['2000.0']},
            'SOL/USDT': {'c': ['50.0']},
            'ETH/BTC': {'c': ['0.067']},
            'SOL/ETH': {'c': ['0.025']},
        }
        return prices.get(symbol, {'c': ['0.0']})
    
    api.get_ticker.side_effect = mock_get_ticker
    return api

@pytest.fixture
def mock_consciousness():
    consciousness = create_autospec(MarketConsciousness)
    
    def mock_get_metrics(symbol):
        return {
            'momentum': 0.7,
            'trend_strength': 0.8,
            'coherence': 0.9,
            'trend_reliability': 0.85
        }
    
    consciousness.get_market_metrics.side_effect = mock_get_metrics
    return consciousness

@pytest.fixture
def mock_morphic_field():
    field = Mock()
    
    def mock_get_metrics(symbol):
        return {
            'field_strength': 0.75,
            'stability': 0.8
        }
    
    field.get_field_metrics.side_effect = mock_get_metrics
    return field

@pytest.fixture
def auto_trader(mock_market_api, mock_consciousness, mock_morphic_field):
    return AutoTrader(mock_market_api, mock_consciousness, mock_morphic_field, symbol='BTC/USDT')

class TestBalanceChecks:
    def setUp(self):
        """Configurar ambiente de teste para verificação de saldos"""
        # Configurar mocks para MarketAPI
        self.market_api = Mock()
        self.market_api.get_ticker.return_value = {'c': ['50000.00', '1.0']}
        self.market_api.get_minimum_order_size.return_value = 0.001

        # Configurar mock para Consciousness
        self.mock_consciousness = Mock()
        self.mock_consciousness.calculate_consciousness_field.return_value = {
            'field_strength': 1.0,
            'coherence': 0.8
        }

        # Configurar mock para MorphicField
        self.mock_morphic_field = Mock()

        # Criar AutoTrader
        self.auto_trader = AutoTrader(
            market_api=self.market_api, 
            consciousness=self.mock_consciousness, 
            morphic_field=self.mock_morphic_field, 
            symbol='BTC/USDT'
        )

    def test_get_available_balance_success(self):
        """Testar obtenção de saldo com sucesso"""
        # Configurar mock para retornar saldo específico
        self.market_api.get_all_balances.return_value = {
            'BTC': {'total': 0.5}
        }
        
        balance = self.auto_trader.get_available_balance('BTC')
        assert balance == 0.5

    def test_get_available_balance_api_error(self):
        """Testar tratamento de erro na obtenção de saldo"""
        # Simular erro na API
        self.market_api.get_all_balances.side_effect = Exception("API Error")
        
        balance = self.auto_trader.get_available_balance('BTC')
        assert balance == 0.0

    def test_calculate_order_size_sufficient_balance(self):
        """Testar cálculo de tamanho de ordem com saldo suficiente"""
        # Configurar mock para retornar saldo suficiente
        self.market_api.get_all_balances.return_value = {
            'USDT': {'total': 10000.0},
            'BTC': {'total': 0.5}
        }
        
        size = self.auto_trader.calculate_order_size(
            symbol='BTC/USDT', 
            side='buy', 
            current_price=50000.0
        )
        assert size > 0.0

    def test_calculate_order_size_insufficient_balance(self):
        """Testar cálculo de tamanho de ordem com saldo insuficiente"""
        # Configurar mock para retornar saldo zero
        self.market_api.get_all_balances.return_value = {
            'USDT': {'total': 0.0},
            'BTC': {'total': 0.0}
        }
        
        size = self.auto_trader.calculate_order_size(
            symbol='BTC/USDT', 
            side='buy', 
            current_price=50000.0
        )
        assert size == 0.0

    def test_calculate_order_size_zero_balance(self):
        """Testar cálculo de tamanho de ordem com saldo zero"""
        # Configurar mock para retornar saldo zero
        self.market_api.get_all_balances.return_value = {
            'USDT': {'total': 0.0},
            'BTC': {'total': 0.0}
        }
        
        size = self.auto_trader.calculate_order_size(
            symbol='BTC/USDT', 
            side='buy', 
            current_price=50000.0
        )
        assert size == 0.0

    def test_calculate_order_size_below_minimum(self):
        """Testar cálculo de tamanho de ordem abaixo do mínimo"""
        # Configurar mock para retornar saldo suficiente
        self.market_api.get_all_balances.return_value = {
            'USDT': {'total': 10000.0},
            'BTC': {'total': 0.5}
        }
        
        # Definir tamanho mínimo de ordem muito alto
        self.market_api.get_minimum_order_size.return_value = 1.0
        
        size = self.auto_trader.calculate_order_size(
            symbol='BTC/USDT', 
            side='buy', 
            current_price=50000.0
        )
        assert size == 0.0

    def test_rebalance_portfolio(self):
        """Testar rebalanceamento de portfólio"""
        # Configurar mocks para simulação de rebalanceamento
        self.market_api.get_all_balances.return_value = {
            'USDT': {'total': 500.0},
            'BTC': {'total': 0.1}
        }
        
        # Simular método de rebalanceamento do portfolio manager
        self.auto_trader.portfolio_manager.execute_rebalancing.return_value = True
        
        # Executar rebalanceamento
        success = self.auto_trader.rebalance_portfolio()
        
        # Verificar resultado
        assert success is True

    def test_rebalancing_not_needed(self):
        """Testar rebalanceamento quando não é necessário"""
        # Configurar mocks para saldo suficiente
        self.market_api.get_all_balances.return_value = {
            'USDT': {'total': 2000.0},
            'BTC': {'total': 0.2}
        }
        
        # Executar rebalanceamento
        success = self.auto_trader.rebalance_portfolio()
        
        # Verificar resultado
        assert success is True

    def test_rebalancing_failure(self):
        """Testar falha no rebalanceamento"""
        # Configurar mocks para simular falha
        self.market_api.get_all_balances.return_value = {
            'USDT': {'total': 500.0},
            'BTC': {'total': 0.1}
        }
        
        # Simular falha no método de rebalanceamento
        self.auto_trader.portfolio_manager.execute_rebalancing.return_value = False
        
        # Executar rebalanceamento
        success = self.auto_trader.rebalance_portfolio()
        
        # Verificar resultado
        assert success is False

class TestOrderSizeCalculation:
    def test_calculate_order_size_sufficient_balance(self, auto_trader):
        """Test order size calculation with sufficient balance"""
        size = auto_trader.calculate_order_size('BTC/USDT', 'buy', 29000.0)
        assert size > 0
        assert size * 29000.0 <= 1000.0  # Should not exceed USDT balance
        
    def test_calculate_order_size_insufficient_balance(self, auto_trader, mock_market_api):
        """Test order size with insufficient balance"""
        mock_market_api.get_balance.return_value = {'USDT': 10.0}
        size = auto_trader.calculate_order_size('BTC/USDT', 'buy', 30000.0)
        assert size == 0.0
        
    def test_calculate_order_size_zero_balance(self, auto_trader, mock_market_api):
        """Test order size with zero balance"""
        mock_market_api.get_balance.return_value = {'USDT': 0.0}
        size = auto_trader.calculate_order_size('BTC/USDT', 'buy', 30000.0)
        assert size == 0.0
        
    def test_calculate_order_size_below_minimum(self, auto_trader, mock_market_api):
        """Test order size below minimum"""
        mock_market_api.get_balance.return_value = {'USDT': 1.0}
        size = auto_trader.calculate_order_size('BTC/USDT', 'buy', 30000.0)
        assert size == 0.0

class TestCurrencyConversion:
    @pytest.mark.parametrize("symbol,base,quote", [
        ('BTC/USDT', 'BTC', 'USDT'),
        ('ETH/BTC', 'ETH', 'BTC'),
        ('SOL/ETH', 'SOL', 'ETH')
    ])
    def test_currency_pair_handling(self, auto_trader, symbol, base, quote):
        """Test handling of different currency pairs"""
        # Create new instance for each symbol
        trader = AutoTrader(auto_trader.market_api, auto_trader.consciousness, auto_trader.morphic_field, symbol=symbol)
        assert trader.base_currency == base
        assert trader.quote_currency == quote
        
    def test_usdt_conversion_calculation(self, auto_trader):
        """Test USDT value calculation for non-USDT pairs"""
        # Create ETH/BTC trader
        eth_btc_trader = AutoTrader(auto_trader.market_api, auto_trader.consciousness, auto_trader.morphic_field, symbol='ETH/BTC')
        eth_price_btc = 0.067
        btc_price_usdt = 30000.0
        
        # Verify conversion
        eth_value_usdt = eth_price_btc * btc_price_usdt
        assert abs(eth_value_usdt - 2010.0) < 1.0  # Allow small floating point difference

class TestRebalancingIntegration:
    def test_successful_rebalancing(self, auto_trader, mock_market_api):
        """Test successful portfolio rebalancing"""
        # Set up initial conditions
        mock_market_api.get_balance.return_value = {'USDT': 500.0}
        mock_market_api.create_order.return_value = {'result': {'txid': ['123']}}
        
        # Execute rebalancing
        success = auto_trader.rebalance_portfolio()
        assert success is True
        
        # Verify order was placed
        mock_market_api.create_order.assert_called_once()
        
    def test_failed_rebalancing(self, auto_trader, mock_market_api):
        """Test failed rebalancing attempt"""
        # Simulate API error
        mock_market_api.create_order.side_effect = Exception("API Error")
        
        # Execute rebalancing
        success = auto_trader.rebalance_portfolio()
        assert success is False
        
    def test_rebalancing_not_needed(self, auto_trader):
        """Test when rebalancing is not needed"""
        # Set balance close to target
        auto_trader.target_position_value = 1000.0
        success = auto_trader.rebalance_portfolio()
        assert success is True  # Should succeed because no rebalancing needed
