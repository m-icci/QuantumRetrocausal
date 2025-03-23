"""Tests for Portfolio Manager functionality"""
import pytest
from unittest.mock import Mock, patch, create_autospec
from datetime import datetime
import time

from quantum_trading.portfolio_manager import PortfolioManager
from quantum_trading.market_api import MarketAPI
from quantum_trading.consciousness import MarketConsciousness


class MockMarketAPI:
    def __init__(self):
        self.balances = {
            'BTC': {'free': 0.5, 'used': 0.0, 'total': 0.5},
            'ETH': {'free': 5.0, 'used': 0.0, 'total': 5.0},
            'USDT': {'free': 1000.0, 'used': 0.0, 'total': 1000.0},
            'SOL': {'free': 20.0, 'used': 0.0, 'total': 20.0}
        }
        self.tickers = {
            'BTC/USDT': {'c': ['40000.0']},
            'ETH/USDT': {'c': ['2000.0']},
            'SOL/USDT': {'c': ['50.0']},
        }

    def get_balance(self, symbol):
        balance = self.balances.get(symbol)
        return balance['total'] if balance else 0.0

    def get_ticker(self, symbol):
        return self.tickers.get(symbol)

    def get_all_balances(self):
        return self.balances


@pytest.fixture
def mock_market_api():
    api = MockMarketAPI()
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
def portfolio_manager(mock_market_api, mock_consciousness, mock_morphic_field):
    return PortfolioManager(mock_market_api, mock_consciousness, mock_morphic_field)


class TestPortfolioStatus:
    def test_get_portfolio_status_success(self, portfolio_manager):
        """Test successful portfolio status retrieval"""
        status = portfolio_manager.get_portfolio_status()

        assert 'assets' in status
        assert 'total_value_usdt' in status

        # Verify BTC calculations
        btc = status['assets']['BTC']
        assert btc['amount'] == 0.5
        assert btc['value_usdt'] == 20000.0  # 0.5 BTC * 40000 USDT

        # Verify total value
        assert status['total_value_usdt'] > 0

    def test_get_portfolio_status_api_error(self, portfolio_manager, mock_market_api):
        """Test portfolio status with API error"""
        mock_market_api.get_all_balances = Mock(side_effect=Exception("API Error"))
        status = portfolio_manager.get_portfolio_status()

        assert status['total_value_usdt'] == 0.0
        assert len(status['assets']) == 0


class TestProfitAnalysis:
    def test_analyze_profit_opportunities_success(self, portfolio_manager):
        """Test profit opportunity analysis"""
        opportunities = portfolio_manager.analyze_profit_opportunities()

        assert len(opportunities) > 0

        # Verify opportunity structure
        opportunity = opportunities[0]
        assert 'symbol' in opportunity
        assert 'profit_potential' in opportunity
        assert 'current_value_usdt' in opportunity

    def test_analyze_profit_opportunities_no_balance(self, portfolio_manager, mock_market_api):
        """Test analysis with empty portfolio"""
        mock_market_api.get_all_balances = Mock(return_value={})
        opportunities = portfolio_manager.analyze_profit_opportunities()
        assert len(opportunities) == 0


class TestSellCandidates:
    def test_find_best_sell_candidate_single_asset(self, portfolio_manager):
        """Test finding single asset to sell"""
        candidate = portfolio_manager.find_best_sell_candidate(1000.0)

        assert candidate is not None
        assert candidate['symbol'] in ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        assert candidate['amount'] > 0
        assert candidate['value_usdt'] >= 1000.0

    def test_find_best_sell_candidate_combined_assets(self, portfolio_manager):
        """Test finding multiple assets to sell"""
        candidate = portfolio_manager.find_best_sell_candidate(20000.0)

        assert candidate is not None
        assert len(candidate['assets']) > 1
        assert candidate['total_value_usdt'] >= 20000.0

    def test_find_best_sell_candidate_insufficient_value(self, portfolio_manager):
        """Test when portfolio value is insufficient"""
        candidate = portfolio_manager.find_best_sell_candidate(100000.0)
        assert candidate is None


class TestRebalancing:
    def test_execute_rebalancing_single_asset(self, portfolio_manager, mock_market_api):
        """Test successful single asset rebalancing"""
        mock_market_api.create_order = Mock(return_value={'result': {'txid': ['123']}})

        success = portfolio_manager.execute_rebalancing(1000.0, 'ETH/USDT')
        assert success is True

        # Verify order was placed
        mock_market_api.create_order.assert_called_once()

    def test_execute_rebalancing_combined_assets(self, portfolio_manager, mock_market_api):
        """Test successful multi-asset rebalancing"""
        mock_market_api.create_order = Mock(return_value={'result': {'txid': ['123']}})

        success = portfolio_manager.execute_rebalancing(20000.0, 'BTC/USDT')
        assert success is True

        # Verify multiple orders were placed
        assert mock_market_api.create_order.call_count > 1

    def test_execute_rebalancing_cooldown(self, portfolio_manager):
        """Test rebalancing cooldown period"""
        # Set last rebalance time to now
        portfolio_manager._last_rebalance = time.time()
        success = portfolio_manager.execute_rebalancing(1000.0, 'ETH/USDT')
        assert success is False

    def test_execute_rebalancing_order_failure(self, portfolio_manager, mock_market_api):
        """Test handling of order execution failure"""
        mock_market_api.create_order = Mock(side_effect=Exception("Order failed"))
        success = portfolio_manager.execute_rebalancing(1000.0, 'ETH/USDT')
        assert success is False


class TestErrorHandling:
    def test_api_timeout(self, portfolio_manager, mock_market_api):
        """Test handling of API timeout"""
        mock_market_api.get_ticker = Mock(side_effect=TimeoutError("API timeout"))
        status = portfolio_manager.get_portfolio_status()
        assert status['total_value_usdt'] == 0.0

    def test_invalid_price_data(self, portfolio_manager, mock_market_api):
        """Test handling of invalid price data"""
        mock_market_api.get_ticker = Mock(return_value={'c': ['invalid']})
        status = portfolio_manager.get_portfolio_status()
        assert status['total_value_usdt'] == 0.0

    def test_insufficient_balance(self, portfolio_manager, mock_market_api):
        """Test handling of insufficient balance"""
        mock_market_api.get_all_balances = Mock(return_value={'USDT': {'free': 0.0, 'used': 0.0, 'total': 0.0}})
        success = portfolio_manager.execute_rebalancing(1000.0, 'ETH/USDT')
        assert success is False
