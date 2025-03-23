"""Test suite for AutoTrader operations"""
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from datetime import datetime, timedelta
from quantum_trading.auto_trader import AutoTrader
from quantum_trading.market_api import MarketAPI
from quantum_trading.consciousness import MarketConsciousness
import time

class TestAutoTrader(unittest.TestCase):
    def setUp(self):
        """Configurar ambiente de teste"""
        # Configurar mocks para MarketAPI
        self.market_api = MagicMock()
        self.market_api.get_ticker.return_value = {'c': ['50000.00', '1.0']}
        self.market_api.get_all_balances.return_value = {
            'USDT': {'total': 10000.0},
            'BTC': {'total': 0.5}
        }
        self.market_api.get_minimum_order_size.return_value = 0.001

        # Configurar mock para Consciousness
        self.mock_consciousness = MagicMock()
        self.mock_consciousness.calculate_consciousness_field.return_value = {
            'field_strength': 1.0,
            'coherence': 0.8
        }

        # Configurar mock para MorphicField
        self.mock_morphic_field = MagicMock()

        # Criar AutoTrader
        self.trader = AutoTrader(
            market_api=self.market_api, 
            consciousness=self.mock_consciousness, 
            morphic_field=self.mock_morphic_field, 
            symbol='BTC/USD'
        )

    def test_edge_cases(self):
        """Testar casos de borda no AutoTrader"""
        # Simular cenário de preço inválido
        with self.assertRaises(ValueError):
            self.trader.calculate_order_size(current_price=-1)

    def test_position_size_calculation(self):
        """Testar cálculo de tamanho de posição"""
        size = self.trader.calculate_order_size(
            symbol='BTC/USD', 
            side='buy', 
            current_price=50000.0
        )
        self.assertGreater(size, 0.0, "Size should be greater than 0")

    def test_trade_execution(self):
        """Testar execução de trade"""
        # Simular condições de mercado
        metrics = {
            'field_strength': 0.8,
            'coherence': 0.7
        }
        
        # Executar trade
        result = self.trader.execute_trade(
            action='buy', 
            metrics=metrics, 
            price=50000.0
        )
        
        # Verificar resultado
        self.assertTrue(result['success'], f"Trade execution failed: {result.get('error', 'Unknown error')}")

    def test_position_lifecycle_management(self):
        """Testar ciclo de vida da posição"""
        # Abrir posição
        open_result = self.trader.open_position(
            side='long', 
            entry_price=50000.0, 
            size=0.1
        )
        
        # Verificar abertura de posição
        self.assertTrue(open_result['success'], "Position opening failed")
        
        # Fechar posição
        close_result = self.trader.close_position()
        
        # Verificar fechamento de posição
        self.assertTrue(close_result['success'], "Position closing failed")

    def test_entry_signals(self):
        """Test entry signal generation"""
        # Test strong entry signal
        metrics = {
            'coherence': 0.7,
            'integration': 0.6,
            'field_strength': 0.8
        }

        patterns = [{
            'scale': 1.0,
            'strength': 1.2,
            'reliability': 0.7
        }]

        should_enter = self.trader.should_enter_long(metrics, patterns)
        self.assertTrue(should_enter)

        # Test weak signal - low coherence
        metrics['coherence'] = 0.4
        should_enter = self.trader.should_enter_long(metrics, patterns)
        self.assertFalse(should_enter)

        # Test weak signal - low pattern strength
        metrics['coherence'] = 0.7
        patterns[0]['strength'] = 0.9
        should_enter = self.trader.should_enter_long(metrics, patterns)
        self.assertFalse(should_enter)

    def test_exit_signals(self):
        """Test exit signal generation"""
        current_time = datetime.now()
        metrics = {
            'coherence': 0.7,
            'integration': 0.6,
            'field_strength': 0.8
        }

        # Setup test position with proper stop loss and take profit
        self.trader.position = {
            'type': 'long',
            'entry_price': 50000.0,
            'size': 0.1,
            'stop_loss': 49500.0,  # 1% below entry
            'take_profit': 50500.0,  # 1% above entry
            'entry_time': current_time,
            'max_duration': 300,
            'order_id': 'test_order'
        }

        # Test stop loss trigger
        should_exit = self.trader.should_exit_long(metrics, 49400.0)  # Price below stop loss
        self.assertTrue(should_exit)
        self.assertEqual(self.trader.position['exit_reason'], 'stop_loss')

        # Reset position for next test
        self.trader.position['exit_reason'] = None

        # Test take profit trigger
        should_exit = self.trader.should_exit_long(metrics, 50600.0)  # Price above take profit
        self.assertTrue(should_exit)
        self.assertEqual(self.trader.position['exit_reason'], 'take_profit')

        # Reset position for next test
        self.trader.position['exit_reason'] = None

        # Test timeout trigger
        with patch('quantum_trading.auto_trader.datetime') as mock_datetime:
            mock_datetime.now.return_value = current_time + timedelta(seconds=301)
            should_exit = self.trader.should_exit_long(metrics, 50000.0)
            self.assertTrue(should_exit)
            self.assertEqual(self.trader.position['exit_reason'], 'timeout')

        # Test market condition trigger
        self.trader.position['exit_reason'] = None
        degraded_metrics = {
            'coherence': 0.3,  # Below threshold
            'integration': 0.6,
            'field_strength': 0.4  # Below threshold
        }
        should_exit = self.trader.should_exit_long(degraded_metrics, 50000.0)
        self.assertTrue(should_exit)
        self.assertEqual(self.trader.position['exit_reason'], 'market_conditions')

    def test_performance_metrics(self):
        """Test performance metrics calculation"""
        # Add some test trades
        self.trader.trade_history = [
            {
                'entry_time': datetime.now() - timedelta(minutes=5),
                'exit_time': datetime.now(),
                'entry_price': 50000.0,
                'exit_price': 50250.0,
                'profit': 25.0,
                'profit_pct': 0.5,
                'duration': 300,
                'exit_reason': 'take_profit'
            },
            {
                'entry_time': datetime.now() - timedelta(minutes=10),
                'exit_time': datetime.now() - timedelta(minutes=5),
                'entry_price': 49800.0,
                'exit_price': 49700.0,
                'profit': -10.0,
                'profit_pct': -0.2,
                'duration': 300,
                'exit_reason': 'stop_loss'
            }
        ]

        metrics = self.trader.get_performance_metrics()

        # Verify metrics
        self.assertEqual(metrics['total_trades'], 2)
        self.assertEqual(metrics['win_rate'], 50.0)
        self.assertEqual(metrics['total_profit'], 15.0)
        self.assertEqual(metrics['avg_duration'], 300)
        self.assertEqual(metrics['exit_reasons']['take_profit'], 1)
        self.assertEqual(metrics['exit_reasons']['stop_loss'], 1)

    def test_quantum_inspired_analysis(self):
        """Test quantum-inspired analysis functionality"""
        # Test quantum metrics calculation
        metrics = self.mock_consciousness.calculate_consciousness_field()
        self.assertIsInstance(metrics, dict)
        self.assertIn('coherence', metrics)
        self.assertIn('integration', metrics)
        self.assertIn('field_strength', metrics)

    def test_stop_loss_take_profit_validation(self):
        """Test stop loss and take profit calculations"""
        # Configure mock consciousness metrics
        metrics = {
            'coherence': 0.8,
            'field_strength': 1.1
        }
        self.mock_consciousness.calculate_consciousness_field.return_value = metrics

        entry_price = 50000.0

        # Test stop loss
        stop_loss = self.trader.calculate_stop_loss(entry_price)
        self.assertLess(stop_loss, entry_price)
        self.assertGreater(stop_loss, entry_price * 0.95)  # Max 5% loss

        # Test take profit
        take_profit = self.trader.calculate_take_profit(entry_price)
        self.assertGreater(take_profit, entry_price)

        # Verify risk/reward ratio
        risk = entry_price - stop_loss
        reward = take_profit - entry_price
        self.assertGreaterEqual(reward / risk, 2.0)

    def test_market_consciousness_degradation(self):
        """Test market consciousness degradation"""
        initial_metrics = {
            'coherence': 0.9,
            'integration': 0.8,
            'field_strength': 0.9
        }

        current_time = time.time()
        future_time = current_time + 3600  # 1 hour later

        degraded = self.trader.apply_consciousness_decay(initial_metrics, future_time)

        # Verify degradation
        self.assertLess(degraded['coherence'], initial_metrics['coherence'])
        self.assertLess(degraded['integration'], initial_metrics['integration'])
        self.assertLess(degraded['field_strength'], initial_metrics['field_strength'])

if __name__ == '__main__':
    unittest.main()