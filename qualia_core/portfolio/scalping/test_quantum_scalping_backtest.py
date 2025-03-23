"""
Tests for quantum scalping backtest system
"""
import unittest
from datetime import datetime, timedelta
import numpy as np

from quantum.core.portfolio.scalping.quantum_scalping_backtest import (
    QuantumScalpingBacktest,
    BacktestResult,
    TradeResult
)
from quantum.core.portfolio.scalping.quantum_scalper import ScalpSignal
from quantum.core.QUALIA import QUALIA, generate_field, PHI

class TestQuantumScalpingBacktest(unittest.TestCase):
    """Testes para sistema de backtesting"""

    def setUp(self):
        """Setup para testes"""
        # Initialize QUALIA with proper field dimensions
        self.qualia = QUALIA(
            dimensions=8,
            field=generate_field(8)
        )

        self.backtest = QuantumScalpingBacktest(
            field_dimensions=8,
            coherence_threshold=0.75,
            resonance_threshold=0.7,
            max_history=1000,
            atr_period=14,
            qualia_system=self.qualia
        )

        # Dados de mercado simulados - tendência de alta
        self.market_data = {
            'opens': [100.0 + i * 0.1 for i in range(100)],
            'highs': [100.0 + i * 0.1 + 0.05 for i in range(100)],
            'lows': [100.0 + i * 0.1 - 0.05 for i in range(100)],
            'closes': [100.0 + i * 0.1 for i in range(100)],
            'volumes': [1000.0 for _ in range(100)]
        }

    def test_backtest_execution(self):
        """Testa execução do backtest"""
        # Create market data for QUALIA update
        for i in range(len(self.market_data['closes'])):
            self.qualia.update({
                'price': self.market_data['closes'][i],
                'volume': self.market_data['volumes'][i],
                'timestamp': datetime.now().timestamp() + i
            })

        results = self.backtest.run_backtest(
            historical_data=self.market_data,
            initial_capital=10000.0,
            risk_per_trade=0.02,
            trading_cost=0.001
        )

        self.assertIsInstance(results, BacktestResult)
        self.assertEqual(results.initial_capital, 10000.0)
        self.assertGreaterEqual(results.total_trades, 0)
        self.assertGreaterEqual(results.winning_trades, 0)
        self.assertGreaterEqual(results.losing_trades, 0)
        self.assertEqual(
            results.total_trades,
            results.winning_trades + results.losing_trades
        )

    def test_equity_curve_calculation(self):
        """Testa cálculo da curva de equity"""
        trades = [
            TradeResult(
                entry_time=datetime.now(),
                exit_time=datetime.now() + timedelta(minutes=5),
                entry_price=100.0,
                exit_price=101.0,
                direction=1,
                size=1.0,
                pnl=1.0,
                field_coherence=0.8,
                phi_resonance=0.7,
                pattern_id="test_pattern"
            ),
            TradeResult(
                entry_time=datetime.now() + timedelta(minutes=10),
                exit_time=datetime.now() + timedelta(minutes=15),
                entry_price=101.0,
                exit_price=100.0,
                direction=-1,
                size=1.0,
                pnl=-1.0,
                field_coherence=0.8,
                phi_resonance=0.7,
                pattern_id="test_pattern"
            )
        ]

        equity = self.backtest._calculate_equity_curve(10000.0, trades)

        self.assertEqual(len(equity), len(trades) + 1)
        self.assertEqual(equity[0], 10000.0)
        self.assertEqual(equity[-1], 10000.0)

    def test_max_drawdown_calculation(self):
        """Testa cálculo do drawdown máximo"""
        # Simula curva de equity com drawdown conhecido
        equity = np.array([10000.0, 10100.0, 10050.0, 10000.0, 10200.0])
        drawdown = self.backtest._calculate_max_drawdown(equity)

        self.assertIsInstance(drawdown, float)
        self.assertGreaterEqual(drawdown, 0.0)
        self.assertLessEqual(drawdown, 1.0)

        # Drawdown esperado: (10100 - 10000) / 10100 = 0.0099
        expected_drawdown = 0.0099
        self.assertAlmostEqual(drawdown, expected_drawdown, places=4)

    def test_qualia_integration(self):
        """Test QUALIA consciousness integration"""
        # Test initial QUALIA state
        metrics = self.qualia.get_metrics()
        self.assertIn('coherence', metrics)
        self.assertIn('resonance', metrics)
        self.assertIn('emergence', metrics)

        # Test QUALIA update with market data
        market_update = {
            'price': 100.0,
            'volume': 1000.0,
            'timestamp': datetime.now().timestamp()
        }
        self.qualia.update(market_update)

        # Verify updated metrics
        updated_metrics = self.qualia.get_metrics()
        self.assertGreaterEqual(updated_metrics['coherence'], 0.0)
        self.assertLessEqual(updated_metrics['coherence'], 1.0)
        self.assertGreaterEqual(updated_metrics['resonance'], 0.0)
        self.assertLessEqual(updated_metrics['resonance'], 1.0)

    def test_phi_resonance(self):
        """Test φ-based resonance in trading patterns"""
        # Create oscillating pattern 
        prices = [100.0 + PHI * np.sin(i/PHI) for i in range(50)]
        volumes = [1000.0 * (1 + 0.1 * np.cos(i/PHI)) for i in range(50)]

        # Update QUALIA with φ-resonant pattern
        for i in range(len(prices)):
            self.qualia.update({
                'price': prices[i],
                'volume': volumes[i],
                'timestamp': datetime.now().timestamp() + i
            })

        # Verify φ-resonance metrics
        metrics = self.qualia.get_metrics()
        self.assertGreaterEqual(metrics['phi_alignment'], 0.5,
                              "Low φ alignment in resonant pattern")

    def test_scalping_pattern_detection(self):
        """Test detection of scalping patterns"""
        # Create rapid price movements
        short_prices = [100.0 + 0.1 * np.sin(i * PHI) for i in range(20)]

        # Update QUALIA with high-frequency data
        for i, price in enumerate(short_prices):
            self.qualia.update({
                'price': price,
                'volume': 1000.0,
                'timestamp': datetime.now().timestamp() + i * 0.001  # millisecond updates
            })

        # Run small backtest
        test_data = {
            'opens': short_prices,
            'highs': [p + 0.02 for p in short_prices],
            'lows': [p - 0.02 for p in short_prices],
            'closes': short_prices,
            'volumes': [1000.0] * len(short_prices)
        }

        results = self.backtest.run_backtest(
            historical_data=test_data,
            initial_capital=10000.0,
            risk_per_trade=0.01
        )

        # Verify pattern detection
        self.assertGreater(results.field_coherence_mean, 0.6,
                          "Low field coherence in scalping patterns")
        self.assertGreater(results.phi_resonance_mean, 0.6,
                          "Low φ resonance in scalping patterns")

    def test_results_analysis(self):
        """Testa análise de resultados"""
        # Cria resultado simulado
        results = BacktestResult(
            initial_capital=10000.0,
            final_capital=11000.0,
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            total_pnl=1000.0,
            max_drawdown=0.1,
            sharpe_ratio=2.0,
            field_coherence_mean=0.8,
            phi_resonance_mean=0.7,
            trades=[
                TradeResult(
                    entry_time=datetime.now(),
                    exit_time=datetime.now() + timedelta(minutes=5),
                    entry_price=100.0,
                    exit_price=101.0,
                    direction=1,
                    size=1.0,
                    pnl=1.0,
                    field_coherence=0.8,
                    phi_resonance=0.7,
                    pattern_id="test_pattern"
                )
            ],
            metrics_history=[],
            patterns=[]
        )

        analysis = self.backtest.analyze_results(results)

        self.assertIsInstance(analysis, dict)
        self.assertIn('performance', analysis)
        self.assertIn('quantum_metrics', analysis)
        self.assertIn('trade_analysis', analysis)

        # Verifica métricas de performance
        self.assertEqual(analysis['performance']['total_return'], 0.1)  # 10% retorno
        self.assertEqual(analysis['performance']['win_rate'], 0.6)  # 60% win rate
        self.assertEqual(analysis['performance']['max_drawdown'], 0.1)
        self.assertEqual(analysis['performance']['sharpe_ratio'], 2.0)

        # Verifica métricas quânticas
        self.assertEqual(analysis['quantum_metrics']['field_coherence'], 0.8)
        self.assertEqual(analysis['quantum_metrics']['phi_resonance'], 0.7)
        self.assertEqual(analysis['quantum_metrics']['pattern_count'], 0)

    def test_backtest_with_sideways_market(self):
        """Testa backtest em mercado lateral"""
        # Cria dados de mercado lateral
        sideways_data = {
            'opens': [100.0 + np.sin(i/10) for i in range(100)],
            'highs': [100.0 + np.sin(i/10) + 0.05 for i in range(100)],
            'lows': [100.0 + np.sin(i/10) - 0.05 for i in range(100)],
            'closes': [100.0 + np.sin(i/10) for i in range(100)],
            'volumes': [1000.0 for _ in range(100)]
        }

        results = self.backtest.run_backtest(
            historical_data=sideways_data,
            initial_capital=10000.0,
            risk_per_trade=0.02
        )

        self.assertIsInstance(results, BacktestResult)

        # Em mercado lateral, esperamos menos trades
        self.assertLess(results.total_trades, 50)

    def test_backtest_with_volatile_market(self):
        """Testa backtest em mercado volátil"""
        # Cria dados de mercado volátil
        volatile_data = {
            'opens': [100.0 + np.random.normal(0, 1) for _ in range(100)],
            'highs': [100.0 + np.random.normal(0, 1) + 0.5 for _ in range(100)],
            'lows': [100.0 + np.random.normal(0, 1) - 0.5 for _ in range(100)],
            'closes': [100.0 + np.random.normal(0, 1) for _ in range(100)],
            'volumes': [1000.0 * (1 + abs(np.random.normal(0, 0.5))) for _ in range(100)]
        }

        results = self.backtest.run_backtest(
            historical_data=volatile_data,
            initial_capital=10000.0,
            risk_per_trade=0.02
        )

        self.assertIsInstance(results, BacktestResult)

        # Em mercado volátil, esperamos mais trades e maior drawdown
        self.assertGreater(results.max_drawdown, 0.05)

    def test_risk_management(self):
        """Testa gestão de risco no backtest"""
        # Executa backtest com diferentes níveis de risco
        risk_levels = [0.01, 0.02, 0.05]

        for risk in risk_levels:
            results = self.backtest.run_backtest(
                historical_data=self.market_data,
                initial_capital=10000.0,
                risk_per_trade=risk
            )

            # Maior risco deve resultar em maior drawdown
            self.assertLessEqual(
                results.max_drawdown,
                risk * 2,  # Drawdown não deve ser mais que 2x o risco
                f"Drawdown muito alto para risco {risk}"
            )

    def test_retrocausal_analysis(self):
        """Testa análise retrocausal do scalping"""
        # Cria padrão de mercado com reversão
        reversal_data = {
            'opens': [100.0 + i * 0.1 for i in range(50)] +
                    [105.0 - i * 0.1 for i in range(50)],
            'highs': [100.0 + i * 0.1 + 0.05 for i in range(50)] +
                    [105.0 - i * 0.1 + 0.05 for i in range(50)],
            'lows': [100.0 + i * 0.1 - 0.05 for i in range(50)] +
                    [105.0 - i * 0.1 - 0.05 for i in range(50)],
            'closes': [100.0 + i * 0.1 for i in range(50)] +
                     [105.0 - i * 0.1 for i in range(50)],
            'volumes': [1000.0 + i * 10 for i in range(50)] +
                      [1500.0 - i * 10 for i in range(50)]
        }

        # Executa backtest com análise retrocausal
        results = self.backtest.run_backtest(
            historical_data=reversal_data,
            initial_capital=10000.0,
            risk_per_trade=0.02
        )

        # Verifica métricas retrocausais
        self.assertGreater(results.phi_resonance_mean, 0.6,
                          "Baixa ressonância φ")
        self.assertGreater(results.field_coherence_mean, 0.7,
                          "Baixa coerência de campo")

        # Verifica se detectou padrão de reversão
        self.assertGreater(len(results.patterns), 0,
                          "Nenhum padrão retrocausal detectado")

        # Verifica ajuste de níveis Fibonacci
        for trade in results.trades:
            if trade.pattern_id:  # Trade baseado em padrão
                # Trades com padrão devem ter maior retorno médio
                self.assertGreater(trade.pnl, 0,
                                 "Trade com padrão teve perda")
                self.assertGreater(trade.phi_resonance, 0.65,
                                 "Baixa ressonância em trade com padrão")

    def test_error_handling_invalid_parameters(self):
        """Test error handling for invalid initialization parameters"""
        with self.assertRaises(ValueError):
            QuantumScalpingBacktest(
                field_dimensions=-1,  # Invalid dimension
                coherence_threshold=0.75,
                resonance_threshold=0.7,
                max_history=1000
            )

        with self.assertRaises(ValueError):
            QuantumScalpingBacktest(
                field_dimensions=8,
                coherence_threshold=2.0,  # Invalid threshold > 1
                resonance_threshold=0.7,
                max_history=1000
            )

    def test_market_data_validation(self):
        """Test validation of market data input"""
        # Test with missing required fields
        invalid_data = {
            'opens': [100.0],
            'highs': [101.0],
            # Missing 'lows'
            'closes': [100.5]
        }
        with self.assertRaises(ValueError):
            self.backtest.run_backtest(invalid_data, 10000.0, 0.02)

        # Test with mismatched array lengths
        invalid_data = {
            'opens': [100.0, 101.0],
            'highs': [101.0],  # Different length
            'lows': [99.0, 98.0],
            'closes': [100.5, 101.5],
            'volumes': [1000.0, 1000.0]
        }
        with self.assertRaises(ValueError):
            self.backtest.run_backtest(invalid_data, 10000.0, 0.02)

    def test_quantum_metrics_extreme_conditions(self):
        """Test quantum metrics under extreme market conditions"""
        # Test with extremely high volatility
        volatile_data = {
            'opens': [100.0 + np.random.normal(0, 5) for _ in range(100)],
            'highs': [100.0 + np.random.normal(0, 5) + 2 for _ in range(100)],
            'lows': [100.0 + np.random.normal(0, 5) - 2 for _ in range(100)],
            'closes': [100.0 + np.random.normal(0, 5) for _ in range(100)],
            'volumes': [1000.0 * (1 + abs(np.random.normal(0, 1))) for _ in range(100)]
        }

        results = self.backtest.run_backtest(
            volatile_data,
            initial_capital=10000.0,
            risk_per_trade=0.02
        )

        self.assertGreater(results.field_coherence_mean, 0.5)
        self.assertLess(results.field_coherence_mean, 1.0)
        self.assertGreater(results.phi_resonance_mean, 0.5)
        self.assertLess(results.phi_resonance_mean, 1.0)

    def test_quantum_state_persistence(self):
        """Test persistence of quantum state across multiple backtest runs"""
        initial_state = self.backtest.qualia_system.get_metrics()

        # Run first backtest
        self.backtest.run_backtest(
            self.market_data,
            initial_capital=10000.0,
            risk_per_trade=0.02
        )

        mid_state = self.backtest.qualia_system.get_metrics()
        self.assertNotEqual(initial_state, mid_state)

        # Run second backtest
        self.backtest.run_backtest(
            self.market_data,
            initial_capital=10000.0,
            risk_per_trade=0.02
        )

        final_state = self.backtest.qualia_system.get_metrics()
        self.assertNotEqual(mid_state, final_state)

    def test_integration_with_external_systems(self):
        """Test integration with external market data and trading systems"""
        class MockExternalSystem:
            def get_market_data(self):
                return {
                    'opens': [100.0 + i for i in range(10)],
                    'highs': [101.0 + i for i in range(10)],
                    'lows': [99.0 + i for i in range(10)],
                    'closes': [100.5 + i for i in range(10)],
                    'volumes': [1000.0 for _ in range(10)]
                }

        external_system = MockExternalSystem()
        external_data = external_system.get_market_data()

        results = self.backtest.run_backtest(
            external_data,
            initial_capital=10000.0,
            risk_per_trade=0.02
        )

        self.assertIsInstance(results, BacktestResult)
        self.assertTrue(hasattr(results, 'trades'))
        self.assertTrue(hasattr(results, 'metrics_history'))

    def test_performance_metrics_calculation(self):
        """Test detailed performance metrics calculation"""
        results = self.backtest.run_backtest(
            self.market_data,
            initial_capital=10000.0,
            risk_per_trade=0.02
        )

        # Test basic metrics
        self.assertGreaterEqual(results.sharpe_ratio, -10)
        self.assertLessEqual(results.sharpe_ratio, 10)
        self.assertGreaterEqual(results.max_drawdown, 0)
        self.assertLessEqual(results.max_drawdown, 1)

        # Test advanced metrics
        analysis = self.backtest.analyze_results(results)
        self.assertIn('performance', analysis)
        self.assertIn('quantum_metrics', analysis)
        self.assertIn('trade_analysis', analysis)

        # Validate specific metrics
        perf = analysis['performance']
        self.assertGreaterEqual(perf['win_rate'], 0)
        self.assertLessEqual(perf['win_rate'], 1)
        self.assertGreaterEqual(perf['profit_factor'], 0)

    def test_quantum_field_alignment(self):
        """Test quantum field alignment with market patterns"""
        # Create phi-based pattern
        phi_pattern = [100.0 + PHI * np.sin(i/PHI) for i in range(50)]
        phi_data = {
            'opens': phi_pattern,
            'highs': [p + 0.5 for p in phi_pattern],
            'lows': [p - 0.5 for p in phi_pattern],
            'closes': phi_pattern,
            'volumes': [1000.0 for _ in range(50)]
        }

        results = self.backtest.run_backtest(
            phi_data,
            initial_capital=10000.0,
            risk_per_trade=0.02
        )

        # Test phi resonance metrics
        self.assertGreater(results.phi_resonance_mean, 0.6)
        self.assertLess(results.phi_resonance_mean, 1.0)

        # Test pattern recognition
        self.assertGreater(len(results.patterns), 0)
        for pattern in results.patterns:
            self.assertGreater(pattern.confidence, 0.5)

if __name__ == '__main__':
    unittest.main()