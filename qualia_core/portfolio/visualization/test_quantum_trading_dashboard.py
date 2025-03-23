"""
Tests for quantum trading dashboard
"""
import unittest
from datetime import datetime
import tempfile
from pathlib import Path
import numpy as np
import asyncio
from unittest.mock import Mock

from .quantum_trading_dashboard import (
    QuantumTradingDashboard,
    DashboardMetrics
)
from ...merge.morphic_field import FieldMetrics

class TestQuantumTradingDashboard(unittest.TestCase):
    """Testes para dashboard de trading"""
    
    def setUp(self):
        """Setup para testes"""
        # Cria diretório temporário
        self.test_dir = tempfile.mkdtemp()
        
        # Configura dashboard
        self.dashboard = QuantumTradingDashboard({
            'base_path': self.test_dir,
            'alert_thresholds': {
                'drawdown': 0.1,
                'coherence': 0.7,
                'resonance': 0.6,
                'pattern_strength': 0.8
            }
        })
        
        # Mock de métricas do campo
        self.field_metrics = FieldMetrics(
            coherence=0.8,
            resonance=0.7,
            strength=0.9,
            patterns=[],
            evolution_rate=0.1
        )
        
    async def test_update_metrics(self):
        """Testa atualização de métricas"""
        # Dados simulados
        capital = 10000.0
        positions = [
            {'id': 1, 'size': 1.0, 'direction': 1}
        ]
        trades = [
            {
                'id': 'test1',
                'entry_time': datetime.now(),
                'exit_time': datetime.now(),
                'pnl': 100.0
            },
            {
                'id': 'test2',
                'entry_time': datetime.now(),
                'exit_time': datetime.now(),
                'pnl': -50.0
            }
        ]
        
        # Atualiza métricas
        await self.dashboard.update_metrics(
            capital=capital,
            positions=positions,
            trades=trades,
            field_metrics=self.field_metrics
        )
        
        # Verifica métricas
        self.assertIsNotNone(self.dashboard.current_metrics)
        self.assertEqual(self.dashboard.current_metrics.capital, capital)
        self.assertEqual(self.dashboard.current_metrics.open_positions, len(positions))
        
    async def test_save_trade(self):
        """Testa salvamento de trade"""
        trade = {
            'id': 'test1',
            'entry_time': datetime.now(),
            'exit_time': datetime.now(),
            'entry_price': 100.0,
            'exit_price': 101.0,
            'direction': 1,
            'size': 1.0,
            'pnl': 1.0
        }
        
        await self.dashboard.save_trade(trade)
        
        # Verifica se trade foi salvo
        self.assertEqual(len(self.dashboard.trades_history), 1)
        trade_file = Path(self.test_dir) / 'trades' / f"trade_{trade['id']}.json"
        self.assertTrue(trade_file.exists())
        
    def test_performance_metrics(self):
        """Testa cálculo de métricas de performance"""
        # Adiciona trades simulados
        self.dashboard.trades_history = [
            {
                'id': f'test{i}',
                'pnl': 1.0 if i % 2 == 0 else -0.5
            }
            for i in range(10)
        ]
        
        metrics = self.dashboard.get_performance_metrics()
        
        self.assertEqual(metrics['total_trades'], 10)
        self.assertEqual(metrics['winning_trades'], 5)
        self.assertGreater(metrics['total_pnl'], 0)
        
    def test_field_analysis(self):
        """Testa análise do campo mórfico"""
        # Adiciona métricas simuladas
        self.dashboard.field_metrics = [
            FieldMetrics(
                coherence=0.8 + i*0.01,
                resonance=0.7 + i*0.01,
                strength=0.9 + i*0.01,
                patterns=[],
                evolution_rate=0.1
            )
            for i in range(10)
        ]
        
        analysis = self.dashboard.get_field_analysis()
        
        self.assertIn('coherence', analysis)
        self.assertIn('resonance', analysis)
        self.assertIn('strength', analysis)
        
        # Verifica tendências
        self.assertGreater(analysis['coherence']['trend'], 0)
        self.assertGreater(analysis['resonance']['trend'], 0)
        self.assertGreater(analysis['strength']['trend'], 0)
        
    def test_alerts(self):
        """Testa geração de alertas"""
        # Configura métricas com valores baixos
        self.dashboard.current_metrics = DashboardMetrics(
            timestamp=datetime.now(),
            capital=10000.0,
            open_positions=1,
            daily_pnl=-1100.0,  # -11% drawdown
            total_pnl=-1100.0,
            win_rate=0.5,
            field_coherence=0.6,  # Abaixo do limiar
            phi_resonance=0.5,    # Abaixo do limiar
            pattern_count=5,
            alerts=[]
        )
        
        alerts = self.dashboard.get_alerts()
        
        self.assertGreater(len(alerts), 0)
        self.assertTrue(any('drawdown' in alert.lower() for alert in alerts))
        self.assertTrue(any('coerência' in alert.lower() for alert in alerts))
        self.assertTrue(any('ressonância' in alert.lower() for alert in alerts))
        
if __name__ == '__main__':
    unittest.main()
