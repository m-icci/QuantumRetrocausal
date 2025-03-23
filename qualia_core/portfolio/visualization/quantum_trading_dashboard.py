"""
Quantum Trading Dashboard

Implementa visualização em tempo real para o sistema de trading quântico
"""
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from dataclasses import asdict, dataclass

from ..scalping.quantum_scalper import ScalpingSignal, ScalpingMetrics
from ...merge.morphic_field import MorphicPattern, FieldMetrics

@dataclass
class DashboardMetrics:
    """Métricas do dashboard"""
    timestamp: datetime
    capital: float
    open_positions: int
    daily_pnl: float
    total_pnl: float
    win_rate: float
    field_coherence: float
    phi_resonance: float
    pattern_count: int
    alerts: List[str]

class QuantumTradingDashboard:
    """
    Dashboard para visualização em tempo real do trading quântico
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa dashboard
        
        Args:
            config: Configuração do dashboard
        """
        self.config = config
        self.base_path = Path(config.get('base_path', '.'))
        
        # Cria diretórios
        self.reports_dir = self.base_path / 'reports'
        self.trades_dir = self.base_path / 'trades'
        self.logs_dir = self.base_path / 'logs'
        
        for dir_path in [self.reports_dir, self.trades_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Configura logging
        self._setup_logging()
        
        # Estado do dashboard
        self.current_metrics: Optional[DashboardMetrics] = None
        self.trades_history: List[Dict[str, Any]] = []
        self.signals_history: List[ScalpingSignal] = []
        self.field_metrics: List[FieldMetrics] = []
        
        # Alertas
        self.alert_thresholds = config.get('alert_thresholds', {
            'drawdown': 0.1,  # 10% drawdown
            'coherence': 0.7,  # Coerência mínima
            'resonance': 0.6,  # Ressonância mínima
            'pattern_strength': 0.8  # Força mínima do padrão
        })
        
    def _setup_logging(self):
        """Configura sistema de logging"""
        log_file = self.logs_dir / f"trading_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('QuantumTrading')
        
    async def update_metrics(self,
                           capital: float,
                           positions: List[Dict[str, Any]],
                           trades: List[Dict[str, Any]],
                           field_metrics: FieldMetrics) -> None:
        """
        Atualiza métricas do dashboard
        
        Args:
            capital: Capital atual
            positions: Posições abertas
            trades: Histórico de trades
            field_metrics: Métricas do campo mórfico
        """
        # Calcula métricas
        daily_pnl = sum(t['pnl'] for t in trades if t['exit_time'].date() == datetime.now().date())
        total_pnl = sum(t['pnl'] for t in trades)
        
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        win_rate = winning_trades / len(trades) if trades else 0
        
        # Gera alertas
        alerts = self._generate_alerts(
            capital=capital,
            daily_pnl=daily_pnl,
            field_metrics=field_metrics
        )
        
        # Atualiza métricas
        self.current_metrics = DashboardMetrics(
            timestamp=datetime.now(),
            capital=capital,
            open_positions=len(positions),
            daily_pnl=daily_pnl,
            total_pnl=total_pnl,
            win_rate=win_rate,
            field_coherence=field_metrics.coherence,
            phi_resonance=field_metrics.resonance,
            pattern_count=len(self.field_metrics),
            alerts=alerts
        )
        
        # Salva métricas
        await self._save_metrics()
        
    def _generate_alerts(self,
                        capital: float,
                        daily_pnl: float,
                        field_metrics: FieldMetrics) -> List[str]:
        """
        Gera alertas baseados nas métricas
        
        Args:
            capital: Capital atual
            daily_pnl: PnL diário
            field_metrics: Métricas do campo
            
        Returns:
            Lista de alertas
        """
        alerts = []
        
        # Alerta de drawdown
        if daily_pnl / capital <= -self.alert_thresholds['drawdown']:
            alerts.append(f"ALERTA: Drawdown diário excedeu {self.alert_thresholds['drawdown']*100}%")
            
        # Alerta de coerência
        if field_metrics.coherence < self.alert_thresholds['coherence']:
            alerts.append(f"ALERTA: Coerência do campo abaixo do limiar ({field_metrics.coherence:.2f})")
            
        # Alerta de ressonância
        if field_metrics.resonance < self.alert_thresholds['resonance']:
            alerts.append(f"ALERTA: Ressonância φ abaixo do limiar ({field_metrics.resonance:.2f})")
            
        return alerts
        
    async def _save_metrics(self) -> None:
        """Salva métricas atuais"""
        if not self.current_metrics:
            return
            
        # Salva relatório diário
        report_file = self.reports_dir / f"trading_{datetime.now().strftime('%Y%m%d')}.json"
        
        metrics_dict = asdict(self.current_metrics)
        metrics_dict['timestamp'] = metrics_dict['timestamp'].isoformat()
        
        async with aiofiles.open(report_file, 'w') as f:
            await f.write(json.dumps(metrics_dict, indent=2))
            
    async def save_trade(self, trade: Dict[str, Any]) -> None:
        """
        Salva informações de trade
        
        Args:
            trade: Informações do trade
        """
        # Adiciona ao histórico
        self.trades_history.append(trade)
        
        # Salva em arquivo
        trade_file = self.trades_dir / f"trade_{trade['id']}.json"
        
        # Converte datetime para string
        trade_dict = trade.copy()
        trade_dict['entry_time'] = trade_dict['entry_time'].isoformat()
        trade_dict['exit_time'] = trade_dict['exit_time'].isoformat()
        
        async with aiofiles.open(trade_file, 'w') as f:
            await f.write(json.dumps(trade_dict, indent=2))
            
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Calcula métricas de performance
        
        Returns:
            Métricas de performance
        """
        if not self.trades_history:
            return {}
            
        # Extrai PnL
        pnl = [t['pnl'] for t in self.trades_history]
        
        # Calcula métricas
        metrics = {
            'total_trades': len(self.trades_history),
            'winning_trades': len([p for p in pnl if p > 0]),
            'total_pnl': sum(pnl),
            'avg_pnl': np.mean(pnl),
            'std_pnl': np.std(pnl),
            'sharpe': np.mean(pnl) / np.std(pnl) if np.std(pnl) > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(pnl)
        }
        
        return metrics
        
    def _calculate_max_drawdown(self, pnl: List[float]) -> float:
        """
        Calcula drawdown máximo
        
        Args:
            pnl: Lista de PnL
            
        Returns:
            Drawdown máximo
        """
        cumulative = np.cumsum(pnl)
        peaks = np.maximum.accumulate(cumulative)
        drawdowns = (peaks - cumulative) / peaks
        return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
        
    def get_field_analysis(self) -> Dict[str, Any]:
        """
        Analisa métricas do campo mórfico
        
        Returns:
            Análise do campo
        """
        if not self.field_metrics:
            return {}
            
        # Extrai séries temporais
        coherence = [m.coherence for m in self.field_metrics]
        resonance = [m.resonance for m in self.field_metrics]
        strength = [m.strength for m in self.field_metrics]
        
        # Calcula métricas
        analysis = {
            'coherence': {
                'current': coherence[-1],
                'mean': np.mean(coherence),
                'trend': np.polyfit(np.arange(len(coherence)), coherence, 1)[0]
            },
            'resonance': {
                'current': resonance[-1],
                'mean': np.mean(resonance),
                'trend': np.polyfit(np.arange(len(resonance)), resonance, 1)[0]
            },
            'strength': {
                'current': strength[-1],
                'mean': np.mean(strength),
                'trend': np.polyfit(np.arange(len(strength)), strength, 1)[0]
            }
        }
        
        return analysis
        
    def get_alerts(self) -> List[str]:
        """
        Retorna alertas ativos
        
        Returns:
            Lista de alertas
        """
        return self.current_metrics.alerts if self.current_metrics else []
