"""
Quantum Scalping Backtest System

Implementa backtesting para o sistema de scalping quântico usando QUALIA e campos mórficos
"""
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
from datetime import datetime
import numpy as np
from dataclasses import dataclass

from .quantum_scalper import QuantumScalper, ScalpSignal, ScalpingMetrics
from ...merge.morphic_field import MorphicField, MorphicPattern, FieldMetrics
from ...QUALIA import QUALIA, QualiaState, generate_field

# Constants for numerical stability
EPSILON = 1e-10
MAX_COND = 1e10

@dataclass
class TradeResult:
    """Resultado de uma operação"""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    direction: int  # 1: long, -1: short
    size: float
    pnl: float
    field_coherence: float
    phi_resonance: float
    pattern_id: Optional[str]

@dataclass
class BacktestResult:
    """Resultado do backtest"""
    initial_capital: float
    final_capital: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    field_coherence_mean: float
    phi_resonance_mean: float
    trades: List[TradeResult]
    metrics_history: List[ScalpingMetrics]
    patterns: List[MorphicPattern]

class QuantumScalpingBacktest:
    """Sistema de backtesting para scalping quântico"""

    def __init__(self,
                 field_dimensions: int = 8,
                 coherence_threshold: float = 0.75,
                 resonance_threshold: float = 0.7,
                 max_history: int = 1000,
                 atr_period: int = 14,
                 qualia_system: Optional[QUALIA] = None,
                 epsilon: float = 1e-10):
        """
        Inicializa sistema de backtesting
        """
        self.epsilon = epsilon
        self.scalper = QuantumScalper()
        self.scalper.configure(
            field_dimensions=field_dimensions,
            coherence_threshold=coherence_threshold,
            resonance_threshold=resonance_threshold,
            max_history=max_history,
            atr_period=atr_period
        )

        # Initialize QUALIA integration with stability
        self.qualia = qualia_system or QUALIA(
            dimensions=field_dimensions,
            field=generate_field(field_dimensions)
        )

        # Initialize metrics tracking
        self.metrics_history: List[ScalpingMetrics] = []

    def run_backtest(self,
                    historical_data: Dict[str, List[float]],
                    initial_capital: float = 10000.0,
                    risk_per_trade: float = 0.02,
                    trading_cost: float = 0.001) -> BacktestResult:
        """
        Executa backtest
        """
        # Validate input data
        try:
            self._validate_historical_data(historical_data)
        except ValueError as e:
            raise ValueError(f"Invalid historical data: {e}")

        # Initialize tracking variables
        capital = initial_capital
        position = 0
        trades: List[TradeResult] = []
        current_trade = None

        # Para cada ponto no tempo
        for i in range(len(historical_data['closes'])):
            try:
                # Update QUALIA state with stability checks
                self.qualia.update({
                    'price': float(historical_data['closes'][i]),
                    'volume': float(historical_data['volumes'][i]),
                    'timestamp': datetime.now().timestamp() + i
                })

                # Prepare market data with stability
                market_data = self._prepare_market_data(historical_data, i)

                # Generate signal with stability checks
                signal = self.scalper.analyze_market(market_data)
                qualia_metrics = self.qualia.get_metrics()

                # Store metrics
                if hasattr(self.scalper, 'metrics'):
                    self.metrics_history.extend(self.scalper.metrics)

                # Process trading logic
                if position == 0:
                    # Entry conditions with stability
                    if (self._check_entry_conditions(signal, qualia_metrics)):
                        # Calculate position size with stability
                        size = self._calculate_position_size(
                            capital, risk_per_trade, signal, market_data
                        )

                        position = signal.direction
                        entry_price = float(market_data['closes'][-1])

                        current_trade = self._create_trade_entry(
                            entry_price, position, size, signal
                        )

                else:
                    # Exit conditions with stability
                    if self._check_exit_conditions(signal, qualia_metrics, position):
                        trades.append(self._close_trade(
                            current_trade, market_data['closes'][-1],
                            position, trading_cost
                        ))
                        capital += trades[-1].pnl
                        position = 0
                        current_trade = None

            except (ValueError, np.linalg.LinAlgError) as e:
                # Log error and continue with next iteration
                print(f"Error processing timestep {i}: {e}")
                continue

        # Calculate final metrics with stability
        try:
            return self._calculate_backtest_results(
                initial_capital, capital, trades
            )
        except Exception as e:
            raise ValueError(f"Error calculating backtest results: {e}")

    def _validate_historical_data(self, data: Dict[str, List[float]]) -> None:
        """Validate historical data structure and values"""
        required_keys = ['opens', 'highs', 'lows', 'closes', 'volumes']
        if not all(key in data for key in required_keys):
            raise ValueError("Missing required data fields")

        if not all(len(data[key]) > 0 for key in required_keys):
            raise ValueError("Empty data arrays")

        if not all(len(data[key]) == len(data['closes']) for key in required_keys):
            raise ValueError("Inconsistent array lengths")

    def _prepare_market_data(self, historical_data: Dict[str, List[float]], index: int) -> Dict:
        """Prepare market data with stability checks"""
        return {
            'prices': historical_data['closes'][:index+1],
            'volumes': historical_data['volumes'][:index+1],
            'highs': historical_data['highs'][:index+1],
            'lows': historical_data['lows'][:index+1],
            'closes': historical_data['closes'][:index+1],
            'qualia_state': self.qualia.get_metrics()
        }

    def _check_entry_conditions(self, signal: ScalpSignal, qualia_metrics: Dict) -> bool:
        """Check entry conditions with stability"""
        try:
            return (signal.direction != 0 and 
                   signal.confidence >= self.scalper.field.coherence_threshold and
                   qualia_metrics['coherence'] >= self.scalper.coherence_threshold)
        except (AttributeError, KeyError):
            return False

    def _calculate_position_size(self, capital: float, risk: float, signal: ScalpSignal, market_data: Dict) -> float:
        """Calculate position size with stability"""
        try:
            size = self.scalper.get_position_size(
                capital=capital,
                risk_per_trade=risk,
                signal=signal
            )
            return float(np.clip(size, 0.0, capital))
        except (ValueError, AttributeError):
            return 0.0

    def _create_trade_entry(self, entry_price: float, position: int, size: float, signal: ScalpSignal) -> Dict:
        """Create trade entry with stability checks"""
        return {
            'entry_time': datetime.now(),
            'entry_price': float(entry_price),
            'direction': int(position),
            'size': float(size),
            'field_coherence': float(getattr(signal, 'field_coherence', 0.0)),
            'phi_resonance': float(getattr(signal, 'phi_resonance', 0.0)),
            'pattern_id': str(getattr(signal, 'pattern_id', ''))
        }

    def _check_exit_conditions(self, signal: ScalpSignal, qualia_metrics: Dict, position: int) -> bool:
        """Check exit conditions with stability"""
        try:
            return (signal.direction == -position or
                   signal.confidence < self.scalper.field.coherence_threshold or
                   qualia_metrics['coherence'] < self.scalper.coherence_threshold)
        except (AttributeError, KeyError):
            return True

    def _close_trade(self, trade: Dict, exit_price: float, position: int, cost: float) -> TradeResult:
        """Close trade with stability checks"""
        exit_price = float(exit_price)
        pnl = position * (exit_price - trade['entry_price']) * trade['size']
        pnl *= (1.0 - cost)

        return TradeResult(
            entry_time=trade['entry_time'],
            exit_time=datetime.now(),
            entry_price=trade['entry_price'],
            exit_price=exit_price,
            direction=trade['direction'],
            size=trade['size'],
            pnl=float(pnl),
            field_coherence=trade['field_coherence'],
            phi_resonance=trade['phi_resonance'],
            pattern_id=trade['pattern_id']
        )

    def _calculate_backtest_results(self, initial_capital: float, final_capital: float, trades: List[TradeResult]) -> BacktestResult:
        """Calculate backtest results with stability"""
        try:
            winning_trades = len([t for t in trades if t.pnl > 0])
            losing_trades = len([t for t in trades if t.pnl <= 0])
            total_pnl = sum(t.pnl for t in trades)

            equity_curve = self._calculate_equity_curve(initial_capital, trades)
            max_drawdown = self._calculate_max_drawdown(equity_curve)

            # Calculate Sharpe with stability
            returns = np.diff(equity_curve) / (equity_curve[:-1] + self.epsilon)
            returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

            sharpe = (np.mean(returns) / (np.std(returns) + self.epsilon)) * np.sqrt(252) if len(returns) > 0 else 0

            # Calculate field metrics with stability
            field_coherence = np.mean([t.field_coherence for t in trades]) if trades else 0
            phi_resonance = np.mean([t.phi_resonance for t in trades]) if trades else 0

            # Get patterns with stability
            patterns = []
            if hasattr(self.scalper.field, 'patterns'):
                patterns = self.scalper.field.patterns

            return BacktestResult(
                initial_capital=initial_capital,
                final_capital=final_capital,
                total_trades=len(trades),
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                total_pnl=total_pnl,
                max_drawdown=max_drawdown,
                sharpe_ratio=float(sharpe),
                field_coherence_mean=float(field_coherence),
                phi_resonance_mean=float(phi_resonance),
                trades=trades,
                metrics_history=self.metrics_history,
                patterns=patterns
            )
        except Exception as e:
            raise ValueError(f"Error in backtest results calculation: {e}")

    def _calculate_equity_curve(self, initial_capital: float, trades: List[TradeResult]) -> np.ndarray:
        """Calculate equity curve with stability"""
        try:
            equity = [initial_capital]
            for trade in trades:
                equity.append(equity[-1] + float(trade.pnl))
            return np.array(equity, dtype=np.float64)
        except Exception:
            return np.array([initial_capital], dtype=np.float64)

    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """Calculate max drawdown with stability"""
        try:
            peaks = np.maximum.accumulate(equity_curve)
            drawdowns = (peaks - equity_curve) / (peaks + self.epsilon)
            drawdowns = np.nan_to_num(drawdowns, nan=0.0, posinf=1.0, neginf=0.0)
            return float(np.clip(np.max(drawdowns), 0.0, 1.0))
        except Exception:
            return 0.0

    def analyze_results(self, results: BacktestResult) -> Dict[str, Any]:
        """Analyze backtest results with stability"""
        try:
            return {
                'performance': {
                    'total_return': (results.final_capital - results.initial_capital) / 
                                  (results.initial_capital + self.epsilon),
                    'win_rate': results.winning_trades / 
                               (results.total_trades + self.epsilon) if results.total_trades > 0 else 0,
                    'profit_factor': self._calculate_profit_factor(results.trades),
                    'max_drawdown': results.max_drawdown,
                    'sharpe_ratio': results.sharpe_ratio
                },
                'quantum_metrics': {
                    'field_coherence': results.field_coherence_mean,
                    'phi_resonance': results.phi_resonance_mean,
                    'pattern_count': len(results.patterns),
                    'qualia_metrics': self.qualia.get_metrics()
                },
                'trade_analysis': {
                    'avg_trade_duration': self._calculate_avg_duration(results.trades),
                    'avg_profit_per_trade': np.mean([t.pnl for t in results.trades]) if results.trades else 0,
                    'best_trade': max(results.trades, key=lambda t: t.pnl).pnl if results.trades else 0,
                    'worst_trade': min(results.trades, key=lambda t: t.pnl).pnl if results.trades else 0
                }
            }
        except Exception as e:
            # Return safe default values on error
            return {
                'performance': {'error': str(e)},
                'quantum_metrics': {'error': str(e)},
                'trade_analysis': {'error': str(e)}
            }

    def _calculate_profit_factor(self, trades: List[TradeResult]) -> float:
        """Calculate profit factor with stability"""
        try:
            wins = sum(t.pnl for t in trades if t.pnl > 0)
            losses = abs(sum(t.pnl for t in trades if t.pnl < 0))
            return float(wins / (losses + self.epsilon))
        except Exception:
            return 0.0

    def _calculate_avg_duration(self, trades: List[TradeResult]) -> float:
        """Calculate average trade duration with stability"""
        try:
            if not trades:
                return 0.0
            durations = [(t.exit_time - t.entry_time).total_seconds() for t in trades]
            return float(np.mean(durations))
        except Exception:
            return 0.0

    def _calculate_equity_curve(self, initial_capital: float, trades: List[TradeResult]) -> np.ndarray:
        """Calculate equity curve with stability"""
        try:
            equity = [initial_capital]
            for trade in trades:
                equity.append(equity[-1] + float(trade.pnl))
            return np.array(equity, dtype=np.float64)
        except Exception:
            return np.array([initial_capital], dtype=np.float64)

    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """Calculate max drawdown with stability"""
        try:
            peaks = np.maximum.accumulate(equity_curve)
            drawdowns = (peaks - equity_curve) / (peaks + self.epsilon)
            drawdowns = np.nan_to_num(drawdowns, nan=0.0, posinf=1.0, neginf=0.0)
            return float(np.clip(np.max(drawdowns), 0.0, 1.0))
        except Exception:
            return 0.0