"""
Quantum Scalp Trading Module
"""
from typing import Dict, Any, Optional, Union, List
from collections import deque
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime

from qualia_core.quantum.quantum_state import QuantumState, GeometricPattern, QualiaState, PHI, phi_normalize, generate_field
from qualia_core.quantum.quantum_computer import QuantumComputer, QUALIA, QuantumCGRConsciousness
from qualia_core.quantum.quantum_memory import QuantumMemory
from qualia_core.quantum.morphic_memory import MorphicMemory, MorphicField
from qualia_core.quantum.insights_analyzer import ConsciousnessMonitor, QualiaMetrics, calculate_field_metrics
from qualia_core.quantum.dark_pool_analyzer import DarkPoolAnalyzer, DarkLiquidityMetrics, analyze_dark_flow, detect_dark_patterns
from qualia_core.quantum.quantum_computer import QuantumCGRConsciousness

from .qualia_scalping import QualiaScalpingIntegration

@dataclass
class ScalpSignal:
    """Sinal de trading identificado"""
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    field_strength: float
    phi_resonance: float
    atr_volatility: float
    dark_liquidity: float
    dark_flow: float
    timestamp: datetime = field(default_factory=datetime.now)
    direction: int = 0
    pattern_id: Optional[str] = None

@dataclass
class ScalpingMetrics:
    """Métricas de scalping"""
    timestamp: datetime
    atr: float
    volatility: float
    field_strength: float
    phi_resonance: float
    pattern_count: int
    dark_metrics: DarkLiquidityMetrics

class QuantumScalper:
    """Implementa scalp trading quântico usando QUALIA."""

    def __init__(self):
        """Inicializa scalper"""
        # Core components
        self.field = MorphicField(
            field=np.zeros(8),  # Campo vazio inicial
            resonance=0.5,      # Ressonância padrão
            coherence=0.5,      # Coerência padrão
            timestamp=datetime.now().timestamp()  # Timestamp atual
        )
        self.monitor = ConsciousnessMonitor()
        self.dimensions = 8

        # Enhanced pattern detection thresholds
        self.coherence_threshold = 0.45  # Further relaxed for better signal detection
        self.resonance_threshold = 0.4   # Further relaxed for better pattern recognition
        self.epsilon = 1e-10
        self.max_history = 1000
        self.atr_period = 14
        self.phi_weight = PHI / (1 + PHI)

        # Initialize state and buffers
        self.history = []
        self.state = {
            'field_strength': 0.0,
            'coherence': 0.0,
            'resonance': 0.0,
            'dark_liquidity': 0.0,
            'dark_flow': 0.0,
            'phi_alignment': 0.5  # Initial neutral alignment
        }

        # Initialize metric buffers
        self.signals = []
        self.metrics = []
        self.metrics_buffer = {
            'field_coherence': deque(maxlen=self.max_history),
            'phi_resonance': deque(maxlen=self.max_history),
            'pattern_strength': deque(maxlen=self.max_history),
            'signal_confidence': deque(maxlen=self.max_history),
            'pnl': deque(maxlen=self.max_history),
            'drawdown': deque(maxlen=self.max_history),
            'phi_alignment': deque(maxlen=self.max_history)
        }

        # Market data buffers
        self.price_buffer = deque(maxlen=self.max_history)
        self.patterns = []
        self.dark_analyzer = DarkPoolAnalyzer()
        self.dark_flow_buffer = deque(maxlen=self.max_history)

        try:
            self._initialize_quantum_components()
        except Exception as e:
            print(f"Warning: Failed to initialize quantum components: {e}")
            self._initialize_default_components()

    def _initialize_default_components(self):
        """Initialize default components for fallback"""
        self.morphic_field = np.zeros((self.dimensions, self.dimensions))
        self.metrics = {
            'field_coherence': 0.0,
            'phi_resonance': 0.0,
            'pattern_strength': 0.0,
            'dark_liquidity': 0.0,
            'dark_flow': 0.0,
            'phi_alignment': 0.5
        }

    def _calculate_phi_alignment(self, prices: List[float]) -> float:
        """Calculate phi-based pattern alignment with enhanced stability"""
        try:
            if len(prices) < 3:
                return 0.5

            # Convert to numpy array and handle price scaling
            prices = np.array(prices, dtype=np.float64)
            prices = np.nan_to_num(prices, nan=0.0, posinf=1e10, neginf=-1e10)

            # Calculate returns with volatility normalization
            log_returns = np.diff(np.log(prices + self.epsilon))
            if len(log_returns) < 2:
                return 0.5

            # Calculate volatility-adjusted returns
            volatility = np.std(log_returns) if len(log_returns) > 1 else 1.0
            norm_returns = log_returns / (volatility + self.epsilon)

            # Multiple timeframe resonance
            alignments = []
            for window in [3, 5, 8, 13]:  # Fibonacci windows
                if len(norm_returns) >= window:
                    # Calculate phi-weighted momentum
                    weights = np.array([PHI ** (-i) for i in range(window)])
                    weights = weights / np.sum(weights)
                    windowed_returns = norm_returns[-window:]
                    momentum = np.sum(windowed_returns * weights[:window])

                    # Calculate resonance with phi levels
                    phi_levels = np.array([PHI ** i for i in range(-2, 3)])
                    resonances = np.exp(-np.abs(momentum - phi_levels))
                    alignments.append(np.max(resonances))

            if not alignments:
                return 0.5

            # Combine alignments with phi weighting
            weights = np.array([PHI ** (-i) for i in range(len(alignments))])
            weights = weights / np.sum(weights)
            combined_alignment = np.sum(alignments * weights)

            # Enhance and normalize alignment
            enhanced = 0.5 + (0.5 * combined_alignment)
            return float(np.clip(enhanced, 0.5, 1.0))

        except Exception as e:
            print(f"Error calculating phi alignment: {e}")
            return 0.5

    def analyze_market(self, market_data: Dict[str, Any]) -> ScalpSignal:
        """Analyze market and generate signal with enhanced pattern detection"""
        try:
            # Update price buffer
            if 'closes' in market_data and len(market_data['closes']) > 0:
                current_price = float(market_data['closes'][-1])
                self.price_buffer.append(current_price)

            # Update state with stability checks
            state = self.update_state(market_data)

            # Calculate phi alignment
            phi_alignment = self._calculate_phi_alignment(list(self.price_buffer))
            state['phi_alignment'] = phi_alignment
            self.metrics_buffer['phi_alignment'].append(phi_alignment)

            # Enhanced coherence check
            coherence_check = (
                state['coherence'] * 0.35 +
                state['resonance'] * 0.35 +
                phi_alignment * 0.3
            )

            if coherence_check < self.coherence_threshold or \
               state['resonance'] < self.resonance_threshold:
                return self._generate_neutral_signal(current_price)

            # Calculate ATR and field strength
            atr = self._calculate_atr(market_data)
            field_strength = (
                state['field_strength'] * 0.35 +
                phi_alignment * 0.35 +
                coherence_check * 0.3
            )

            # Create signal
            signal = ScalpSignal(
                entry_price=current_price,
                stop_loss=current_price - (atr * PHI),
                take_profit=current_price + (atr * PHI * PHI),
                confidence=float(coherence_check),
                field_strength=float(field_strength),
                phi_resonance=float(phi_alignment),
                atr_volatility=float(atr),
                dark_liquidity=float(state['dark_liquidity']),
                dark_flow=float(state['dark_flow']),
                direction=int(self._calculate_signal_direction(state)),
                pattern_id=None
            )

            # Update metrics
            self._update_drawdown(signal)
            self._track_signal(signal)

            # Track field coherence
            self.metrics_buffer['field_coherence'].append(field_strength)
            self.metrics_buffer['phi_resonance'].append(phi_alignment)

            return signal

        except Exception as e:
            print(f"Error analyzing market: {e}")
            return self._generate_neutral_signal(market_data.get('closes', [0.0])[-1])

    def _update_drawdown(self, signal: ScalpSignal):
        """Update drawdown metrics with enhanced volatility tracking"""
        try:
            if len(self.signals) < 2:
                return

            # Calculate returns and volatility
            prices = np.array([s.entry_price for s in self.signals])
            directions = np.array([s.direction for s in self.signals])
            returns = np.diff(np.log(prices + self.epsilon)) * directions[1:]

            # Volatility-adjusted PnL
            volatility = np.std(returns) if len(returns) > 1 else 1.0
            norm_returns = returns / (volatility + self.epsilon)

            # Calculate cumulative returns
            weights = np.array([PHI ** (-i) for i in range(len(norm_returns))])
            weights = weights / np.sum(weights)
            weighted_returns = norm_returns * weights[:len(norm_returns)]
            cumulative_returns = np.exp(np.cumsum(weighted_returns)) - 1

            # Calculate drawdown
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (peak - cumulative_returns) / (np.abs(peak) + self.epsilon)
            max_drawdown = float(np.clip(np.max(drawdown), 0.0, 1.0))

            # Update metrics buffers
            self.metrics_buffer['drawdown'].append(max_drawdown)
            self.metrics_buffer['pnl'].append(
                float(weighted_returns[-1]) if len(weighted_returns) > 0 else 0.0
            )

        except Exception as e:
            print(f"Error updating drawdown: {e}")

    def update_state(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Update state with enhanced stability and resonance tracking"""
        try:
            if not self._validate_market_data(market_data):
                return self.state.copy()

            # Update history
            self.history.append(market_data)
            if len(self.history) > self.max_history:
                self.history.pop(0)

            # Get component updates
            field_metrics = self._safe_field_update(market_data)
            consciousness_metrics = self._safe_consciousness_update(market_data)
            qualia_metrics = self._safe_qualia_update(market_data)

            # Calculate phi alignment
            phi_alignment = self._calculate_phi_alignment(
                [d.get('closes', [0.0])[-1] for d in self.history[-13:]]
            )

            # Update state
            new_state = {
                'field_strength': self._validate_metric(field_metrics.get('field_energy', 0.0)),
                'coherence': self._validate_metric(consciousness_metrics.get('coherence', 0.0)),
                'resonance': self._validate_metric(consciousness_metrics.get('resonance', 0.0)),
                'dark_liquidity': self._validate_metric(qualia_metrics.get('dark_liquidity', 0.0)),
                'dark_flow': self._validate_metric(qualia_metrics.get('dark_flow', 0.0)),
                'phi_alignment': self._validate_metric(phi_alignment)
            }

            self.state = new_state
            return new_state

        except Exception as e:
            print(f"Error updating state: {e}")
            return self.state.copy()

    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics with enhanced stability"""
        try:
            metrics = self.state.copy()

            # Calculate field coherence mean
            if len(self.metrics_buffer['field_coherence']) > 0:
                coherence_values = np.array(list(self.metrics_buffer['field_coherence']))
                weights = np.array([PHI ** (-i) for i in range(len(coherence_values))])
                weights = weights / np.sum(weights)
                metrics['field_coherence_mean'] = float(np.sum(coherence_values * weights))
            else:
                metrics['field_coherence_mean'] = 0.0

            # Calculate maximum drawdown
            if len(self.metrics_buffer['drawdown']) > 0:
                metrics['max_drawdown'] = float(np.max(list(self.metrics_buffer['drawdown'])))
            else:
                metrics['max_drawdown'] = 0.0

            return metrics

        except Exception:
            return self.state.copy()

    def _safe_field_update(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Update field with enhanced stability"""
        try:
            if self.field is None:
                return {'field_energy': 0.0}

            result = self.field.update(market_data)
            metrics = {k: self._validate_metric(v) for k, v in result.items()}

            if len(self.price_buffer) > 2:
                phi_factor = self._calculate_phi_alignment(list(self.price_buffer))
                field_energy = metrics.get('field_energy', 0.0)
                metrics['field_energy'] = (field_energy * 0.6 + phi_factor * 0.4)

            return metrics
        except Exception:
            return {'field_energy': 0.0}

    def _safe_consciousness_update(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Update consciousness with stability"""
        try:
            if self.consciousness is None:
                return {'coherence': 0.0, 'resonance': 0.0}
            result = self.consciousness.update(market_data)
            return {k: self._validate_metric(v) for k, v in result.items()}
        except Exception:
            return {'coherence': 0.0, 'resonance': 0.0}

    def _safe_qualia_update(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Update QUALIA with stability"""
        try:
            if self.qualia is None:
                return {'dark_liquidity': 0.0, 'dark_flow': 0.0}
            result = self.qualia.update(market_data)
            if not isinstance(result, dict):
                return {'dark_liquidity': 0.0, 'dark_flow': 0.0}
            return {k: self._validate_metric(v) for k, v in result.items()}
        except Exception:
            return {'dark_liquidity': 0.0, 'dark_flow': 0.0}

    def _track_signal(self, signal: ScalpSignal) -> None:
        """Track trading signal"""
        try:
            self.signals.append(signal)
            if len(self.signals) > self.max_history:
                self.signals.pop(0)

        except Exception as e:
            print(f"Error tracking signal: {e}")

    def _calculate_signal_direction(self, state: Dict[str, float]) -> float:
        """Calculate signal direction with enhanced stability"""
        try:
            direction_factors = [
                state['field_strength'] - 0.5,
                (state['dark_flow'] - 0.5) * 2,
                state['resonance'] - 0.5,
                state['phi_alignment'] - 0.5
            ]

            weights = [0.35, 0.25, 0.25, 0.15]
            combined = sum(f * w for f, w in zip(direction_factors, weights))

            if abs(combined) < self.epsilon:
                return 0.0

            direction = np.sign(combined)
            confidence = abs(combined)

            return float(direction) if confidence > self.epsilon else 0.0

        except Exception:
            return 0.0

    def _calculate_signal_confidence(self, state: Dict[str, float]) -> float:
        """Calculate signal confidence with stability"""
        try:
            confidence = (
                state['coherence'] * 0.35 +
                state['resonance'] * 0.25 +
                state['field_strength'] * 0.25 +
                state['phi_alignment'] * 0.15
            )

            return self._validate_metric(confidence)
        except Exception:
            return 0.0

    def _generate_neutral_signal(self, current_price: float = 0.0) -> ScalpSignal:
        """Generate neutral signal"""
        try:
            return ScalpSignal(
                entry_price=float(current_price),
                stop_loss=float(current_price * 0.99),
                take_profit=float(current_price * 1.01),
                confidence=0.0,
                field_strength=0.0,
                phi_resonance=0.5,
                atr_volatility=0.0,
                dark_liquidity=0.0,
                dark_flow=0.0,
                direction=0,
                pattern_id=None
            )
        except Exception:
            return ScalpSignal(
                entry_price=0.0,
                stop_loss=0.0,
                take_profit=0.0,
                confidence=0.0,
                field_strength=0.0,
                phi_resonance=0.5,
                atr_volatility=0.0,
                dark_liquidity=0.0,
                dark_flow=0.0,
                direction=0,
                pattern_id=None
            )

    def _validate_metric(self, value: float) -> float:
        """Validate and bound metric values"""
        try:
            if np.isnan(value) or np.isinf(value):
                return 0.0
            return float(np.clip(value, 0.0, 1.0))
        except Exception:
            return 0.0

    def _calculate_atr(self, data: Dict[str, Any]) -> float:
        """Calculate ATR with stability"""
        try:
            if len(data.get('highs', [])) < self.atr_period:
                return 0.0

            highs = np.array(data['highs'][-self.atr_period:], dtype=np.float64)
            lows = np.array(data['lows'][-self.atr_period:], dtype=np.float64)
            closes = np.array(data['closes'][-self.atr_period:], dtype=np.float64)

            highs = np.nan_to_num(highs, nan=0.0, posinf=np.inf, neginf=-np.inf)
            lows = np.nan_to_num(lows, nan=0.0, posinf=np.inf, neginf=-np.inf)
            closes = np.nan_to_num(closes, nan=0.0, posinf=np.inf, neginf=-np.inf)

            tr1 = highs - lows
            tr2 = np.abs(highs - np.roll(closes, 1))
            tr3 = np.abs(lows - np.roll(closes, 1))

            true_ranges = np.maximum(tr1, np.maximum(tr2, tr3))
            true_ranges[0] = tr1[0]

            weights = np.array([PHI ** (-i) for i in range(len(true_ranges))])
            weights = weights / (np.sum(weights) + self.epsilon)

            return float(np.clip(np.sum(true_ranges * weights), 0.0, np.inf))

        except Exception:
            return 0.0

    def clear_history(self) -> None:
        """Clear historical data with stability"""
        try:
            self.history = []
            self.signals = []
            self.metrics = []
            self.metrics_buffer = {
                'field_coherence': deque(maxlen=self.max_history),
                'phi_resonance': deque(maxlen=self.max_history),
                'pattern_strength': deque(maxlen=self.max_history),
                'signal_confidence': deque(maxlen=self.max_history),
                'pnl': deque(maxlen=self.max_history),
                'drawdown': deque(maxlen=self.max_history),
                'phi_alignment': deque(maxlen=self.max_history)
            }
            self.price_buffer.clear()
            if self.field is not None:
                self.field.clear_history()
        except Exception as e:
            print(f"Error clearing history: {e}")

    def _initialize_quantum_components(self):
        """Initialize quantum trading components with stability"""
        try:
            # Initialize morphic field with validation
            self.morphic_field = generate_field(self.dimensions)
        except Exception:
            self.morphic_field = np.zeros((self.dimensions, self.dimensions))

        self.metrics = {
            'field_coherence': 0.0,
            'phi_resonance': 0.0,
            'pattern_strength': 0.0,
            'dark_liquidity': 0.0,
            'dark_flow': 0.0,
            'phi_alignment': 0.5  # Initialize with neutral alignment
        }
        self.qualia_integration = QualiaScalpingIntegration(
            field_dimensions=self.dimensions,
            coherence_threshold=self.coherence_threshold,
            resonance_threshold=self.resonance_threshold
        )
        self.consciousness = QuantumCGRConsciousness(
            field_dimensions=self.dimensions,
            coherence_threshold=self.coherence_threshold,
            retrocausal_depth=21,
            max_history=self.max_history
        )
        self.qualia = QUALIA(
            consciousness=self.consciousness,
            field=self.field
        )
        self.field = MorphicField(
            field_dimensions=self.dimensions,
            field_strength=0.8,
            coherence_threshold=self.coherence_threshold,
            resonance_threshold=self.resonance_threshold,
            max_history=self.max_history
        )

    def _safe_consciousness_update(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Update consciousness with stability"""
        try:
            if self.consciousness is None:
                return {'coherence': 0.0, 'resonance': 0.0}
            result = self.consciousness.update(market_data)
            return {k: self._validate_metric(v) for k, v in result.items()}
        except Exception:
            return {'coherence': 0.0, 'resonance': 0.0}

    def _safe_qualia_update(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Update QUALIA with stability"""
        try:
            if self.qualia is None:
                return {'dark_liquidity': 0.0, 'dark_flow': 0.0}
            result = self.qualia.update(market_data)
            if not isinstance(result, dict):
                return {'dark_liquidity': 0.0, 'dark_flow': 0.0}
            return {k: self._validate_metric(v) for k, v in result.items()}
        except Exception:
            return {'dark_liquidity': 0.0, 'dark_flow': 0.0}

    def _track_signal(self, signal: ScalpSignal) -> None:
        """Track trading signal"""
        try:
            self.signals.append(signal)
            if len(self.signals) > self.max_history:
                self.signals.pop(0)

        except Exception as e:
            print(f"Error tracking signal: {e}")

    def configure(self,
                 field_dimensions: int = 8,
                 coherence_threshold: float = 0.75,
                 resonance_threshold: float = 0.7,
                 max_history: int = 1000,
                 atr_period: int = 14):
        """Configure scalper parameters"""
        # Update parameters
        self.dimensions = field_dimensions
        self.coherence_threshold = coherence_threshold
        self.resonance_threshold = resonance_threshold
        self.max_history = max_history
        self.atr_period = atr_period

        # Reset metrics and re-initialize components
        self._initialize_quantum_components()

    def get_position_size(self,
                         capital: float,
                         risk_per_trade: float,
                         signal: ScalpSignal) -> float:
        """Calculate position size with stability"""
        try:
            risk_amount = capital * risk_per_trade
            price_risk = abs(signal.entry_price - signal.stop_loss)

            if price_risk < self.epsilon:
                return 0.0

            # Adjust size by confidence and resonance
            position_size = (risk_amount / price_risk) * signal.confidence * signal.phi_resonance

            # Apply stability bounds
            return float(np.clip(position_size, 0.0, capital))
        except (ValueError, ZeroDivisionError, AttributeError, TypeError):
            return 0.0

    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown with stability"""
        try:
            if not self.metrics_buffer['pnl']:
                return 0.0

            equity_curve = np.cumsum(list(self.metrics_buffer['pnl']))
            peaks = np.maximum.accumulate(equity_curve)
            drawdowns = (peaks - equity_curve) / (peaks + self.epsilon)

            return float(np.clip(np.max(drawdowns), 0.0, 1.0))
        except Exception:
            return 0.0
    def get_signals(self,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   min_strength: Optional[float] = None) -> List[ScalpSignal]:
        """
        Retorna sinais filtrados

        Args:
            start_time: Tempo inicial opcional
            end_time: Tempo final opcional
            min_strength: Força mínima opcional

        Returns:
            Lista de sinais filtrada
        """
        try:
            filtered = self.signals

            if start_time:
                filtered = [s for s in filtered if s.timestamp >= start_time]

            if end_time:
                filtered = [s for s in filtered if s.timestamp <= end_time]

            if min_strength:
                filtered = [s for s in filtered if s.field_strength >= min_strength]

            return filtered
        except Exception:
            return []
    def get_metrics(self,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None) -> List[ScalpingMetrics]:
        """
        Retorna métricas filtradas por tempo

        Args:
            start_time: Tempo inicial opcional
            end_time: Tempo final opcional

        Returns:
            Lista de métricas filtrada
        """
        try:
            filtered = self.metrics

            if start_time:
                filtered = [m for m in filtered if m.timestamp >= start_time]

            if end_time:
                filtered = [m for m in filtered if m.timestamp <= end_time]

            return filtered
        except Exception:
            return []

    def _validate_market_data(self, data: Dict[str, Any]) -> bool:
        """Validate market data structure"""
        try:
            required = ['closes', 'volumes', 'highs', 'lows']

            # Check required fields
            if not all(k in data for k in required):
                return False

            # Validate data types
            if not all(isinstance(data[k], (list, np.ndarray)) for k in required):
                return False

            # Check for non-empty arrays
            if not all(len(data[k]) > 0 for k in required):
                return False

            # Validate numeric values
            for k in required:
                if not all(isinstance(x, (int, float)) for x in data[k]):
                    return False
                if not all(abs(x) < 1e10 for x in data[k]):  # Prevent extreme values
                    return False

            return True

        except Exception:
            return False