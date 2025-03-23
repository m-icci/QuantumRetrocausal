"""
Integração QUALIA com Scalping Quântico
"""
from typing import Dict, Optional, List, Tuple
import numpy as np
from datetime import datetime

from qualia_core.quantum.quantum_state import GeometricPattern, QualiaState, PHI, phi_normalize, generate_field
from qualia_core.quantum.quantum_computer import QUALIA
from qualia_core.quantum.insights_analyzer import ConsciousnessMonitor as QualiaMonitor
from qualia_core.quantum.morphic_memory import MorphicField as QualiaField

class QualiaScalpingIntegration:
    """Integra QUALIA com estratégia de scalping usando M-ICCI"""

    def __init__(self, 
                 field_dimensions: int = 8,
                 coherence_threshold: float = 0.45,  # Relaxed for better detection
                 resonance_threshold: float = 0.4,   # Relaxed for better resonance
                 buffer_size: int = 1000,
                 epsilon: float = 1e-10):
        """Inicializa integração QUALIA-Scalping"""
        self.field = QualiaField(
            field=np.zeros(field_dimensions),  # Campo vazio inicial
            resonance=resonance_threshold,     # Ressonância inicial
            coherence=coherence_threshold,     # Coerência inicial
            timestamp=datetime.now().timestamp()  # Timestamp atual
        )
        self.monitor = QualiaMonitor()
        self.dimensions = field_dimensions
        self.coherence_threshold = coherence_threshold
        self.resonance_threshold = resonance_threshold
        self.epsilon = epsilon

        # Enhanced buffers for pattern detection
        self.price_buffer = []
        self.volume_buffer = []
        self.buffer_size = buffer_size
        self.phi_buffer = []
        self.oscillation_buffer = []  # Track oscillating patterns

        # Initialize field with safety
        try:
            self.morphic_field = generate_field(field_dimensions)
        except Exception:
            self.morphic_field = np.zeros((field_dimensions, field_dimensions))

        # Initialize metrics
        self.consciousness_metrics = {
            'field_coherence': 0.0,
            'phi_resonance': 0.5,
            'pattern_strength': 0.0,
            'dark_liquidity': 0.0,
            'dark_flow': 0.0,
            'coherence_time': 0.0,
            'phi_alignment': 0.5  # Initialize with neutral alignment
        }

    def update_buffers(self, price: float, volume: float) -> None:
        """Update buffers with enhanced pattern tracking"""
        try:
            price = float(price)
            volume = float(volume)

            # Update main buffers
            if len(self.price_buffer) >= self.buffer_size:
                self.price_buffer.pop(0)
                self.volume_buffer.pop(0)
                if self.phi_buffer:
                    self.phi_buffer.pop(0)
                if self.oscillation_buffer:
                    self.oscillation_buffer.pop(0)

            self.price_buffer.append(price)
            self.volume_buffer.append(volume)

            # Calculate and track oscillation metrics
            if len(self.price_buffer) >= 3:
                oscillation = self._calculate_oscillation_strength(self.price_buffer[-3:])
                self.oscillation_buffer.append(oscillation)

            # Calculate phi pattern
            if len(self.price_buffer) > 1:
                phi_pattern = self._calculate_phi_pattern(self.price_buffer[-2:])
                self.phi_buffer.append(phi_pattern)

        except (TypeError, ValueError):
            return

    def _calculate_oscillation_strength(self, prices: List[float]) -> float:
        """Calculate oscillation strength with phi-based pattern recognition"""
        try:
            if len(prices) < 3:
                return 0.0

            # Convert to array and calculate differences
            prices = np.array(prices, dtype=np.float64)
            diff1 = prices[1] - prices[0]
            diff2 = prices[2] - prices[1]

            # Detect direction changes (oscillation)
            direction_change = np.sign(diff1) != np.sign(diff2)

            # Calculate strength using golden ratio proportions
            if direction_change:
                ratio = min(abs(diff1), abs(diff2)) / (max(abs(diff1), abs(diff2)) + self.epsilon)
                strength = ratio * PHI
                return float(np.clip(strength, 0.0, 1.0))

            return 0.0

        except Exception:
            return 0.0

    def _calculate_phi_pattern(self, prices: List[float]) -> float:
        """Calculate phi-based pattern with enhanced oscillation detection"""
        try:
            if len(prices) < 2:
                return 0.5

            # Calculate normalized price movement
            prices = np.array(prices, dtype=np.float64)
            movement = (prices[-1] - prices[-2]) / (prices[-2] + self.epsilon)

            # Calculate resonance with multiple phi levels
            phi_levels = [PHI ** i for i in range(-3, 4)]
            resonances = [np.exp(-np.abs(movement - level)) for level in phi_levels]

            # Weight resonances by phi
            weights = np.array([PHI ** (-i) for i in range(len(resonances))])
            weights = weights / np.sum(weights)

            # Calculate weighted resonance
            phi_resonance = np.sum(resonances * weights)

            # Enhance resonance based on oscillation pattern
            if len(self.oscillation_buffer) > 0:
                oscillation_factor = np.mean(self.oscillation_buffer[-3:]) if len(self.oscillation_buffer) >= 3 else 0.0
                phi_resonance = phi_resonance * (1.0 + oscillation_factor)

            return float(np.clip(0.5 + (0.5 * phi_resonance), 0.5, 1.0))

        except Exception:
            return 0.5

    def analyze_market(self, market_data: Dict) -> QualiaState:
        """Analyze market with enhanced pattern recognition"""
        try:
            # Validate market data
            if not self._validate_market_data(market_data):
                return self._generate_default_state()

            # Update buffers
            self.update_buffers(
                float(market_data['closes'][-1]),
                float(market_data['volumes'][-1])
            )

            # Convert data to tensor
            market_tensor = self._market_to_tensor(market_data)
            market_tensor = np.nan_to_num(market_tensor, nan=0.0, posinf=1.0, neginf=-1.0)

            # Evolve state
            state = self.field.evolve(market_tensor)

            # Update field and metrics
            self.morphic_field = self._update_morphic_field(state)
            self.consciousness_metrics = self._calculate_consciousness_metrics(state)

            return state if isinstance(state, QualiaState) else self._generate_default_state()

        except Exception as e:
            print(f"Error in market analysis: {e}")
            return self._generate_default_state()

    def _generate_default_state(self) -> QualiaState:
        """Generate neutral state"""
        return QualiaState(
            geometric_coherence=0.0,
            philosophical_resonance=0.5,
            consciousness_level=0.0,
            field_tensor=np.zeros((self.dimensions, self.dimensions))
        )

    def _market_to_tensor(self, market_data: Dict) -> np.ndarray:
        """Convert market data to quantum tensor"""
        try:
            # Extract and validate data
            opens = np.array(market_data['opens'], dtype=np.float64)
            highs = np.array(market_data['highs'], dtype=np.float64)
            lows = np.array(market_data['lows'], dtype=np.float64)
            closes = np.array(market_data['closes'], dtype=np.float64)
            volumes = np.array(market_data['volumes'], dtype=np.float64)

            # Handle invalid values
            arrays = [opens, highs, lows, closes, volumes]
            for arr in arrays:
                arr = np.nan_to_num(arr, nan=0.0, posinf=np.inf, neginf=-np.inf)

            # Calculate enhanced metrics
            typical_price = (highs + lows + closes) / 3.0
            price_momentum = np.gradient(typical_price)
            volume_momentum = np.gradient(volumes)

            # Initialize tensor
            tensor = np.zeros((8, 8), dtype=np.float64)

            # Enhanced tensor filling with oscillation awareness
            tensor[0:2, :] = self._normalize_with_phi(typical_price[-8:])
            tensor[2:4, :] = self._normalize_with_phi(price_momentum[-8:])
            tensor[4:6, :] = self._normalize_with_phi(volume_momentum[-8:])
            tensor[6:8, :] = self._normalize_with_phi(volumes[-8:])

            # Add oscillation pattern information
            if len(self.oscillation_buffer) >= 8:
                oscillation_pattern = np.array(self.oscillation_buffer[-8:])
                tensor *= (1.0 + 0.5 * oscillation_pattern.reshape(-1, 1))

            return phi_normalize(tensor)

        except Exception as e:
            print(f"Error in tensor conversion: {e}")
            return np.zeros((8, 8))

    def _normalize_with_phi(self, data: np.ndarray) -> np.ndarray:
        """Normalize data with phi-based scaling"""
        try:
            data = np.array(data, dtype=np.float64)

            if np.all(data == data[0]):
                return np.zeros_like(data)

            data_range = np.max(data) - np.min(data)
            if data_range < self.epsilon:
                return np.zeros_like(data)

            normalized = (data - np.min(data)) / data_range

            # Enhanced phi-based sigmoid
            return PHI / (1.0 + np.exp(-6.0 * (normalized - 0.5)))

        except Exception:
            return np.zeros_like(data)

    def _update_morphic_field(self, state: QualiaState) -> np.ndarray:
        """Update morphic field with enhanced coherence"""
        try:
            field = state.field_tensor.copy()
            field = np.nan_to_num(field, nan=0.0, posinf=1.0, neginf=-1.0)

            # Calculate enhanced metrics
            resonance = self._calculate_morphic_resonance(state)
            coherence = self._calculate_field_coherence(state)

            # Enhanced field update with oscillation awareness
            if len(self.oscillation_buffer) > 0:
                oscillation_factor = np.mean(self.oscillation_buffer)
                resonance *= (1.0 + oscillation_factor)

            # Update field with weighted combination
            updated_field = (
                self.morphic_field * PHI +
                field * resonance * PHI +
                field * coherence
            ) / (PHI * 2 + 1.0 + self.epsilon)

            return phi_normalize(updated_field)

        except Exception:
            return self.morphic_field.copy()

    def _calculate_morphic_resonance(self, state: QualiaState) -> float:
        """Calculate morphic resonance with enhanced pattern detection"""
        try:
            if len(self.phi_buffer) < 2:
                return 0.0

            # Calculate weighted resonance
            resonances = np.array(self.phi_buffer, dtype=np.float64)
            weights = np.array([PHI ** (-i) for i in range(len(resonances))])
            weights = weights / np.sum(weights)

            # Enhanced resonance with oscillation
            base_resonance = np.sum(resonances * weights)
            if len(self.oscillation_buffer) > 0:
                oscillation_factor = np.mean(self.oscillation_buffer)
                base_resonance *= (1.0 + oscillation_factor)

            return float(np.clip(base_resonance, 0.0, 1.0))

        except Exception:
            return 0.0

    def _calculate_field_coherence(self, state: QualiaState) -> float:
        """Calculate field coherence with enhanced stability"""
        try:
            field = np.nan_to_num(state.field_tensor, nan=0.0, posinf=1.0, neginf=-1.0)

            # Calculate base coherence
            coherence = np.abs(np.mean(field)) * PHI
            pattern_strength = np.max(np.abs(field))

            # Enhance with oscillation patterns
            if len(self.oscillation_buffer) > 0:
                oscillation_factor = np.mean(self.oscillation_buffer)
                coherence *= (1.0 + oscillation_factor)

            # Combine metrics with phi weighting
            return float(np.clip(
                (coherence * PHI + pattern_strength) / (PHI + 1.0),
                0.0,
                1.0
            ))

        except Exception:
            return 0.0

    def _calculate_consciousness_metrics(self, state: QualiaState) -> Dict[str, float]:
        """Calculate consciousness metrics with enhanced stability"""
        try:
            coherence = self._calculate_field_coherence(state)
            resonance = self._calculate_morphic_resonance(state)
            pattern_strength = float(np.max(np.abs(state.field_tensor)))

            # Calculate enhanced dark metrics
            dark_metrics = self._calculate_dark_metrics()

            # Calculate phi alignment
            phi_alignment = 0.5
            if len(self.phi_buffer) > 0:
                phi_values = np.array(self.phi_buffer)
                weights = np.array([PHI ** (-i) for i in range(len(phi_values))])
                weights = weights / np.sum(weights)
                phi_alignment = float(np.clip(np.sum(phi_values * weights), 0.5, 1.0))

            # Enhance metrics with oscillation awareness
            if len(self.oscillation_buffer) > 0:
                oscillation_factor = np.mean(self.oscillation_buffer)
                coherence *= (1.0 + oscillation_factor)
                resonance *= (1.0 + oscillation_factor)
                pattern_strength *= (1.0 + oscillation_factor)

            metrics = {
                'field_coherence': coherence,
                'phi_resonance': resonance,
                'pattern_strength': pattern_strength,
                'dark_liquidity': dark_metrics['liquidity'],
                'dark_flow': dark_metrics['flow'],
                'coherence_time': self._calculate_coherence_time(),
                'phi_alignment': phi_alignment
            }

            return {k: float(np.clip(v, 0.0, 1.0)) for k, v in metrics.items()}

        except Exception:
            return {
                'field_coherence': 0.0,
                'phi_resonance': 0.5,
                'pattern_strength': 0.0,
                'dark_liquidity': 0.0,
                'dark_flow': 0.0,
                'coherence_time': 0.0,
                'phi_alignment': 0.5
            }

    def _calculate_coherence_time(self) -> float:
        """Calculate coherence time with enhanced stability"""
        try:
            if len(self.phi_buffer) < 2:
                return 0.0

            coherence_values = np.array(self.phi_buffer)
            weights = np.array([PHI ** (-i) for i in range(len(coherence_values))])
            weights = weights / np.sum(weights)

            # Enhanced coherence time with oscillation
            base_coherence = np.sum(coherence_values * weights)
            if len(self.oscillation_buffer) > 0:
                oscillation_factor = np.mean(self.oscillation_buffer)
                base_coherence *= (1.0 + oscillation_factor)

            return float(np.clip(base_coherence, 0.0, 1.0))

        except Exception:
            return 0.0

    def _calculate_dark_metrics(self) -> Dict[str, float]:
        """Calculate dark metrics with enhanced oscillation awareness"""
        try:
            if len(self.volume_buffer) < 2:
                return {'liquidity': 0.0, 'flow': 0.0}

            volumes = np.array(self.volume_buffer, dtype=np.float64)
            prices = np.array(self.price_buffer, dtype=np.float64)

            # Calculate enhanced relative volume
            avg_volume = np.mean(volumes) + self.epsilon
            relative_volume = volumes / avg_volume

            # Enhanced dark liquidity
            liquidity = float(np.clip(
                np.mean(relative_volume) * PHI / (PHI + 1.0),
                0.0,
                1.0
            ))

            # Enhanced dark flow with oscillation
            flow = 0.0
            if len(prices) > 1:
                price_changes = np.diff(prices)
                volume_changes = np.diff(volumes)
                if len(price_changes) > 0 and len(volume_changes) > 0:
                    correlation = np.corrcoef(price_changes, volume_changes)[0, 1]
                    flow = float(np.clip((correlation + 1.0) / 2.0, 0.0, 1.0))

                    # Enhance flow with oscillation awareness
                    if len(self.oscillation_buffer) > 0:
                        oscillation_factor = np.mean(self.oscillation_buffer)
                        flow *= (1.0 + oscillation_factor)

                    flow = (flow * PHI) / (PHI + 1.0)

            return {
                'liquidity': liquidity,
                'flow': flow
            }

        except Exception:
            return {'liquidity': 0.0, 'flow': 0.0}

    def get_metrics(self) -> Dict[str, float]:
        """Return current metrics with enhanced stability"""
        return self.consciousness_metrics.copy()

    def _validate_market_data(self, data: Dict) -> bool:
        """Validate market data with enhanced checks"""
        try:
            required = ['opens', 'highs', 'lows', 'closes', 'volumes']

            # Check required fields
            if not all(key in data for key in required):
                return False

            # Validate data types and lengths
            if not all(isinstance(data[key], (list, np.ndarray)) for key in required):
                return False

            if not all(len(data[key]) > 0 for key in required):
                return False

            # Validate numeric values and ranges
            for key in required:
                values = np.array(data[key], dtype=np.float64)
                if np.any(np.isnan(values)) or np.any(np.isinf(values)):
                    return False
                if np.any(np.abs(values) > 1e10):
                    return False

            return True

        except Exception:
            return False
    def _calculate_geometric_signal(self, state: QualiaState) -> float:
        """Calcula componente geométrica do sinal"""
        if not self.price_buffer:
            return 0.0

        try:
            # Analyze phi patterns with stability
            price_diffs = np.diff(np.array(self.price_buffer, dtype=np.float64))
            if len(price_diffs) == 0:
                return 0.0

            phi_levels = [PHI ** i for i in range(-3, 4)]

            # Calculate stability-aware proximity
            mean_diff = np.mean(price_diffs) if len(price_diffs) > 0 else 0.0
            phi_proximities = [abs(mean_diff - level) for level in phi_levels]
            min_proximity = min(phi_proximities) + self.epsilon

            return float(np.clip(1.0 - min_proximity, 0.0, 1.0))
        except (ValueError, np.linalg.LinAlgError):
            return 0.0

    def _calculate_philosophical_signal(self, state: QualiaState) -> float:
        """Calcula componente filosófica do sinal"""
        if not self.volume_buffer:
            return 0.0

        try:
            # Calculate volume momentum with stability
            volume_array = np.array(self.volume_buffer, dtype=np.float64)
            if len(volume_array) < 2:
                return 0.0

            volume_momentum = np.gradient(volume_array)
            if len(volume_momentum) == 0:
                return 0.0

            philosophical_momentum = np.mean(volume_momentum) * PHI if len(volume_momentum) > 0 else 0.0

            return float(np.clip(np.tanh(philosophical_momentum), -1.0, 1.0))
        except (ValueError, np.linalg.LinAlgError):
            return 0.0

    def _calculate_consciousness_signal(self) -> float:
        """Calcula componente de consciência do sinal"""
        try:
            # Combine consciousness metrics with stability
            if not all(k in self.consciousness_metrics for k in ['field_coherence', 'phi_resonance', 'pattern_strength']):
                return 0.0

            field_coherence = float(self.consciousness_metrics.get('field_coherence', 0.0))
            phi_resonance = float(self.consciousness_metrics.get('phi_resonance', 0.0))
            pattern_strength = float(self.consciousness_metrics.get('pattern_strength', 0.0))

            if any(np.isnan([field_coherence, phi_resonance, pattern_strength])):
                return 0.0

            consciousness_level = (
                field_coherence * PHI ** 2 +
                phi_resonance * PHI +
                pattern_strength
            ) / (PHI ** 2 + PHI + 1.0 + self.epsilon)

            return float(np.clip(consciousness_level, 0.0, 1.0))
        except (KeyError, ValueError, np.linalg.LinAlgError):
            return 0.0

    def get_trading_signal(self, state: QualiaState) -> Dict[str, float]:
        """
        Gera sinal de trading baseado no estado QUALIA e M-ICCI
        """
        try:
            # Calculate signal components with stability
            geometric_signal = self._calculate_geometric_signal(state)
            philosophical_signal = self._calculate_philosophical_signal(state)
            consciousness_signal = self._calculate_consciousness_signal()

            # Combine signals using golden ratio with stability
            combined_signal = (
                geometric_signal * PHI ** 2 +
                philosophical_signal * PHI +
                consciousness_signal
            ) / (PHI ** 2 + PHI + 1.0 + self.epsilon)

            # Apply thresholds for signal validation
            if (self.consciousness_metrics['field_coherence'] < self.coherence_threshold or 
                self.consciousness_metrics['phi_resonance'] < self.resonance_threshold):
                direction = 0  # Return neutral signal if thresholds not met
                confidence = 0.0
            else:
                # Calculate direction with stability
                direction = np.sign(combined_signal) if np.abs(combined_signal) > self.epsilon else 0
                confidence = float(np.clip(abs(combined_signal), 0.0, 1.0))

            # Ensure numerical stability in output
            return {
                'direction': float(direction),  # Ensure int direction
                'confidence': float(np.clip(confidence, 0.0, 1.0)),
                'geometric_coherence': float(np.clip(state.geometric_coherence, 0.0, 1.0)),
                'philosophical_resonance': float(np.clip(state.philosophical_resonance, 0.0, 1.0)),
                'field_energy': float(np.clip(np.mean(state.field_tensor), -1.0, 1.0)),
                'consciousness_level': float(np.clip(
                    self.consciousness_metrics['field_coherence'], 0.0, 1.0
                ))
            }
        except Exception as e:
            print(f"Error generating trading signal: {e}")
            # Return neutral signal on numerical errors
            return {
                'direction': 0.0,
                'confidence': 0.0,
                'geometric_coherence': 0.0,
                'philosophical_resonance': 0.0,
                'field_energy': 0.0,
                'consciousness_level': 0.0
            }