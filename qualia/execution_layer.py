"""
Execution Layer for QUALIA Trading System.
Implements high-frequency trading with quantum state validation and M-ICCI framework integration.
"""
import logging
import ccxt
import numpy as np
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto

from .quantum_state_manager import QuantumStateManager
from .core.holographic_memory import HolographicMemory
from .validation_layer import ValidationLayer
from .utils import (
    calculate_dark_finance_metrics,
    predict_market_state,
    validate_market_distribution,
    calculate_quantum_coherence,
    calculate_morphic_resonance,
    calculate_field_entropy,
    calculate_integration_index
)
from .utils.quantum_field import (
    create_quantum_field,
    calculate_phi_resonance,
    calculate_financial_decoherence
)
from .monte_carlo import QuantumMonteCarlo

# Enums for execution types
class OrderType(Enum):
    """Enum for order types supported by the execution layer"""
    MARKET = auto()
    LIMIT = auto()
    STOP_LOSS = auto()
    TAKE_PROFIT = auto()
    TRAILING_STOP = auto()

class ExecutionStatus(Enum):
    """Enum for execution status values"""
    PENDING = auto()
    EXECUTING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    REJECTED = auto()

# Constants for normalization
COHERENCE_THRESHOLD = 0.5
PHI_RESONANCE_THRESHOLD = 0.4
MAX_FIELD_ENTROPY = 10.0
MAX_INTEGRATION_INDEX = 2.0
MAX_DARK_RATIO = 1.5

def normalize_to_range(value: float, max_value: float = 1.0, min_value: float = 0.0) -> float:
    """Normalize a value to [0,1] range"""
    return max(0.0, min(1.0, (value - min_value) / (max_value - min_value)))

@dataclass
class ExecutionResult:
    """Result of a quantum trade execution"""
    status: str
    order: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    signal: Optional[Dict[str, Any]] = None
    pattern_id: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    consciousness_metrics: Optional[Dict[str, float]] = None
    stops: Optional[Dict[str, float]] = None
    timestamp: datetime = datetime.now()

class ExecutionLayer:
    """Quantum-aware execution layer with M-ICCI framework integration."""
    def __init__(
        self,
        symbols: List[str] = None,
        timeframe: str = "1m",
        risk_threshold: float = 0.7,
        exchange_name: str = None,
        quantum_dimension: int = 64,
        kraken_enabled: bool = True
    ):
        """
        Initialize the execution layer with trading parameters.

        Args:
            symbols: List of trading pairs
            timeframe: Trading timeframe
            risk_threshold: Risk tolerance level
            exchange_name: Name of exchange to use
            quantum_dimension: Dimension of quantum states
            kraken_enabled: Whether to use Kraken exchange
        """
        # Core trading parameters
        self.symbols = symbols or ["BTC/USD", "ETH/USD"]
        self.timeframe = timeframe
        self.risk_threshold = risk_threshold
        
        # Exchange configuration
        self.exchange_id = str(exchange_name or 'kraken')
        self.kraken_enabled = kraken_enabled
        self.exchange = None
        
        # Quantum parameters
        self.quantum_dimension = quantum_dimension

        # Initialize quantum components with fixed dimension
        try:
            self.quantum_field = create_quantum_field(size=64)
            self.state_manager = QuantumStateManager()
            self.memory = HolographicMemory(dimension=64)
            self.validator = ValidationLayer()

            # Initialize exchange with fallback
            self._initialize_exchange_with_fallback()

            # Initialize normalized consciousness metrics
            raw_metrics = {
                'coherence': calculate_quantum_coherence(self.quantum_field),
                'morphic_resonance': calculate_morphic_resonance(self.quantum_field),
                'field_entropy': calculate_field_entropy(self.quantum_field),
                'integration_index': calculate_integration_index(self.quantum_field),
                'phi_resonance': calculate_phi_resonance(self.quantum_field),
                'dark_ratio': calculate_financial_decoherence(
                    self.quantum_field,
                    create_quantum_field(64)  
                )
            }

            # Normalize all metrics to [0,1]
            self.consciousness_metrics = {
                'coherence': normalize_to_range(raw_metrics['coherence'], max_value=2.0),
                'morphic_resonance': normalize_to_range(raw_metrics['morphic_resonance']),
                'field_entropy': normalize_to_range(raw_metrics['field_entropy'], max_value=MAX_FIELD_ENTROPY),
                'integration_index': normalize_to_range(raw_metrics['integration_index'], max_value=MAX_INTEGRATION_INDEX),
                'phi_resonance': normalize_to_range(raw_metrics['phi_resonance']),
                'dark_ratio': normalize_to_range(raw_metrics['dark_ratio'], max_value=MAX_DARK_RATIO)
            }

        except Exception as e:
            logging.error(f"Failed to initialize execution layer: {str(e)}")
            raise

        # Trading state
        self.active_trades: Dict[str, Dict] = {}
        self.trade_metrics: Dict[str, List[float]] = {
            "coherence": [],
            "decoherence": [],
            "phi_resonance": [],
            "profit_loss": [],
            "consciousness_level": [],
            "morphic_resonance": [],
            "integration_index": []
        }

        # Initialize Monte Carlo simulation with quantum awareness
        self.monte_carlo = QuantumMonteCarlo(
            n_simulations=1000,
            time_steps=100,
            confidence_level=0.95
        )

    def execute_quantum_trade(self, 
                            symbol: str,
                            quantum_state: np.ndarray,
                            risk_metrics: Dict[str, float]) -> ExecutionResult:
        """Execute trade with quantum validation and error handling."""
        try:
            # Validate inputs
            if quantum_state is None:
                return ExecutionResult(
                    status='error',
                    error='Invalid quantum state: None provided',
                    metrics=self.consciousness_metrics
                )

            if risk_metrics is None:
                return ExecutionResult(
                    status='error',
                    error='Missing risk metrics',
                    metrics=self.consciousness_metrics
                )

            try:
                # Validate quantum state dimensions and type
                quantum_state = np.asarray(quantum_state)
                if not isinstance(quantum_state, np.ndarray):
                    return ExecutionResult(
                        status='error',
                        error='Invalid quantum state type: must be numpy array',
                        metrics=self.consciousness_metrics
                    )

                # Check for valid 2D shape before resizing
                if len(quantum_state.shape) != 2:
                    return ExecutionResult(
                        status='error',
                        error=f'Invalid quantum state shape: expected 2D array, got {len(quantum_state.shape)}D',
                        metrics=self.consciousness_metrics
                    )

                original_size = quantum_state.size
                if original_size != 64 and original_size != self.quantum_dimension ** 2:
                    return ExecutionResult(
                        status='error',
                        error=f'Invalid quantum state dimensions: expected 64 or {self.quantum_dimension}x{self.quantum_dimension}, got {quantum_state.shape}',
                        metrics=self.consciousness_metrics
                    )

                # Standardize to 64 dimensions if needed
                if quantum_state.size > 64:
                    quantum_state = quantum_state.flatten()[:64].reshape(-1, 1)
                elif quantum_state.size < 64:
                    padding = np.zeros(64 - quantum_state.size, dtype=quantum_state.dtype)
                    quantum_state = np.concatenate([quantum_state.flatten(), padding]).reshape(-1, 1)

            except Exception as e:
                return ExecutionResult(
                    status='error',
                    error=f'Quantum state validation failed: {str(e)}',
                    metrics=self.consciousness_metrics
                )

            # Validate symbol
            if symbol not in self.symbols:
                return ExecutionResult(
                    status='error',
                    error=f'Invalid symbol: {symbol}',
                    metrics=self.consciousness_metrics
                )

            # Check risk metrics
            if not self._validate_risk_metrics(risk_metrics):
                return ExecutionResult(
                    status='rejected',
                    error='Trade rejected due to risk metrics',
                    metrics=risk_metrics,
                    consciousness_metrics=self.consciousness_metrics
                )

            # Generate trading signal with standardized quantum state
            signal = self._generate_trading_signal(
                symbol,
                quantum_state,
                predict_market_state(
                    quantum_state,
                    self.state_manager.hamiltonian,  
                    dt=0.01  
                ),
                risk_metrics
            )

            # Execute trade based on signal
            if self.kraken_enabled:
                result = self._execute_trade_signal(symbol, signal, risk_metrics)
            else:
                result = ExecutionResult(
                    status='simulated',
                    signal=signal,
                    metrics=risk_metrics,
                    consciousness_metrics=self.consciousness_metrics,
                    timestamp=datetime.now()
                )

            return result

        except Exception as e:
            logging.error(f"Trade execution error: {str(e)}")
            return ExecutionResult(
                status='error',
                error=str(e),
                metrics=self.consciousness_metrics
            )

    def _validate_risk_metrics(self, metrics: Dict[str, float]) -> bool:
        """Validate risk metrics against thresholds."""
        try:
            required_metrics = ['coherence', 'phi_resonance']
            if not all(metric in metrics for metric in required_metrics):
                return False

            thresholds = {
                'coherence': COHERENCE_THRESHOLD,
                'phi_resonance': PHI_RESONANCE_THRESHOLD
            }

            return all(
                metrics.get(metric, 0) >= threshold 
                for metric, threshold in thresholds.items()
            )

        except Exception as e:
            logging.error(f"Risk validation error: {str(e)}")
            return False

    def _generate_trading_signal(self,
                                 symbol: str,
                                 current_state: np.ndarray,
                                 future_state: np.ndarray,
                                 risk_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Generate trading signal with normalized metrics"""
        try:
            # Calculate base signal strength
            raw_signal_strength = 0.0
            weights = {
                'coherence': 0.3,
                'phi_resonance': 0.3,
                'consciousness_level': 0.4
            }

            # Add weighted metrics
            for metric, weight in weights.items():
                if metric == 'consciousness_level':
                    level = (
                        self.consciousness_metrics['coherence'] * 0.4 +
                        self.consciousness_metrics['morphic_resonance'] * 0.3 +
                        self.consciousness_metrics['integration_index'] * 0.3
                    )
                    raw_signal_strength += weight * level
                else:
                    raw_signal_strength += weight * risk_metrics.get(metric, 0.5)

            # Normalize signal strength
            signal_strength = normalize_to_range(raw_signal_strength)

            return {
                'symbol': symbol,
                'action': 'BUY' if signal_strength > self.risk_threshold else 'SELL',
                'confidence': signal_strength,
                'timestamp': datetime.now(),
                'metrics': risk_metrics,
                'consciousness_metrics': self.consciousness_metrics
            }

        except Exception as e:
            logging.error(f"Error generating trading signal: {str(e)}")
            return {
                'symbol': symbol,
                'action': 'NONE',
                'confidence': 0.0,
                'error': str(e)
            }
    def _execute_trade_signal(self, symbol: str, signal: Dict[str, Any], risk_metrics: Dict[str, float]) -> ExecutionResult:
        try:
            # Get market data for analysis
            try:
                market_data = np.array(self.exchange.fetch_ohlcv(
                    symbol, self.timeframe, limit=100
                ))
                close_prices = market_data[:, 4]
            except Exception as e:
                logging.error(f"Failed to fetch market data: {str(e)}")
                return ExecutionResult(
                    status='error',
                    error=f'Market data fetch failed: {str(e)}',
                    metrics=self.consciousness_metrics
                )

            # Calculate dynamic stops with error handling
            try:
                current_price = float(close_prices[-1])
                volatility = float(np.std(np.diff(close_prices) / close_prices[:-1]))
                stop_loss, take_profit = self.validator.calculate_dynamic_stops(
                    current_price,
                    risk_metrics,
                    volatility,
                    self.consciousness_metrics
                )
            except Exception as e:
                logging.error(f"Stop calculation failed: {str(e)}")
                return ExecutionResult(
                    status='error',
                    error=f'Stop calculation failed: {str(e)}',
                    metrics=self.consciousness_metrics
                )

            # Execute trade based on signal
            try:
                position_size = self._calculate_position_size(signal)

                if not isinstance(position_size, (int, float)) or position_size <= 0:
                    raise ValueError(f"Invalid position size calculated: {position_size}")

                if signal.get('action') == 'BUY':
                    logging.info(f"Executing BUY order for {symbol}, size: {position_size}")
                    order = self.exchange.create_market_buy_order(
                        symbol=symbol,
                        amount=position_size,
                        params={
                            'stopLoss': {'stopPrice': stop_loss},
                            'takeProfit': {'takeProfitPrice': take_profit}
                        }
                    )
                elif signal.get('action') == 'SELL':
                    logging.info(f"Executing SELL order for {symbol}, size: {position_size}")
                    order = self.exchange.create_market_sell_order(
                        symbol=symbol,
                        amount=position_size,
                        params={
                            'stopLoss': {'stopPrice': stop_loss},
                            'takeProfit': {'takeProfitPrice': take_profit}
                        }
                    )
                else:
                    return ExecutionResult(
                        status='no_action',
                        signal=signal,
                        metrics=risk_metrics
                    )

                if not isinstance(order, dict):
                    raise ValueError("Invalid order response format")

                order_id = str(order.get('id', ''))
                if not order_id:
                    raise ValueError("Missing order ID in response")

                logging.info(f"Order executed successfully: {order_id}")

                # Store trade pattern
                pattern_id = self.memory.store_pattern(
                    quantum_state,
                    {
                        'timestamp': datetime.now(),
                        'metrics': risk_metrics,
                        'signal': signal,
                        'consciousness_metrics': self.consciousness_metrics,
                        'stops': {
                            'stop_loss': stop_loss,
                            'take_profit': take_profit
                        }
                    }
                )

                # Update active trades with proper type checking
                self.active_trades[order_id] = {
                    'entry_price': float(current_price),
                    'stop_loss': float(stop_loss),
                    'take_profit': float(take_profit),
                    'metrics': risk_metrics,
                    'consciousness_metrics': self.consciousness_metrics,
                    'pattern_id': pattern_id
                }

                # Update metrics history
                self._update_metrics(risk_metrics)

                return ExecutionResult(
                    status='executed',
                    order=order,
                    signal=signal,
                    pattern_id=pattern_id,
                    metrics=risk_metrics,
                    consciousness_metrics=self.consciousness_metrics,
                    stops={
                        'stop_loss': float(stop_loss),
                        'take_profit': float(take_profit)
                    }
                )

            except Exception as e:
                error_msg = f"Order execution failed: {str(e)}"
                logging.error(error_msg)
                return ExecutionResult(
                    status='error',
                    error=error_msg,
                    signal=signal,
                    metrics=risk_metrics
                )

        except Exception as e:
            logging.error(f"Trade execution error: {str(e)}")
            return ExecutionResult(
                status='error',
                error=str(e),
                metrics=self.consciousness_metrics
            )

    def _validate_trade_conditions(self,
                                  signal: Dict[str, Any],
                                  metrics: Dict[str, float]) -> bool:
        """Validate trading conditions using quantum and consciousness metrics."""
        try:
            # Base conditions with consciousness integration
            conditions = [
                metrics['coherence'] >= self.risk_threshold,
                metrics['decoherence'] <= 2.0,
                metrics['phi_resonance'] >= 0.6,
                metrics['p_value'] >= 0.05,
                metrics['stability_index'] >= 0.4,
                signal['confidence'] >= self.risk_threshold,
                self.consciousness_metrics['coherence'] >= 0.5,
                self.consciousness_metrics['morphic_resonance'] >= 0.4,
                self.consciousness_metrics['integration_index'] >= 0.5
            ]

            # Add Monte Carlo specific conditions
            if 'monte_carlo' in metrics:
                mc_metrics = metrics['monte_carlo']
                conditions.extend([
                    mc_metrics['value_at_risk'] <= 0.05 * self._calculate_position_size(signal),
                    mc_metrics['expected_shortfall'] <= 0.07 * self._calculate_position_size(signal),
                    mc_metrics['skewness'] > -0.5
                ])

            # Additional checks based on market regime
            if metrics['market_regime'] == 'VOLATILE':
                conditions.extend([
                    metrics['stability_index'] >= 0.6,
                    self.consciousness_metrics['coherence'] >= 0.6,
                    self.consciousness_metrics['morphic_resonance'] >= 0.5
                ])

            return all(conditions)

        except Exception as e:
            logging.error(f"Error validating trade conditions: {e}")
            return False

    def _calculate_position_size(self, signal: Dict[str, Any]) -> float:
        """Calculate position size based on quantum metrics and consciousness."""
        base_size = 0.01  
        confidence = signal['confidence']
        metrics = signal['metrics']
        consciousness_level = (
            self.consciousness_metrics['coherence'] * 0.4 +
            self.consciousness_metrics['morphic_resonance'] * 0.3 +
            self.consciousness_metrics['integration_index'] * 0.3
        )

        # Adjust size based on market regime and consciousness
        regime_factor = {
            'TRENDING': 1.2,
            'RANGING': 1.0,
            'VOLATILE': 0.7
        }.get(metrics['market_regime'], 1.0)

        consciousness_factor = consciousness_level * 1.5
        stability_factor = metrics['stability_index']

        size = (
            base_size *
            confidence *
            stability_factor *
            regime_factor *
            consciousness_factor
        )
        return min(size, 0.05)  

    def _update_metrics(self, metrics: Dict[str, float]) -> None:
        """Update trading metrics history with consciousness integration."""
        for key in self.trade_metrics:
            if key in metrics:
                self.trade_metrics[key].append(metrics[key])
            elif key in self.consciousness_metrics:
                self.trade_metrics[key].append(self.consciousness_metrics[key])
            # Keep last 1000 values
            if len(self.trade_metrics[key]) > 1000:
                self.trade_metrics[key].pop(0)

    def get_trading_metrics(self) -> Dict[str, Any]:
        """Return current trading metrics including consciousness metrics."""
        base_metrics = {
            'metrics': self.trade_metrics,
            'active_trades': len(self.active_trades),
            'total_patterns': len(self.memory.patterns),
            'consciousness_metrics': self.consciousness_metrics
        }
        mc_metrics = self.monte_carlo.get_simulation_metrics()

        return {
            'market_metrics': base_metrics,
            'monte_carlo_metrics': mc_metrics
        }

    def set_kraken_enabled(self, enabled: bool) -> None:
        """Enable or disable Kraken connection"""
        if self.kraken_enabled != enabled:
            self.kraken_enabled = enabled
            self.exchange_id = 'kraken' if enabled else 'coinbase'
            self._initialize_exchange_with_fallback()

    def is_kraken_enabled(self) -> bool:
        """Check if Kraken is enabled"""
        return self.kraken_enabled

    def _initialize_exchange_with_fallback(self) -> None:
        """Initialize exchange with automatic fallback to Coinbase if Kraken fails."""
        try:
            if not isinstance(self.exchange_id, str):
                raise ValueError(f"exchange_id must be a string, got {type(self.exchange_id)}")

            exchange_class = getattr(ccxt, self.exchange_id)
            config = {
                'enableRateLimit': True,
                'timeout': 30000
            }

            if self.exchange_id == 'kraken':
                if not self.kraken_enabled:
                    logging.info("Kraken is disabled, switching to Coinbase simulation mode")
                    self.exchange_id = 'coinbase'
                    config['sandbox'] = True
                else:
                    config.update({
                        'apiKey': os.getenv('KRAKEN_API_KEY'),
                        'secret': os.getenv('KRAKEN_API_SECRET'),
                        'passphrase': os.getenv('KRAKEN_API_PASSPHRASE')
                    })

            self.exchange = exchange_class(config)
            self.exchange.load_markets()
            logging.info(f"Successfully initialized {self.exchange_id} exchange")

        except Exception as e:
            error_msg = str(e)
            if self._detect_geo_restriction(error_msg) or not self.kraken_enabled:
                if self.exchange_id == 'kraken':
                    logging.warning("Kraken unavailable, falling back to Coinbase simulation mode")
                    self.exchange_id = 'coinbase'
                    try:
                        config = {
                            'enableRateLimit': True,
                            'timeout': 30000,
                            'sandbox': True
                        }
                        self.exchange = ccxt.coinbase(config)
                        self.exchange.load_markets()
                        logging.info("Successfully switched to Coinbase simulation mode")
                    except Exception as coinbase_error:
                        raise ExchangeInitializationError(
                            f"Failed to initialize both Kraken and Coinbase: {str(coinbase_error)}"
                        )
            else:
                raise ExchangeInitializationError(f"Failed to initialize {self.exchange_id}: {error_msg}")

    def _detect_geo_restriction(self, error_msg: str) -> bool:
        """Detect if error is due to geographic restrictions."""
        geo_keywords = [
            "unavailable in your region",
            "unavailable in the U.S",
            "IP address is restricted",
            "geographic restrictions",
            "restricted country/region"
        ]
        return any(keyword.lower() in error_msg.lower() for keyword in geo_keywords)

    def _validate_quantum_state(self) -> bool:
        """Validate quantum state using M-ICCI framework principles."""
        try:
            # Update consciousness metrics
            raw_metrics = {
                'coherence': calculate_quantum_coherence(self.quantum_field),
                'morphic_resonance': calculate_morphic_resonance(self.quantum_field),
                'field_entropy': calculate_field_entropy(self.quantum_field),
                'integration_index': calculate_integration_index(self.quantum_field),
                'phi_resonance': calculate_phi_resonance(self.quantum_field),
                'dark_ratio': calculate_financial_decoherence(
                    self.quantum_field,
                    create_quantum_field(self.quantum_dimension)
                )
            }
            self.consciousness_metrics = {
                'coherence': normalize_to_range(raw_metrics['coherence'], max_value=2.0),
                'morphic_resonance': normalize_to_range(raw_metrics['morphic_resonance']),
                'field_entropy': normalize_to_range(raw_metrics['field_entropy'], max_value=MAX_FIELD_ENTROPY),
                'integration_index': normalize_to_range(raw_metrics['integration_index'], max_value=MAX_INTEGRATION_INDEX),
                'phi_resonance': normalize_to_range(raw_metrics['phi_resonance']),
                'dark_ratio': normalize_to_range(raw_metrics['dark_ratio'], max_value=MAX_DARK_RATIO)
            }

            # Calculate consciousness level
            consciousness_level = (
                self.consciousness_metrics['coherence'] * 0.3 +
                self.consciousness_metrics['morphic_resonance'] * 0.3 +
                self.consciousness_metrics['integration_index'] * 0.4
            )

            # Validate metrics
            conditions = [
                consciousness_level >= 0.4,  
                self.consciousness_metrics['coherence'] >= 0.5,  
                self.consciousness_metrics['morphic_resonance'] >= 0.4,  
                self.consciousness_metrics['dark_ratio'] <= 0.6,  
                self.consciousness_metrics['phi_resonance'] >= 0.5,  
                self.consciousness_metrics['integration_index'] >= 0.4  
            ]

            return all(conditions)

        except Exception as e:
            logging.error(f"Error in quantum state validation: {e}")
            return False

class ExchangeInitializationError(Exception):
    pass

class GeoRestrictionError(Exception):
    pass