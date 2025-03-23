"""
Market Data Provider and related classes with enhanced quantum consciousness integration
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime
import time
import ccxt
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

@dataclass
class MarketState:
    """Current market state with quantum-enhanced consciousness metrics"""
    def __init__(
        self, 
        symbol: str = '', 
        price: float = 0.0, 
        volume: float = 0.0, 
        timestamp: float = 0.0,
        ohlcv: np.ndarray = None,
        quantum_features: np.ndarray = None,
        market_phase: str = '',
        entropy: float = 0.0,
        complexity: float = 0.0,
        coherence: float = 0.0,  # Quantum coherence metric
        decoherence_rate: float = 0.0,  # Rate of quantum decoherence
        consciousness_level: float = 0.0  # Market consciousness level
    ):
        # Use OHLCV data if provided
        if ohlcv is not None and len(ohlcv) > 0:
            # Assume standard OHLCV format: [Open, High, Low, Close, Volume, Timestamp]
            self.symbol = symbol or 'UNKNOWN'
            self.open_price = float(ohlcv[0, 0]) if ohlcv.ndim > 1 else float(ohlcv[0])
            self.high_price = float(ohlcv[0, 1]) if ohlcv.ndim > 1 else float(ohlcv[0])
            self.low_price = float(ohlcv[0, 2]) if ohlcv.ndim > 1 else float(ohlcv[0])
            self.close_price = float(ohlcv[0, 3]) if ohlcv.ndim > 1 else float(ohlcv[0])
            self.volume = float(ohlcv[0, 4]) if ohlcv.ndim > 1 else 0.0
            self.timestamp = float(ohlcv[0, 5]) if ohlcv.ndim > 1 and ohlcv.shape[1] > 5 else timestamp
            self.current_price = self.close_price
        else:
            # Traditional initialization
            self.symbol = symbol
            self.current_price = float(price)
            self.volume = float(volume)
            self.timestamp = float(timestamp)
            self.high_price = float(price)
            self.low_price = float(price)
            self.open_price = float(price)
            self.close_price = float(price)

        # Enhanced quantum metrics
        self.data = np.array([
            self.current_price,
            self.volume,
            self.timestamp,
            self.high_price,
            self.low_price
        ], dtype=np.float32).reshape(1, -1)

        # Quantum consciousness metrics
        self.quantum_features = quantum_features
        self.market_phase = market_phase
        self.entropy = self._normalize_metric(entropy)
        self.complexity = self._normalize_metric(complexity)
        self.coherence = self._normalize_metric(coherence)
        self.decoherence_rate = self._normalize_metric(decoherence_rate)
        self.consciousness_level = self._normalize_metric(consciousness_level)

        # Field metrics
        self.phi = 1.618033988749895  # Golden ratio for quantum protection
        self.morphic_field_strength = self._calculate_morphic_strength()

        self.trades: int = 0
        self.metrics: Dict[str, float] = {}
        self.features: Dict[str, np.ndarray] = {}

        # 24h metrics
        self.price_change_24h = 0.0
        self.high_24h = 0.0
        self.low_24h = 0.0

    def _normalize_metric(self, value: float) -> float:
        """Normalize metric to [0,1] range with quantum protection"""
        try:
            return np.clip(float(value), 0.0, 1.0)
        except (ValueError, TypeError):
            return 0.0

    def _calculate_morphic_strength(self) -> float:
        """Calculate morphic field strength using quantum metrics"""
        try:
            # Integrate consciousness metrics with golden ratio protection
            strength = (
                self.entropy * 0.3 +
                self.complexity * 0.3 +
                (1 - self.decoherence_rate) * 0.4
            ) * self.phi
            return self._normalize_metric(strength)
        except Exception as e:
            logger.error(f"Error calculating morphic strength: {e}")
            return 0.5

    def get_quantum_state(self) -> Dict[str, float]:
        """Get current quantum state metrics"""
        return {
            'entropy': self.entropy,
            'complexity': self.complexity,
            'coherence': self.coherence,
            'decoherence_rate': self.decoherence_rate,
            'consciousness_level': self.consciousness_level,
            'morphic_field_strength': self.morphic_field_strength
        }

class MarketDataProvider:
    """Market data provider with enhanced quantum consciousness integration"""
    def __init__(
        self,
        exchange: Any,
        symbols: List[str],
        retry_delay: int = 2,
        max_retries: int = 5,
        quantum_dimension: int = 64
    ):
        self.exchange = exchange
        self.symbols = symbols
        self.last_market_states: Dict[str, MarketState] = {}
        self.quantum_dimension = quantum_dimension

        # Retry settings
        self.retry_delay = retry_delay
        self.max_retries = max_retries

        # Initialize quantum protection
        self.phi = 1.618033988749895

        logger.info(f"Initializing Enhanced MarketDataProvider")
        logger.info(f"Monitored symbols: {', '.join(symbols)}")
        logger.info(f"Quantum dimension: {quantum_dimension}")

    def get_state(self, symbol: str) -> Optional[MarketState]:
        """Get current market state with quantum consciousness metrics"""
        try:
            if symbol not in self.symbols:
                logger.warning(f"Symbol {symbol} not in monitored symbols")
                return None

            # Fetch market data with quantum awareness
            market_data = self.exchange.fetch_market_data(symbol)
            if market_data is None:
                logger.warning(f"No market data available for {symbol}")
                return self.last_market_states.get(symbol)

            # Calculate quantum metrics
            coherence = self._calculate_quantum_coherence(market_data)
            decoherence = self._calculate_decoherence_rate(market_data)
            consciousness = self._calculate_consciousness_level(
                coherence, 
                decoherence
            )

            # Create enhanced market state
            market_state = MarketState(
                symbol=symbol,
                ohlcv=market_data,
                coherence=coherence,
                decoherence_rate=decoherence,
                consciousness_level=consciousness
            )

            self.last_market_states[symbol] = market_state
            return market_state

        except Exception as e:
            logger.error(f"Error getting market state for {symbol}: {e}")
            return self.last_market_states.get(symbol)

    def _calculate_quantum_coherence(self, market_data: np.ndarray) -> float:
        """Calculate quantum coherence from market data"""
        try:
            if market_data is None or len(market_data) == 0:
                return 0.5

            # Normalize data with quantum protection
            prices = market_data[:, 4]  # Close prices
            normalized = (prices - np.mean(prices)) / (np.std(prices) + 1e-8)

            # Calculate quantum autocorrelation with phi protection
            autocorr = np.correlate(normalized, normalized, mode='full')
            coherence = float(np.max(np.abs(autocorr[len(autocorr)//2:])))

            return np.clip(coherence / self.phi, 0.0, 1.0)

        except Exception as e:
            logger.error(f"Error calculating quantum coherence: {e}")
            return 0.5

    def _calculate_decoherence_rate(self, market_data: np.ndarray) -> float:
        """Calculate market decoherence rate"""
        try:
            if market_data is None or len(market_data) < 2:
                return 0.5

            # Calculate price changes
            prices = market_data[:, 4]  # Close prices
            changes = np.diff(prices) / prices[:-1]

            # Measure decoherence as volatility with quantum protection
            volatility = np.std(changes) * self.phi
            return np.clip(volatility, 0.0, 1.0)

        except Exception as e:
            logger.error(f"Error calculating decoherence: {e}")
            return 0.5

    def _calculate_consciousness_level(
        self,
        coherence: float,
        decoherence: float
    ) -> float:
        """Calculate market consciousness level"""
        try:
            # Integrate coherence and decoherence with golden ratio
            consciousness = (
                coherence * (1 - decoherence) * self.phi
            )
            return np.clip(consciousness, 0.0, 1.0)

        except Exception as e:
            logger.error(f"Error calculating consciousness: {e}")
            return 0.5

    def validate_market_state(self, symbol: str) -> bool:
        """Validate market state with quantum consciousness checks"""
        try:
            state = self.get_state(symbol)
            if state is None:
                return False

            # Validate price data
            price_valid = state.current_price > 0

            # Validate quantum metrics
            quantum_metrics = state.get_quantum_state()
            metrics_valid = all(
                0 <= v <= 1 for v in quantum_metrics.values()
            )

            return price_valid and metrics_valid

        except Exception as e:
            logger.error(f"Error validating market state for {symbol}: {e}")
            return False