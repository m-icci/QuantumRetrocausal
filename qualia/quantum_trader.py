"""
QUALIA Trading System - Quantum Trader Module
Main trading system implementation with quantum-enhanced decision making.
"""
from typing import Optional, Dict, List, Any, Union, Tuple
import logging
import numpy as np
from datetime import datetime
import os
import ccxt
from dotenv import load_dotenv
import threading
import time
from scipy import linalg as la
import traceback

from .core.market_data import MarketDataProvider, MarketState
from .core.exchange_interface import ExchangeInterface
from .analysis.quantum_analysis import QuantumAnalyzer
from .core.risk_analyzer import RiskAnalyzer
from .core.holographic_memory import HolographicMemory
from .execution_layer import ExecutionLayer

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class TradeSignal:
    """Represents a trading signal with quantum metrics"""
    def __init__(
        self,
        symbol: str,
        action: str,  # 'BUY', 'SELL', or 'HOLD'
        price: float,
        confidence: float,
        quantum_metrics: Dict[str, float],
        risk_metrics: Dict[str, float],
        consciousness_metrics: Dict[str, float],
        timestamp: Optional[datetime] = None,
        alerts: Optional[List[Dict[str, Any]]] = None
    ):
        self.symbol = symbol
        self.action = action
        self.price = price
        self.confidence = confidence
        self.quantum_metrics = quantum_metrics
        self.risk_metrics = risk_metrics
        self.consciousness_metrics = consciousness_metrics or {}
        self.timestamp = timestamp or datetime.now()
        self.alerts = alerts or []

    def validate(self) -> bool:
        """Validate signal parameters"""
        try:
            if not (0 <= self.confidence <= 1):
                return False
            if self.action not in ['BUY', 'SELL', 'HOLD']:
                return False
            if self.price <= 0:
                return False
            return True
        except Exception as e:
            logger.error(f"Signal validation failed: {e}")
            return False

class QuantumTrader:
    """Main quantum trading system implementation"""

    def __init__(
        self,
        trading_pairs: Optional[List[str]] = None,
        quantum_dimension: int = 64,
        simulation_mode: bool = True,
        kraken_enabled: bool = False,
        consciousness_threshold: float = 0.7  # Add consciousness threshold parameter
    ):
        """Initialize quantum trading system"""
        try:
            logger.info("Initializing QuantumTrader...")
            self.trading_pairs = trading_pairs or ['BTC/USD', 'ETH/USD', 'SOL/USD']
            self.quantum_dimension = quantum_dimension
            self.simulation_mode = simulation_mode
            self.consciousness_threshold = consciousness_threshold

            # Initialize exchange interface with Kraken support
            self.exchange = ExchangeInterface(
                simulation_mode=simulation_mode,
                quantum_dimension=quantum_dimension,
                kraken_enabled=kraken_enabled
            )

            # Initialize market data provider
            self.market_data = MarketDataProvider(
                exchange=self.exchange,
                symbols=self.trading_pairs
            )

            # Initialize analysis components
            self.analyzer = QuantumAnalyzer(dimension=quantum_dimension)
            self.risk_analyzer = RiskAnalyzer()
            self.memory = HolographicMemory(dimension=quantum_dimension)

            # Initialize execution layer
            self.execution_layer = ExecutionLayer(
                symbols=self.trading_pairs,
                quantum_dimension=quantum_dimension
            )

            logger.info(f"QuantumTrader initialized in {'simulation' if simulation_mode else 'live'} mode")

        except Exception as e:
            logger.error(f"Error initializing QuantumTrader: {e}")
            logger.error(traceback.format_exc())
            raise

    def get_quantum_metrics(self) -> Dict[str, float]:
        """Get current quantum metrics
        Returns:
            Dict with metrics including coherence, entropy, consciousness, market_stability, morphic_resonance
        """
        try:
            market_state = self.market_data.get_state(self.trading_pairs[0])
            if not market_state:
                return {
                    'coherence': 0.0,
                    'entropy': 0.0,
                    'consciousness': 0.0,
                    'market_stability': 0.0,
                    'morphic_resonance': 0.0
                }

            quantum_state = self.analyzer.analyze(market_state)
            return self._calculate_quantum_metrics(market_state)

        except Exception as e:
            logger.error(f"Error getting quantum metrics: {e}")
            return {
                'coherence': 0.0,
                'entropy': 0.0,
                'consciousness': 0.0,
                'market_stability': 0.0,
                'morphic_resonance': 0.0
            }

    def calculate_risk_metrics(self, market_state: Optional[MarketState]) -> Dict[str, float]:
        """Calculate risk metrics for given market state"""
        try:
            if not market_state:
                return {
                    'risk_level': 1.0,
                    'stability_index': 0.0,
                    'market_volatility': 1.0,
                    'quantum_adjusted_risk': 1.0
                }

            # Get quantum metrics
            quantum_metrics = self._calculate_quantum_metrics(market_state)

            # Calculate base risk metrics
            risk_metrics = {
                'risk_level': 1.0 - quantum_metrics.get('coherence', 0.0),
                'stability_index': quantum_metrics.get('market_stability', 0.0),
                'market_volatility': 1.0 - quantum_metrics.get('morphic_resonance', 0.0),
                'quantum_adjusted_risk': 1.0 - quantum_metrics.get('consciousness', 0.0)
            }

            # Normalize values
            for key in risk_metrics:
                risk_metrics[key] = max(0.0, min(1.0, risk_metrics[key]))

            return risk_metrics

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {
                'risk_level': 1.0,
                'stability_index': 0.0,
                'market_volatility': 1.0,
                'quantum_adjusted_risk': 1.0
            }

    def _calculate_quantum_metrics(self, market_state: Optional[MarketState]) -> Dict[str, float]:
        """Calculate quantum metrics from market state"""
        try:
            if not market_state:
                return {
                    'coherence': 0.0,
                    'entropy': 0.0,
                    'consciousness': 0.0,
                    'market_stability': 0.0,
                    'morphic_resonance': 0.0
                }

            # Get quantum analysis
            quantum_state = self.analyzer.analyze(market_state)
            if not quantum_state:
                return {
                    'coherence': 0.0,
                    'entropy': 0.0,
                    'consciousness': 0.0,
                    'market_stability': 0.0,
                    'morphic_resonance': 0.0
                }

            # Calculate metrics from quantum state
            metrics = {
                'coherence': float(np.abs(quantum_state.get('coherence', 0.0))),
                'entropy': float(quantum_state.get('entropy', 1.0)),
                'consciousness': float(quantum_state.get('consciousness', 0.0)),
                'market_stability': float(quantum_state.get('stability', 0.0)),
                'morphic_resonance': float(quantum_state.get('resonance', 0.0))
            }

            # Normalize values
            for key in metrics:
                metrics[key] = max(0.0, min(1.0, metrics[key]))

            return metrics

        except Exception as e:
            logger.error(f"Error calculating quantum metrics: {e}")
            return {
                'coherence': 0.0,
                'entropy': 0.0,
                'consciousness': 0.0,
                'market_stability': 0.0,
                'morphic_resonance': 0.0
            }

    def get_aggressive_scalp_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Generate aggressive scalping signal"""
        try:
            # Get market data
            market_state = self.market_data.get_state(symbol)
            if not market_state:
                return None

            # Perform quantum analysis
            quantum_state = self.analyzer.analyze(market_state)
            if quantum_state is None:
                return None

            # Calculate risk metrics
            risk_metrics = self.risk_analyzer.analyze(market_state)

            # Generate signal
            price = market_state.current_price
            action = self._determine_action(quantum_state, risk_metrics)
            confidence = self._calculate_confidence(quantum_state)

            # Calculate stop levels
            stop_loss, take_profit = self._calculate_dynamic_targets(price, action)

            return {
                'symbol': symbol,
                'decision': action,
                'price': price,
                'confidence': confidence * 100,  # Convert to percentage
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'quantum_metrics': self.analyzer.get_metrics(quantum_state),
                'risk_metrics': risk_metrics
            }

        except Exception as e:
            logger.error(f"Error generating scalp signal: {e}")
            return None

    def _determine_action(self, quantum_state: Dict[str, Any], risk_metrics: Dict[str, float]) -> str:
        """Determine trading action based on quantum state and risk"""
        try:
            if risk_metrics.get('risk_level', 1.0) > 0.8:
                return 'HOLD'

            if not quantum_state or not isinstance(quantum_state, dict):
                return 'HOLD'

            # Extract strength value
            strength = quantum_state.get('strength', 0.0)
            if strength > 0.6:
                return 'BUY'
            elif strength < 0.4:
                return 'SELL'
            return 'HOLD'

        except Exception as e:
            logger.error(f"Error determining action: {e}")
            return 'HOLD'

    def _calculate_confidence(self, quantum_state: Dict[str, Any]) -> float:
        """Calculate confidence level from quantum state"""
        try:
            if not quantum_state or not isinstance(quantum_state, dict):
                return 0.0

            # Extract strength value from quantum state if available
            if 'strength' in quantum_state:
                return float(quantum_state['strength'])

            # Otherwise use metrics to calculate confidence
            metrics = quantum_state.get('metrics', {})
            if metrics:
                # Use coherence and market_stability as confidence indicators
                coherence = float(metrics.get('coherence', 0.0))
                stability = float(metrics.get('market_stability', 0.0))
                return (coherence + stability) / 2.0

            return 0.0
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.0

    def _calculate_dynamic_targets(self, price: float, action: str) -> Tuple[float, float]:
        """Calculate dynamic stop loss and take profit levels"""
        if action == 'BUY':
            stop_loss = price * 0.995  # 0.5% below entry
            take_profit = price * 1.015  # 1.5% above entry
        elif action == 'SELL':
            stop_loss = price * 1.005  # 0.5% above entry
            take_profit = price * 0.985  # 1.5% below entry
        else:
            stop_loss = price * 0.99
            take_profit = price * 1.01

        return stop_loss, take_profit

    def validate_trading(self, symbol: str) -> Dict[str, Any]:
        """Validate if trading is possible"""
        try:
            if symbol not in self.trading_pairs:
                return {'status': 'error', 'message': f'Symbol {symbol} not in trading pairs'}

            if not self.market_data.validate_market_state(symbol):
                return {'status': 'error', 'message': f'No market data for {symbol}'}

            return {'status': 'success', 'message': f'Trading validated for {symbol}'}

        except Exception as e:
            logger.error(f"Trading validation failed: {e}")
            return {'status': 'error', 'message': str(e)}

    def execute_trade(
        self,
        symbol: str,
        tipo: str,
        quantidade: float,
        stop_loss: float,
        take_profit: float
    ) -> Dict[str, Any]:
        """Execute trade"""
        try:
            if not all([symbol, quantidade, stop_loss, take_profit]):
                return {'status': 'error', 'message': 'Missing required parameters'}

            order = self.exchange.execute_market_order(
                symbol=symbol,
                side=tipo,
                amount=quantidade
            )

            if not order:
                return {'status': 'error', 'message': 'Order execution failed'}

            return {
                'status': 'success',
                'ordem': order,
                'message': f'Trade executed successfully'
            }

        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return {'status': 'error', 'message': str(e)}

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price"""
        try:
            market_state = self.market_data.get_state(symbol)
            if market_state:
                return market_state.current_price
            return None
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
            return None

    def _calculate_position_size(self, symbol: str, price: float) -> Optional[float]:
        """Calculate position size"""
        try:
            balance = self.exchange.get_balance()
            if not balance:
                return None

            risk_amount = balance['total'].get('USDT', 0) * 0.01
            position_size = risk_amount / price

            market_info = self.exchange.get_market_info(symbol)
            if market_info and 'precision' in market_info:
                decimals = market_info['precision'].get('amount', 8)
                position_size = round(position_size, decimals)

            return position_size

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return None

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get open positions"""
        try:
            return self.exchange.get_open_positions()
        except Exception as e:
            logger.error(f"Error getting open positions: {e}")
            return []


# Export core components
__all__ = ['QuantumTrader', 'TradeSignal']