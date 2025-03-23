"""
Risk Analyzer for quantum trading system.
"""
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class RiskAnalyzer:
    """Risk analyzer that computes risk metrics based on market state"""

    def __init__(
        self,
        market_data_provider = None,
        config: Dict[str, Any] = None
    ):
        """
        Initialize risk analyzer with config dict and market data provider.

        Args:
            market_data_provider: Market data provider (optional)
            config: Risk analyzer configurations
        """
        # Default configurations
        default_config = {
            'volatility_window': 20,
            'max_risk_score': 1.0,
            'risk_threshold': 0.7,
            'volume_threshold': 1000,
            'price_change_threshold': 0.05
        }

        # Merge config with defaults
        self.config = {**default_config, **(config or {})}

        # Extract specific configurations
        self.volatility_window = self.config['volatility_window']
        self.max_risk_score = self.config['max_risk_score']
        self.risk_threshold = self.config['risk_threshold']
        self.volume_threshold = self.config['volume_threshold']
        self.price_change_threshold = self.config['price_change_threshold']

        # Initialize histories
        self.price_history = {}
        self.volume_history = {}

        # Store market data provider
        self.market_data_provider = market_data_provider

        # Validate provider
        self.validate_market_data_provider()

        logger.info(f"RiskAnalyzer initialized")
        if market_data_provider:
            logger.info(f"Market data provider configured: {type(market_data_provider).__name__}")

    def validate_market_data_provider(self):
        """Validate market_data_provider accessibility and configuration"""
        if self.market_data_provider is None:
            logger.warning("No market_data_provider configured")
            return False

        try:
            logger.info(f"Market Data Provider: {type(self.market_data_provider).__name__}")
            logger.info(f"Exchange: {getattr(self.market_data_provider, 'exchange', 'Not defined')}")
            logger.info(f"Symbols: {getattr(self.market_data_provider, 'symbols', 'Not defined')}")
            return True
        except Exception as e:
            logger.error(f"Error validating market_data_provider: {e}")
            return False

    def analyze(self, market_state) -> Dict[str, float]:
        """
        Analyze current market risk based on provided state.

        Args:
            market_state: Current market state with prices and metrics

        Returns:
            Dict[str, float]: Computed risk metrics
        """
        try:
            # Check if market_state has required attributes
            if not hasattr(market_state, 'data'):
                logger.warning("Market state missing 'data' attribute")
                return self._get_safe_metrics()

            # Extract data from market state
            price_data = market_state.data[:, 0]  # First column is price
            volume_data = market_state.data[:, 1]  # Second column is volume

            # Calculate volatility
            if len(price_data) >= 2:
                returns = np.diff(price_data) / price_data[:-1]
                volatility = np.std(returns) if len(returns) > 0 else 0
            else:
                volatility = 0

            # Calculate relative volume
            avg_volume = np.mean(volume_data) if len(volume_data) > 0 else 0
            volume_ratio = volume_data[-1] / avg_volume if avg_volume > 0 else 1

            # Robust division by zero prevention
            if len(price_data) == 0:
                price_mean = 1e-6
            else:
                price_mean = np.mean(price_data)
                if abs(price_mean) < 1e-9:  # Avoid infinitesimal values
                    price_mean = 1e-6 if price_mean >= 0 else -1e-6

            # Safe price risk calculation
            price_risk = abs(price_data[-1] - price_mean) / price_mean if len(price_data) > 0 else 0.0

            # Combine metrics into risk score
            risk_score = min(
                self.max_risk_score,
                (0.4 * volatility + 0.3 * volume_ratio + 0.3 * price_risk)
            )

            # Generate risk metrics
            return {
                'risk_score': float(risk_score),
                'volatility': float(volatility),
                'volume_ratio': float(volume_ratio),
                'price_risk': float(price_risk)
            }

        except Exception as e:
            logger.error(f"Error analyzing risk: {str(e)}")
            return self._get_safe_metrics()

    def compute_risk(self, market_state) -> Dict[str, float]:
        """Alias for analyze method to maintain compatibility"""
        return self.analyze(market_state)

    def _get_safe_metrics(self) -> Dict[str, float]:
        """Return safe default risk metrics in case of error"""
        return {
            'risk_score': 0.5,
            'volatility': 0.0,
            'volume_ratio': 1.0,
            'price_risk': 0.0
        }