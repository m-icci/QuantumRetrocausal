"""
Quantum Analyzer for Trading Decisions
Implements quantum-inspired analysis for market signals
"""
import numpy as np
from typing import Dict, Any, Optional, Union
import logging
from datetime import datetime, timedelta
import pandas as pd
from .market_data import MarketState

logger = logging.getLogger(__name__)

class QuantumAnalyzer:
    """Analyzes market using quantum principles to generate trading signals"""

    def __init__(self, dimension: int = 64):
        """Initialize analyzer with quantum dimension"""
        self.dimension = dimension
        self.last_signal_time = None
        self.min_signal_interval = 60  # seconds
        self.current_state = None
        self.epsilon = 1e-10  # Numerical stability
        logger.info(f"Initialized QuantumAnalyzer with dimension {dimension}")

    def analyze(self, market_state: MarketState) -> Dict[str, Any]:
        """
        Analyze market state and generate trading signals

        Args:
            market_state: Current market state

        Returns:
            Dict with metrics and signals
        """
        try:
            # Check minimum interval between signals
            current_time = datetime.now()
            if (self.last_signal_time and 
                (current_time - self.last_signal_time).seconds < self.min_signal_interval):
                return {
                    'should_trade': False, 
                    'reason': 'Minimum interval not reached',
                    'metrics': self._get_safe_metrics()
                }

            # Calculate primary quantum metrics
            metrics = self.calculate_metrics(market_state)

            # Analyze signal strength
            signal_strength = self._calculate_signal_strength(metrics)

            # Define signal threshold
            signal_threshold = 0.7

            if abs(signal_strength) >= signal_threshold:
                self.last_signal_time = current_time
                return {
                    'should_trade': True,
                    'side': 'buy' if signal_strength > 0 else 'sell',
                    'metrics': metrics,
                    'strength': abs(signal_strength)
                }

            return {
                'should_trade': False,
                'reason': 'Insufficient signal',
                'metrics': metrics,
                'strength': abs(signal_strength)
            }

        except Exception as e:
            logger.error(f"Error analyzing market: {e}")
            return {
                'should_trade': False,
                'reason': f'Error: {str(e)}',
                'metrics': self._get_safe_metrics(),
                'strength': 0.0
            }

    def calculate_metrics(self, market_state: MarketState) -> Dict[str, float]:
        """Calculate quantum metrics from market state"""
        try:
            if not self._validate_market_state(market_state):
                return self._get_safe_metrics()

            # Normalize data
            normalized_data = self._normalize_data(market_state.data)

            # Calculate density matrix
            density_matrix = self._calculate_density_matrix(normalized_data)

            # Calculate metrics
            metrics = self._calculate_quantum_metrics(density_matrix)

            return metrics

        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return self._get_safe_metrics()

    def _validate_market_state(self, market_state: MarketState) -> bool:
        """Validate market state data"""
        try:
            return (hasattr(market_state, 'data') and 
                   market_state.data is not None and 
                   market_state.data.size > 0)
        except Exception as e:
            logger.error(f"Error validating market state: {e}")
            return False

    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize market data"""
        try:
            if data.size == 0:
                return np.zeros(self.dimension)

            normalized = (data - np.mean(data)) / (np.std(data) + self.epsilon)
            if normalized.size > self.dimension:
                normalized = normalized[:self.dimension]
            elif normalized.size < self.dimension:
                padding = np.zeros(self.dimension - normalized.size)
                normalized = np.concatenate([normalized, padding])

            return normalized / (np.linalg.norm(normalized) + self.epsilon)

        except Exception as e:
            logger.error(f"Error normalizing data: {e}")
            return np.zeros(self.dimension)

    def _calculate_density_matrix(self, normalized_data: np.ndarray) -> np.ndarray:
        """Calculate quantum density matrix"""
        try:
            density_matrix = np.outer(normalized_data.flatten(), 
                                    normalized_data.flatten().conj())
            return density_matrix / (np.trace(density_matrix) + self.epsilon)
        except Exception as e:
            logger.error(f"Error calculating density matrix: {e}")
            return np.eye(self.dimension) / self.dimension

    def _calculate_quantum_metrics(self, density_matrix: np.ndarray) -> Dict[str, float]:
        """Calculate quantum metrics from density matrix"""
        try:
            eigenvalues = np.linalg.eigvalsh(density_matrix)
            eigenvalues = eigenvalues[eigenvalues > self.epsilon]

            # von Neumann entropy
            entropy = -np.sum(eigenvalues * np.log2(eigenvalues + self.epsilon))
            entropy = np.clip(entropy / np.log2(self.dimension), 0, 1)

            # Quantum coherence
            coherence = np.abs(np.trace(density_matrix @ density_matrix))
            coherence = float(np.clip(coherence, 0, 1))

            metrics = {
                'coherence': coherence,
                'entropy': entropy,
                'market_stability': 1.0 - entropy,
                'quantum_alignment': float(np.clip(np.abs(np.trace(density_matrix)), 0, 1)),
                'morphic_resonance': float(np.clip(np.abs(coherence - entropy), 0, 1))
            }

            return {k: float(v) for k, v in metrics.items()}

        except Exception as e:
            logger.error(f"Error calculating quantum metrics: {e}")
            return self._get_safe_metrics()

    def _calculate_signal_strength(self, metrics: Dict[str, float]) -> float:
        """Calculate signal strength based on metrics"""
        try:
            weights = {
                'coherence': 0.3,
                'entropy': -0.2,
                'market_stability': 0.2,
                'quantum_alignment': 0.15,
                'morphic_resonance': 0.15
            }

            signal = sum(metrics[k] * w for k, w in weights.items() if k in metrics)
            return float(np.clip(signal, -1, 1))

        except Exception as e:
            logger.error(f"Error calculating signal strength: {e}")
            return 0.0

    def _get_safe_metrics(self) -> Dict[str, float]:
        """Return safe metrics in case of error"""
        return {
            'coherence': 0.0,
            'entropy': 0.0,
            'market_stability': 0.0,
            'quantum_alignment': 0.0,
            'morphic_resonance': 0.0
        }

    def get_metrics(self, quantum_state: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """
        Get metrics from quantum state with enhanced validation

        Args:
            quantum_state: Dictionary containing quantum state information

        Returns:
            Dictionary of quantum metrics with safe fallback values
        """
        try:
            if not quantum_state or not isinstance(quantum_state, dict):
                logger.warning("Invalid quantum state format")
                return self._get_safe_metrics()

            if 'metrics' in quantum_state:
                metrics = quantum_state['metrics']
                if isinstance(metrics, dict):
                    return {k: float(v) for k, v in metrics.items()}
                logger.warning("Invalid metrics format in quantum state")
                return self._get_safe_metrics()

            if 'should_trade' in quantum_state:
                # If it's an analysis result, try to extract metrics
                return quantum_state.get('metrics', self._get_safe_metrics())

            logger.warning("No valid metrics found in quantum state")
            return self._get_safe_metrics()

        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return self._get_safe_metrics()
    def _calculate_rsi(self, market_state: MarketState, period: int = 14) -> float:
        """Calcula RSI"""
        try:
            changes = np.diff([market_state.current_price])
            gains = np.where(changes > 0, changes, 0)
            losses = np.where(changes < 0, -changes, 0)
            
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0
            
            if avg_loss == 0:
                return 100
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi)
            
        except Exception as e:
            logger.error(f"Erro calculando RSI: {e}")
            return 50.0
    
    def _calculate_macd(self, market_state: MarketState) -> Dict[str, float]:
        """Calcula MACD"""
        try:
            # Simulação simples de MACD
            fast = market_state.current_price
            slow = market_state.open_price
            signal = (fast + slow) / 2
            
            macd_line = fast - slow
            signal_line = macd_line - signal
            histogram = macd_line - signal_line
            
            return {
                'macd': float(macd_line),
                'signal': float(signal_line),
                'histogram': float(histogram)
            }
            
        except Exception as e:
            logger.error(f"Erro calculando MACD: {e}")
            return {'macd': 0, 'signal': 0, 'histogram': 0}
    
    def _calculate_bollinger_bands(self, 
                                 market_state: MarketState, 
                                 period: int = 20,
                                 std_dev: float = 2.0) -> Dict[str, float]:
        """Calcula Bandas de Bollinger"""
        try:
            # Preço médio
            typical_price = (market_state.high_price + 
                           market_state.low_price + 
                           market_state.current_price) / 3
            
            # Desvio padrão
            std = np.std([market_state.high_price,
                         market_state.low_price,
                         market_state.current_price])
            
            middle_band = typical_price
            upper_band = middle_band + (std_dev * std)
            lower_band = middle_band - (std_dev * std)
            
            return {
                'upper': float(upper_band),
                'middle': float(middle_band),
                'lower': float(lower_band)
            }
            
        except Exception as e:
            logger.error(f"Erro calculando Bollinger Bands: {e}")
            return {
                'upper': market_state.current_price * 1.02,
                'middle': market_state.current_price,
                'lower': market_state.current_price * 0.98
            }
    
    def _analyze_volume(self, market_state: MarketState) -> float:
        """Analisa volume para sinal"""
        try:
            # Volume atual vs média
            volume_signal = 0.0
            
            if market_state.volume > 0:
                # Volume acima da média é positivo
                volume_signal = 0.5
                
                # Volume muito alto é mais positivo
                if market_state.volume > market_state.volume * 1.5:
                    volume_signal = 1.0
            
            return volume_signal
            
        except Exception as e:
            logger.error(f"Erro analisando volume: {e}")
            return 0.0
    
    def _analyze_trend(self, market_state: MarketState, window: int = 12) -> float:
        """Analisa tendência"""
        try:
            # Tendência baseada em preços
            if market_state.current_price > market_state.open_price:
                return 1.0  # Tendência de alta
            elif market_state.current_price < market_state.open_price:
                return -1.0  # Tendência de baixa
            return 0.0  # Lateral
            
        except Exception as e:
            logger.error(f"Erro analisando tendência: {e}")
            return 0.0
    
    def _calculate_base_price(self, 
                            market_state: MarketState,
                            bb: Dict[str, float]) -> float:
        """Calcula preço base para ordem"""
        try:
            # Usa média das Bandas de Bollinger como referência
            base_price = bb['middle']
            
            # Ajusta com o preço atual
            base_price = (base_price + market_state.current_price) / 2
            
            return float(base_price)
            
        except Exception as e:
            logger.error(f"Erro calculando preço base: {e}")
            return float(market_state.current_price)