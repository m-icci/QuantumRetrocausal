"""
Quantum Analysis Implementation with Enhanced Alert System
Implements quantum state evolution and analysis following QUALIA framework principles
"""
import numpy as np
import logging
import time
import traceback
from scipy import stats
from scipy.linalg import expm
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import warnings

from ..core.holographic_memory import HolographicMemory, HolographicPattern

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class QuantumState:
    """Data class for quantum state information"""
    state_vector: np.ndarray
    density_matrix: np.ndarray
    coherence: float
    entropy: float
    phase_space: Optional[np.ndarray] = None
    entanglement_measure: Optional[float] = None
    morphic_resonance: Optional[float] = None
    alerts: Optional[List[Any]] = None

class QuantumAnalyzer:
    """Enhanced Quantum state analyzer"""
    def __init__(
        self,
        dimension: int = 64,
        planck_constant: float = 1.0,
        memory_capacity: int = 1000,
        phi: float = 1.618033988749895  # Golden ratio
    ):
        self.dimension = dimension
        self.planck_constant = planck_constant
        self.phi = phi
        self.epsilon = 1e-10
        self.last_signal_time = None
        self.min_signal_interval = 60  # seconds

        # Initialize operators
        self.operators = self._initialize_operators()

        # Initialize holographic memory
        self.holographic_memory = HolographicMemory(
            dimension=dimension,
            memory_capacity=memory_capacity,
            phi=phi
        )

        # Initialize quantum state
        self.current_state = self._initialize_safe_state()

    def analyze(self, market_state: Any) -> Dict[str, Any]:
        """
        Main analysis method that processes market state and generates quantum metrics

        Args:
            market_state: Current market state with price and volume data

        Returns:
            Dict containing quantum analysis results and metrics
        """
        try:
            # Check signal interval
            current_time = time.time()
            if (self.last_signal_time and 
                (current_time - self.last_signal_time) < self.min_signal_interval):
                return {
                    'should_trade': False,
                    'reason': 'Minimum interval not reached',
                    'metrics': self._get_safe_metrics()
                }

            # Extract and validate market data
            if hasattr(market_state, 'data'):
                data = market_state.data
            else:
                data = np.array([market_state], dtype=np.float32)

            # Calculate quantum metrics
            metrics = self.calculate_metrics(data)

            # Analyze signal strength
            signal_strength = self._calculate_signal_strength(metrics)

            # Define threshold for signal
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
                'reason': 'Insufficient signal strength',
                'metrics': metrics,
                'strength': abs(signal_strength)
            }

        except Exception as e:
            logger.error(f"Error in quantum analysis: {e}")
            logger.error(traceback.format_exc())
            return {
                'should_trade': False,
                'reason': f'Error: {str(e)}',
                'metrics': self._get_safe_metrics()
            }

    def _calculate_signal_strength(self, metrics: Dict[str, float]) -> float:
        """Calculate trading signal strength from quantum metrics"""
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

    def _initialize_safe_state(self):
        """Initialize a safe quantum state"""
        state_vector = np.ones(self.dimension, dtype=complex) / np.sqrt(self.dimension)
        density_matrix = np.eye(self.dimension, dtype=complex) / self.dimension
        phase_space = np.zeros((self.dimension, self.dimension), dtype=complex)
        return QuantumState(
            state_vector=state_vector,
            density_matrix=density_matrix,
            coherence=0.5,
            entropy=0.5,
            phase_space=phase_space,
            entanglement_measure=0.5,
            morphic_resonance=0.5,
            alerts=[]
        )

    def calculate_metrics(self, data: np.ndarray) -> Dict[str, float]:
        """Calculate quantum metrics with enhanced stability"""
        try:
            # Ensure data is a numpy array
            data = np.asarray(data, dtype=np.float64)
            if not isinstance(data, np.ndarray):
                return self._get_safe_metrics()

            # Normalize data and ensure proper shape
            if data.size > 0:
                # Reshape data if it's 1D
                if data.ndim == 1:
                    data = data.reshape(-1, 1)

                # Normalize each dimension independently
                data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + self.epsilon)

                # Ensure proper dimension
                if data.shape[0] > self.dimension:
                    data = data[:self.dimension]
                elif data.shape[0] < self.dimension:
                    padding = np.zeros((self.dimension - data.shape[0], data.shape[1]))
                    data = np.vstack([data, padding])

                # Create density matrix
                state_vector = data.mean(axis=1)  # Collapse multiple features into single vector
                state_vector = state_vector / (np.linalg.norm(state_vector) + self.epsilon)
                density_matrix = np.outer(state_vector, state_vector.conj())
                density_matrix = density_matrix / (np.trace(density_matrix) + self.epsilon)

                # Calculate metrics
                eigenvalues = np.linalg.eigvalsh(density_matrix)
                eigenvalues = eigenvalues[eigenvalues > self.epsilon]

                if len(eigenvalues) > 0:
                    # Von Neumann entropy
                    entropy = -np.sum(eigenvalues * np.log2(eigenvalues + self.epsilon))
                    entropy = entropy / np.log2(self.dimension)  # Normalize

                    # Quantum coherence
                    coherence = np.abs(np.trace(density_matrix @ density_matrix))

                    # Calculate other metrics
                    metrics = {
                        'coherence': float(np.clip(coherence, 0, 1)),
                        'entropy': float(np.clip(entropy, 0, 1)),
                        'market_stability': float(1.0 - entropy),
                        'quantum_alignment': float(np.clip(np.abs(np.trace(
                            density_matrix @ self.operators['emergence']
                        )), 0, 1)),
                        'entanglement': float(np.clip(self._calculate_entanglement(density_matrix), 0, 1)),
                        'morphic_resonance': float(np.clip(self._calculate_resonance(density_matrix), 0, 1))
                    }

                    return metrics

            return self._get_safe_metrics()

        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return self._get_safe_metrics()

    def _initialize_operators(self) -> Dict[str, np.ndarray]:
        """Initialize quantum operators"""
        try:
            field_op = np.zeros((self.dimension, self.dimension), dtype=complex)
            for i in range(self.dimension):
                phase = 2 * np.pi * i / self.dimension
                field_op[i, i] = np.exp(1j * phase * self.planck_constant)

            identity = np.eye(self.dimension, dtype=complex)

            return {
                'field': field_op / (np.abs(np.trace(field_op)) + self.epsilon),
                'morphic': identity.copy(),
                'emergence': identity.copy()
            }

        except Exception as e:
            logger.error(f"Error initializing operators: {str(e)}")
            return {
                'field': np.eye(self.dimension, dtype=complex),
                'morphic': np.eye(self.dimension, dtype=complex),
                'emergence': np.eye(self.dimension, dtype=complex)
            }

    def _calculate_entanglement(self, state: np.ndarray) -> float:
        """Calculate quantum entanglement measure"""
        try:
            eigenvalues = np.linalg.eigvalsh(state)
            eigenvalues = eigenvalues[eigenvalues > self.epsilon]
            if len(eigenvalues) == 0:
                return 0.5

            entropy = -np.sum(eigenvalues * np.log2(eigenvalues + self.epsilon))
            return float(entropy / np.log2(self.dimension))

        except Exception as e:
            logger.error(f"Error calculating entanglement: {str(e)}")
            return 0.5

    def _calculate_resonance(self, state: np.ndarray) -> float:
        """Calculate morphic resonance"""
        try:
            resonance = np.abs(np.trace(self.operators['morphic'] @ state))
            norm = np.sqrt(np.abs(np.trace(self.operators['morphic'] @ self.operators['morphic'].conj().T)) * 
                         np.abs(np.trace(state @ state.conj().T)))
            return float(resonance / (norm + self.epsilon))

        except Exception as e:
            logger.error(f"Error calculating resonance: {str(e)}")
            return 0.5

    def _get_safe_metrics(self) -> Dict[str, float]:
        """Return safe metrics for error cases"""
        return {
            'coherence': 0.5,
            'entropy': 0.5,
            'market_stability': 0.5,
            'quantum_alignment': 0.5,
            'entanglement': 0.5,
            'morphic_resonance': 0.5
        }

    def evolve_state(self, market_data: np.ndarray) -> Optional[QuantumState]:
        """Evolve quantum state based on market data with enhanced stability"""
        try:
            # Normalize market data with improved stability
            normalized_data = self._normalize_data(market_data)

            # Store pattern in holographic memory with stability check
            pattern = normalized_data.flatten()
            if np.all(np.isfinite(pattern)):
                self.holographic_memory.store_pattern(pattern)

            # Time evolution operator with improved stability
            time_step = 0.1
            ham = self.operators['field']
            eigenvalues = np.linalg.eigvalsh(ham)
            max_eigenval = np.max(np.abs(eigenvalues))
            scaled_ham = ham * min(1.0, 50.0 / (max_eigenval + self.epsilon))

            evolution_op = expm(-1j * scaled_ham * time_step)

            # Apply quantum operations with enhanced numerical stability
            state = np.eye(self.dimension, dtype=complex) / self.dimension # Initialize state safely

            # Safe matrix multiplication with intermediate normalization
            state = evolution_op @ state @ evolution_op.conj().T
            trace = np.abs(np.trace(state)) + self.epsilon
            state = state / trace

            # Calculate quantum metrics with improved stability
            metrics = self.calculate_metrics(state)
            state_vector = np.diagonal(state).real
            phase_space = np.zeros((self.dimension, self.dimension))

            # Update current state with validation
            new_state = QuantumState(
                state_vector=state_vector,
                density_matrix=state,
                coherence=metrics['coherence'],
                entropy=metrics['entropy'],
                phase_space=phase_space,
                entanglement_measure=metrics['entanglement'],
                morphic_resonance=metrics['morphic_resonance'],
                alerts=[]
            )

            self.current_state = new_state
            return self.current_state

        except Exception as e:
            logger.error(f"State evolution failed: {str(e)}")
            return self.current_state

    def _normalize_data(self, market_state: Any) -> np.ndarray:
        """
        Normaliza dados do mercado para análise quântica

        Args:
            market_state: Estado do mercado

        Returns:
            np.ndarray: Dados normalizados
        """
        try:
            # Extract data from market_state
            if hasattr(market_state, 'data') and isinstance(market_state.data, np.ndarray):
                data = market_state.data
            else:
                # If market_state is already an ndarray, use it directly
                data = np.asarray(market_state, dtype=np.float32)

            # Reshape data if needed
            if data.ndim == 1:
                data = data.reshape(-1, 1)

            # Ensure we have at least some data
            if data.size == 0:
                return np.zeros((1, 5), dtype=np.float32)

            # Normalize each column separately
            normalized = np.zeros_like(data, dtype=np.float32)
            for i in range(data.shape[1]):
                col = data[:, i]
                col_min, col_max = np.min(col), np.max(col)

                if col_max > col_min:
                    normalized[:, i] = (col - col_min) / (col_max - col_min)
                else:
                    normalized[:, i] = 0.5  # Valor neutro se não houver variação

            return normalized

        except Exception as e:
            logger.error(f"Error in normalization: {str(e)}")
            return np.zeros((1, 5), dtype=np.float32)

    def get_current_metrics(self) -> Dict[str, float]:
        """Get current quantum metrics with validation"""
        try:
            if self.current_state is None:
                return self._get_safe_metrics()

            metrics = self.calculate_metrics(self.current_state.density_matrix)

            # Ensure all metrics are real and within [0,1]
            return {k: float(np.clip(np.real(v), 0, 1)) for k, v in metrics.items()}

        except Exception as e:
            logger.error(f"Error getting current metrics: {str(e)}")
            return self._get_safe_metrics()

    def calculate_dark_ratio(self) -> float:
        """Calculate dark energy ratio"""
        try:
            if self.current_state is None:
                return 0.5

            # Total system energy
            total_energy = np.trace(np.abs(self.current_state.density_matrix))

            # Dark energy (off-diagonal elements)
            dark_energy = np.sum(np.abs(self.current_state.density_matrix)) - np.trace(np.abs(self.current_state.density_matrix))

            # Ratio between dark and total energy
            dark_ratio = dark_energy / (total_energy + 1e-10)
            return float(np.clip(dark_ratio, 0, 1))

        except Exception as e:
            logger.error(f"Error calculating dark ratio: {str(e)}")
            return 0.5

    def calculate_consciousness(self) -> float:
        """Calculate system consciousness level"""
        try:
            if self.current_state is None:
                return 0.5

            # Calculate through information integration
            # Higher integration = higher consciousness
            eigenvalues = np.linalg.eigvals(self.current_state.density_matrix)
            consciousness = np.abs(np.prod(eigenvalues)) ** (1.0/len(eigenvalues))
            return float(np.clip(consciousness, 0, 1))

        except Exception as e:
            logger.error(f"Error calculating consciousness: {str(e)}")
            return 0.5
    def calculate_coherence(self) -> float:
        """Calcula coerência do estado quântico"""
        try:
            if self.current_state is None:
                return 0.5

            # Calcula coerência como medida de ordem no sistema
            eigenvalues = np.linalg.eigvals(self.current_state.density_matrix)
            coherence = np.abs(np.sum(eigenvalues)) / (np.sum(np.abs(eigenvalues)) + 1e-10)
            return float(np.clip(coherence, 0, 1))

        except Exception as e:
            logger.error(f"Erro calculando coerência: {e}")
            return 0.5

    def calculate_field_entropy(self) -> float:
        """Calcula entropia do campo quântico"""
        try:
            if not hasattr(self, 'current_state') or self.current_state is None:
                return 0.5

            if not hasattr(self.current_state, 'density_matrix'):
                logger.error("Current state does not have density_matrix attribute")
                return 0.5

            # Calcula entropia de von Neumann com validação adicional
            try:
                eigenvalues = np.linalg.eigvals(self.current_state.density_matrix)
                # Remove valores muito pequenos para evitar problemas numéricos
                eigenvalues = eigenvalues[np.abs(eigenvalues) > 1e-10]

                if len(eigenvalues) == 0:
                    return 0.5

                # Normaliza probabilidades
                probabilities = np.abs(eigenvalues)
                probabilities = probabilities / np.sum(probabilities)

                # Calcula entropia
                entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))

                # Normaliza para [0,1]
                max_entropy = np.log2(len(eigenvalues))
                if max_entropy == 0:
                    return 0.5

                normalized_entropy = entropy / (max_entropy + 1e-10)
                return float(np.clip(normalized_entropy, 0, 1))

            except np.linalg.LinAlgError as e:
                logger.error(f"Linear algebra error in entropy calculation: {e}")
                return 0.5

        except Exception as e:
            logger.error(f"Error calculating entropy: {str(e)}")
            logger.error(traceback.format_exc())
            return 0.5

    def calculate_dark_ratio(self) -> float:
        """Calcula razão de energia escura"""
        try:
            if self.current_state is None:
                return 0.5

            # Energia total do sistema
            total_energy = np.trace(np.abs(self.current_state.density_matrix))

            # Energia "escura" (elementos fora da diagonal)
            dark_energy = np.sum(np.abs(self.current_state.density_matrix)) - np.trace(np.abs(self.current_state.density_matrix))

            # Razão entre energia escura e total
            dark_ratio = dark_energy / (total_energy + 1e-10)
            return float(np.clip(dark_ratio, 0, 1))

        except Exception as e:
            logger.error(f"Erro calculando dark ratio: {e}")
            return 0.5

    def calculate_consciousness(self) -> float:
        """Calcula nível de consciência do sistema"""
        try:
            if self.current_state is None:
                return 0.5

            # Calcula através da integração de informação
            # Maior integração = maior consciência
            eigenvalues = np.linalg.eigvals(self.current_state.density_matrix)
            consciousness = np.abs(np.prod(eigenvalues)) ** (1.0/len(eigenvalues))
            return float(np.clip(consciousness, 0, 1))

        except Exception as e:
            logger.error(f"Erro calculando consciência: {e}")
            return 0.5


    def get_quantum_metrics(self, market_state: Any) -> Dict[str, float]:
        """
        Calcula métricas quânticas do estado atual do mercado

        Args:
            market_state: Estado atual do mercado

        Returns:
            Dict com métricas quânticas
        """
        try:
            # Extrai dados do market_state
            if hasattr(market_state, 'data'):
                data = market_state.data
            else:
                data = market_state

            # Calcula métricas usando os dados diretamente
            metrics = self.calculate_metrics(data)

            return {
                'coherence': metrics['coherence'],
                'dark_ratio': self.calculate_dark_ratio(),
                'consciousness_level': self.calculate_consciousness(),
                'field_entropy': metrics['entropy']
            }

        except Exception as e:
            logger.error(f"Erro calculando métricas quânticas: {e}")
            return {
                'coherence': 0.5,
                'dark_ratio': 0.5,
                'consciousness_level': 0.5,
                'field_entropy': 0.5
            }

    def get_metrics(self, quantum_state: Dict[str, Any]) -> Dict[str, float]:
        """Get quantum metrics from analyzed state

        Args:
            quantum_state: Dictionary containing analyzed quantum state and metrics

        Returns:
            Dict containing normalized quantum metrics
        """
        try:
            if not quantum_state:
                return self._get_safe_metrics()

            # Extract base metrics if available
            if 'metrics' in quantum_state:
                return quantum_state['metrics']

            # Calculate metrics from raw state
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
            logger.error(f"Error getting metrics from quantum state: {e}")
            return self._get_safe_metrics()