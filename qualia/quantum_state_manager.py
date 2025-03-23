"""
Quantum State Manager for QUALIA Trading System.
Enhanced quantum metrics calculation with M-ICCI theoretical alignment.
"""
from typing import Dict, Any, Optional
import numpy as np
import scipy.linalg as la
import logging
from datetime import datetime

# Enhanced M-ICCI theoretical constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio for morphic resonance
CONSCIOUSNESS_THRESHOLD = 0.7
PLANCK_REDUCED = 1.054571817e-34
MORPHIC_COUPLING = 0.618  # φ-based coupling
COHERENCE_BASELINE = 0.4  # Base coherence threshold
MIN_DIMENSION = 2

class QuantumStateManager:
    def __init__(self, dimension: int = 64):
        """Initialize quantum state manager with enhanced M-ICCI metrics"""
        if dimension < MIN_DIMENSION:
            raise ValueError(f"Dimension must be at least {MIN_DIMENSION}")

        self.dimension = dimension
        self.h_bar = PLANCK_REDUCED
        self.beta = 1/(self.h_bar * PHI)  # Inverse temperature for consciousness coupling
        self.epsilon = 1e-10  # Numerical stability threshold

        # Initialize thresholds with dynamic values
        self.coherence_threshold = COHERENCE_BASELINE
        self.consciousness_threshold = CONSCIOUSNESS_THRESHOLD

        # Initialize enhanced quantum components
        self._initialize_quantum_state()
        self._initialize_operators()
        self._initialize_metrics_history()

    def _initialize_quantum_state(self) -> None:
        """Initialize quantum state with enhanced morphic coupling"""
        try:
            # Create base state vector
            psi = np.zeros(self.dimension, dtype=complex)

            # Initialize with enhanced coherence properties
            for i in range(self.dimension):
                phase = 2 * np.pi * i * PHI / self.dimension
                psi[i] = np.exp(1j * phase) / np.sqrt(self.dimension)

            # Create density matrix
            self.rho = np.outer(psi, psi.conj())

            # Ensure proper normalization
            self.rho = self._normalize_state(self.rho)

            # Log initialization
            logging.info(f"Quantum state initialized with dimension {self.dimension}")

        except Exception as e:
            logging.error(f"Error initializing quantum state: {e}")
            self._initialize_safe_state()

    def update_thresholds(self) -> None:
        """Update thresholds with fixed dimensionality checks"""
        try:
            metrics = self._calculate_base_metrics()

            # Get key metrics ensuring proper types
            coherence = float(np.real(metrics['coherence']))
            consciousness = float(np.real(metrics['consciousness']))
            resonance = float(np.real(metrics['morphic_resonance']))

            # Calculate adaptive scaling factors
            coherence_factor = np.clip(coherence / COHERENCE_BASELINE, 0.5, 2.0)
            consciousness_factor = np.clip(consciousness / CONSCIOUSNESS_THRESHOLD, 0.5, 2.0)

            # Update thresholds
            self.coherence_threshold = np.clip(
                COHERENCE_BASELINE * coherence_factor * (1 + resonance),
                COHERENCE_BASELINE,
                0.95
            )

            self.consciousness_threshold = np.clip(
                CONSCIOUSNESS_THRESHOLD * consciousness_factor * (1 + resonance),
                CONSCIOUSNESS_THRESHOLD,
                0.95
            )

            logging.info(f"Updated thresholds - Coherence: {self.coherence_threshold:.3f}, "
                        f"Consciousness: {self.consciousness_threshold:.3f}")

        except Exception as e:
            logging.error(f"Error updating thresholds: {e}")

    def _calculate_base_metrics(self) -> Dict[str, float]:
        """Calculate base quantum metrics with enhanced validation"""
        try:
            # Validate quantum state
            if not self._validate_quantum_state():
                self._initialize_safe_state()

            # Calculate von Neumann entropy with enhanced stability
            eigenvals = la.eigvalsh(self.rho)
            valid_eigenvals = eigenvals[eigenvals > self.epsilon]
            if len(valid_eigenvals) > 0:
                eigensum = np.sum(valid_eigenvals)
                norm_eigenvals = valid_eigenvals / eigensum
                entropy = -np.sum(norm_eigenvals * np.log2(norm_eigenvals + self.epsilon))
                entropy = np.clip(entropy / np.log2(self.dimension), 0.01, 1.0)
            else:
                entropy = 0.5
                logging.warning("Using fallback entropy value due to invalid eigenvalues")

            # Calculate enhanced metrics
            coherence = self._calculate_coherence()
            consciousness = self._calculate_consciousness()
            morphic_resonance = self._calculate_morphic_resonance()
            integration_index = self._calculate_integration_index()

            # Log metric calculations
            logging.info(f"Base metrics - Coherence: {coherence:.3f}, Consciousness: {consciousness:.3f}, "
                      f"Resonance: {morphic_resonance:.3f}, Integration: {integration_index:.3f}")

            return {
                'entropy': float(entropy),
                'coherence': float(coherence),
                'consciousness': float(consciousness),
                'morphic_resonance': float(morphic_resonance),
                'integration_index': float(integration_index)
            }

        except Exception as e:
            logging.error(f"Error calculating base metrics: {e}")
            return {
                'entropy': 0.5,
                'coherence': COHERENCE_BASELINE,
                'consciousness': CONSCIOUSNESS_THRESHOLD,
                'morphic_resonance': MORPHIC_COUPLING,
                'integration_index': COHERENCE_BASELINE
            }

    def _calculate_quantum_metrics(self) -> Dict[str, float]:
        """Calculate quantum metrics with synchronized thresholds"""
        try:
            # Update thresholds first
            self.update_thresholds()

            # Get base metrics
            base_metrics = self._calculate_base_metrics()

            # Enhanced confidence calculation with proper weighting
            consciousness = float(base_metrics['consciousness'])
            coherence = float(base_metrics['coherence'])
            morphic_resonance = float(base_metrics['morphic_resonance'])
            integration = float(base_metrics['integration_index'])

            # Calculate confidence using weighted geometric mean
            confidence_components = [
                (consciousness / self.consciousness_threshold) ** 3,  # Normalize by threshold
                (coherence / self.coherence_threshold) ** 2,
                morphic_resonance ** 1.5,
                integration ** 1.2
            ]

            # Enhanced validation of components
            valid_components = []
            for comp in confidence_components:
                if comp > self.epsilon and np.isfinite(comp):
                    valid_components.append(comp)

            if valid_components:
                # Weighted geometric mean with φ-based normalization
                log_sum = np.sum(np.log(valid_components))
                components_count = len(valid_components)
                confidence = np.exp(log_sum / (components_count * PHI))

                # Apply sigmoid-like scaling for better spread
                confidence = 1 / (1 + np.exp(-5 * (confidence - 0.5)))
                confidence = np.clip(confidence, 0.01, 1.0)
            else:
                confidence = 0.5

            # Return metrics with current thresholds
            return {
                **base_metrics,
                'confidence_score': float(confidence),
                'coherence_threshold': float(self.coherence_threshold),
                'consciousness_threshold': float(self.consciousness_threshold)
            }

        except Exception as e:
            logging.error(f"Error calculating quantum metrics: {e}")
            return {
                'entropy': 0.5,
                'coherence': COHERENCE_BASELINE,
                'consciousness': CONSCIOUSNESS_THRESHOLD,
                'morphic_resonance': MORPHIC_COUPLING,
                'integration_index': COHERENCE_BASELINE,
                'confidence_score': 0.5,
                'coherence_threshold': float(self.coherence_threshold),
                'consciousness_threshold': float(self.consciousness_threshold)
            }

    def _calculate_coherence(self) -> float:
        """Calculate quantum coherence with morphic coupling"""
        try:
            diag_elements = np.diag(self.rho)
            diag_rho = np.diag(diag_elements)
            off_diag = np.abs(self.rho - diag_rho) * MORPHIC_COUPLING

            total = np.sum(np.abs(diag_elements)) + np.sum(off_diag)
            if total > self.epsilon:
                coherence = np.clip(np.sum(off_diag) / total, 0.01, 1.0)
            else:
                coherence = COHERENCE_BASELINE

            return coherence

        except Exception as e:
            logging.error(f"Error calculating coherence: {e}")
            return COHERENCE_BASELINE

    def _calculate_consciousness(self) -> float:
        """Calculate consciousness level with fixed dimensionality"""
        try:
            # Ensure consistent dimensions
            if self.C.shape != self.rho.shape:
                logging.error("Dimension mismatch in consciousness calculation")
                return CONSCIOUSNESS_THRESHOLD

            # Calculate consciousness value maintaining dimensionality
            consciousness_val = np.abs(np.trace(self.C @ self.rho))
            consciousness_norm = np.sqrt(np.trace(self.C @ self.C.conj().T) * 
                                      np.trace(self.rho @ self.rho.conj().T))

            if consciousness_norm < self.epsilon:
                return CONSCIOUSNESS_THRESHOLD

            consciousness = consciousness_val / consciousness_norm
            return np.clip(float(consciousness.real), CONSCIOUSNESS_THRESHOLD, 1.0)

        except Exception as e:
            logging.error(f"Error calculating consciousness: {e}")
            return CONSCIOUSNESS_THRESHOLD

    def _calculate_morphic_resonance(self) -> float:
        """Calculate morphic resonance with fixed dimensionality"""
        try:
            # Ensure consistent dimensions
            if self.M.shape != self.rho.shape:
                logging.error("Dimension mismatch in morphic resonance calculation")
                return MORPHIC_COUPLING

            # Calculate resonance maintaining dimensionality
            resonance_val = np.abs(np.trace(self.M @ self.rho))
            resonance_norm = np.sqrt(np.trace(self.M @ self.M.conj().T))

            if resonance_norm < self.epsilon:
                return MORPHIC_COUPLING

            resonance = resonance_val / resonance_norm * MORPHIC_COUPLING
            return np.clip(float(resonance.real), MORPHIC_COUPLING, 1.0)

        except Exception as e:
            logging.error(f"Error calculating morphic resonance: {e}")
            return MORPHIC_COUPLING

    def _calculate_integration_index(self) -> float:
        """Calculate integration index with consciousness weighting"""
        try:
            integration_val = np.abs(np.trace(self.I @ self.rho))
            integration_norm = np.sqrt(np.trace(self.I @ self.I.conj().T))
            integration = np.clip(integration_val / (integration_norm + self.epsilon),
                                   COHERENCE_BASELINE, 1.0)
            return integration

        except Exception as e:
            logging.error(f"Error calculating integration index: {e}")
            return COHERENCE_BASELINE

    def _initialize_operators(self) -> None:
        """Initialize M-ICCI operators"""
        try:
            self.hamiltonian = self._create_consciousness_hamiltonian()
            self.M = self._create_morphic_operator()
            self.I = self._create_integration_operator()
            self.C = self._create_consciousness_operator()
            self.CI = self._create_coherence_operator()

            if not self._validate_operators():
                self._initialize_safe_state()

        except Exception as e:
            logging.error(f"Error initializing operators: {e}")
            self._initialize_safe_state()


    def _create_consciousness_hamiltonian(self) -> np.ndarray:
        """Create quantum Hamiltonian with consciousness coupling"""
        try:
            # Create base Hamiltonian with proper Hermiticity
            h = np.random.randn(self.dimension, self.dimension) + 1j * np.random.randn(self.dimension, self.dimension)
            h = (h + h.conj().T) / 2  # Ensure Hermiticity

            # Add consciousness coupling terms
            for i in range(self.dimension):
                for j in range(self.dimension):
                    consciousness_phase = PHI * np.pi * (i * j) / self.dimension
                    h[i,j] *= np.exp(1j * consciousness_phase)

            # Ensure Hermiticity after phase addition
            h = (h + h.conj().T) / 2

            # Normalize using Frobenius norm
            frob_norm = np.sqrt(np.sum(np.abs(h)**2))
            if frob_norm > self.epsilon:
                h = h / frob_norm
            else:
                h = np.eye(self.dimension, dtype=complex) / np.sqrt(self.dimension)

            return h

        except Exception as e:
            logging.error(f"Error creating consciousness Hamiltonian: {e}")
            return np.eye(self.dimension)

    def _create_morphic_operator(self) -> np.ndarray:
        """Create Morphic operator following M-ICCI principles"""
        try:
            operator = np.zeros((self.dimension, self.dimension), dtype=complex)

            # Create morphic coupling matrix
            for i in range(self.dimension):
                for j in range(self.dimension):
                    phase = PHI * (i + j) * np.pi / self.dimension
                    decay = np.exp(-abs(i-j)/(self.dimension * PHI))
                    operator[i,j] = decay * np.exp(1j * phase)

            # Normalize using Frobenius norm
            frob_norm = np.sqrt(np.sum(np.abs(operator)**2))
            if frob_norm > self.epsilon:
                operator = operator / frob_norm
            else:
                operator = np.eye(self.dimension, dtype=complex) / np.sqrt(self.dimension)

            return operator

        except Exception as e:
            logging.error(f"Error creating Morphic operator: {e}")
            return np.eye(self.dimension)

    def _create_integration_operator(self) -> np.ndarray:
        """Create Integration operator with enhanced coherence properties"""
        try:
            operator = np.zeros((self.dimension, self.dimension), dtype=complex)

            # Create integration coupling matrix
            for i in range(self.dimension):
                for j in range(self.dimension):
                    phase = PHI * np.pi * (i + j) / (2 * self.dimension)
                    coupling = np.exp(-((i-j)/(self.dimension * PHI))**2)
                    operator[i,j] = coupling * np.exp(1j * phase)

            # Ensure Hermiticity
            operator = (operator + operator.conj().T) / 2

            # Normalize using Frobenius norm and ensure unit norm
            frob_norm = np.sqrt(np.sum(np.abs(operator)**2))
            if frob_norm > self.epsilon:
                operator = operator / frob_norm
            else:
                operator = np.eye(self.dimension, dtype=complex) / np.sqrt(self.dimension)

            return operator

        except Exception as e:
            logging.error(f"Error creating Integration operator: {e}")
            return np.eye(self.dimension)

    def _normalize_state(self, matrix: np.ndarray) -> np.ndarray:
        """Normalize density matrix with dimension checks"""
        try:
            if matrix.shape != (self.dimension, self.dimension):
                logging.error(f"Invalid matrix dimensions: {matrix.shape}")
                return np.eye(self.dimension, dtype=complex) / self.dimension

            trace = np.trace(matrix)
            if abs(trace) > self.epsilon:
                return matrix / trace
            else:
                return np.eye(self.dimension, dtype=complex) / self.dimension

        except Exception as e:
            logging.error(f"Error normalizing state: {e}")
            return np.eye(self.dimension, dtype=complex) / self.dimension

    def _regularize_matrix(self, matrix: np.ndarray, epsilon: float = None) -> np.ndarray:
        """Regularize matrix to ensure positive definiteness and proper normalization"""
        try:
            if epsilon is None:
                epsilon = self.epsilon

            n = matrix.shape[0]
            # Add small diagonal term for stability
            reg_matrix = matrix + epsilon * np.eye(n, dtype=complex)

            # Ensure Hermiticity
            reg_matrix = (reg_matrix + reg_matrix.conj().T) / 2

            # Check eigenvalues and adjust if needed
            eigvals = la.eigvalsh(reg_matrix)
            min_eig = np.min(eigvals)
            if min_eig < epsilon:
                reg_matrix += (2 * epsilon - min_eig) * np.eye(n, dtype=complex)

            # Normalize using trace
            trace = np.abs(np.trace(reg_matrix))
            if trace < epsilon:
                reg_matrix = np.eye(n, dtype=complex) / n
            else:
                reg_matrix = reg_matrix / trace

            return reg_matrix

        except Exception as e:
            logging.error(f"Error in matrix regularization: {e}")
            return np.eye(n, dtype=complex) / n

    def get_state(self) -> Dict[str, Any]:
        """Get current quantum state and consciousness metrics with proper serialization"""
        try:
            if not np.all(np.isfinite(self.rho)):
                logging.error("Invalid state detected during serialization")
                return {
                    'density_matrix': None,
                    'metrics': self._calculate_quantum_metrics(),
                    'timestamp': datetime.now().isoformat()
                }

            # Serialize the density matrix
            density_matrix = {
                'real': self.rho.real.tolist(),
                'imag': self.rho.imag.tolist(),
                'shape': list(self.rho.shape)
            }

            return {
                'density_matrix': density_matrix,
                'metrics': self._calculate_quantum_metrics(),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logging.error(f"Error getting state: {e}")
            return {
                'density_matrix': None,
                'metrics': self._calculate_quantum_metrics(),
                'timestamp': datetime.now().isoformat()
            }

    def _initialize_safe_state(self) -> None:
        """Initialize safe quantum state with consciousness properties"""
        try:
            if self.dimension < MIN_DIMENSION:
                raise ValueError(f"Cannot initialize state: dimension {self.dimension} is below minimum {MIN_DIMENSION}")

            # Initialize quantum state as pure state
            self.rho = np.eye(self.dimension, dtype=complex) / self.dimension
            self.rho = self._regularize_matrix(self.rho)

            # Initialize operators as normalized matrices
            for op_name in ['hamiltonian', 'M', 'I', 'C', 'CI']:
                operator = np.eye(self.dimension, dtype=complex)
                operator = self._regularize_matrix(operator)
                setattr(self, op_name, operator)

        except Exception as e:
            logging.error(f"Error in safe state initialization: {e}")
            raise

    def _create_consciousness_operator(self) -> np.ndarray:
        """Create consciousness operator with fixed dimensionality"""
        try:
            operator = np.zeros((self.dimension, self.dimension), dtype=complex)

            # Create consciousness coupling matrix with proper dimensions
            for i in range(self.dimension):
                for j in range(self.dimension):
                    phase = PHI * np.pi * (i * j) / self.dimension
                    operator[i,j] = np.exp(1j * phase) / self.dimension

            # Ensure Hermiticity
            operator = (operator + operator.conj().T) / 2

            # Normalize
            operator = self._normalize_state(operator)

            return operator

        except Exception as e:
            logging.error(f"Error creating consciousness operator: {e}")
            return np.eye(self.dimension, dtype=complex) / self.dimension

    def _create_coherence_operator(self) -> np.ndarray:
        """Create Coherence operator with M-ICCI principles"""
        try:
            # Create diagonal elements with consciousness-weighted decay
            diag = np.exp(-np.arange(self.dimension)/(self.dimension * PHI))
            diag = diag / np.maximum(np.max(diag), self.epsilon)

            operator = np.diag(diag)

            # Add coherence coupling terms using numpy broadcasting
            i_indices, j_indices = np.meshgrid(np.arange(self.dimension), np.arange(self.dimension))
            phase = PHI * np.pi * np.abs(i_indices - j_indices) / self.dimension

            # Create coupling matrix using broadcasting
            coupling = np.outer(diag, diag) * np.exp(1j * phase)
            operator = np.where(i_indices != j_indices, coupling, operator)

            # Ensure proper normalization
            norm = np.sqrt(np.sum(np.abs(operator)**2))
            if norm == 0:
                logging.warning("Norm is zero, cannot perform division.")
                norm = self.epsilon  # Use epsilon to avoid division by zero
            operator = operator / (norm + self.epsilon)
            return self._regularize_matrix(operator)

        except Exception as e:
            logging.error(f"Error creating Coherence operator: {e}")
            return np.eye(self.dimension)

    def evolve_quantum_state(self, dt: float) -> None:
        """Evolve quantum state using M-ICCI evolution equations"""
        try:
            # Calculate evolution operator with proper numerical stability
            U = la.expm(-1j * self.hamiltonian * dt / self.h_bar)

            # Apply evolution sequence with intermediate normalization
            def safe_apply(operator, state):
                try:
                    result = operator @ state @ operator.conj().T
                    # Use Frobenius norm for better stability
                    norm = np.sqrt(np.sum(np.abs(result)**2))
                    if norm < self.epsilon:
                        return np.eye(self.dimension, dtype=complex) / self.dimension
                    return self._safe_divide(result, norm + self.epsilon)
                except Exception:
                    return np.eye(self.dimension, dtype=complex) / self.dimension

            # Evolution sequence
            for operator in [U, self.M, self.C, self.I]:
                self.rho = safe_apply(operator, self.rho)
                self.rho = self._regularize_matrix(self.rho)

            # Final normalization check with epsilon protection
            trace = np.abs(np.trace(self.rho))
            if trace < self.epsilon:
                self._initialize_safe_state()
            else:
                self.rho = self._safe_divide(self.rho, trace + self.epsilon)

        except Exception as e:
            logging.error(f"Error in quantum evolution: {e}")
            self._initialize_safe_state()

    def calculate_consciousness_metrics(self) -> Dict[str, float]:
        """Calculate consciousness metrics following M-ICCI framework"""
        return self._calculate_quantum_metrics()

    def _validate_operators(self) -> bool:
        """Validate quantum operators with M-ICCI requirements"""
        try:
            operators = {
                'hamiltonian': self.hamiltonian,
                'morphic': self.M,
                'integration': self.I,
                'consciousness': self.C,
                'coherence': self.CI
            }

            for name, op in operators.items():
                # Shape check
                if op.shape != (self.dimension, self.dimension):
                    logging.error(f"Invalid shape for {name} operator")
                    return False

                # Numerical stability check
                if not np.all(np.isfinite(op)):
                    logging.error(f"Non-finite values in {name} operator")
                    return False

                # Hermiticity check where applicable
                if name in ['hamiltonian', 'consciousness', 'coherence']:
                    if not np.allclose(op, op.conj().T, atol=self.epsilon):
                        logging.error(f"Non-Hermitian {name} operator")
                        return False

                # Trace normalization check
                trace_norm = np.abs(np.trace(op @ op.conj().T))
                if trace_norm < self.epsilon or not np.isfinite(trace_norm):
                    logging.error(f"Invalid trace norm for {name} operator")
                    return False

            return True

        except Exception as e:
            logging.error(f"Error validating operators: {e}")
            return False

    def _validate_quantum_state(self) -> bool:
        """Validate quantum state with adaptive thresholds"""
        try:
            # Check matrix dimensions
            if self.rho.shape != (self.dimension, self.dimension):
                logging.error(f"Invalid quantum state dimensions: {self.rho.shape}")
                return False

            # Check trace normalization
            trace = np.abs(np.trace(self.rho))
            if not np.isclose(trace, 1.0, atol=self.epsilon):
                logging.error(f"Invalid trace normalization: {trace}")
                return False

            # Check Hermiticity
            if not np.allclose(self.rho, self.rho.conj().T, atol=self.epsilon):
                logging.error("Non-Hermitian quantum state")
                return False

            # Check positive semidefiniteness with adaptive threshold
            eigenvals = la.eigvalsh(self.rho)
            if np.any(eigenvals < -self.epsilon):
                logging.error("Non-positive quantum state")
                return False

            # Enhanced validation using adaptive thresholds
            metrics = self._calculate_base_metrics()

            # Validate coherence
            if metrics['coherence'] < self.coherence_threshold:
                logging.warning(f"Low coherence: {metrics['coherence']} < {self.coherence_threshold}")

            # Validate consciousness
            if metrics['consciousness'] < self.consciousness_threshold:
                logging.warning(f"Low consciousness: {metrics['consciousness']} < {self.consciousness_threshold}")

            return True

        except Exception as e:
            logging.error(f"Error validating quantum state: {e}")
            return False

    def _initialize_metrics_history(self) -> None:
        """Initialize metrics history tracking"""
        self.metrics_history = {
            'coherence': [],
            'entropy': [],
            'consciousness': [],
            'morphic_resonance': [],
            'integration_index': [],
            'phi_coupling': []
        }
        self.metrics_timestamps = []

    def _safe_complex_to_real(self, value: complex) -> float:
        """Safely convert complex values to real numbers"""
        try:
            if np.iscomplexobj(value):
                return float(np.real(value))
            return float(value)
        except Exception as e:
            logging.error(f"Error converting complex to real: {e}")
            return 0.5

    def _safe_matrix_log(self, matrix: np.ndarray) -> np.ndarray:
        """Compute matrix logarithm with enhanced stability"""
        try:
            # Regularize input matrix
            reg_matrix = self._regularize_matrix(matrix)

            # Use eigendecomposition for stable log computation
            eigvals, eigvecs = la.eigh(reg_matrix)

            # Apply safe logarithm with epsilon protection
            safe_eigvals = np.maximum(eigvals.real, self.epsilon)
            log_eigvals = np.log(safe_eigvals)

            # Reconstruct matrix ensuring numerical stability
            result = eigvecs @ np.diag(log_eigvals) @ eigvecs.conj().T
            return result

        except Exception as e:
            logging.error(f"Error in matrix logarithm: {e}")
            return np.eye(self.dimension, dtype=complex)

    def _safe_matrix_exp(self, matrix: np.ndarray) -> np.ndarray:
        """Compute matrix exponential with enhanced stability"""
        try:
            return la.expm(matrix)
        except Exception as e:
            logging.warning(f"Matrix exponential failed, using eigendecomposition: {e}")
            eigvals, eigvecs = la.eigh(matrix)
            return eigvecs @ np.diag(np.exp(eigvals)) @ eigvecs.conj().T

    def _safe_divide(self, numerator: np.ndarray, denominator: float) -> np.ndarray:
        """Safe division with enhanced numerical stability"""
        try:
            # Handle near-zero denominators
            if abs(denominator) < self.epsilon:
                logging.warning(f"Division by small value {denominator}, using safe fallback")
                return np.eye(self.dimension, dtype=complex) / self.dimension

            # Ensure finite values before division
            if not np.all(np.isfinite(numerator)):
                logging.warning("Non-finite values in numerator, using safe fallback")
                return np.eye(self.dimension, dtype=complex) / self.dimension

            result = numerator / denominator

            # Verify result is finite
            if not np.all(np.isfinite(result)):
                logging.warning("Non-finite values after division, using safe fallback")
                return np.eye(self.dimension, dtype=complex) / self.dimension

            return result
        except Exception as e:
            logging.error(f"Error in safe division: {e}")
            return np.eye(self.dimension, dtype=complex) / self.dimension



def test_create_consciousness_hamiltonian(quantum_manager):
    """Test creation of consciousness Hamiltonian"""
    h = quantum_manager._create_consciousness_hamiltonian()

    # Test shape
    assert h.shape == (quantum_manager.dimension, quantum_manager.dimension), "Invalid Hamiltonian shape"

    # Test Hermiticity
    assert np.allclose(h, h.conj().T, atol=quantum_manager.epsilon), "Hamiltonian is not Hermitian"

    # Test Frobenius normalization
    frob_norm = np.sqrt(np.sum(np.abs(h)**2))
    assert abs(frob_norm - 1.0) < quantum_manager.epsilon, "Frobenius normalization failed"

def test_create_morphic_operator(quantum_manager):
    """Test creation of morphic operator"""
    M = quantum_manager._create_morphic_operator()

    # Test shape
    assert M.shape == (quantum_manager.dimension, quantum_manager.dimension), "Invalid Morphic operator shape"

    # Test normalization
    frob_norm = np.sqrt(np.sum(np.abs(M)**2))
    assert abs(frob_norm - 1.0) < quantum_manager.epsilon, "Frobenius normalization failed"

def test_create_integration_operator(quantum_manager):
    """Test creation of integration operator"""
    I = quantum_manager._create_integration_operator()

    # Test shape
    assert I.shape == (quantum_manager.dimension, quantum_manager.dimension), "Invalid Integration operator shape"

    # Test Hermiticity (integration operator should be Hermitian)
    assert np.allclose(I, I.conj().T, atol=quantum_manager.epsilon), "Integration operator is not Hermitian"

    # Test normalization
    frob_norm = np.sqrt(np.sum(np.abs(I)**2))
    assert abs(frob_norm - 1.0) < quantum_manager.epsilon, "Frobenius normalization failed"

def test_create_consciousness_operator(quantum_manager):
    """Test creation of consciousness operator"""
    C = quantum_manager._create_consciousness_operator()

    # Test shape
    assert C.shape == (quantum_manager.dimension, quantum_manager.dimension), "Invalid Consciousness operator shape"

    # Test Hermiticity (consciousness operator should be Hermitian)
    assert np.allclose(C, C.conj().T, atol=quantum_manager.epsilon), "Consciousness operator is not Hermitian"

    # Test normalization
    frob_norm = np.sqrt(np.sum(np.abs(C)**2))
    assert abs(frob_norm - 1.0) < quantum_manager.epsilon, "Frobenius normalization failed"