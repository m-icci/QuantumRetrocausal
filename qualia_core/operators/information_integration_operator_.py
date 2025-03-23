import numpy as np
from typing import List, Tuple, Dict
from scipy.linalg import logm
from .coherence_operator import CoherenceOperator

class InformationIntegrationOperator:
    """
    Implements the Information Integration Operator (OII) as defined in M-ICCI framework.

    Enhanced implementation incorporating:
    1. IIT principles for quantum systems
    2. Multiple integration measures
    3. Proper handling of quantum correlations

    This operator quantifies how quantum information is processed and integrated across
    different components of the system, following principles from both quantum information
    theory and integrated information theory as described in M-ICCI sections 1.3 and 2.
    """

    def __init__(self, precision: float = 1e-15):
        """
        Initialize component operators and parameters

        Args:
            precision: Numerical precision threshold for calculations
        """
        self.coherence_op = CoherenceOperator(precision=precision)
        self.precision = precision

    def quantum_mutual_information(self, density_matrix: np.ndarray, 
                                 partition: Tuple[int, ...]) -> float:
        """
        Calculates quantum mutual information between subsystems.
        I(A:B) = S(ρA) + S(ρB) - S(ρAB)

        Enhanced with:
        1. Proper handling of quantum correlations
        2. Numerical stability in entropy calculations
        3. Validation of quantum state properties

        Args:
            density_matrix: Full system density matrix
            partition: Tuple specifying dimensions of subsystems

        Returns:
            float: Quantum mutual information value
        """
        if not self._is_valid_density_matrix(density_matrix):
            raise ValueError("Invalid density matrix")

        # Calculate reduced density matrices with proper partial trace
        rho_a = self._partial_trace(density_matrix, partition, 'A')
        rho_b = self._partial_trace(density_matrix, partition, 'B')

        # Calculate von Neumann entropies with numerical stability
        s_a = self._von_neumann_entropy(rho_a)
        s_b = self._von_neumann_entropy(rho_b)
        s_ab = self._von_neumann_entropy(density_matrix)

        # Handle numerical precision
        mutual_info = float(s_a + s_b - s_ab)
        return max(0.0, mutual_info)  # Ensure non-negative

    def integrated_information(self, density_matrix: np.ndarray, 
                             partitions: List[Tuple[int, ...]]) -> Dict[str, float]:
        """
        Calculates integrated information (Φ) across multiple partitions.
        Enhanced implementation of quantum integrated information theory.

        New features:
        1. Multiple integration measures
        2. Proper handling of quantum correlations
        3. Enhanced partition analysis

        Args:
            density_matrix: System density matrix
            partitions: List of possible system partitions to consider

        Returns:
            Dict[str, float]: Integration measures including Φ and additional metrics
        """
        # Validate input state
        if not self._is_valid_density_matrix(density_matrix):
            raise ValueError("Invalid density matrix")

        # Calculate baseline entropy with proper numerical handling
        total_entropy = self._von_neumann_entropy(density_matrix)

        # Calculate integration across partitions with enhanced metrics
        partition_info = []
        effective_info = []
        for partition in partitions:
            # Calculate quantum mutual information
            mutual_info = self.quantum_mutual_information(density_matrix, partition)
            partition_info.append(mutual_info)

            # Calculate effective information (new metric)
            eff_info = self._calculate_effective_information(density_matrix, partition)
            effective_info.append(eff_info)

        # Calculate Φ as minimum information loss across partitions
        phi = min(partition_info) if partition_info else 0.0

        # Calculate effective Φ (new metric incorporating quantum effects)
        effective_phi = min(effective_info) if effective_info else 0.0

        # Calculate relative integration metrics
        relative_phi = phi / total_entropy if total_entropy > self.precision else 0.0
        relative_effective_phi = effective_phi / total_entropy if total_entropy > self.precision else 0.0

        return {
            'phi': float(phi),
            'effective_phi': float(effective_phi),
            'relative_phi': float(relative_phi),
            'relative_effective_phi': float(relative_effective_phi),
            'total_entropy': float(total_entropy),
            'partition_mutual_info': [float(x) for x in partition_info],
            'effective_information': [float(x) for x in effective_info]
        }

    def _calculate_effective_information(self, density_matrix: np.ndarray, 
                                      partition: Tuple[int, ...]) -> float:
        """
        Calculates effective information for a partition incorporating quantum effects

        Args:
            density_matrix: System density matrix
            partition: System partition

        Returns:
            float: Effective information value
        """
        # Calculate reduced states
        rho_a = self._partial_trace(density_matrix, partition, 'A')
        rho_b = self._partial_trace(density_matrix, partition, 'B')

        # Calculate quantum discord-like measure
        mutual_info = self.quantum_mutual_information(density_matrix, partition)
        classical_corr = self._classical_correlations(density_matrix, partition)

        # Effective information includes both classical and quantum correlations
        return float(mutual_info + (mutual_info - classical_corr))

    def _classical_correlations(self, density_matrix: np.ndarray, 
                              partition: Tuple[int, ...]) -> float:
        """
        Estimates classical correlations in the quantum state
        """
        # Simplified classical correlations estimate
        # This could be enhanced with proper POVM measurements in future versions
        rho_a = self._partial_trace(density_matrix, partition, 'A')
        rho_b = self._partial_trace(density_matrix, partition, 'B')

        eigenvals_a = np.linalg.eigvalsh(rho_a)
        eigenvals_b = np.linalg.eigvalsh(rho_b)

        # Classical correlations based on eigenvalue distributions
        classical_corr = 0.0
        for p_a in eigenvals_a:
            for p_b in eigenvals_b:
                if p_a > self.precision and p_b > self.precision:
                    classical_corr -= p_a * p_b * np.log2(p_a * p_b)

        return float(max(0.0, classical_corr))

    def _von_neumann_entropy(self, density_matrix: np.ndarray) -> float:
        """
        Calculates von Neumann entropy S(ρ) = -Tr(ρ log₂ ρ)
        Enhanced with proper numerical stability
        """
        eigenvals = np.linalg.eigvalsh(density_matrix)
        eigenvals = eigenvals[eigenvals > self.precision]
        entropy = 0.0
        for p in eigenvals:
            if p > self.precision:
                entropy -= p * np.log2(p)
        return float(entropy)

    def _is_valid_density_matrix(self, matrix: np.ndarray) -> bool:
        """
        Enhanced validation of density matrix properties
        """
        if not isinstance(matrix, np.ndarray):
            return False
        if matrix.shape[0] != matrix.shape[1]:
            return False
        if not np.allclose(matrix, matrix.conj().T, atol=self.precision):
            return False
        if not np.isclose(np.trace(matrix), 1.0, atol=self.precision):
            return False
        eigenvals = np.linalg.eigvalsh(matrix)
        if np.any(eigenvals < -self.precision):
            return False
        return True

    def _partial_trace(self, density_matrix: np.ndarray, 
                      dims: Tuple[int, ...], subsystem: str = 'A') -> np.ndarray:
        """
        Calculates partial trace over subsystem with enhanced numerical stability
        """
        d_a, d_b = dims
        if subsystem.upper() == 'A':
            shape = (d_a, d_b, d_a, d_b)
            rho = density_matrix.reshape(shape)
            return np.trace(rho, axis1=1, axis2=3)
        else:
            shape = (d_a, d_b, d_a, d_b)
            rho = density_matrix.reshape(shape)
            return np.trace(rho, axis1=0, axis2=2)