import numpy as np
from typing import Tuple, Optional
from scipy.linalg import sqrtm

class EntanglementOperator:
    """
    Implements the Entanglement Operator (OE) as defined in M-ICCI framework.
    Quantifies quantum entanglement through various metrics including:
    - Concurrence for 2-qubit systems
    - Von Neumann entropy of reduced density matrices
    - Quantum mutual information

    Enhanced with proper dimension handling and normalization for M-ICCI alignment.
    """

    def concurrence(self, density_matrix: np.ndarray) -> float:
        """
        Calculates the concurrence for a two-qubit state as a measure of entanglement.
        For a mixed state ρ, C(ρ) = max(0, λ₁ - λ₂ - λ₃ - λ₄)
        where λᵢ are square roots of eigenvalues of ρ(σy⊗σy)ρ*(σy⊗σy) in decreasing order

        Args:
            density_matrix: 4x4 density matrix representing two-qubit state

        Returns:
            float: Concurrence value between 0 (separable) and 1 (maximally entangled)
        """
        if density_matrix.shape != (4, 4):
            raise ValueError("Concurrence is defined only for two-qubit states (4x4 density matrices)")

        # Pauli Y matrix
        sigma_y = np.array([[0, -1j], [1j, 0]])

        # Calculate R = ρ(σy⊗σy)ρ*(σy⊗σy)
        sigma_y_tensor = np.kron(sigma_y, sigma_y)
        R = density_matrix @ sigma_y_tensor @ density_matrix.conj() @ sigma_y_tensor

        # Get eigenvalues and sort in decreasing order
        eigenvals = np.sqrt(np.abs(np.linalg.eigvals(R)))
        eigenvals = np.sort(eigenvals)[::-1]

        # Calculate concurrence with proper normalization
        concurrence = max(0.0, float(eigenvals[0] - eigenvals[1] - eigenvals[2] - eigenvals[3]))
        return concurrence

    def entanglement_entropy(self, density_matrix: np.ndarray, partition: Optional[Tuple[int, ...]] = None) -> float:
        """
        Calculates the entanglement entropy for a bipartite split of the system.
        S = -Tr(ρₐ log₂ ρₐ) where ρₐ is the reduced density matrix of subsystem A

        Enhanced with:
        1. Proper partition dimension handling
        2. Automatic partitioning for standard cases
        3. Improved numerical stability

        Args:
            density_matrix: Density matrix of the complete system
            partition: Optional tuple specifying dimensions of subsystems (dᴀ, dᴮ)

        Returns:
            float: Normalized entanglement entropy [0,1]
        """
        dim = density_matrix.shape[0]

        # Handle default partitioning
        if partition is None:
            # For power of 2 dimensions, split evenly
            if (dim & (dim - 1)) == 0:  # Check if power of 2
                d = int(np.sqrt(dim))
                partition = (d, d)
            else:
                # For other dimensions, use largest possible bipartition
                d = int(np.floor(np.sqrt(dim)))
                partition = (d, dim//d)

        # Validate partition dimensions
        d_a, d_b = partition
        if d_a * d_b != dim:
            raise ValueError(f"Invalid partition dimensions {partition} for system dimension {dim}")

        # Calculate reduced density matrix by partial trace
        rho_a = self._partial_trace(density_matrix, partition, subsystem='A')

        # Calculate von Neumann entropy with improved numerical stability
        eigenvals = np.linalg.eigvalsh(rho_a)
        eigenvals = eigenvals[eigenvals > 1e-15]  # Remove numerical noise
        entropy = float(-np.sum(eigenvals * np.log2(eigenvals + 1e-15)))

        # Normalize to [0,1] range
        # Maximum entropy for d_a dimensional system is log2(d_a)
        max_entropy = np.log2(d_a)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        return float(normalized_entropy)

    def _partial_trace(self, density_matrix: np.ndarray, dims: Tuple[int, ...], subsystem: str = 'A') -> np.ndarray:
        """
        Calculates the partial trace over subsystem B (or A) to get reduced density matrix.
        Enhanced with proper dimension handling and validation.

        Args:
            density_matrix: Full density matrix
            dims: Tuple of (dᴀ, dᴮ) for subsystem dimensions
            subsystem: Which subsystem to keep ('A' or 'B')

        Returns:
            np.ndarray: Reduced density matrix
        """
        d_a, d_b = dims
        if d_a * d_b != density_matrix.shape[0]:
            raise ValueError("Dimensions mismatch in partial trace calculation")

        try:
            if subsystem.upper() == 'A':
                shape = (d_a, d_b, d_a, d_b)
                rho = density_matrix.reshape(shape)
                return np.trace(rho, axis1=1, axis2=3)
            else:  # trace out system A
                shape = (d_a, d_b, d_a, d_b)
                rho = density_matrix.reshape(shape)
                return np.trace(rho, axis1=0, axis2=2)
        except ValueError as e:
            raise ValueError(f"Error in partial trace: {str(e)}. Check if dimensions {dims} are compatible.")