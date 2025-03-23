"""
Quantum Portfolio Optimizer

Implements portfolio optimization using quantum algorithms and QUALIA framework.
"""
from typing import Dict, Any, Optional
import numpy as np

from quantum_trading.core.quantum_state import QuantumState
from quantum_trading.morphic_field import MorphicField
from quantum_trading.analysis.holographic_memory import HolographicMemory

class QuantumPortfolioOptimizer:
    """
    Quantum-enhanced portfolio optimization with QUALIA integration
    """

    def __init__(self, n_assets: int = 8):
        """
        Initialize portfolio optimizer

        Args:
            n_assets: Number of assets in portfolio
        """
        self.n_assets = n_assets
        self.n_qubits = int(np.log2(n_assets * 8))

        # Initialize quantum state
        initial_state = np.zeros(2**self.n_qubits)
        initial_state[0] = 1.0  # Base state |0⟩
        self.state = QuantumState(state_vector=initial_state)

        # QUALIA components
        self.morphic_field = MorphicField()
        self.memory = HolographicMemory()

    def optimize_portfolio(self, 
                         assets: Dict[str, float],
                         constraints: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Optimize portfolio allocation using quantum algorithms

        Args:
            assets: Current asset allocations
            constraints: Optional trading constraints

        Returns:
            Optimized portfolio allocations
        """
        # Encode portfolio state
        portfolio_state = self._encode_portfolio(assets)

        # Apply morphic field optimization
        optimized_state = self.morphic_field.apply_field(portfolio_state, 
                                                        field_type="optimization",
                                                        constraints=constraints)

        # Store optimization pattern in memory
        self.memory.store(optimized_state)

        # Decode optimized allocations
        return self._decode_portfolio(optimized_state)

    def _encode_portfolio(self, assets: Dict[str, float]) -> QuantumState:
        """
        Encode portfolio data into quantum state

        Args:
            assets: Current asset allocations

        Returns:
            Encoded quantum state
        """
        encoded_data = np.array(list(assets.values()), dtype=np.float64)
        norm = np.linalg.norm(encoded_data)
        if norm > 0:
            encoded_data = encoded_data / norm

        new_state = np.zeros(2**self.n_qubits)
        new_state[:len(encoded_data)] = encoded_data

        self.state.state_vector = new_state
        return self.state

    def _decode_portfolio(self, state: QuantumState) -> Dict[str, float]:
        """
        Decode quantum state into portfolio allocations

        Args:
            state: Optimized quantum state

        Returns:
            Dictionary of optimized asset allocations
        """
        allocations = state.state_vector[:self.n_assets]

        # Normalize allocations
        norm = np.linalg.norm(allocations)
        if norm > 0:
            allocations = allocations / norm

        return {
            f"asset_{i}": float(alloc)
            for i, alloc in enumerate(allocations)
        }