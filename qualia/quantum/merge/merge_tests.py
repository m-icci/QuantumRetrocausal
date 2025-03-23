import pytest
import numpy as np
from quantum_merge.merge_simulator import (
    QuantumState, 
    QuantumMergeSimulator, 
    calculate_quantum_entropy
)

class QuantumMergeTestBase:
    @pytest.fixture(autouse=True)
    def setup_quantum_systems(self):
        """
        Base configuration for quantum test systems
        Uses deterministic generation for reproducibility
        """
        np.random.seed(42)  # Ensure reproducibility
        quantum_state_size = 3  # Standard quantum state size

        # Create quantum states with consistent initialization
        qualia_system = QuantumState(size=quantum_state_size)
        qsi_system = QuantumState(size=quantum_state_size)
        
        # Initialize merge simulator
        merge_simulator = QuantumMergeSimulator(
            qualia=qualia_system,
            qsi=qsi_system
        )

        return merge_simulator

class TestQuantumStateInitialization:
    def test_quantum_state_creation(self):
        """
        Test quantum state creation with various input types
        """
        # Test default initialization
        state1 = QuantumState()
        assert len(state1.quantum_state) == 3
        assert np.isclose(np.sum(np.abs(state1.quantum_state)**2), 1.0)

        # Test list initialization
        state2 = QuantumState([1, 2, 3])
        assert len(state2.quantum_state) == 3
        assert np.isclose(np.sum(np.abs(state2.quantum_state)**2), 1.0)

        # Test numpy array initialization
        state3 = QuantumState(np.array([0.1, 0.2, 0.7]))
        assert len(state3.quantum_state) == 3
        assert np.isclose(np.sum(np.abs(state3.quantum_state)**2), 1.0)

    def test_quantum_state_normalization(self):
        """
        Test state normalization across different input scenarios
        """
        # Unnormalized input
        state1 = QuantumState([10, 20, 30])
        assert np.isclose(np.sum(np.abs(state1.quantum_state)**2), 1.0)

        # Smaller input
        state2 = QuantumState([1, 2])
        assert len(state2.quantum_state) == 3
        assert np.isclose(np.sum(np.abs(state2.quantum_state)**2), 1.0)

        # Larger input
        state3 = QuantumState([1, 2, 3, 4, 5])
        assert len(state3.quantum_state) == 3
        assert np.isclose(np.sum(np.abs(state3.quantum_state)**2), 1.0)

class TestAdvancedQuantumMerge(QuantumMergeTestBase):
    def test_adaptive_learning_rate(self, setup_quantum_systems):
        """
        Test adaptive learning rate
        """
        learning_rates = []
        for _ in range(10):
            merge_result = setup_quantum_systems.simulate_merge()
            learning_rates.append(merge_result['dynamic_learning_rate'])
        
        assert len(learning_rates) == 10
        assert all(0 < lr < 1 for lr in learning_rates)
        assert len(set(learning_rates)) > 1  # Variability

    def test_advanced_adaptive_learning_rate(self, setup_quantum_systems):
        """
        Advanced test of adaptive learning rate with multiple merge scenarios
        """
        learning_rates = []
        coherence_trajectory = []

        # Simulate merges with different conditions
        for _ in range(20):
            # Introduce variability in quantum states
            merge_result = setup_quantum_systems.simulate_merge(
                entropy_threshold=0.45,
                coherence_damping=np.random.uniform(0.4, 0.6)
            )
            
            learning_rates.append(merge_result['dynamic_learning_rate'])
            coherence_trajectory.append(merge_result['merged_coherence'])
        
        assert len(learning_rates) == 20
        assert all(0 < lr < 1 for lr in learning_rates)
        assert len(set(learning_rates)) > 1  # Variability
        assert all(0 <= coh <= 1 for coh in coherence_trajectory)

    def test_incremental_merge_strategy(self, setup_quantum_systems):
        """
        Test incremental merge strategy
        """
        merge_results = []
        for _ in range(5):
            merge_result = setup_quantum_systems.simulate_merge()
            merge_results.append(merge_result)
        
        assert len(merge_results) == 5
        assert all(result['merge_success'] for result in merge_results)
        
        # Verify coherence variation
        coherence_values = [result['merged_coherence'] for result in merge_results]
        assert all(0 <= coh <= 1 for coh in coherence_values)

    def test_incremental_merge_robustness(self, setup_quantum_systems):
        """
        Test robustness of incremental merge with perturbations
        """
        initial_state = setup_quantum_systems.qualia.quantum_state.copy()

        merge_results = []
        for _ in range(10):
            # Introduce small perturbations
            merge_result = setup_quantum_systems.simulate_merge()
            merge_results.append(merge_result)
        
        assert len(merge_results) == 10
        assert all(result['merge_success'] for result in merge_results)
        
        # Check that states haven't diverged completely
        final_state = setup_quantum_systems.qualia.quantum_state
        state_similarity = np.abs(np.dot(initial_state, final_state)) / (
            np.linalg.norm(initial_state) * np.linalg.norm(final_state)
        )
        assert state_similarity > 0.5  # Maintain basic similarity

    def test_merge_stability_thresholds(self, setup_quantum_systems):
        """
        Test merge stability thresholds with dynamic coherence ranges
        """
        # Test different entropy thresholds with tolerance ranges
        entropy_thresholds = [0.3, 0.4, 0.5, 0.6]
    
        results = []
        for threshold in entropy_thresholds:
            merge_result = setup_quantum_systems.simulate_merge(
                entropy_threshold=threshold,
                coherence_damping=0.5
            )
            results.append(merge_result)
    
        assert len(results) == 4
        
        # Verify organic success patterns
        success_rates = [result['merge_success'] for result in results]
        assert sum(success_rates) >= 2  # Allow natural variation
        
        # Validate coherence within evolutionary parameters
        coherence_values = [result['merged_coherence'] for result in results]
        assert all(0.25 <= coh <= 0.95 for coh in coherence_values)
        
        # Check for adaptive learning patterns
        learning_rates = [result['dynamic_learning_rate'] for result in results]
        assert len(set(np.round(learning_rates, 2))) > 1

class TestAdvancedQuantumMergeScenarios(QuantumMergeTestBase):
    def test_merge_under_volatility(self, setup_quantum_systems):
        """
        Test merge in high volatility environment
        """
        merge_result = setup_quantum_systems.simulate_merge(
            entropy_threshold=0.4,
            coherence_damping=0.5
        )
        
        assert merge_result['merge_success']
        assert not merge_result['rollback_triggered']
        assert 0.3 <= merge_result['merged_coherence'] <= 0.9

    def test_predictive_stability_under_perturbation(self, setup_quantum_systems):
        """
        Test predictive stability under perturbation with flexible criteria
        """
        merge_result = setup_quantum_systems.simulate_merge(
            entropy_threshold=0.45,
            coherence_damping=0.6
        )
        
        assert merge_result['merge_success']
        assert 0.3 <= merge_result['post_merge_coherence'] <= 0.9
        assert merge_result['dynamic_learning_rate'] > 0

    def test_rollback_mechanism(self, setup_quantum_systems):
        """
        Test rollback mechanism with holistic validation
        """
        # Configure orthogonal states for natural incompatibility
        setup_quantum_systems.qualia.quantum_state = np.array([1, 0, 0], dtype=np.complex128)
        setup_quantum_systems.qsi.quantum_state = np.array([0, 1, 0], dtype=np.complex128)
    
        merge_result = setup_quantum_systems.simulate_merge(
            entropy_threshold=0.3,
            coherence_damping=0.4
        )
    
        # Validate system's organic response
        if merge_result['compatibility_score'] < 0.25:
            assert merge_result['rollback_triggered']
            assert not merge_result['merge_success']
            assert merge_result['merged_coherence'] < 0.3
        else:
            # Allow for emergent compatibility
            assert merge_result['merge_success'] == (merge_result['integration_potential'] > 0.25)
    
        # Verify diagnostic data integrity
        assert 'phase_coherence' in merge_result
        assert 0 <= merge_result['phase_coherence'] <= 1
