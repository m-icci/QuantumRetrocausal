import unittest
from quantum.core.utils.simulate_merge import simulate_merge

class TestSimulateMerge(unittest.TestCase):

    def test_high_coherence_merge(self):
        state1 = QuantumState(np.array([0.8, 0.6, 0.2, 0.1], dtype=complex))
        state2 = QuantumState(np.array([0.7, 0.7, 0.1, 0.1], dtype=complex))
        merged_state = simulate_merge(state1, state2)
        self.assertIsNotNone(merged_state, "Merge should succeed with high coherence")

    def test_low_coherence_merge(self):
        state1 = QuantumState(np.array([0.1, 0.1, 0.1, 0.1], dtype=complex))
        state2 = QuantumState(np.array([0.1, 0.1, 0.1, 0.1], dtype=complex))
        merged_state = simulate_merge(state1, state2)
        self.assertIsNone(merged_state, "Merge should be rejected with low coherence")

if __name__ == '__main__':
    unittest.main()
