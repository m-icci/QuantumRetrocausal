import unittest
from quantum.core.consciousness.pattern_resonance import PatternResonance
from quantum.core.state.quantum_state import QuantumState

class TestPatternResonance(unittest.TestCase):
    def setUp(self):
        self.pattern_resonance = PatternResonance(threshold=0.5)

    def test_detect_pattern_above_threshold(self):
        quantum_state = QuantumState(amplitude=[1, 0], phase=0, coherence=0.6)  # Coherence above threshold
        result = self.pattern_resonance.detect_pattern(quantum_state)
        self.assertTrue(result)
        self.assertIn(quantum_state, self.pattern_resonance.get_patterns())

    def test_detect_pattern_below_threshold(self):
        quantum_state = QuantumState(amplitude=[0, 1], phase=0, coherence=0.4)  # Coherence below threshold
        result = self.pattern_resonance.detect_pattern(quantum_state)
        self.assertFalse(result)
        self.assertNotIn(quantum_state, self.pattern_resonance.get_patterns())

if __name__ == '__main__':
    unittest.main()
