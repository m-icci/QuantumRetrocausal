# tests/test_predictor.py
import unittest
from qualia.predictor import QualiaPredictor

class TestQualiaPredictor(unittest.TestCase):
    def setUp(self):
        self.predictor = QualiaPredictor()

    def test_predict_nonce_returns_valid_range(self):
        block = "test_block"
        nonce = self.predictor.predict_nonce(block)
        self.assertIsInstance(nonce, int)
        self.assertTrue(0 <= nonce < 2**32)

    def test_predict_nonce_deterministic(self):
        block = "test_block"
        nonce1 = self.predictor.predict_nonce(block)
        nonce2 = self.predictor.predict_nonce(block)
        self.assertEqual(nonce1, nonce2)

    def test_predict_nonce_invalid_input(self):
        with self.assertRaises(ValueError):
            self.predictor.predict_nonce("")
        with self.assertRaises(ValueError):
            self.predictor.predict_nonce(None)

if __name__ == '__main__':
    unittest.main()