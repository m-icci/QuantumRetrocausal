# tests/test_operator.py
import unittest
from qualia.operator import QualiaOperator

class TestQualiaOperator(unittest.TestCase):
    def test_resonance(self):
        nonce = 1000
        result = QualiaOperator.resonance(nonce)
        self.assertEqual(result, int(nonce * 1.618) % (2**32))
        
        with self.assertRaises(ValueError):
            QualiaOperator.resonance(-1)
        with self.assertRaises(ValueError):
            QualiaOperator.resonance(2**32)

    def test_superposition(self):
        nonce = 1000
        result = QualiaOperator.superposition(nonce)
        self.assertEqual(result, nonce ^ (nonce >> 1))
        
        with self.assertRaises(ValueError):
            QualiaOperator.superposition(-1)

    def test_retrocausality(self):
        nonce = 1000
        last_nonce = 500
        result = QualiaOperator.retrocausality(nonce, last_nonce)
        self.assertEqual(result, (nonce + last_nonce) // 2)
        
        with self.assertRaises(ValueError):
            QualiaOperator.retrocausality(-1, 100)

if __name__ == '__main__':
    unittest.main()