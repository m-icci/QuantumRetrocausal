# tests/test_miner.py
import unittest
import pandas as pd
from qualia.miner import QualiaAdaptiveMiner

class TestQualiaAdaptiveMiner(unittest.TestCase):
    def setUp(self):
        self.miner = QualiaAdaptiveMiner(difficulty=2)

    def test_initialization(self):
        with self.assertRaises(ValueError):
            QualiaAdaptiveMiner(difficulty=0)
        with self.assertRaises(ValueError):
            QualiaAdaptiveMiner(gran_min=100, gran_max=50)

    def test_hash_function(self):
        nonce = 1000
        hash_value = self.miner.hash_function(nonce)
        self.assertEqual(len(hash_value), 64)  # SHA-256 produces 64 hex chars
        self.assertTrue(all(c in '0123456789abcdef' for c in hash_value))

    def test_mine_block(self):
        block_header = "test_block"
        predicted_nonce, results_df, total_time = self.miner.mine_block(
            block_header, max_nonces=2)
        
        self.assertIsInstance(predicted_nonce, int)
        self.assertIsInstance(results_df, pd.DataFrame)
        self.assertIsInstance(total_time, float)
        self.assertEqual(len(results_df), 2)

        with self.assertRaises(ValueError):
            self.miner.mine_block("", max_nonces=1)
        with self.assertRaises(ValueError):
            self.miner.mine_block("test", max_nonces=0)

if __name__ == '__main__':
    unittest.main()