import unittest
import sys
import os
import logging
import threading
import time
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mining_metrics_api import MiningMetricsAPI
from start_mining import MiningController
from qualia_miner import QualiaMiner
from wallet import WalletManager
from test_utils import MockPool, EntropyGenerator, FractalAnalyzer, RetrocausalitySimulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestMiningIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create mock pool
        cls.mock_pool = MockPool()
        cls.pool_port = cls.mock_pool.actual_port
        
        # Initialize test wallet
        cls.test_wallet_path = "test_wallet.json"
        cls.wallet = WalletManager(cls.test_wallet_path)
        
        # Initialize components
        cls.metrics_api = MiningMetricsAPI()
        cls.mining_controller = MiningController(
            pool_address="localhost",
            pool_port=cls.pool_port,
            wallet_address=cls.wallet.wallet_data["address"]
        )
        
        # Initialize test utilities
        cls.entropy_gen = EntropyGenerator(seed=42)
        cls.fractal_analyzer = FractalAnalyzer()
        cls.retrocausality_sim = RetrocausalitySimulator()

    def setUp(self):
        self.start_time = datetime.now()

    def tearDown(self):
        # Clean up any test files
        if os.path.exists(self.test_wallet_path):
            os.remove(self.test_wallet_path)

    @classmethod
    def tearDownClass(cls):
        cls.mock_pool.stop()

    def test_basic_integration(self):
        """Test 1: Basic Integration Test"""
        logger.info("Starting basic integration test...")
        
        # Start mining
        mining_thread = threading.Thread(
            target=self.mining_controller.start_mining
        )
        mining_thread.daemon = True
        mining_thread.start()
        
        # Wait for mining to initialize
        time.sleep(5)
        
        try:
            # Check mining status
            status = self.mining_controller.get_mining_status()
            self.assertEqual(status["status"], "running")
            
            # Check metrics collection
            metrics = self.metrics_api.monitor_resources()
            self.assertIsNotNone(metrics.get("cpu_usage_percent"))
            self.assertIsNotNone(metrics.get("memory_usage_percent"))
            
            # Test wallet operations
            initial_balance = self.wallet.wallet_data["balance"]
            test_transaction = {
                "timestamp": datetime.now().isoformat(),
                "amount": 1.0,
                "sender": "test_sender",
                "recipient": self.wallet.wallet_data["address"],
                "signature": "0" * 128
            }
            
            # Validate transaction
            self.assertTrue(self.wallet.validate_transaction(test_transaction))
            
        finally:
            self.mining_controller.stop_mining()
            mining_thread.join(timeout=5)

    def test_fractal_hpc_integration(self):
        """Test 2: Fractal Analysis and HPC Integration"""
        logger.info("Starting fractal and HPC integration test...")
        
        # Generate test data with varying entropy
        sequence_length = 1000
        high_entropy_seq = self.entropy_gen.generate_hash_sequence(
            sequence_length, entropy_level=0.9
        )
        low_entropy_seq = self.entropy_gen.generate_hash_sequence(
            sequence_length, entropy_level=0.3
        )
        
        # Calculate fractal dimensions
        high_entropy_fd = self.fractal_analyzer.calculate_fractal_dimension(high_entropy_seq)
        low_entropy_fd = self.fractal_analyzer.calculate_fractal_dimension(low_entropy_seq)
        
        # Verify fractal analysis
        self.assertGreater(high_entropy_fd, low_entropy_fd)
        
        # Test mining with fractal analysis
        miner = QualiaMiner()
        miner.performance_metrics["hash_rate"] = 1000
        
        # Simulate different entropy conditions
        for sequence in [high_entropy_seq, low_entropy_seq]:
            fd = self.fractal_analyzer.calculate_fractal_dimension(sequence)
            if fd < 1.5:  # Threshold for low complexity
                # Adjust mining parameters
                miner.config["thread_count"] = max(1, miner.config["thread_count"] - 1)
            
            metrics = miner.monitor_performance()
            self.assertIn("efficiency_metrics", metrics)

    def test_retrocausality(self):
        """Test 3: Retrocausality Integration"""
        logger.info("Starting retrocausality test...")
        
        # Generate future states
        future_states = self.retrocausality_sim.generate_future_states(datetime.now())
        self.assertGreater(len(future_states), 0)
        
        # Test mining with future state injection
        miner = QualiaMiner()
        
        for i in range(5):
            # Get future state prediction
            future_state = self.retrocausality_sim.get_future_state(i * 10)
            
            # Adjust mining based on future state
            if future_state["network_difficulty"] > 1.1:
                # Reduce intensity if difficulty is predicted to increase
                miner.config["max_cpu_usage"] = max(50, miner.config["max_cpu_usage"] - 5)
            
            # Monitor performance
            metrics = miner.monitor_performance()
            self.assertIn("mining_metrics", metrics)
            
            # Update future buffer
            self.retrocausality_sim.update_buffer()
            time.sleep(1)

    def test_security_and_payments(self):
        """Test 4: Security and Payment Verification"""
        logger.info("Starting security and payment verification test...")
        
        # Test transaction validation
        valid_transaction = {
            "timestamp": datetime.now().isoformat(),
            "amount": 1.0,
            "sender": "valid_sender",
            "recipient": self.wallet.wallet_data["address"],
            "signature": "0" * 128  # Valid format
        }
        
        invalid_transaction = {
            "timestamp": datetime.now().isoformat(),
            "amount": -1.0,  # Invalid amount
            "sender": "invalid_sender",
            "signature": "invalid"  # Invalid format
        }
        
        # Test valid transaction
        self.assertTrue(self.wallet.validate_transaction(valid_transaction))
        
        # Test invalid transaction
        self.assertFalse(self.wallet.validate_transaction(invalid_transaction))
        
        # Test payment ID extraction
        valid_payment = {"payment_id": "0" * 64}
        invalid_payment = {"payment_id": "invalid"}
        
        self.assertIsNotNone(self.wallet.extract_payment_id(valid_payment))
        self.assertIsNone(self.wallet.extract_payment_id(invalid_payment))
        
        # Test concurrent transactions
        def process_transaction(tx):
            return self.wallet.validate_transaction(tx)
            
        threads = []
        results = []
        
        for i in range(10):
            tx = valid_transaction.copy()
            tx["timestamp"] = (datetime.now() + timedelta(seconds=i)).isoformat()
            thread = threading.Thread(
                target=lambda: results.append(process_transaction(tx))
            )
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # All transactions should be validated successfully
        self.assertTrue(all(results))

if __name__ == '__main__':
    unittest.main()
