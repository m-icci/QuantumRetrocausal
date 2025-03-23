import unittest
import sys
import os
import logging
import threading
import time
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import psutil
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import queue

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mining_metrics_api import MiningMetricsAPI
from start_mining import MiningController
from qualia_miner import QualiaMiner
from wallet import WalletManager
from test_utils import MockPool, EntropyGenerator, FractalAnalyzer, RetrocausalitySimulator
from network_simulator import NetworkSimulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestAdvancedMining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize network simulator
        cls.network_sim = NetworkSimulator()
        cls.network_sim.start()
        
        # Initialize test wallet
        cls.test_wallet_path = "test_wallet.json"
        cls.wallet = WalletManager(cls.test_wallet_path)
        
        # Initialize components
        cls.metrics_api = MiningMetricsAPI()
        cls.mining_controller = MiningController(
            pool_address="localhost",
            pool_port=cls.network_sim.actual_port,
            wallet_address=cls.wallet.wallet_data["address"]
        )
        
        # Initialize test utilities
        cls.entropy_gen = EntropyGenerator(seed=42)
        cls.fractal_analyzer = FractalAnalyzer()
        cls.retrocausality_sim = RetrocausalitySimulator()
        
        # Performance metrics storage
        cls.performance_metrics = {
            "shares_accepted": 0,
            "shares_rejected": 0,
            "avg_hashrate": 0.0,
            "network_changes": [],
            "entropy_levels": [],
            "resource_usage": []
        }
        
        # Metric collection queue
        cls.metric_queue = queue.Queue()

    def setUp(self):
        self.start_time = datetime.now()
        self.test_duration = timedelta(minutes=5)  # Default test duration

    def tearDown(self):
        self.mining_controller.stop_mining()
        time.sleep(2)  # Allow time for cleanup

    @classmethod
    def tearDownClass(cls):
        cls.network_sim.stop()
        if os.path.exists(cls.test_wallet_path):
            os.remove(cls.test_wallet_path)

    def _collect_metrics(self, duration: timedelta):
        """Collect system metrics for specified duration."""
        end_time = datetime.now() + duration
        
        while datetime.now() < end_time:
            try:
                metrics = self.metrics_api.monitor_resources()
                self.metric_queue.put(metrics)
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error collecting metrics: {str(e)}")

    def test_network_resilience(self):
        """Test system resilience under dynamic network conditions."""
        logger.info("Starting network resilience test...")
        
        # Start metric collection
        metric_thread = threading.Thread(
            target=self._collect_metrics,
            args=(self.test_duration,)
        )
        metric_thread.daemon = True
        metric_thread.start()
        
        # Start mining
        mining_thread = threading.Thread(
            target=self.mining_controller.start_mining
        )
        mining_thread.daemon = True
        mining_thread.start()
        
        try:
            # Monitor performance under changing conditions
            test_end = datetime.now() + self.test_duration
            while datetime.now() < test_end:
                # Get current metrics
                try:
                    metrics = self.metric_queue.get_nowait()
                    self.performance_metrics["resource_usage"].append({
                        "timestamp": datetime.now().isoformat(),
                        "cpu": metrics.get("cpu_usage_percent", 0),
                        "memory": metrics.get("memory_usage_percent", 0)
                    })
                except queue.Empty:
                    pass
                
                # Record network changes
                self.performance_metrics["network_changes"].append({
                    "timestamp": datetime.now().isoformat(),
                    "difficulty": self.network_sim.current_difficulty,
                    "latency": self.network_sim.current_latency
                })
                
                time.sleep(1)
            
            # Verify performance metrics
            cpu_usage = [m["cpu"] for m in self.performance_metrics["resource_usage"]]
            self.assertLess(np.std(cpu_usage), 20, "CPU usage too volatile")
            
            # Verify network adaptation
            difficulties = [n["difficulty"] for n in self.performance_metrics["network_changes"]]
            self.assertTrue(len(difficulties) > 0, "No network changes recorded")
            
        finally:
            metric_thread.join(timeout=5)
            mining_thread.join(timeout=5)

    def test_hpc_convergence(self):
        """Test HPC workload distribution and convergence."""
        logger.info("Starting HPC convergence test...")
        
        def run_worker(worker_id: int, entropy_level: float):
            """Simulate HPC worker processing."""
            sequence = self.entropy_gen.generate_hash_sequence(1000, entropy_level)
            fd = self.fractal_analyzer.calculate_fractal_dimension(sequence)
            return {
                "worker_id": worker_id,
                "entropy_level": entropy_level,
                "fractal_dimension": fd
            }
        
        # Test with different entropy levels
        entropy_levels = [0.3, 0.6, 0.9]
        workers = 3
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for entropy in entropy_levels:
                for worker_id in range(workers):
                    futures.append(
                        executor.submit(run_worker, worker_id, entropy)
                    )
            
            # Collect results
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=30)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Worker error: {str(e)}")
            
            # Verify results
            for entropy in entropy_levels:
                worker_results = [r for r in results if r["entropy_level"] == entropy]
                dimensions = [r["fractal_dimension"] for r in worker_results]
                
                # Check convergence
                self.assertLess(
                    np.std(dimensions),
                    0.1,
                    f"Workers did not converge for entropy {entropy}"
                )

    def test_retrocausality_efficiency(self):
        """Compare mining efficiency with and without retrocausality."""
        logger.info("Starting retrocausality efficiency test...")
        
        def run_test_configuration(retro_enabled: bool) -> dict:
            """Run mining test with specific configuration."""
            miner = QualiaMiner()
            
            if retro_enabled:
                # Initialize retrocausality simulation
                self.retrocausality_sim.update_buffer()
            
            start_time = datetime.now()
            end_time = start_time + timedelta(minutes=2)
            
            metrics = []
            while datetime.now() < end_time:
                if retro_enabled:
                    # Get future state prediction
                    future_state = self.retrocausality_sim.get_future_state(5)
                    
                    # Adjust mining parameters based on prediction
                    if future_state["network_difficulty"] > 1.1:
                        miner.config["max_cpu_usage"] = max(50, miner.config["max_cpu_usage"] - 5)
                    else:
                        miner.config["max_cpu_usage"] = min(90, miner.config["max_cpu_usage"] + 5)
                
                # Collect performance metrics
                current_metrics = miner.monitor_performance()
                metrics.append(current_metrics)
                
                time.sleep(1)
            
            # Calculate average performance
            return {
                "avg_hashrate": np.mean([m["mining_metrics"]["hash_rate"] for m in metrics]),
                "efficiency": np.mean([m["efficiency_metrics"]["acceptance_rate"] for m in metrics])
            }
        
        # Run tests with retrocausality on and off
        retro_results = run_test_configuration(retro_enabled=True)
        normal_results = run_test_configuration(retro_enabled=False)
        
        # Compare results
        self.assertGreaterEqual(
            retro_results["efficiency"],
            normal_results["efficiency"],
            "Retrocausality did not improve efficiency"
        )

    def test_long_duration(self):
        """Test system stability over long duration."""
        logger.info("Starting long duration test...")
        
        # Extend test duration for burn-in
        self.test_duration = timedelta(minutes=30)
        
        # Start metric collection
        metric_thread = threading.Thread(
            target=self._collect_metrics,
            args=(self.test_duration,)
        )
        metric_thread.daemon = True
        metric_thread.start()
        
        # Start mining
        mining_thread = threading.Thread(
            target=self.mining_controller.start_mining
        )
        mining_thread.daemon = True
        mining_thread.start()
        
        try:
            # Monitor system over extended period
            test_end = datetime.now() + self.test_duration
            memory_usage = []
            
            while datetime.now() < test_end:
                # Check for memory leaks
                process = psutil.Process()
                memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB
                
                # Get current metrics
                try:
                    metrics = self.metric_queue.get_nowait()
                    self.performance_metrics["resource_usage"].append({
                        "timestamp": datetime.now().isoformat(),
                        "cpu": metrics.get("cpu_usage_percent", 0),
                        "memory": metrics.get("memory_usage_percent", 0)
                    })
                except queue.Empty:
                    pass
                
                time.sleep(10)  # Check every 10 seconds
            
            # Verify system stability
            self.assertLess(
                np.std(memory_usage),
                50,  # MB
                "Memory usage shows potential leak"
            )
            
            # Verify consistent performance
            cpu_usage = [m["cpu"] for m in self.performance_metrics["resource_usage"]]
            self.assertLess(
                np.std(cpu_usage),
                15,
                "CPU usage shows instability over time"
            )
            
        finally:
            metric_thread.join(timeout=5)
            mining_thread.join(timeout=5)

if __name__ == '__main__':
    unittest.main()
