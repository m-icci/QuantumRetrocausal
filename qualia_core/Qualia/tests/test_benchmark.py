# tests/test_benchmark.py
import unittest
import timeit
import psutil
import time
from typing import List
import numpy as np
from qualia.miner import QualiaAdaptiveMiner
from qualia.predictor import QualiaPredictor
from qualia.operator import QualiaOperator

def get_process_memory() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def generate_test_blocks(n: int) -> List[str]:
    """Generate test block headers."""
    return [f"TEST_BLOCK_{i}_{time.time()}" for i in range(n)]

def count_valid_hashes(hashes: List[str], difficulty: int) -> int:
    """Count hashes starting with required number of zeros."""
    target = "0" * difficulty
    return sum(1 for h in hashes if h.startswith(target))

class TestQualiaBenchmark(unittest.TestCase):
    def setUp(self):
        self.miner = QualiaAdaptiveMiner(difficulty=3)
        self.predictor = QualiaPredictor()
        self.test_block = "BENCHMARK_TEST_BLOCK"

    def test_mining_speed(self):
        """Benchmark mining speed."""
        def mine_operation():
            return self.miner.mine_block(self.test_block, max_nonces=3)

        # Run mining operation multiple times and measure average speed
        number = 3
        total_time = timeit.timeit(mine_operation, number=number)
        avg_time = total_time / number
        
        print(f"\nMining Speed Benchmark:")
        print(f"Average time per mining operation: {avg_time:.3f} seconds")
        print(f"Mining operations per second: {1/avg_time:.2f}")
        
        self.assertGreater(1/avg_time, 0)  # Ensure positive mining rate

    def test_predictor_performance(self):
        """Benchmark predictor performance with different block sizes."""
        blocks = [
            "small" * 10,
            "medium" * 100,
            "large" * 1000
        ]
        
        times = []
        for block in blocks:
            start_time = time.time()
            self.predictor.predict_nonce(block)
            elapsed = time.time() - start_time
            times.append(elapsed)
            
        print(f"\nPredictor Performance Benchmark:")
        for size, t in zip(["Small", "Medium", "Large"], times):
            print(f"{size} block prediction time: {t:.6f} seconds")
            
        self.assertTrue(all(t > 0 for t in times))

    def test_operator_performance(self):
        """Benchmark QUALIA operators performance."""
        nonces = list(range(0, 1000000, 100000))
        
        def run_operators():
            results = []
            for nonce in nonces:
                r = QualiaOperator.resonance(nonce)
                s = QualiaOperator.superposition(r)
                results.append(QualiaOperator.retrocausality(s, nonce))
            return results
            
        # Measure operator execution time
        number = 100
        total_time = timeit.timeit(run_operators, number=number)
        avg_time = total_time / number
        ops_per_second = len(nonces) / avg_time
        
        print(f"\nOperator Performance Benchmark:")
        print(f"Average time per operation set: {avg_time:.6f} seconds")
        print(f"Operations per second: {ops_per_second:.2f}")
        
        self.assertGreater(ops_per_second, 0)

    def test_memory_usage(self):
        """Benchmark memory usage under different loads."""
        initial_memory = get_process_memory()
        blocks = generate_test_blocks(5)
        memory_readings = []
        
        for block in blocks:
            before_memory = get_process_memory()
            self.miner.mine_block(block, max_nonces=2)
            after_memory = get_process_memory()
            memory_readings.append(after_memory - before_memory)
            
        avg_memory = np.mean(memory_readings)
        peak_memory = max(memory_readings)
        
        print(f"\nMemory Usage Benchmark:")
        print(f"Initial memory: {initial_memory:.2f} MB")
        print(f"Average memory per block: {avg_memory:.2f} MB")
        print(f"Peak memory usage: {peak_memory:.2f} MB")
        
        self.assertGreater(avg_memory, 0)

    def test_accuracy_metrics(self):
        """Benchmark accuracy metrics at different difficulty levels."""
        for difficulty in [2, 3, 4]:
            miner = QualiaAdaptiveMiner(difficulty=difficulty)
            blocks = generate_test_blocks(3)
            total_hashes = []
            
            for block in blocks:
                _, results_df, _ = miner.mine_block(block, max_nonces=2)
                total_hashes.extend(results_df['Hash'].tolist())
                
            valid_count = count_valid_hashes(total_hashes, difficulty)
            accuracy = valid_count / len(total_hashes)
            
            print(f"\nAccuracy Metrics (Difficulty {difficulty}):")
            print(f"Total hashes: {len(total_hashes)}")
            print(f"Valid hashes: {valid_count}")
            print(f"Success rate: {accuracy:.2%}")
            
            self.assertGreater(accuracy, 0)

    def test_load_performance(self):
        """Benchmark performance under different load conditions."""
        blocks = generate_test_blocks(5)
        times = []
        
        for i, block in enumerate(blocks, 1):
            start_time = time.time()
            self.miner.mine_block(block, max_nonces=2)
            elapsed = time.time() - start_time
            times.append(elapsed)
            
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"\nLoad Performance Benchmark:")
        print(f"Average processing time: {avg_time:.3f} seconds")
        print(f"Standard deviation: {std_time:.3f} seconds")
        print(f"Throughput: {1/avg_time:.2f} blocks/second")
        
        self.assertGreater(1/avg_time, 0)

if __name__ == '__main__':
    unittest.main()