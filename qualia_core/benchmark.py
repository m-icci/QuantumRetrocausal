"""
Módulo de benchmark para o sistema adaptativo
"""
import time
import hashlib
import psutil
from typing import List, Tuple, Dict
from dataclasses import dataclass
from core.adaptive_granularity import GranularityMetrics

@dataclass
class BenchmarkResult:
    """Resultado de um benchmark para uma granularidade específica"""
    granularity: int
    execution_time: float
    cpu_percent: float
    memory_percent: float
    entropy: float
    coherence: float
    resonance: float
    stability: float

class Benchmark:
    """Classe para realizar benchmarks do sistema"""
    
    def __init__(self):
        self.process = psutil.Process()
        
    def sha256_mining(self, granularity: int) -> str:
        """Simula mineração com SHA-256"""
        data = "data" * granularity
        return hashlib.sha256(data.encode()).hexdigest()
    
    def calculate_metrics(self, granularity: int, execution_time: float) -> Dict[str, float]:
        """Calcula métricas de qualidade"""
        # Simulando métricas baseadas no tempo de execução e granularidade
        base_score = 1.0 / (1 + execution_time)  # Quanto menor o tempo, melhor o score
        
        # Coerência diminui com granularidades muito altas
        coherence = base_score * (1 - (granularity / 200))
        
        # Entropia aumenta com granularidades maiores
        entropy = granularity / 200
        
        # Ressonância é melhor em granularidades intermediárias
        resonance = 1 - abs(granularity - 34) / 100
        
        # Estabilidade diminui com tempos de execução muito longos
        stability = 1 / (1 + execution_time * 10)
        
        return {
            "coherence": max(0.1, min(0.99, coherence)),
            "entropy": max(0.1, min(0.99, entropy)),
            "resonance": max(0.1, min(0.99, resonance)),
            "stability": max(0.1, min(0.99, stability))
        }
    
    def benchmark_granularity(self, granularity: int) -> BenchmarkResult:
        """Realiza benchmark para uma granularidade específica"""
        # Mede tempo e recursos
        start_time = time.time()
        start_cpu = self.process.cpu_percent()
        start_memory = psutil.virtual_memory().percent
        
        # Executa mineração
        self.sha256_mining(granularity)
        
        # Calcula métricas
        execution_time = time.time() - start_time
        cpu_percent = self.process.cpu_percent() - start_cpu
        memory_percent = psutil.virtual_memory().percent - start_memory
        
        # Calcula métricas de qualidade
        metrics = self.calculate_metrics(granularity, execution_time)
        
        return BenchmarkResult(
            granularity=granularity,
            execution_time=execution_time,
            cpu_percent=max(0, cpu_percent),
            memory_percent=max(0, memory_percent),
            entropy=metrics["entropy"],
            coherence=metrics["coherence"],
            resonance=metrics["resonance"],
            stability=metrics["stability"]
        )
    
    def run_benchmark(self, granularities: List[int]) -> List[BenchmarkResult]:
        """Executa benchmark para múltiplas granularidades"""
        results = []
        
        print("\nIniciando benchmark...")
        print(f"{'Granularidade':<12} {'Tempo (s)':<10} {'CPU (%)':<8} {'Mem (%)':<8} {'Coer.':<8} {'Entr.':<8} {'Res.':<8} {'Estab.':<8}")
        print("-" * 80)
        
        for granularity in granularities:
            result = self.benchmark_granularity(granularity)
            results.append(result)
            
            # Exibe resultado
            print(f"{result.granularity:<12} {result.execution_time:<10.5f} {result.cpu_percent:<8.1f} "
                  f"{result.memory_percent:<8.1f} {result.coherence:<8.2f} {result.entropy:<8.2f} "
                  f"{result.resonance:<8.2f} {result.stability:<8.2f}")
            
            # Pequena pausa para CPU estabilizar
            time.sleep(0.1)
        
        return results
    
    def to_granularity_metrics(self, result: BenchmarkResult) -> GranularityMetrics:
        """Converte resultado do benchmark para métricas de granularidade"""
        return GranularityMetrics(
            coherence=result.coherence,
            resonance=result.resonance,
            entropy=result.entropy,
            stability=result.stability,
            execution_time=result.execution_time
        )
