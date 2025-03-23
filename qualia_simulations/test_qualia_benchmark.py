#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List

# Importa o sistema de configuração QUALIA
from qualia_config_production import (
    get_config,
    QuantumParametersConfig,
    RetrocausalConfig,
    RetrocausalBridge
)

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("qualia.benchmark")

class QualiaBenchmark:
    """Benchmark do sistema QUALIA com configuração quântica"""
    
    def __init__(
        self,
        test_duration: int = 300,  # 5 minutos
        difficulty_prefix: str = "000",
        use_quantum_config: bool = True
    ):
        self.test_duration = test_duration
        self.difficulty_prefix = difficulty_prefix
        self.use_quantum_config = use_quantum_config
        
        # Carrega configuração QUALIA
        self.config = get_config(testnet_mode=True)
        
        # Inicializa componentes quânticos
        self.quantum_params = self.config.get_quantum_parameters()
        self.retrocausal_config = self.config.get_retrocausal_config()
        self.retrocausal_bridge = RetrocausalBridge(self.quantum_params)
        
        # Métricas
        self.metrics = {
            "start_time": 0,
            "end_time": 0,
            "total_hashes": 0,
            "valid_nonces": 0,
            "rejected_shares": 0,
            "best_difficulty": 0,
            "average_hashrate": 0,
            "peak_hashrate": 0,
            "quantum_coherence": [],
            "share_difficulties": [],
            "blocks_found": 0,
            "adaptation_events": 0,
            "quantum_state_changes": 0
        }
        
        # Estado do sistema
        self.current_field_coherence = 0.7
        self.current_retrocausal_factor = 0.5
    
    def _calculate_hash(self, nonce: int) -> str:
        """Calcula hash SHA-256 para o nonce"""
        import hashlib
        return hashlib.sha256(str(nonce).encode()).hexdigest()
    
    def _meets_difficulty(self, nonce: int) -> bool:
        """Verifica se o hash atende à dificuldade"""
        hash_result = self._calculate_hash(nonce)
        hash_value = int(hash_result, 16)
        
        # Usa dificuldade real do Monero
        network_difficulty = 300000000  # Dificuldade aproximada da rede
        target = (2**256) // network_difficulty
        
        return hash_value < target
    
    def _update_quantum_state(self):
        """Atualiza o estado quântico do sistema"""
        if not self.use_quantum_config:
            return
        
        # Sincroniza com campo retrocausal
        bit_configs = self.retrocausal_bridge.synchronize_with_retrocausal_field(
            self.current_field_coherence,
            self.current_retrocausal_factor
        )
        
        # Atualiza métricas
        self.metrics["quantum_state_changes"] += 1
        
        # Simula flutuação do campo
        self.current_field_coherence += np.random.normal(0, 0.01)
        self.current_field_coherence = max(0.1, min(0.9, self.current_field_coherence))
        
        self.current_retrocausal_factor += np.random.normal(0, 0.01)
        self.current_retrocausal_factor = max(0.1, min(0.9, self.current_retrocausal_factor))
    
    def _simulate_mining(self) -> Dict[str, Any]:
        """Simula uma iteração de mineração com parâmetros quânticos"""
        # Simular hashrate com variação realista
        base_hashrate = 2000  # Hashrate base por thread
        variation = np.random.uniform(0.8, 1.2)  # Variação de ±20%
        current_hashrate = base_hashrate * variation
        
        # Simular encontrar nonce válido baseado na dificuldade
        share_difficulty = np.random.randint(1000, 10000)
        found_nonce = np.random.random() < (current_hashrate / 1000000)  # Simulação simplificada
        
        return {
            "hashrate": current_hashrate,
            "found_nonce": found_nonce,
            "share_difficulty": share_difficulty
        }
    
    def run_benchmark(self):
        """Executa o benchmark de mineração com QUALIA"""
        print("\n=== Benchmark de Mineração QUALIA ===")
        print(f"Duração: {self.test_duration}s")
        print(f"Dificuldade: {self.difficulty_prefix}")
        print(f"Configuração Quântica: {'Ativada' if self.use_quantum_config else 'Desativada'}")
        print("\n" + "="*50 + "\n")
        
        try:
            self.metrics["start_time"] = time.time()
            
            # Loop principal do benchmark
            start_time = time.time()
            while (time.time() - start_time) < self.test_duration:
                # Atualiza estado quântico
                self._update_quantum_state()
                
                # Simula mineração
                mining_result = self._simulate_mining()
                
                # Atualiza métricas
                self.metrics["total_hashes"] += mining_result["hashrate"]
                if mining_result["found_nonce"]:
                    self.metrics["valid_nonces"] += 1
                    self.metrics["share_difficulties"].append(mining_result["share_difficulty"])
                
                # Registra coerência quântica
                self.metrics["quantum_coherence"].append(self.current_field_coherence)
                
                # Atualiza progresso
                elapsed = time.time() - start_time
                progress = (elapsed / self.test_duration) * 100
                print(f"\rProgresso: {progress:.1f}% | Nonces: {self.metrics['valid_nonces']} | "
                      f"Hashrate: {mining_result['hashrate']:.0f} H/s", end="")
                
                time.sleep(0.1)  # Pequena pausa para não sobrecarregar
            
            print("\n")  # Nova linha após o progresso
            
            # Finaliza benchmark
            self.metrics["end_time"] = time.time()
            duration = self.metrics["end_time"] - self.metrics["start_time"]
            
            # Calcula métricas finais
            self.metrics["average_hashrate"] = self.metrics["total_hashes"] / duration
            self.metrics["peak_hashrate"] = max(self.metrics["share_difficulties"]) if self.metrics["share_difficulties"] else 0
            
            # Gera relatório
            self._generate_report()
            
            return True
            
        except Exception as e:
            logger.error(f"Erro durante benchmark: {e}")
            return False
    
    def _generate_report(self):
        """Gera relatório detalhado do benchmark"""
        duration = self.metrics["end_time"] - self.metrics["start_time"]
        
        # Dados para visualização
        coherence_data = self.metrics["quantum_coherence"]
        time_points = np.linspace(0, duration, len(coherence_data))
        
        # Cria gráfico de coerência quântica
        plt.figure(figsize=(12, 6))
        plt.plot(time_points, coherence_data, label="Coerência Quântica")
        plt.axhline(y=np.mean(coherence_data), color='r', linestyle='--', label="Média")
        plt.title("Evolução da Coerência Quântica Durante o Benchmark")
        plt.xlabel("Tempo (s)")
        plt.ylabel("Coerência")
        plt.legend()
        plt.grid(True)
        plt.savefig("qualia_benchmark_coherence.png")
        plt.close()
        
        # Imprime relatório
        print("\n=== Relatório do Benchmark QUALIA ===")
        print(f"Duração Total: {duration:.2f}s")
        print(f"Hashrate Médio: {self.metrics['average_hashrate']:.2f} H/s")
        print(f"Nonces Válidos: {self.metrics['valid_nonces']}")
        print(f"Melhor Dificuldade: {self.metrics['peak_hashrate']}")
        print(f"Eventos de Adaptação: {self.metrics['adaptation_events']}")
        print(f"Mudanças de Estado Quântico: {self.metrics['quantum_state_changes']}")
        
        # Métricas de adaptação
        adaptation_metrics = self.retrocausal_bridge.get_adaptation_metrics()
        print("\nMétricas de Adaptação:")
        print(f"Total de Adaptações: {adaptation_metrics['adaptations_count']}")
        print(f"Coerência Média: {adaptation_metrics['average_coherence']:.3f}")
        print(f"Fator Retrocausal Médio: {adaptation_metrics['average_retrocausal_factor']:.3f}")
        print(f"Estabilidade das Configurações: {adaptation_metrics['configs_stability']:.3f}")
        
        # Top configurações usadas
        print("\nTop Configurações de Bits:")
        for config, count in adaptation_metrics['top_configurations']:
            print(f"  {config}: {count} vezes")

def main():
    """Função principal para executar o benchmark"""
    # Executa benchmark com configuração quântica
    benchmark = QualiaBenchmark(
        test_duration=300,  # 5 minutos
        difficulty_prefix="000",
        use_quantum_config=True
    )
    
    success = benchmark.run_benchmark()
    
    if success:
        print("\nBenchmark concluído com sucesso!")
    else:
        print("\nErro durante o benchmark!")

if __name__ == "__main__":
    main() 