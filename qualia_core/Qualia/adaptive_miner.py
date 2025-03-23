# qualia/adaptive_miner.py
import hashlib
import time
import logging
import random
import numpy as np
from typing import Tuple, Optional, Dict, Any
import pandas as pd
from .predictor import QualiaPredictor
from .operator import QualiaOperator

logger = logging.getLogger(__name__)

class QUALIANeuralAdapter:
    """
    Sistema neural adaptativo para QUALIA que permite ajuste dinâmico
    baseado em feedback do ambiente computacional e resultados de mineração
    """
    def __init__(self, miner):
        self.miner = miner
        self.learning_rate = 0.05
        self.memory_window = 120  # 2 minutos de memória adaptativa
        
        # Histórico de desempenho por parâmetro
        self.performance_history = {
            'granularity': {},      # granularidade -> [taxa_hash, validez, tempo]
            'coherence_levels': {}, # níveis de coerência -> desempenho
            'environment': []       # capturas do ambiente computacional
        }
        
        # Parâmetros adaptativos
        self.dynamic_params = {
            'current_granularity': 21,
            'coherence_target': 0.65,
            'mutation_rate': 0.02,
            'adaptation_speed': 0.1
        }
        
        # Estado inicial do ambiente
        self.env_snapshot = self._capture_environment()
        
        # Inicializar rede neural simples (pesos para cada fator ambiental)
        self.weights = {
            'cpu_load': 0.3,
            'memory_usage': 0.2,
            'process_count': 0.15,
            'hash_success_rate': 0.4,
            'hash_rate': 0.35,
            'system_temperature': 0.25  # Se disponível
        }
    
    def _capture_environment(self):
        """Captura estado atual do ambiente computacional"""
        env = {}
        try:
            import psutil
            
            # Carga de CPU (média de todos os cores)
            env['cpu_load'] = psutil.cpu_percent(interval=0.1) / 100.0
            
            # Uso de memória
            mem = psutil.virtual_memory()
            env['memory_usage'] = mem.percent / 100.0
            
            # Contagem de processos (excluindo threads)
            env['process_count'] = len(psutil.pids()) / 500  # Normalizado
            
            # Temperatura do sistema (se disponível)
            if hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()
                if temps:
                    # Pegar a média das temperaturas disponíveis
                    all_temps = []
                    for name, entries in temps.items():
                        for entry in entries:
                            if entry.current:
                                all_temps.append(entry.current)
                    
                    if all_temps:
                        # Normalizar com base em 100°C como máximo
                        env['system_temperature'] = sum(all_temps) / len(all_temps) / 100.0
            
            # Disco I/O (normalizado)
            disk_io = psutil.disk_io_counters()
            if disk_io:
                env['disk_activity'] = (disk_io.read_bytes + disk_io.write_bytes) % 10000000 / 10000000
            
            # Rede (atividade normalizada)
            net_io = psutil.net_io_counters()
            if net_io:
                env['network_activity'] = (net_io.bytes_sent + net_io.bytes_recv) % 1000000 / 1000000
                
        except (ImportError, AttributeError):
            # Fallback para estimativas básicas se psutil não estiver disponível
            env['cpu_load'] = 0.5  # valor estimado
            env['memory_usage'] = 0.5
            env['process_count'] = 0.3
        
        # Adicionar ruído quântico simulado (inspirado por flutuações de vácuo quântico)
        import random
        env['quantum_noise'] = random.random() * 0.05
        
        return env
    
    def suggest_granularity(self, entropy, current_performance):
        """
        Determina dinamicamente a granularidade ideal com base no ambiente
        e no histórico de desempenho
        """
        # Capturar ambiente atual
        current_env = self._capture_environment()
        
        # Registrar ambiente para análise futura
        self.performance_history['environment'].append(current_env)
        if len(self.performance_history['environment']) > self.memory_window:
            self.performance_history['environment'].pop(0)
        
        # Calcular "pressão ambiental" - quanto maior, mais precisamos nos adaptar
        environmental_pressure = (
            current_env['cpu_load'] * self.weights['cpu_load'] +
            current_env['memory_usage'] * self.weights['memory_usage'] +
            current_env.get('process_count', 0) * self.weights['process_count']
        )
        
        # Incluir desempenho atual na decisão
        performance_score = (
            current_performance.get('success_rate', 0.5) * self.weights['hash_success_rate'] +
            (1.0 - min(1.0, current_performance.get('hashrate', 500) / 1000)) * self.weights['hash_rate']
        )
        
        # Adicionar temperatura se disponível (mais quente = mais cautela)
        if 'system_temperature' in current_env:
            performance_score += current_env['system_temperature'] * self.weights['system_temperature']
        
        # Combinar todos os fatores
        combined_factor = (environmental_pressure * 0.4) + (performance_score * 0.6)
        
        # Adicionar um componente de exploração (comportamento emergente)
        if random.random() < self.dynamic_params['mutation_rate']:
            exploration_factor = (random.random() - 0.5) * 0.2
            combined_factor += exploration_factor
        
        # Determinar modo de operação baseado nos fatores combinados
        # Fibonacci como base biológica para granularidade (encontrado na natureza)
        fibonacci = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
        
        # Segmentação adaptativa
        if combined_factor < 0.3:
            # Modo ultra-rápido - preferir velocidade absoluta
            segment = fibonacci[:4]
            idx = min(3, int(entropy * 4))
        elif combined_factor < 0.5:
            # Modo rápido - boa velocidade com alguma precisão
            segment = fibonacci[2:6]
            idx = min(3, int(entropy * 4))
        elif combined_factor < 0.7:
            # Modo balanceado - compromisso entre velocidade e precisão
            segment = fibonacci[4:8]
            idx = min(3, int(entropy * 4))
        else:
            # Modo preciso - preferir qualidade sobre velocidade
            segment = fibonacci[6:10]
            idx = min(3, int((1-entropy) * 4))  # Invertido para priorizar precisão
        
        # Escolher granularidade
        granularity = segment[idx % len(segment)]
        
        # Registrar escolha para aprendizado
        current_granularity_key = str(granularity)
        if current_granularity_key not in self.performance_history['granularity']:
            self.performance_history['granularity'][current_granularity_key] = []
        
        # Armazenar dados para análise (max 20 entradas por granularidade)
        perf_entry = {
            'hashrate': current_performance.get('hashrate', 0),
            'success_rate': current_performance.get('success_rate', 0),
            'timestamp': time.time(),
            'env_pressure': environmental_pressure
        }
        
        self.performance_history['granularity'][current_granularity_key].append(perf_entry)
        if len(self.performance_history['granularity'][current_granularity_key]) > 20:
            self.performance_history['granularity'][current_granularity_key].pop(0)
        
        # Ajustar taxa de mutação (mais alta quando desempenho é fraco)
        if current_performance.get('success_rate', 0) < 0.3:
            self.dynamic_params['mutation_rate'] = min(0.1, self.dynamic_params['mutation_rate'] * 1.05)
        else:
            self.dynamic_params['mutation_rate'] = max(0.01, self.dynamic_params['mutation_rate'] * 0.95)
        
        # Atualizar o alvo de coerência quantum com base no desempenho
        if current_performance.get('success_rate', 0) > 0.7:
            self.dynamic_params['coherence_target'] = min(0.9, 
                                                         self.dynamic_params['coherence_target'] + 0.01)
        else:
            self.dynamic_params['coherence_target'] = max(0.4, 
                                                         self.dynamic_params['coherence_target'] - 0.01)
        
        # Atualizar pesos da rede neural baseado em resultados (aprendizado contínuo)
        self._update_neural_weights(current_performance)
        
        return granularity, self.dynamic_params['coherence_target']
    
    def _update_neural_weights(self, performance):
        """Atualiza os pesos da rede neural baseado no desempenho recente"""
        # Só atualiza se tivermos dados suficientes
        if len(self.performance_history['environment']) < 5:
            return
        
        # Simplificação de backpropagation para ajustar pesos
        # Baseado na correlação entre fatores ambientais e desempenho
        
        # Calcular sinal de erro (quanto maior sucesso, menor o erro)
        error_signal = 1.0 - performance.get('success_rate', 0.5)
        
        # Ajustar pesos com base no erro e no ambiente recente
        recent_env = self.performance_history['environment'][-1]
        for factor in ['cpu_load', 'memory_usage', 'process_count']:
            if factor in recent_env:
                # Correlação do fator com erro (aumentar peso se correlacionado)
                correlation = recent_env[factor] * error_signal
                adjustment = correlation * self.learning_rate
                
                # Aplicar ajuste limitado
                self.weights[factor] = max(0.05, min(0.5, self.weights[factor] + adjustment))
        
        # Normalizar pesos para somar 1
        weight_sum = sum(self.weights.values())
        for factor in self.weights:
            self.weights[factor] /= weight_sum
            
    def update_performance_metrics(self, metrics):
        """Atualiza o modelo com novos dados de desempenho"""
        # Implementar métricas para feedback do sistema
        # Será chamado periodicamente durante a mineração
        
        # Atualizar velocidade de adaptação baseado na volatilidade do ambiente
        env_history = self.performance_history['environment']
        if len(env_history) > 10:
            # Calcular variação na carga da CPU como indicador de volatilidade
            cpu_variations = [abs(env_history[i]['cpu_load'] - env_history[i-1]['cpu_load']) 
                             for i in range(1, len(env_history))]
            
            avg_variation = sum(cpu_variations) / len(cpu_variations)
            
            # Ambiente mais volátil = adaptação mais rápida
            self.dynamic_params['adaptation_speed'] = min(0.3, max(0.05, avg_variation * 2))

class QualiaAdaptiveMiner:
    """
    QualiaAdaptiveMiner implements an advanced mining system with:
    - Adaptive granularity adjustment
    - QUALIA operators integration
    - Predictive nonce estimation
    """
    
    def __init__(self, 
                 difficulty: int = 3, 
                 gran_min: int = 8, 
                 gran_max: int = 128, 
                 gran_step: int = 8):
        """
        Initialize the miner with configurable parameters.
        
        Args:
            difficulty (int): Mining difficulty (number of leading zeros)
            gran_min (int): Minimum granularity value
            gran_max (int): Maximum granularity value
            gran_step (int): Step size for granularity adjustment
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not 1 <= difficulty <= 64:
            raise ValueError("Difficulty must be between 1 and 64")
        if gran_min >= gran_max:
            raise ValueError("Minimum granularity must be less than maximum")
        if gran_step <= 0:
            raise ValueError("Granularity step must be positive")
            
        self.difficulty = difficulty
        self.gran_min = gran_min
        self.gran_max = gran_max
        self.gran_step = gran_step
        self.gran_current = 32
        self.target_prefix = "0" * difficulty
        self.predictor = QualiaPredictor()
        self.neural_adapter = QUALIANeuralAdapter(self)
        
        logger.info(f"Initialized QualiaAdaptiveMiner with difficulty={difficulty}, "
                   f"gran_min={gran_min}, gran_max={gran_max}, gran_step={gran_step}")

    def hash_function(self, nonce: int) -> str:
        """
        Compute SHA-256 hash of a nonce.
        
        Args:
            nonce (int): Nonce value to hash
            
        Returns:
            str: Hexadecimal hash string
        """
        return hashlib.sha256(str(nonce).encode()).hexdigest()

    def mine_block(self, block_header: str, max_nonces: int = 5) -> Tuple[int, pd.DataFrame, float]:
        """
        Execute QUALIA mining with adaptive granularity.
        
        Args:
            block_header (str): Block data to mine
            max_nonces (int): Number of valid nonces to find
            
        Returns:
            Tuple[int, pd.DataFrame, float]: Predicted nonce, results DataFrame, and total time
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not block_header:
            raise ValueError("Block header cannot be empty")
        if max_nonces <= 0:
            raise ValueError("max_nonces must be positive")
            
        logger.info(f"Starting mining block with header={block_header}, max_nonces={max_nonces}")
        logger.info(f"Iniciando a mineração com cabeçalho: {block_header}")
        
        predicted_nonce = self.predictor.predict_nonce(block_header)
        logger.info(f"Nonce previsto: {predicted_nonce}")
        logger.info(f"Previsão de nonce realizada com sucesso")
        successful_hashes = 0
        attempts = 0
        results = []
        start_time = time.time()
        last_nonce = predicted_nonce

        while successful_hashes < max_nonces:
            logger.info(f"Iniciando nova tentativa de mineração...")
            nonce_candidate = QualiaOperator.resonance(last_nonce)
            logger.info(f"Aplicando operador de ressonância: {nonce_candidate}")
            nonce_candidate = QualiaOperator.superposition(nonce_candidate)
            logger.info(f"Aplicando operador de superposição: {nonce_candidate}")
            nonce_candidate = QualiaOperator.retrocausality(nonce_candidate, last_nonce)
            logger.info(f"Aplicando operador de retrocausalidade: {nonce_candidate}")
            logger.info(f"Tentativa de nonce: {nonce_candidate}")

            hash_value = self.hash_function(nonce_candidate)
            logger.info(f"Hash calculado: {hash_value}")
            
            if hash_value.startswith(self.target_prefix):
                successful_hashes += 1
                elapsed = time.time() - start_time
                results.append((self.gran_current, nonce_candidate, hash_value, elapsed))
                last_nonce = nonce_candidate
                logger.info(f"Found valid nonce: {nonce_candidate}, hash: {hash_value}")
                logger.info(f"Nonce válido encontrado com sucesso")

            attempts += 1
            efficiency = successful_hashes / attempts if attempts > 0 else 0
            logger.info(f"Número de tentativas: {attempts}")
            logger.info(f"Taxa de sucesso atual: {efficiency:.2f}")
            
            logger.info(f"Granularidade antes: {self.gran_current}")
            self._adjust_granularity(efficiency)
            logger.info(f"Granularidade depois: {self.gran_current}")

        total_time = time.time() - start_time
        results_df = pd.DataFrame(results, columns=["Granularity", "Nonce", "Hash", "Tempo (s)"])
        
        logger.info(f"Mining completed in {total_time:.2f}s, found {successful_hashes} nonces")
        logger.info(f"Mineração concluída com sucesso")
        return predicted_nonce, results_df, total_time

    def _adjust_granularity(self, efficiency: float) -> None:
        """
        Adjust granularity based on mining efficiency.
        
        Args:
            efficiency (float): Current mining efficiency ratio
        """
        old_gran = self.gran_current
        
        # Use the neural adapter to suggest a new granularity
        self.gran_current, _ = self.neural_adapter.suggest_granularity(efficiency, {'success_rate': efficiency})
        
        if old_gran != self.gran_current:
            logger.info(f"Adjusted granularity from {old_gran} to {self.gran_current}")
            logger.info(f"Ajustando granularidade para: {self.gran_current}")
            logger.info(f"Granularidade atual: {self.gran_current}")
            logger.info(f"Ajuste de granularidade realizado com sucesso")