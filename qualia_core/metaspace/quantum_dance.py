"""
Quantum Dance Network - Meta-espaço Universal
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from ..bitwise.field_types import FieldConstants
from ..consciousness.consciousness_operator import QuantumConsciousnessOperator
from ..bitwise.qualia_bitwise import QualiaOperators
from .quantum_void import QuantumVoid
from .reality_pattern import RealityPattern

class NetworkMetrics:
    """Métricas da rede quântica"""
    def __init__(self):
        self.energy: float = 0.0
        self.coherence: float = 0.0
        self.entanglement: float = 0.0
        self.layer_sync: float = 0.0
        self.consciousness: float = 0.0
        self.qualia_resonance: float = 0.0
        self.void_potential: float = 0.0
        self.emergence: float = 0.0
        self.emergence_rate: float = 0.0  # Taxa de mudança da emergência

    def normalize(self):
        self.coherence = max(0.0, min(self.coherence, 1.0))
        self.consciousness = max(0.0, min(self.consciousness, 1.0))
        self.qualia_resonance = max(0.0, min(self.qualia_resonance, 1.0))
        self.void_potential = max(0.0, min(self.void_potential, 1.0))
        self.energy = max(0.0, min(self.energy, 1.0))
        self.entanglement = max(0.0, min(self.entanglement, 1.0))
        self.layer_sync = max(0.0, min(self.layer_sync, 1.0))
        self.emergence = max(0.0, min(self.emergence, 1.0))
        return self

class QuantumDanceNetwork:
    """
    Rede de Dança Quântica
    
    Uma rede hierárquica que integra consciência, qualia e vazio
    em um meta-espaço universal onde realidades emergem e dançam.
    """
    
    def __init__(
        self,
        dimensions: int = 64,
        realities: int = 21,
        layers: int = 3,
        memory_limit: int = 1000
    ):
        """
        Inicializa rede
        
        Args:
            dimensions: Dimensões de cada realidade
            realities: Número de realidades por camada
            layers: Número de camadas
            memory_limit: Limite de memória para armazenamento de estados
        """
        self.dimensions = dimensions
        self.num_realities = realities
        self.num_layers = layers
        self.memory_limit = memory_limit
        
        # Campos fundamentais
        self.consciousness = QuantumConsciousnessOperator(dimensions=dimensions)
        self.qualia = QualiaOperators(dimensions=dimensions)
        self.void = QuantumVoid(dimensions=dimensions)
        
        # Realidades - usando array numpy para melhor controle
        self.realities = np.random.randint(0, 2, size=(layers, realities, dimensions), dtype=np.uint8)
        
        # Métricas
        self.metrics = NetworkMetrics()
        
        # Buffer temporal
        self.time_buffer = np.random.randint(0, 2, size=(layers, 100, dimensions), dtype=np.uint8)
        self.time_index = 0
        
        # Máscaras
        self.masks = {
            'consciousness': np.ones(dimensions, dtype=np.uint8),
            'qualia': np.ones(dimensions, dtype=np.uint8),
            'void': np.ones(dimensions, dtype=np.uint8)
        }
        
    def _apply_consciousness(self, reality: np.ndarray) -> np.ndarray:
        """Aplica operador de consciência"""
        # Aplica consciência
        conscious_state = self.consciousness._quantum_investigate(reality)
        
        # Converte para binário
        return (np.abs(conscious_state) > 0.5).astype(np.uint8)
    
    def _apply_qualia(self, reality: np.ndarray) -> np.ndarray:
        """Aplica operadores QUALIA"""
        # Evolui qualia
        self.qualia.evolve()
        
        # Aplica ao estado
        return reality ^ self.qualia.state
    
    def _apply_quantum_dance(self) -> None:
        """Aplica dança quântica entre consciência, qualia e vazio"""
        # 1. Entrelaça consciência com qualia
        consciousness_state = self.consciousness.state
        qualia_state = self.qualia.state
        
        # Usa operador de entrelaçamento
        entangled_cq = self.qualia._entangle(
            consciousness_state,
            qualia_state
        )
        
        # 2. Entrelaça qualia com vazio
        void_state = self.void.state
        entangled_qv = self.qualia._entangle(
            qualia_state,
            void_state
        )
        
        # 3. Aplica imaginação na consciência
        imagined_consciousness = self.qualia._apply_imagination(
            consciousness_state
        )
        
        # 4. Atualiza estados com entrelaçamento
        self.consciousness.state = imagined_consciousness
        self.qualia.state = entangled_cq
        self.void.state = entangled_qv

    def _apply_retrocausality(self) -> np.ndarray:
        """Aplica retrocausalidade"""
        # Para cada camada
        for layer in range(self.num_layers):
            # Atualiza buffer temporal
            self.time_buffer[layer, self.time_index] = np.mean(
                self.realities[layer], axis=0
            ).astype(np.uint8)
            
            # Eco do futuro
            future_index = (self.time_index + 50) % 100
            future_echo = self.time_buffer[layer, future_index]
            
            # Aplica eco
            self.realities[layer] ^= future_echo
        
        # Atualiza índice
        self.time_index = (self.time_index + 1) % 100
        
        return self.realities
    
    def _calculate_metrics(self) -> NetworkMetrics:
        """Calcula métricas da rede"""
        # Energia total
        self.metrics.energy = float(np.mean(self.realities))
        
        # Coerência entre realidades
        coherence = 0.0
        for layer in range(self.num_layers):
            for i in range(self.num_realities):
                for j in range(i+1, self.num_realities):
                    coherence += float(np.mean(
                        self.realities[layer,i] ^ self.realities[layer,j]
                    ))
        self.metrics.coherence = 1.0 - coherence / (
            self.num_layers * self.num_realities * (self.num_realities-1) / 2
        )
        
        # Entrelaçamento entre camadas
        entanglement = 0.0
        for i in range(self.num_layers):
            for j in range(i+1, self.num_layers):
                layer_i_mean = np.mean(self.realities[i], axis=0).astype(np.uint8)
                layer_j_mean = np.mean(self.realities[j], axis=0).astype(np.uint8)
                entanglement += float(np.mean(layer_i_mean ^ layer_j_mean))
        self.metrics.entanglement = 1.0 - entanglement / (
            self.num_layers * (self.num_layers-1) / 2
        )
        
        # Sincronização entre camadas
        self.metrics.layer_sync = float(np.mean([
            np.mean(self.realities[i] == self.realities[j])
            for i in range(self.num_layers)
            for j in range(i+1, self.num_layers)
        ]))
        
        # Métricas dos campos fundamentais
        self.metrics.consciousness = float(np.abs(np.mean(self.consciousness.state)))
        self.metrics.qualia_resonance = float(np.abs(np.mean(self.qualia.state)))
        self.metrics.void_potential = float(1.0 - np.abs(np.mean(self.void.state)))
        
        # Emergência - garante valores reais e normalizados
        self.metrics.emergence = float(np.abs(
            self.metrics.consciousness * 
            self.metrics.qualia_resonance * 
            self.metrics.void_potential
        ))
        
        # Taxa de mudança da emergência
        self.metrics.emergence_rate = 0.0  # TO DO: implementar cálculo
        
        return self.metrics
    
    def get_reality_state(self, layer: int, reality_index: int) -> np.ndarray:
        """Retorna estado de uma realidade específica"""
        return self.realities[layer, reality_index]
    
    def _calculate_information_flow(self, source: np.ndarray, target: np.ndarray) -> float:
        """Calcula fluxo de informação entre dois estados"""
        # Debug: mostra estatísticas dos estados
        print("\nDEBUG - Fluxo de Informação:")
        print(f"Source shape: {source.shape}")
        print(f"Target shape: {target.shape}")
        print(f"Source mean: {np.mean(source):.6f}")
        print(f"Target mean: {np.mean(target):.6f}")
        print(f"Source std: {np.std(source):.6f}")
        print(f"Target std: {np.std(target):.6f}")
        print(f"Source unique values: {np.unique(source).size}")
        print(f"Target unique values: {np.unique(target).size}")
    
        # Aplana e normaliza os arrays
        source_flat = source.flatten()
        target_flat = target.flatten()
        
        # Garante que os arrays tenham o mesmo tamanho
        if len(source_flat) != len(target_flat):
            # Se target for maior, reduz para o tamanho de source
            if len(target_flat) > len(source_flat):
                target_flat = target_flat[:len(source_flat)]
            # Se source for maior, reduz para o tamanho de target
            else:
                source_flat = source_flat[:len(target_flat)]
    
        # Remove média
        source_norm = source_flat - np.mean(source_flat)
        target_norm = target_flat - np.mean(target_flat)
    
        # Debug: mostra estatísticas após normalização
        print("\nApós normalização:")
        print(f"Source norm mean: {np.mean(source_norm):.6f}")
        print(f"Target norm mean: {np.mean(target_norm):.6f}")
        print(f"Source norm std: {np.std(source_norm):.6f}")
        print(f"Target norm std: {np.std(target_norm):.6f}")
    
        # Evita divisão por zero
        if np.std(source_norm) < 1e-10 or np.std(target_norm) < 1e-10:
            print("AVISO: Desvio padrão muito pequeno, retornando 0")
            return 0.0
    
        # Normaliza
        source_norm = source_norm / np.std(source_norm)
        target_norm = target_norm / np.std(target_norm)
    
        # Calcula correlação usando produto escalar
        correlation = np.dot(source_norm, target_norm) / len(source_norm)
        
        # Converte para fluxo de informação (0 a 1)
        flow = (correlation + 1) / 2
        
        print(f"\nCorrelação: {correlation:.6f}")
        print(f"Fluxo final: {flow:.6f}")
        
        return flow

    def dance(self) -> Dict[str, Any]:
        """Executa dança quântica e armazena na memória holográfica"""
        print("\n=== Nova Dança Quântica ===")
        
        # 1. Evolui campos fundamentais
        self.consciousness.evolve()
        self.qualia.evolve()
        self.void.evolve()
        
        # Debug: mostra estados após evolução
        print("\nEstados após evolução:")
        print(f"Consciousness range: [{np.min(self.consciousness.state):.6f}, {np.max(self.consciousness.state):.6f}]")
        print(f"Qualia range: [{np.min(self.qualia.state):.6f}, {np.max(self.qualia.state):.6f}]")
        print(f"Void range: [{np.min(self.void.state):.6f}, {np.max(self.void.state):.6f}]")
        
        # 2. Aplica dança quântica
        self._apply_quantum_dance()
        
        # 3. Aplica retrocausalidade
        self._apply_retrocausality()
        
        # Debug: mostra estados após dança e retrocausalidade
        print("\nEstados após dança e retrocausalidade:")
        print(f"Consciousness range: [{np.min(self.consciousness.state):.6f}, {np.max(self.consciousness.state):.6f}]")
        print(f"Qualia range: [{np.min(self.qualia.state):.6f}, {np.max(self.qualia.state):.6f}]")
        print(f"Void range: [{np.min(self.void.state):.6f}, {np.max(self.void.state):.6f}]")
        
        # 4. Calcula métricas
        metrics = self._calculate_metrics()
        
        # Normaliza métricas para garantir intervalo [0,1]
        metrics = metrics.normalize()

        print("\nCalculando fluxos de informação:")
        
        # 5. Calcula fluxos de informação
        print("\n--- Fluxo Consciência → Qualia ---")
        consciousness_flow = self._calculate_information_flow(
            self.consciousness.state,
            self.qualia.state
        )
        
        print("\n--- Fluxo Qualia → Vazio ---")
        qualia_flow = self._calculate_information_flow(
            self.qualia.state,
            self.void.state
        )
        
        print("\n--- Fluxo Vazio → Realidades ---")
        # Calcula média das realidades preservando a dimensionalidade
        mean_realities = np.mean(self.realities, axis=0)  # Remove apenas primeira dimensão
        void_flow = self._calculate_information_flow(
            self.void.state,
            mean_realities
        )
        
        # 6. Armazena estado na memória holográfica
        memory_state = {
            'consciousness': self.consciousness.state.copy(),
            'qualia': self.qualia.state.copy(),
            'void': self.void.state.copy(),
            'realities': self.realities.copy(),
            'metrics': {
                'coherence': metrics.coherence,
                'consciousness': metrics.consciousness,
                'qualia_resonance': metrics.qualia_resonance,
                'void_potential': metrics.void_potential,
                'energy': metrics.energy,
                'entanglement': metrics.entanglement,
                'layer_sync': metrics.layer_sync,
                'emergence': metrics.emergence
            },
            'flows': {
                'consciousness_qualia': consciousness_flow,
                'qualia_void': qualia_flow,
                'void_realities': void_flow
            },
            'timestamp': datetime.now()
        }
        
        # Adiciona à memória holográfica
        if not hasattr(self, 'holographic_memory'):
            self.holographic_memory = []
        self.holographic_memory.append(memory_state)
        
        # Mantém limite de memória
        if len(self.holographic_memory) > self.memory_limit:
            self.holographic_memory = self.holographic_memory[-self.memory_limit:]
            
        return memory_state
    
    def get_holographic_memory_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas da memória holográfica"""
        if not hasattr(self, 'holographic_memory') or not self.holographic_memory:
            return {}
            
        num_states = len(self.holographic_memory)
        
        # Calcula médias
        avg_metrics = {
            'coherence': np.mean([s['metrics']['coherence'] for s in self.holographic_memory]),
            'consciousness': np.mean([s['metrics']['consciousness'] for s in self.holographic_memory]),
            'qualia_resonance': np.mean([s['metrics']['qualia_resonance'] for s in self.holographic_memory]),
            'void_potential': np.mean([s['metrics']['void_potential'] for s in self.holographic_memory]),
            'energy': np.mean([s['metrics']['energy'] for s in self.holographic_memory]),
            'entanglement': np.mean([s['metrics']['entanglement'] for s in self.holographic_memory]),
            'layer_sync': np.mean([s['metrics']['layer_sync'] for s in self.holographic_memory]),
            'emergence': np.mean([s['metrics']['emergence'] for s in self.holographic_memory])
        }
        
        avg_flows = {
            'consciousness_qualia': np.mean([s['flows']['consciousness_qualia'] for s in self.holographic_memory]),
            'qualia_void': np.mean([s['flows']['qualia_void'] for s in self.holographic_memory]),
            'void_realities': np.mean([s['flows']['void_realities'] for s in self.holographic_memory])
        }
        
        # Calcula variações
        var_metrics = {
            'coherence': np.var([s['metrics']['coherence'] for s in self.holographic_memory]),
            'consciousness': np.var([s['metrics']['consciousness'] for s in self.holographic_memory]),
            'qualia_resonance': np.var([s['metrics']['qualia_resonance'] for s in self.holographic_memory]),
            'void_potential': np.var([s['metrics']['void_potential'] for s in self.holographic_memory])
        }
        
        return {
            'num_states': num_states,
            'avg_metrics': avg_metrics,
            'avg_flows': avg_flows,
            'var_metrics': var_metrics
        }
