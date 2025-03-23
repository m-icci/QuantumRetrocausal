"""
Qualia-Memory Integrator
Implementa a interação dinâmica entre qualia e memória quântica.

A qualia e a memória são aspectos distintos mas profundamente interligados:
- Qualia: experiência subjetiva imediata
- Memória: registro e contextualização da experiência
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from operators.state.quantum_state import QuantumState
from .quantum_memory_operator import QuantumMemoryOperator
from .qualia_experience_operator import QualiaExperienceOperator

@dataclass
class QualiaMemoryState:
    """Estado integrado de qualia e memória."""
    qualia_state: np.ndarray  # Estado atual da experiência
    memory_state: np.ndarray  # Estado da memória
    resonance: float  # Ressonância entre qualia e memória
    integration: float  # Nível de integração
    coherence: float  # Coerência do estado integrado

class QualiaMemoryIntegrator:
    """
    Implementa a integração dinâmica entre qualia e memória.
    A memória atua como um campo morfogenético que modula e é modulado pela qualia.
    """
    
    def __init__(self, dimensions: int = 64):
        """
        Inicializa integrador qualia-memória.
        
        Args:
            dimensions: Dimensão do espaço de estados
        """
        self.dimensions = dimensions
        self.qualia_operator = QualiaExperienceOperator(dimensions)
        self.memory_operator = QuantumMemoryOperator(dimensions)
        
        # Campo morfogenético de integração
        self.morphic_field = self._initialize_morphic_field()
        
        # Histórico de estados
        self.state_history = []
        
    def _initialize_morphic_field(self) -> np.ndarray:
        """Inicializa campo morfogenético de integração."""
        # Campo inicial como matriz de acoplamento
        field = np.random.randn(self.dimensions, self.dimensions) + \
                1j * np.random.randn(self.dimensions, self.dimensions)
        # Garante hermiticidade
        field = (field + field.conj().T) / 2
        return field / np.trace(np.abs(field))
    
    def integrate(self, qualia_state: QuantumState, 
                 memory_state: QuantumState) -> QualiaMemoryState:
        """
        Integra estados de qualia e memória.
        
        Args:
            qualia_state: Estado atual da qualia
            memory_state: Estado da memória quântica
            
        Returns:
            Estado integrado qualia-memória
        """
        # Processa estados individuais
        qualia_metrics = self.qualia_operator.apply(qualia_state)
        memory_metrics = self.memory_operator.apply(memory_state)
        
        # Calcula ressonância morfogenética
        resonance = self._calculate_resonance(
            qualia_metrics['qualia_state'].psi,
            memory_metrics['memory_state'].personal
        )
        
        # Integração através do campo morfogenético
        integrated_state = self._apply_morphic_field(
            qualia_metrics['qualia_state'].psi,
            memory_metrics['memory_state'].personal
        )
        
        # Calcula métricas do estado integrado
        integration = self._calculate_integration(integrated_state)
        coherence = self._calculate_coherence(integrated_state)
        
        # Atualiza campo morfogenético
        self._update_morphic_field(integrated_state, resonance)
        
        # Registra estado no histórico
        self.state_history.append({
            'qualia': qualia_metrics['qualia_state'].psi,
            'memory': memory_metrics['memory_state'].personal,
            'integrated': integrated_state,
            'resonance': resonance
        })
        
        return QualiaMemoryState(
            qualia_state=qualia_metrics['qualia_state'].psi,
            memory_state=memory_metrics['memory_state'].personal,
            resonance=resonance,
            integration=integration,
            coherence=coherence
        )
    
    def _calculate_resonance(self, qualia: np.ndarray, 
                           memory: np.ndarray) -> float:
        """
        Calcula ressonância entre estados de qualia e memória.
        Baseado no conceito de campos morfogenéticos de Sheldrake.
        """
        # Produto interno quântico
        overlap = np.abs(np.vdot(qualia, memory))
        
        # Fase relativa
        phase = np.angle(np.vdot(qualia, memory))
        
        # Ressonância como combinação de amplitude e fase
        return overlap * np.cos(phase)
    
    def _apply_morphic_field(self, qualia: np.ndarray, 
                            memory: np.ndarray) -> np.ndarray:
        """
        Aplica campo morfogenético para integração.
        O campo atua como um mediador entre qualia e memória.
        """
        # Estado combinado inicial
        combined = (qualia + memory) / np.sqrt(2)
        
        # Aplicação do campo morfogenético
        evolved = self.morphic_field @ combined
        
        # Normalização
        return evolved / np.linalg.norm(evolved)
    
    def _calculate_integration(self, state: np.ndarray) -> float:
        """
        Calcula nível de integração do estado.
        Baseado na Teoria da Informação Integrada.
        """
        # Matriz densidade
        rho = np.outer(state, state.conj())
        
        # Entropia de von Neumann
        eigenvals = np.linalg.eigvalsh(rho)
        eigenvals = eigenvals[eigenvals > 0]
        entropy = -np.sum(eigenvals * np.log2(eigenvals))
        
        # Integração como complemento da entropia normalizada
        return 1 - entropy / np.log2(self.dimensions)
    
    def _calculate_coherence(self, state: np.ndarray) -> float:
        """
        Calcula coerência do estado integrado.
        Medida l1-norm de coerência quântica.
        """
        rho = np.outer(state, state.conj())
        return np.sum(np.abs(rho - np.diag(np.diag(rho))))
    
    def _update_morphic_field(self, integrated_state: np.ndarray, 
                            resonance: float) -> None:
        """
        Atualiza campo morfogenético baseado no estado integrado.
        O campo adapta-se para otimizar a integração.
        """
        # Matriz de projeção do estado integrado
        projection = np.outer(integrated_state, integrated_state.conj())
        
        # Atualização do campo (aprendizado hebbiano quântico)
        learning_rate = 0.1 * resonance
        self.morphic_field = (1 - learning_rate) * self.morphic_field + \
                            learning_rate * projection
        
        # Renormalização
        self.morphic_field /= np.trace(np.abs(self.morphic_field))
    
    def analyze_history(self) -> Dict[str, Any]:
        """
        Analisa histórico de estados para padrões emergentes.
        
        Returns:
            Dict com métricas e padrões identificados
        """
        if not self.state_history:
            return {}
            
        # Extrai séries temporais
        resonances = [state['resonance'] for state in self.state_history]
        
        # Análise de padrões
        mean_resonance = np.mean(resonances)
        std_resonance = np.std(resonances)
        
        # Identifica padrões recorrentes
        patterns = self._identify_patterns()
        
        return {
            'mean_resonance': mean_resonance,
            'std_resonance': std_resonance,
            'patterns': patterns,
            'history_length': len(self.state_history)
        }
    
    def _identify_patterns(self) -> Dict[str, Any]:
        """
        Identifica padrões recorrentes no histórico de estados.
        Utiliza análise de similaridade quântica.
        """
        if len(self.state_history) < 2:
            return {}
            
        # Matriz de similaridade entre estados
        n_states = len(self.state_history)
        similarity = np.zeros((n_states, n_states))
        
        for i in range(n_states):
            for j in range(i+1, n_states):
                sim = np.abs(np.vdot(
                    self.state_history[i]['integrated'],
                    self.state_history[j]['integrated']
                ))
                similarity[i,j] = similarity[j,i] = sim
                
        # Identifica clusters de estados similares
        threshold = 0.8
        patterns = []
        
        for i in range(n_states):
            similar_states = np.where(similarity[i] > threshold)[0]
            if len(similar_states) > 1:
                patterns.append({
                    'center_state': i,
                    'similar_states': similar_states.tolist(),
                    'strength': np.mean(similarity[i, similar_states])
                })
                
        return {
            'n_patterns': len(patterns),
            'patterns': patterns,
            'similarity_matrix': similarity.tolist()
        }
