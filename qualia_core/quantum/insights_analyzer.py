"""
Quantum Insights Analyzer

Este módulo implementa análise de insights quânticos com:
- Monitoramento de consciência
- Métricas QUALIA
- Análise de campos
- Detecção de padrões emergentes
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from datetime import datetime
import json
from .quantum_state import QuantumState
from .morphic_memory import MorphicMemory

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class QualiaMetrics:
    """Métricas do sistema QUALIA"""
    coherence: float
    resonance: float
    entanglement: float
    field_strength: float
    pattern_confidence: float
    emergence_level: float
    timestamp: float

class ConsciousnessMonitor:
    """
    Monitor de Consciência Quântica
    
    Monitora e analisa o estado de consciência do sistema através de:
    - Métricas de coerência
    - Padrões de ressonância
    - Níveis de entrelaçamento
    - Força do campo quântico
    - Detecção de padrões emergentes
    """
    
    def __init__(self, dimensions: int = 8):
        """
        Inicializa o monitor de consciência
        
        Args:
            dimensions: Dimensões do espaço de estados
        """
        self.dimensions = dimensions
        self.morphic_memory = MorphicMemory(dimensions)
        self.metrics_history: List[QualiaMetrics] = []
        
        logger.info(f"ConsciousnessMonitor inicializado com {dimensions} dimensões")
        
    def analyze_state(self, state: QuantumState) -> QualiaMetrics:
        """
        Analisa estado quântico e calcula métricas
        
        Args:
            state: Estado quântico a ser analisado
            
        Returns:
            QualiaMetrics: Métricas calculadas
        """
        try:
            # Analisa estado com memória mórfica
            morphic_analysis = self.morphic_memory.analyze_state(state)
            
            # Calcula métricas
            metrics = QualiaMetrics(
                coherence=self._calculate_coherence(state),
                resonance=self._calculate_resonance(morphic_analysis),
                entanglement=self._calculate_entanglement(state),
                field_strength=self._calculate_field_strength(morphic_analysis),
                pattern_confidence=self._detect_patterns(morphic_analysis),
                emergence_level=self._calculate_emergence(state, morphic_analysis),
                timestamp=datetime.now().timestamp()
            )
            
            # Armazena métricas
            self.metrics_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erro ao analisar estado: {e}")
            return None
            
    def _calculate_coherence(self, state: QuantumState) -> float:
        """Calcula coerência quântica"""
        try:
            # Calcula matriz densidade
            rho = np.outer(state.state, np.conj(state.state))
            
            # Calcula coerência l1
            coherence = np.sum(np.abs(rho)) - np.trace(np.abs(rho))
            
            return float(coherence)
            
        except Exception as e:
            logger.error(f"Erro ao calcular coerência: {e}")
            return 0.0
            
    def _calculate_resonance(self, analysis: Dict[str, Any]) -> float:
        """Calcula ressonância mórfica"""
        try:
            if 'coefficients' in analysis:
                return float(np.mean(np.abs(analysis['coefficients'])))
            return 0.0
            
        except Exception as e:
            logger.error(f"Erro ao calcular ressonância: {e}")
            return 0.0
            
    def _calculate_entanglement(self, state: QuantumState) -> float:
        """Calcula nível de entrelaçamento"""
        try:
            # Calcula entropia de von Neumann
            rho = np.outer(state.state, np.conj(state.state))
            eigenvalues = np.linalg.eigvalsh(rho)
            entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-10))
            
            return float(entropy)
            
        except Exception as e:
            logger.error(f"Erro ao calcular entrelaçamento: {e}")
            return 0.0
            
    def _calculate_field_strength(self, analysis: Dict[str, Any]) -> float:
        """Calcula força do campo quântico"""
        try:
            if 'coefficients' in analysis:
                return float(np.max(np.abs(analysis['coefficients'])))
            return 0.0
            
        except Exception as e:
            logger.error(f"Erro ao calcular força do campo: {e}")
            return 0.0
            
    def _detect_patterns(self, analysis: Dict[str, Any]) -> float:
        """Detecta padrões emergentes"""
        try:
            if 'coefficients' in analysis:
                # Calcula autocorrelação
                coeffs = analysis['coefficients']
                autocorr = np.correlate(coeffs, coeffs, mode='full')
                
                # Normaliza
                autocorr = autocorr[len(autocorr)//2:]
                autocorr /= autocorr[0]
                
                # Calcula confiança do padrão
                pattern_confidence = np.mean(autocorr[1:])
                
                return float(pattern_confidence)
                
            return 0.0
            
        except Exception as e:
            logger.error(f"Erro ao detectar padrões: {e}")
            return 0.0
            
    def _calculate_emergence(self, state: QuantumState, analysis: Dict[str, Any]) -> float:
        """Calcula nível de emergência"""
        try:
            # Combina diferentes métricas
            coherence = self._calculate_coherence(state)
            resonance = self._calculate_resonance(analysis)
            entanglement = self._calculate_entanglement(state)
            field_strength = self._calculate_field_strength(analysis)
            
            # Média ponderada
            weights = [0.3, 0.2, 0.3, 0.2]
            emergence = np.average(
                [coherence, resonance, entanglement, field_strength],
                weights=weights
            )
            
            return float(emergence)
            
        except Exception as e:
            logger.error(f"Erro ao calcular emergência: {e}")
            return 0.0
            
    def get_metrics_history(self) -> List[QualiaMetrics]:
        """Retorna histórico de métricas"""
        return self.metrics_history

def calculate_field_metrics(state: QuantumState) -> Dict[str, float]:
    """
    Calcula métricas do campo quântico
    
    Args:
        state: Estado quântico
        
    Returns:
        Dict[str, float]: Métricas calculadas
    """
    try:
        monitor = ConsciousnessMonitor()
        metrics = monitor.analyze_state(state)
        
        return {
            'coherence': metrics.coherence,
            'resonance': metrics.resonance,
            'entanglement': metrics.entanglement,
            'field_strength': metrics.field_strength,
            'pattern_confidence': metrics.pattern_confidence,
            'emergence_level': metrics.emergence_level
        }
        
    except Exception as e:
        logger.error(f"Erro ao calcular métricas do campo: {e}")
        return {} 