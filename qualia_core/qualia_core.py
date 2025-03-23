"""
qualia_core.py
--------------------------------------
Módulo central do sistema QUALIA com constantes e classes fundamentais.

Este módulo define as constantes geométricas, classes e tipos fundamentais
utilizados em todo o sistema QUALIA para processamento quântico adaptativo.
"""

import numpy as np
import math
import logging
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional, Union

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GeometricConstants:
    """
    Constantes geométricas fundamentais usadas pelo sistema QUALIA.
    
    Estas constantes são utilizadas para cálculos quânticos e operações
    morfogenéticas em todo o sistema.
    """
    # Constantes primárias
    PHI = (1 + np.sqrt(5)) / 2  # Proporção áurea (1.618...)
    PI = np.pi
    E = np.e
    
    # Constantes derivadas
    PHI_INVERSE = 1 / PHI
    PHI_SQUARED = PHI * PHI
    SQRT_PHI = np.sqrt(PHI)
    
    # Constantes específicas para operações QUALIA
    QUANTUM_RESONANCE = PHI * PI
    MORPHIC_THRESHOLD = E / PI
    COHERENCE_BASELINE = 0.5 + (PHI_INVERSE / 2)
    
    # Sequência de Fibonacci para níveis quânticos
    QUANTUM_GRANULARITY = [3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
    
    @classmethod
    def fibonacci(cls, n: int) -> int:
        """
        Calcula o n-ésimo número de Fibonacci.
        
        Args:
            n: Posição na sequência (começando de 0)
            
        Returns:
            Número de Fibonacci na posição n
        """
        if n <= 0:
            return 0
        elif n == 1:
            return 1
            
        a, b = 0, 1
        for _ in range(2, n+1):
            a, b = b, a + b
        return b
    
    @classmethod
    def golden_sequence(cls, n: int) -> List[float]:
        """
        Gera uma sequência de n números derivados da proporção áurea.
        
        Args:
            n: Tamanho da sequência
            
        Returns:
            Lista de valores derivados da proporção áurea
        """
        return [cls.PHI ** i % 1 for i in range(1, n+1)]


class OperatorType:
    # Valores unificados para operadores QUALIA
    FOLD = "F"
    MERGE = "M"
    EMERGE = "E"
    COLLAPSE = "C"
    DECOHERE = "D"
    OBSERVE = "O"
    TRANSCEND = "T"
    RETARD = "R"
    ACCELERATE = "A"
    RETROCAUSE = "Z"
    NARRATE = "N"
    ENTRAIN = "X"


class QualiaMode(str, Enum):
    """
    Modos de operação do sistema QUALIA.
    
    Cada modo configura o sistema para um equilíbrio diferente
    entre performance, coerência e adaptabilidade.
    """
    ATOMIC = "atomic"       # Alta performance, baixa coerência
    BALANCED = "balanced"   # Equilíbrio entre performance e coerência
    EMERGENT = "emergent"   # Alta coerência, emergência máxima


class QualiaObserver:
    """
    Observador quântico para o sistema QUALIA.
    
    O observador monitora e influencia o comportamento do sistema,
    implementando o princípio do observador quântico de maneira simulada.
    """
    
    def __init__(self, 
                sensitivity: float = 0.5, 
                coherence_weight: float = 0.4,
                field_weight: float = 0.3,
                entropy_weight: float = 0.3):
        """
        Inicializa o observador QUALIA.
        
        Args:
            sensitivity: Sensibilidade do observador (0.0-1.0)
            coherence_weight: Peso da coerência na observação
            field_weight: Peso do campo na observação
            entropy_weight: Peso da entropia na observação
        """
        self.sensitivity = max(0.0, min(1.0, sensitivity))
        
        # Normalizar pesos
        total_weight = coherence_weight + field_weight + entropy_weight
        self.weights = {
            "coherence": coherence_weight / total_weight,
            "field": field_weight / total_weight,
            "entropy": entropy_weight / total_weight
        }
        
        self.observations = []
        self.state = {
            "coherence": 0.5,
            "field_strength": 0.5,
            "entropy": 0.5
        }
        
        logger.info(f"QualiaObserver inicializado com sensibilidade {sensitivity}")
        
    def observe(self, 
                coherence: Optional[float] = None,
                field_strength: Optional[float] = None,
                entropy: Optional[float] = None) -> float:
        """
        Realiza uma observação do sistema QUALIA.
        
        Args:
            coherence: Valor de coerência (0.0-1.0)
            field_strength: Valor da força do campo (0.0-1.0)
            entropy: Valor da entropia (0.0-1.0)
            
        Returns:
            Valor da influência do observador (0.0-1.0)
        """
        # Atualizar estado se valores forem fornecidos
        if coherence is not None:
            self.state["coherence"] = max(0.0, min(1.0, coherence))
        if field_strength is not None:
            self.state["field_strength"] = max(0.0, min(1.0, field_strength))
        if entropy is not None:
            self.state["entropy"] = max(0.0, min(1.0, entropy))
            
        # Calcular valor de observação ponderado
        observation_value = (
            self.state["coherence"] * self.weights["coherence"] +
            self.state["field_strength"] * self.weights["field"] +
            self.state["entropy"] * self.weights["entropy"]
        )
        
        # Aplicar sensibilidade do observador
        influence = observation_value * self.sensitivity
        
        # Registrar observação
        self.observations.append({
            "timestamp": np.datetime64('now'),
            "coherence": self.state["coherence"],
            "field_strength": self.state["field_strength"],
            "entropy": self.state["entropy"],
            "influence": influence
        })
        
        return influence
    
    def get_observation_history(self) -> List[Dict[str, Any]]:
        """
        Retorna o histórico de observações.
        
        Returns:
            Lista de observações registradas
        """
        return self.observations
    
    def calibrate(self, sensitivity: float) -> None:
        """
        Recalibra a sensibilidade do observador.
        
        Args:
            sensitivity: Nova sensibilidade (0.0-1.0)
        """
        old_sensitivity = self.sensitivity
        self.sensitivity = max(0.0, min(1.0, sensitivity))
        logger.info(f"Observador recalibrado: {old_sensitivity:.2f} -> {self.sensitivity:.2f}")


class QualiaState:
    """
    Representa o estado quântico do sistema QUALIA.
    
    Esta classe encapsula os valores que definem o estado
    atual do sistema, facilitando a propagação entre componentes.
    """
    
    def __init__(self,
                coherence: float = 0.5,
                field_strength: float = 0.5,
                entropy: float = 0.5,
                granularity: int = 21):
        """
        Inicializa um estado QUALIA.
        
        Args:
            coherence: Coerência quântica (0.0-1.0)
            field_strength: Força do campo morfogenético (0.0-1.0)
            entropy: Entropia do sistema (0.0-1.0)
            granularity: Granularidade quântica (em bits)
        """
        self.coherence = max(0.0, min(1.0, coherence))
        self.field_strength = max(0.0, min(1.0, field_strength))
        self.entropy = max(0.0, min(1.0, entropy))
        self.granularity = granularity
        self.creation_time = np.datetime64('now')
        self.history = []
        
        # Registro inicial
        self._log_state("initialize")
    
    def update(self, **kwargs):
        """
        Atualiza o estado com novos valores.
        
        Args:
            **kwargs: Parâmetros de estado para atualizar
        """
        if "coherence" in kwargs:
            self.coherence = max(0.0, min(1.0, kwargs["coherence"]))
        if "field_strength" in kwargs:
            self.field_strength = max(0.0, min(1.0, kwargs["field_strength"]))
        if "entropy" in kwargs:
            self.entropy = max(0.0, min(1.0, kwargs["entropy"]))
        if "granularity" in kwargs:
            self.granularity = kwargs["granularity"]
            
        # Registrar atualização
        self._log_state("update")
    
    def clone(self):
        """
        Cria uma cópia do estado atual.
        
        Returns:
            Novo objeto QualiaState com os mesmos valores
        """
        new_state = QualiaState(
            coherence=self.coherence,
            field_strength=self.field_strength,
            entropy=self.entropy,
            granularity=self.granularity
        )
        return new_state
    
    def _log_state(self, action: str):
        """
        Registra o estado atual no histórico.
        
        Args:
            action: Descrição da ação que gerou o estado
        """
        self.history.append({
            "timestamp": np.datetime64('now'),
            "action": action,
            "coherence": self.coherence,
            "field_strength": self.field_strength,
            "entropy": self.entropy,
            "granularity": self.granularity
        })
    
    def __str__(self):
        """String representation of the state"""
        return (f"QualiaState(coherence={self.coherence:.2f}, "
                f"field_strength={self.field_strength:.2f}, "
                f"entropy={self.entropy:.2f}, "
                f"granularity={self.granularity})")


class QualiaMiner:
    """
    Minerador de padrões no campo morfológico.
    Extrai padrões e métricas para análise e otimização.
    """
    def __init__(self, field: np.ndarray):
        self.field = field

    def detect_patterns(self, window_size: int = 8) -> Dict[str, Any]:
        """
        Detecta padrões no campo morfológico.
        
        Args:
            window_size: Tamanho da janela para análise
            
        Returns:
            Informações sobre padrões detectados
        """
        # Converter campo para representação binária
        threshold = np.mean(self.field)
        binary_field = (self.field > threshold).astype(int)
        
        # Detectar padrões horizontais, verticais e diagonais
        patterns = {"horizontal": {}, "vertical": {}, "diagonal": {}}
        dimension = binary_field.shape[0]
        
        # Padrões horizontais
        for i in range(dimension):
            for j in range(dimension - window_size + 1):
                pattern = tuple(binary_field[i, j:j+window_size])
                patterns["horizontal"][pattern] = patterns["horizontal"].get(pattern, 0) + 1
                
        # Padrões verticais
        for i in range(dimension - window_size + 1):
            for j in range(dimension):
                pattern = tuple(binary_field[i:i+window_size, j])
                patterns["vertical"][pattern] = patterns["vertical"].get(pattern, 0) + 1
                
        # Padrões diagonais (direção principal)
        for i in range(dimension - window_size + 1):
            for j in range(dimension - window_size + 1):
                pattern = tuple(binary_field[i+k, j+k] for k in range(window_size))
                patterns["diagonal"][pattern] = patterns["diagonal"].get(pattern, 0) + 1
                
        # Encontrar padrões dominantes
        results = {}
        for direction, dir_patterns in patterns.items():
            if not dir_patterns:
                results[direction] = {"dominant": None, "count": 0, "diversity": 0}
                continue
                
            # Padrão dominante
            dominant_pattern = max(dir_patterns.items(), key=lambda x: x[1])
            
            # Diversidade de padrões (normalizada)
            max_possible = dimension * (dimension - window_size + 1)
            diversity = len(dir_patterns) / min(max_possible, 2**window_size)
            
            results[direction] = {
                "dominant": list(dominant_pattern[0]),
                "count": dominant_pattern[1],
                "diversity": diversity,
                "unique_patterns": len(dir_patterns)
            }
            
        return results
