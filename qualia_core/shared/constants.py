#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QUALIA Sistema Unificado - Constantes Fundamentais
--------------------------------------------------

Este módulo contém todas as constantes fundamentais utilizadas pelo sistema QUALIA.
Centralizar as constantes aqui permite manter consistência em todo o sistema e
facilita alterações futuras.

Constantes incluem:
- Operadores quânticos (FOLD, MERGE, etc.)
- Constantes geométricas (PHI, PI, etc.)
- Configurações de sistema
- Constantes matemáticas especiais
"""

from enum import Enum
import numpy as np
from typing import Dict, Any, List, Set

# ----- Operadores QUALIA -----

class OperatorType(Enum):
    """
    Operadores QUALIA unificados
    
    Cada operador representa uma transformação quântica fundamental
    que pode ser aplicada a diferentes tipos de dados e estados.
    """
    FOLD = "F"          # Dobramento - dobra o espaço de dados sobre si mesmo
    MERGE = "M"         # Mesclagem - combina diferentes estados
    EMERGE = "E"        # Emergência - faz emergir padrões não-evidentes
    COLLAPSE = "C"      # Colapso - reduz estados de superposição a um valor definido
    DECOHERE = "D"      # Decoerência - introduz ruído quântico controlado
    OBSERVE = "O"       # Observação - causa colapso consciente do sistema
    TRANSCEND = "T"     # Transcendência - transforma para um nível superior
    RETARD = "R"        # Retardo - introduz atraso temporal
    ACCELERATE = "A"    # Aceleração - aumenta a velocidade de convergência
    RETROCAUSE = "Z"    # Retrocausalidade - influência do futuro no presente
    NARRATE = "N"       # Narrativa - cria contexto histórico para dados
    ENTRAIN = "X"       # Entrelaçamento - liga estados de forma não-local

# ----- Constantes Geométricas -----

class GeometricConstants:
    """
    Constantes geométricas fundamentais
    
    Estas constantes são utilizadas em cálculos quânticos e transformações
    de campo. A precisão destas constantes é importante para a estabilidade
    do sistema QUALIA.
    """
    PHI = 1.618033988749895       # Proporção áurea
    PHI_INVERSE = 0.618033988749895  # Inverso da proporção áurea
    PI = 3.141592653589793        # Pi
    TAU = 6.283185307179586       # Tau (2π)
    E = 2.718281828459045         # Número de Euler
    SQRT2 = 1.4142135623730951    # Raiz quadrada de 2
    SQRT3 = 1.7320508075688772    # Raiz quadrada de 3
    SQRT5 = 2.236067977499790     # Raiz quadrada de 5
    LN2 = 0.6931471805599453      # Logaritmo natural de 2
    LN10 = 2.302585092994046      # Logaritmo natural de 10
    GOLDEN_ANGLE = 2.399963229728653  # Ângulo áureo (137.5º em radianos)

# ----- Constantes de Sistema -----

class SystemDefaults:
    """
    Configurações padrão do sistema
    
    Valores padrão para diversos parâmetros do sistema QUALIA.
    Estes valores podem ser sobrescritos por configurações específicas.
    """
    DEFAULT_FIELD_DIMENSION = 64      # Dimensão padrão para campos mórficos
    DEFAULT_COHERENCE = 0.5           # Nível de coerência inicial
    DEFAULT_FIELD_STRENGTH = 0.5      # Força de campo inicial
    DEFAULT_HASH_SIZE = 32            # Tamanho de hash padrão (bytes)
    MAX_STATES = 128                  # Número máximo de estados quânticos
    HISTORY_BUFFER_SIZE = 1000        # Tamanho do buffer de histórico
    CONVERGENCE_THRESHOLD = 0.0001    # Limiar de convergência
    RETROCAUSALITY_LIMIT = 7          # Limite de passos retrocausais
    
    # Intervalos de tempo (em segundos)
    AUTO_SAVE_INTERVAL = 300          # Salvar estado a cada 5 minutos
    FIELD_UPDATE_INTERVAL = 0.1       # Atualizar campo a cada 100ms
    COHERENCE_CHECK_INTERVAL = 5      # Verificar coerência a cada 5 segundos

# ----- Sequências de Operadores -----

class OperatorSequences:
    """
    Sequências pré-definidas de operadores QUALIA
    
    Cada sequência representa um padrão específico de transformações
    que produz um efeito distinto no sistema.
    """
    BASIC = "FMEDT"        # Sequência básica (Fold-Merge-Emerge-Decohere-Transcend)
    EMERGENT = "MEZRT"     # Ênfase em Emergência e Retrocausalidade
    TRANSCENDENT = "TFMEC" # Transcendência inicial para maior expansão
    DECOHERENT = "DMEFZ"   # Decoerência com Retrocausalidade
    COLLAPSE = "FEMCR"     # Abordagem de colapso e retardo
    FOLDING = "MEDFT"      # Ênfase em Emergência e Dobramento
    RETROCAUSAL = "RMETZ"  # Abordagem com Retardo e Retrocausalidade
    TIME_LOOP = "FTEMZ"    # Transcendência e Retrocausalidade
    OBSERVATIONAL = "OMED" # Ênfase em Observação
    HARMONIC = "TFDM"      # Transcendência-Dobramento-Decoerência
    QUANTUM_WELL = "MORMOR" # Observação-Retardo-Mesclagem recursiva
    ENTANGLED = "XMEXF"    # Entrelaçamento com Emergência e Dobramento
    
    # Sequências completas ordenadas por efeito de campo
    SEQUENCES = [
        BASIC, EMERGENT, TRANSCENDENT, DECOHERENT, COLLAPSE,
        FOLDING, RETROCAUSAL, TIME_LOOP, OBSERVATIONAL, HARMONIC,
        QUANTUM_WELL, ENTANGLED
    ]

# ----- Mapeamento de Interfaces -----

# Mapeamento entre nomes de campos antigos e novos para compatibilidade
FIELD_MAPPING = {
    "morph_field": "morphic_field",
    "quantum_field": "morphic_field.quantum_state",
    "coherence_level": "morphic_field.metrics.coherence",
    "field_state": "morphic_field.field_state",
    "complex_state": "morphic_field.complex_state",
    "operator_mapping": "operators.mapping",
    "metrics": "morphic_field.metrics",
}

# Indicadores de algoritmos e suas implementações
ALGORITHM_IMPLEMENTATIONS = {
    "nonce_generation": [
        "basic", "quantum", "adaptive", "retroactive", 
        "coherent", "unified", "holographic"
    ],
    "hash_verification": [
        "standard", "probabilistic", "quantum", "morphic"
    ],
    "difficulty_adjustment": [
        "fixed", "adaptive", "emergent", "harmonic"
    ]
}

# Codificação de estados quânticos complexos
QUANTUM_STATE_ENCODING = {
    "superposition": "S",
    "entanglement": "E",
    "decoherence": "D",
    "collapse": "C",
    "measurement": "M",
    "retroactive": "R",
    "transcendent": "T"
}

# Definir pesos para combinação harmônica de constantes
HARMONIC_WEIGHTS = {
    "phi": 0.618033988749895,
    "pi": 0.318309886183791,
    "e": 0.367879441171442,
    "sqrt2": 0.414213562373095,
    "sqrt3": 0.577350269189626,
    "ln2": 0.693147180559945,
}
