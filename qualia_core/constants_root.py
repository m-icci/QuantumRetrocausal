#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo de constantes para o QUALIAMiner
"""

import numpy as np

class GeometricConstants:
    """Constantes geométricas fundamentais"""
    PHI = (1 + np.sqrt(5)) / 2
    PHI_INVERSE = 2 / (1 + np.sqrt(5))
    SQRT_5 = np.sqrt(5)
    E = np.e
    PI = np.pi


class OperatorType:
    """Tipos de operadores QUALIA"""
    FOLD = "F"       # Dobramento
    MERGE = "M"      # Mesclagem
    EMERGE = "E"     # Emergência
    COLLAPSE = "C"   # Colapso
    DECOHERE = "D"   # Decoerência 
    OBSERVE = "O"    # Observação
    TRANSCEND = "T"  # Transcendência
    RETARD = "R"     # Retardo
    ACCELERATE = "A" # Aceleração
    RETROCAUSE = "Z" # Retrocausalidade
    NARRATE = "N"    # Narrativa
    ENTRAIN = "X"    # Entrelaçamento
