# qualia/operator.py
import logging
import random
import math
import numpy as np
from typing import Union, Tuple, Dict, List, Any, Optional

# Importar constantes geométricas
try:
    from .constants import GeometricConstants, OperatorType
except ImportError:
    # Fallback para operação standalone
    class GeometricConstants:
        GOLDEN_RATIO = 1.618
        PHI = (1 + np.sqrt(5)) / 2
        PI = np.pi
        E = np.e
    
    class OperatorType:
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

logger = logging.getLogger(__name__)

class QualiaOperator:
    """
    Implementation of basic QUALIA operators for quantum-inspired mining.
    
    Esta classe implementa os operadores QUALIA fundamentais que permitem
    a transformação de estados quânticos simulados para mineração adaptativa.
    
    Operadores implementados:
    - Ressonância: Ajusta valores usando a proporção áurea
    - Superposição: Aplica transformação de código Gray para criar superposição
    - Retrocausalidade: Implementa influência retroativa entre estados
    - Dobramento: Dobra o espaço de estados criando simetrias
    - Transcendência: Expande o espaço de busca para incluir novas possibilidades
    - Colapso: Colapsa múltiplos estados em um único estado definido
    - Decoerência: Simula interação com o ambiente, induzindo ruído controlado
    """
    
    GOLDEN_RATIO = GeometricConstants.PHI if hasattr(GeometricConstants, 'PHI') else 1.618
    MAX_NONCE = 2**32
    
    @staticmethod
    def resonance(nonce: int) -> int:
        """
        Resonance operator: ajusta o nonce usando a proporção áurea.
        
        Args:
            nonce (int): Valor de nonce de entrada
            
        Returns:
            int: Valor de nonce ajustado
            
        Raises:
            ValueError: Se o nonce for negativo ou exceder 32 bits
        """
        if not 0 <= nonce < QualiaOperator.MAX_NONCE:
            logger.error(f"Invalid nonce value: {nonce}")
            raise ValueError("Nonce must be between 0 and 2^32")
            
        result = int(nonce * QualiaOperator.GOLDEN_RATIO) % QualiaOperator.MAX_NONCE
        logger.debug(f"Resonance: {nonce} -> {result}")
        return result

    @staticmethod
    def superposition(nonce: int) -> int:
        """
        Superposition operator: aplica um mapeamento estilo código Gray.
        
        Args:
            nonce (int): Valor de nonce de entrada
            
        Returns:
            int: Valor de nonce transformado
            
        Raises:
            ValueError: Se o nonce for negativo ou exceder 32 bits
        """
        if not 0 <= nonce < QualiaOperator.MAX_NONCE:
            logger.error(f"Invalid nonce value: {nonce}")
            raise ValueError("Nonce must be between 0 and 2^32")
            
        result = nonce ^ (nonce >> 1)
        logger.debug(f"Superposition: {nonce} -> {result}")
        return result

    @staticmethod
    def retrocausality(nonce: int, last_nonce: int) -> int:
        """
        Retrocausality operator: calcula média entre nonce atual e anterior com variação.
        
        Args:
            nonce (int): Valor de nonce atual
            last_nonce (int): Valor de nonce anterior
            
        Returns:
            int: Média dos dois nonces com variação
            
        Raises:
            ValueError: Se algum nonce for inválido
        """
        if not (0 <= nonce < QualiaOperator.MAX_NONCE and 
                0 <= last_nonce < QualiaOperator.MAX_NONCE):
            logger.error(f"Invalid nonce values: current={nonce}, last={last_nonce}")
            raise ValueError("Nonce values must be between 0 and 2^32")
            
        variation = random.randint(-50000, 50000)  # Introduz variação para evitar repetição
        result = ((nonce + last_nonce) // 2) + variation
        logger.debug(f"Retrocausality: {nonce}, {last_nonce} -> {result % QualiaOperator.MAX_NONCE}")
        return result % QualiaOperator.MAX_NONCE
        
    @staticmethod
    def fold(nonce: int, axis: int = None) -> int:
        """
        Folding operator: dobra o espaço de estados criando simetrias.
        
        Args:
            nonce (int): Valor de nonce de entrada
            axis (int, optional): Eixo de dobramento (bit de referência)
            
        Returns:
            int: Valor de nonce transformado
        """
        if not 0 <= nonce < QualiaOperator.MAX_NONCE:
            logger.error(f"Invalid nonce value: {nonce}")
            raise ValueError("Nonce must be between 0 and 2^32")
            
        if axis is None:
            # Usar proporção áurea para determinar eixo de dobramento
            axis = int((QualiaOperator.GOLDEN_RATIO * 32) % 32)
        
        # Criar máscara para bits à direita do eixo
        mask = (1 << axis) - 1
        
        # Dobrar: invertendo os bits à direita do eixo
        right_bits = nonce & mask
        inverted_right = (~right_bits) & mask
        left_bits = nonce & ~mask
        
        result = left_bits | inverted_right
        logger.debug(f"Fold (axis={axis}): {nonce} -> {result}")
        return result
        
    @staticmethod
    def transcend(nonce: int, expansion_factor: float = 1.5) -> int:
        """
        Transcendence operator: expande o espaço de busca.
        
        Args:
            nonce (int): Valor de nonce de entrada
            expansion_factor (float): Fator de expansão (>1.0)
            
        Returns:
            int: Valor de nonce transformado
        """
        if not 0 <= nonce < QualiaOperator.MAX_NONCE:
            logger.error(f"Invalid nonce value: {nonce}")
            raise ValueError("Nonce must be between 0 and 2^32")
            
        if expansion_factor <= 0:
            logger.error(f"Invalid expansion factor: {expansion_factor}")
            raise ValueError("Expansion factor must be positive")
            
        # Aplicar expansão não-linear baseada em série de Taylor
        scaled = nonce / QualiaOperator.MAX_NONCE  # Normalizar para [0,1]
        
        # Aplicar transformação transcendente (baseada em série de Taylor para e^x)
        transcended = scaled
        for i in range(1, 5):  # Usar primeiros 4 termos da série
            transcended += (scaled ** i) / math.factorial(i)
            
        # Aplicar fator de expansão e normalizar de volta
        transcended = (transcended * expansion_factor) % 1.0
        result = int(transcended * QualiaOperator.MAX_NONCE)
        
        logger.debug(f"Transcend: {nonce} -> {result}")
        return result
        
    @staticmethod
    def collapse(nonces: List[int], weights: List[float] = None) -> int:
        """
        Collapse operator: colapsa múltiplos estados em um único estado.
        
        Args:
            nonces (List[int]): Lista de valores de nonce em superposição
            weights (List[float], optional): Pesos relativos para cada nonce
            
        Returns:
            int: Valor de nonce colapsado único
        """
        if not nonces:
            logger.error("Empty nonce list provided to collapse operator")
            raise ValueError("At least one nonce must be provided")
            
        # Validar todos os nonces
        for n in nonces:
            if not 0 <= n < QualiaOperator.MAX_NONCE:
                logger.error(f"Invalid nonce value: {n}")
                raise ValueError("All nonces must be between 0 and 2^32")
        
        # Se não houver pesos, usar pesos iguais
        if weights is None:
            weights = [1.0] * len(nonces)
        elif len(weights) != len(nonces):
            logger.error(f"Mismatch between nonces and weights: {len(nonces)} vs {len(weights)}")
            raise ValueError("Number of weights must match number of nonces")
            
        # Normalizar pesos
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Colapsar usando média ponderada
        result = sum(n * w for n, w in zip(nonces, normalized_weights))
        result = int(result) % QualiaOperator.MAX_NONCE
        
        logger.debug(f"Collapse: {nonces} -> {result}")
        return result
        
    @staticmethod
    def decohere(nonce: int, gamma: float = 0.05) -> int:
        """
        Decoherence operator: induz ruído controlado para simular interação com ambiente.
        
        Args:
            nonce (int): Valor de nonce de entrada
            gamma (float): Coeficiente de decoerência (0.0-1.0)
            
        Returns:
            int: Valor de nonce com decoerência
        """
        if not 0 <= nonce < QualiaOperator.MAX_NONCE:
            logger.error(f"Invalid nonce value: {nonce}")
            raise ValueError("Nonce must be between 0 and 2^32")
            
        if not 0 <= gamma <= 1:
            logger.error(f"Invalid gamma value: {gamma}")
            raise ValueError("Gamma must be between 0.0 and 1.0")
            
        # Calcular magnitude do ruído com base em gamma
        noise_amplitude = int(gamma * QualiaOperator.MAX_NONCE)
        noise = random.randint(-noise_amplitude, noise_amplitude)
        
        # Aplicar ruído
        result = (nonce + noise) % QualiaOperator.MAX_NONCE
        
        logger.debug(f"Decohere (gamma={gamma}): {nonce} -> {result}")
        return result
        
    @staticmethod
    def entangle(nonce1: int, nonce2: int, strength: float = 0.5) -> Tuple[int, int]:
        """
        Entanglement operator: entrelaça dois nonces, criando correlação quântica simulada.
        
        Args:
            nonce1 (int): Primeiro valor de nonce
            nonce2 (int): Segundo valor de nonce
            strength (float): Força do entrelaçamento (0.0-1.0)
            
        Returns:
            Tuple[int, int]: Par de nonces entrelaçados
        """
        if not (0 <= nonce1 < QualiaOperator.MAX_NONCE and 
                0 <= nonce2 < QualiaOperator.MAX_NONCE):
            logger.error(f"Invalid nonce values: nonce1={nonce1}, nonce2={nonce2}")
            raise ValueError("Nonce values must be between 0 and 2^32")
            
        if not 0 <= strength <= 1:
            logger.error(f"Invalid entanglement strength: {strength}")
            raise ValueError("Entanglement strength must be between 0.0 and 1.0")
            
        # Calcular média ponderada
        combined = int((nonce1 + nonce2) / 2)
        
        # Criar novos valores com correlação
        entangled1 = int((1 - strength) * nonce1 + strength * combined) % QualiaOperator.MAX_NONCE
        entangled2 = int((1 - strength) * nonce2 + strength * combined) % QualiaOperator.MAX_NONCE
        
        # Garantir correlação adicional usando XOR
        if strength > 0.7:
            shared_bits = random.randint(1, 8)  # Número de bits correlacionados
            mask = (1 << shared_bits) - 1
            common_pattern = random.randint(0, mask)
            
            # Aplicar padrão comum aos bits menos significativos
            entangled1 = (entangled1 & ~mask) | common_pattern
            entangled2 = (entangled2 & ~mask) | common_pattern
            
        logger.debug(f"Entangle: ({nonce1}, {nonce2}) -> ({entangled1}, {entangled2})")
        return entangled1, entangled2
        
    @staticmethod
    def apply_sequence(nonce: int, sequence: str) -> int:
        """
        Aplica uma sequência de operadores QUALIA a um nonce.
        
        Args:
            nonce (int): Valor inicial de nonce
            sequence (str): Sequência de códigos de operadores (ex: "FMECDT")
            
        Returns:
            int: Nonce final após aplicação da sequência
        """
        if not 0 <= nonce < QualiaOperator.MAX_NONCE:
            logger.error(f"Invalid nonce value: {nonce}")
            raise ValueError("Nonce must be between 0 and 2^32")
            
        result = nonce
        last_result = nonce
        
        logger.info(f"Applying operator sequence: {sequence}")
        
        for op in sequence:
            if op == OperatorType.FOLD:
                result = QualiaOperator.fold(result)
            elif op == OperatorType.MERGE:
                result = QualiaOperator.collapse([result, last_result])
            elif op == OperatorType.EMERGE:
                result = QualiaOperator.resonance(result)
            elif op == OperatorType.COLLAPSE:
                result = QualiaOperator.collapse([result, result ^ 0xFFFFFFFF])
            elif op == OperatorType.DECOHERE:
                result = QualiaOperator.decohere(result)
            elif op == OperatorType.OBSERVE:
                # Observe não altera o valor, apenas registra
                logger.info(f"Observed state: {result}")
            elif op == OperatorType.TRANSCEND:
                result = QualiaOperator.transcend(result)
            elif op == OperatorType.RETARD:
                result = result
                # Demoramos mais para aplicar retardo (sem efeito real)
            elif op == OperatorType.ACCELERATE:
                # Aceleração aplica duas vezes o mesmo operador
                result = QualiaOperator.resonance(QualiaOperator.resonance(result))
            elif op == OperatorType.RETROCAUSE:
                result = QualiaOperator.retrocausality(result, last_result)
            elif op == OperatorType.NARRATE:
                # Narrativa não altera o valor, apenas registra
                binary = bin(result)[2:].zfill(32)
                pattern = f"{binary[:8]}...{binary[-8:]}"
                logger.info(f"Narrative pattern: {pattern}")
            elif op == OperatorType.ENTRAIN:
                result, _ = QualiaOperator.entangle(result, last_result)
            else:
                logger.warning(f"Unknown operator: {op}")
                
            # Armazenar resultado anterior para operadores que precisam de histórico
            last_result = result
            
        logger.info(f"Sequence result: {nonce} -> {result}")
        return result