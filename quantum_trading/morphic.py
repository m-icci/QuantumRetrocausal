"""
Campo Mórfico para o Sistema QUALIA
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime
from collections import deque

# Configure logging
logger = logging.getLogger("morphic_field")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class MorphicField:
    """
    Implementa campo mórfico para trading
    """
    
    def __init__(self, 
                 field_dimension: int = 2048,
                 memory_capacity: int = 1000,
                 stability_threshold: float = 0.7):
        """
        Inicializa campo mórfico
        
        Args:
            field_dimension: Dimensão do campo
            memory_capacity: Capacidade da memória
            stability_threshold: Limiar de estabilidade
        """
        logger.info("Inicializando Campo Mórfico")
        
        # Parâmetros
        self.field_dimension = field_dimension
        self.memory_capacity = memory_capacity
        self.stability_threshold = stability_threshold
        
        # Estado
        self.field_buffer = deque(maxlen=memory_capacity)
        self.current_state = {
            'field': np.zeros(field_dimension),
            'stability': 1.0,
            'resonance': 0.0,
            'last_update': None
        }
        
        logger.info(f"Campo inicializado com dimensão {field_dimension}")
    
    def atualizar_campo(self, quantum_field: np.ndarray) -> Dict[str, Any]:
        """
        Atualiza campo mórfico com novo campo quântico
        
        Args:
            quantum_field: Campo quântico
            
        Returns:
            Estado atualizado do campo
        """
        try:
            # Validar campo
            if not self._validate_field(quantum_field):
                return {
                    'error': 'Campo inválido',
                    'stability': 0.0,
                    'field': None
                }
            
            # Atualizar buffer
            self.field_buffer.append(quantum_field)
            
            # Calcular campo mórfico
            field = self._calculate_morphic_field(
                np.array(list(self.field_buffer))
            )
            
            # Calcular métricas
            stability = self._calculate_stability(field)
            resonance = self._calculate_resonance(field)
            coherence = self._calculate_coherence(field)
            
            # Atualizar estado
            self.current_state.update({
                'field': field,
                'stability': stability,
                'resonance': resonance,
                'coherence': coherence,
                'last_update': datetime.now()
            })
            
            return {
                'stability': float(stability),
                'resonance': float(resonance),
                'coherence': float(coherence),
                'field': field.tolist()
            }
            
        except Exception as e:
            logger.error(f"Erro na atualização do campo: {str(e)}")
            return {
                'error': str(e),
                'stability': 0.0,
                'field': None
            }
    
    def validar_campo(self, quantum_field: np.ndarray) -> bool:
        """
        Valida campo quântico contra campo mórfico
        
        Args:
            quantum_field: Campo quântico
            
        Returns:
            Se o campo é válido
        """
        try:
            # Validar campo
            if not self._validate_field(quantum_field):
                return False
            
            # Calcular estabilidade
            stability = self._calculate_stability(quantum_field)
            
            # Verificar limiar
            return stability >= self.stability_threshold
            
        except Exception as e:
            logger.error(f"Erro na validação do campo: {str(e)}")
            return False
    
    def is_stable(self) -> bool:
        """
        Verifica se o campo está estável
        
        Returns:
            Se o campo está estável
        """
        return self.current_state['stability'] >= self.stability_threshold
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Retorna estado atual do campo
        
        Returns:
            Estado atual
        """
        return {
            'stability': float(self.current_state['stability']),
            'resonance': float(self.current_state['resonance']),
            'coherence': float(self.current_state['coherence']),
            'last_update': self.current_state['last_update']
        }
    
    def _validate_field(self, field: np.ndarray) -> bool:
        """
        Valida campo quântico
        
        Args:
            field: Campo a validar
            
        Returns:
            Se o campo é válido
        """
        try:
            # Verificar tipo
            if not isinstance(field, np.ndarray):
                logger.warning("Campo deve ser numpy array")
                return False
            
            # Verificar dimensão
            if field.shape != (self.field_dimension,):
                logger.warning(f"Dimensão incorreta: {field.shape}")
                return False
            
            # Verificar valores
            if np.any(~np.isfinite(field)):
                logger.warning("Campo contém valores inválidos")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Erro na validação: {str(e)}")
            return False
    
    def _calculate_morphic_field(self, fields: np.ndarray) -> np.ndarray:
        """
        Calcula campo mórfico
        
        Args:
            fields: Array de campos quânticos
            
        Returns:
            Campo mórfico
        """
        try:
            # Calcular média dos campos
            mean_field = np.mean(fields, axis=0)
            
            # Normalizar campo
            field = mean_field / np.linalg.norm(mean_field)
            
            return field
            
        except Exception as e:
            logger.error(f"Erro no cálculo do campo: {str(e)}")
            return np.zeros(self.field_dimension)
    
    def _calculate_stability(self, field: np.ndarray) -> float:
        """
        Calcula estabilidade do campo
        
        Args:
            field: Campo mórfico
            
        Returns:
            Valor da estabilidade
        """
        try:
            # Calcular diferença com campo atual
            if len(self.field_buffer) > 0:
                current = self.current_state['field']
                diff = np.abs(field - current)
                stability = 1.0 - np.mean(diff)
            else:
                stability = 1.0
            
            return float(stability)
            
        except Exception as e:
            logger.error(f"Erro no cálculo da estabilidade: {str(e)}")
            return 0.0
    
    def _calculate_resonance(self, field: np.ndarray) -> float:
        """
        Calcula ressonância do campo
        
        Args:
            field: Campo mórfico
            
        Returns:
            Valor da ressonância
        """
        try:
            # Calcular autocorrelação
            corr = np.correlate(field, field, mode='full')
            
            # Normalizar
            resonance = np.max(np.abs(corr)) / len(field)
            
            return float(resonance)
            
        except Exception as e:
            logger.error(f"Erro no cálculo da ressonância: {str(e)}")
            return 0.0
    
    def _calculate_coherence(self, field: np.ndarray) -> float:
        """
        Calcula coerência do campo
        
        Args:
            field: Campo mórfico
            
        Returns:
            Valor da coerência
        """
        try:
            # Calcular matriz densidade
            rho = np.outer(field, field.conj())
            
            # Calcular pureza
            coherence = np.abs(np.trace(np.matmul(rho, rho)))
            
            return float(coherence)
            
        except Exception as e:
            logger.error(f"Erro no cálculo da coerência: {str(e)}")
            return 0.0
