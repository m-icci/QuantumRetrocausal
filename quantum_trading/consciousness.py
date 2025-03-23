"""
Consciência de Mercado para o Sistema QUALIA
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime
from collections import deque

# Configure logging
logger = logging.getLogger("market_consciousness")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class MarketConsciousness:
    """
    Implementa consciência de mercado para trading
    """
    
    def __init__(self, 
                 memory_capacity: int = 1000,
                 coherence_threshold: float = 0.3,
                 entropy_threshold: float = 0.7):
        """
        Inicializa consciência de mercado
        
        Args:
            memory_capacity: Capacidade da memória
            coherence_threshold: Limiar de coerência
            entropy_threshold: Limiar de entropia
        """
        logger.info("Inicializando Consciência de Mercado")
        
        # Parâmetros
        self.memory_capacity = memory_capacity
        self.coherence_threshold = coherence_threshold
        self.entropy_threshold = entropy_threshold
        
        # Estado
        self.state_buffer = deque(maxlen=memory_capacity)
        self.current_state = {
            'coherence': 1.0,
            'entropy': 0.0,
            'field': np.zeros(memory_capacity),
            'last_update': None
        }
        
        logger.info(f"Consciência inicializada com capacidade {memory_capacity}")
    
    def calculate_consciousness_field(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Calcula campo de consciência do mercado
        
        Args:
            data: Array de dados do mercado
            
        Returns:
            Campo de consciência
        """
        try:
            # Validar dados
            if not self._validate_data(data):
                return {
                    'error': 'Dados inválidos',
                    'coherence': 0.0,
                    'field': None
                }
            
            # Atualizar buffer
            for value in data:
                self.state_buffer.append(value)
            
            # Calcular campo
            field = self._calculate_field(
                np.array(list(self.state_buffer))
            )
            
            # Calcular métricas
            coherence = self._calculate_coherence(field)
            entropy = self._calculate_entropy(field)
            trend = self._calculate_trend(field)
            
            # Atualizar estado
            self.current_state.update({
                'coherence': coherence,
                'entropy': entropy,
                'field': field,
                'last_update': datetime.now()
            })
            
            return {
                'coherence': float(coherence),
                'entropy': float(entropy),
                'field': field.tolist(),
                'trend': float(trend)
            }
            
        except Exception as e:
            logger.error(f"Erro no cálculo do campo: {str(e)}")
            return {
                'error': str(e),
                'coherence': 0.0,
                'field': None
            }
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Retorna estado atual da consciência
        
        Returns:
            Estado atual
        """
        return {
            'coherence': float(self.current_state['coherence']),
            'entropy': float(self.current_state['entropy']),
            'last_update': self.current_state['last_update']
        }
    
    def _validate_data(self, data: np.ndarray) -> bool:
        """
        Valida dados de entrada
        
        Args:
            data: Dados a validar
            
        Returns:
            Se os dados são válidos
        """
        try:
            # Verificar tipo
            if not isinstance(data, np.ndarray):
                logger.warning("Dados devem ser numpy array")
                return False
            
            # Verificar dimensão
            if data.ndim != 1:
                logger.warning("Dados devem ser unidimensionais")
                return False
            
            # Verificar valores
            if np.any(~np.isfinite(data)):
                logger.warning("Dados contêm valores inválidos")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Erro na validação: {str(e)}")
            return False
    
    def _calculate_field(self, data: np.ndarray) -> np.ndarray:
        """
        Calcula campo de consciência
        
        Args:
            data: Dados do mercado
            
        Returns:
            Campo de consciência
        """
        try:
            # Normalizar dados
            normalized = (data - np.mean(data)) / (np.std(data) + 1e-8)
            
            # Calcular transformada de Fourier
            fft = np.fft.fft(normalized)
            
            # Calcular amplitudes
            amplitudes = np.abs(fft)
            
            # Calcular fases
            phases = np.angle(fft)
            
            # Criar campo
            field = amplitudes * np.exp(1j * phases)
            
            # Normalizar campo
            field = field / np.linalg.norm(field)
            
            return field
            
        except Exception as e:
            logger.error(f"Erro no cálculo do campo: {str(e)}")
            return np.zeros(len(data))
    
    def _calculate_coherence(self, field: np.ndarray) -> float:
        """
        Calcula coerência do campo
        
        Args:
            field: Campo de consciência
            
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
    
    def _calculate_entropy(self, field: np.ndarray) -> float:
        """
        Calcula entropia do campo
        
        Args:
            field: Campo de consciência
            
        Returns:
            Valor da entropia
        """
        try:
            # Calcular probabilidades
            probs = np.abs(field) ** 2
            probs = probs / np.sum(probs)
            
            # Calcular entropia
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            
            # Normalizar
            return entropy / np.log2(len(field))
            
        except Exception as e:
            logger.error(f"Erro no cálculo da entropia: {str(e)}")
            return 1.0
    
    def _calculate_trend(self, field: np.ndarray) -> float:
        """
        Calcula tendência do campo
        
        Args:
            field: Campo de consciência
            
        Returns:
            Valor da tendência
        """
        try:
            # Calcular fases
            phases = np.angle(field)
            
            # Calcular diferenças de fase
            phase_diff = np.diff(phases)
            
            # Ajustar para [-pi, pi]
            phase_diff = np.where(
                phase_diff > np.pi,
                phase_diff - 2*np.pi,
                phase_diff
            )
            phase_diff = np.where(
                phase_diff < -np.pi,
                phase_diff + 2*np.pi,
                phase_diff
            )
            
            # Calcular tendência média
            trend = np.mean(phase_diff) / np.pi
            
            return float(trend)
            
        except Exception as e:
            logger.error(f"Erro no cálculo da tendência: {str(e)}")
            return 0.0
