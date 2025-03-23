"""
Analisador de Padrões Quânticos para o Sistema QUALIA
Versão 2.0 - Integração com Memória Holográfica Avançada
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime
from .holographic_memory import HolographicMemory
from .morphic import MorphicField
from .consciousness import MarketConsciousness

# Configure logging
logger = logging.getLogger("quantum_pattern_analyzer")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class QuantumPatternAnalyzer:
    """
    Analisador de padrões quânticos para trading
    """
    
    def __init__(self,
                 memory_dimension: int = 2048,
                 max_memory_states: int = 1000,
                 similarity_threshold: float = 0.7,
                 coherence_threshold: float = 0.8):
        """
        Inicializa o analisador
        
        Args:
            memory_dimension: Dimensão da memória holográfica
            max_memory_states: Número máximo de estados
            similarity_threshold: Limiar de similaridade
            coherence_threshold: Limiar de coerência
        """
        logger.info("Inicializando Analisador de Padrões Quânticos")
        
        # Parâmetros
        self.memory_dimension = memory_dimension
        self.max_memory_states = max_memory_states
        self.similarity_threshold = similarity_threshold
        self.coherence_threshold = coherence_threshold
        
        # Componentes
        self.memory = HolographicMemory(
            dimension=memory_dimension,
            max_states=max_memory_states,
            similarity_threshold=similarity_threshold
        )
        self.morphic_field = MorphicField()
        self.consciousness = MarketConsciousness()
        
        # Estado
        self.current_state = {
            'total_patterns': 0,
            'last_analysis': None,
            'coherence': 1.0,
            'entropy': 0.0
        }
        
        logger.info("Analisador inicializado com sucesso")
    
    def analyze_pattern(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisa padrão de mercado
        
        Args:
            market_data: Dados do mercado
            
        Returns:
            Resultado da análise
        """
        try:
            # Validar dados
            if not self._validate_market_data(market_data):
                return self._get_default_result()
            
            # Extrair campo quântico
            field = self._extract_quantum_field(market_data)
            
            # Calcular métricas
            entropy = self._calculate_entropy(field)
            coherence = self._calculate_coherence(field)
            
            # Criar estado
            state = {
                'price': market_data.get('price', 0.0),
                'field': field.tolist(),
                'entropy': entropy,
                'coherence': coherence,
                'timestamp': datetime.now().timestamp()
            }
            
            # Armazenar estado
            self.memory.store_state(state)
            
            # Buscar estados similares
            similar_states = self.memory.retrieve_similar_states(
                field, max_results=5
            )
            
            # Detectar padrões mórficos
            morphic_patterns = self.morphic_field.detect_patterns(field)
            
            # Integrar consciência
            consciousness_state = self.consciousness.analyze_state(state)
            
            # Atualizar estado atual
            self.current_state.update({
                'total_patterns': self.memory.get_current_state()['total_states'],
                'last_analysis': datetime.now(),
                'coherence': coherence,
                'entropy': entropy
            })
            
            # Preparar resultado
            result = {
                'state': state,
                'similar_states': similar_states,
                'morphic_patterns': morphic_patterns,
                'consciousness': consciousness_state,
                'metrics': {
                    'entropy': entropy,
                    'coherence': coherence,
                    'total_patterns': self.current_state['total_patterns']
                },
                'recommendation': self._generate_recommendation(
                    state, similar_states, morphic_patterns, consciousness_state
                )
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Erro na análise: {str(e)}")
            return self._get_default_result()
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Retorna estado atual do analisador
        
        Returns:
            Estado atual
        """
        return {
            'total_patterns': self.current_state['total_patterns'],
            'last_analysis': self.current_state['last_analysis'],
            'coherence': float(self.current_state['coherence']),
            'entropy': float(self.current_state['entropy'])
        }
    
    def save_state(self, filename: str) -> bool:
        """
        Salva estado do analisador
        
        Args:
            filename: Nome do arquivo
            
        Returns:
            Se salvou com sucesso
        """
        try:
            # Salvar memória
            self.memory.save_to_file(filename)
            
            logger.info(f"Estado salvo em {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao salvar estado: {str(e)}")
            return False
    
    def load_state(self, filename: str) -> bool:
        """
        Carrega estado do analisador
        
        Args:
            filename: Nome do arquivo
            
        Returns:
            Se carregou com sucesso
        """
        try:
            # Carregar memória
            self.memory.load_from_file(filename)
            
            # Atualizar estado atual
            memory_state = self.memory.get_current_state()
            self.current_state.update({
                'total_patterns': memory_state['total_states'],
                'last_analysis': memory_state['last_update'],
                'coherence': memory_state['coherence']
            })
            
            logger.info(f"Estado carregado de {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar estado: {str(e)}")
            return False
    
    def _validate_market_data(self, data: Dict[str, Any]) -> bool:
        """
        Valida dados do mercado
        
        Args:
            data: Dados a validar
            
        Returns:
            Se os dados são válidos
        """
        try:
            # Verificar campos obrigatórios
            required = ['price', 'volume', 'timestamp']
            if not all(k in data for k in required):
                logger.warning("Dados não contêm campos obrigatórios")
                return False
            
            # Verificar tipos
            if not isinstance(data['price'], (int, float)):
                logger.warning("Preço inválido")
                return False
                
            if not isinstance(data['volume'], (int, float)):
                logger.warning("Volume inválido")
                return False
                
            if not isinstance(data['timestamp'], (int, float)):
                logger.warning("Timestamp inválido")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Erro na validação: {str(e)}")
            return False
    
    def _extract_quantum_field(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Extrai campo quântico dos dados
        
        Args:
            data: Dados do mercado
            
        Returns:
            Campo quântico
        """
        try:
            # Extrair métricas
            price = data['price']
            volume = data['volume']
            timestamp = data['timestamp']
            
            # Criar vetor base
            field = np.zeros(self.memory_dimension)
            
            # Codificar preço
            price_idx = int(price * 100) % self.memory_dimension
            field[price_idx] = 1.0
            
            # Codificar volume
            volume_idx = int(volume * 100) % self.memory_dimension
            field[volume_idx] = 0.5
            
            # Codificar tempo
            time_idx = int(timestamp) % self.memory_dimension
            field[time_idx] = 0.25
            
            # Normalizar
            field = field / np.linalg.norm(field)
            
            return field
            
        except Exception as e:
            logger.error(f"Erro na extração: {str(e)}")
            return np.zeros(self.memory_dimension)
    
    def _calculate_entropy(self, field: np.ndarray) -> float:
        """
        Calcula entropia do campo
        
        Args:
            field: Campo quântico
            
        Returns:
            Valor da entropia
        """
        try:
            # Calcular probabilidades
            probs = np.abs(field) ** 2
            probs = probs[probs > 0]
            
            # Calcular entropia
            entropy = -np.sum(probs * np.log2(probs))
            
            return float(entropy)
            
        except Exception as e:
            logger.error(f"Erro no cálculo: {str(e)}")
            return 0.0
    
    def _calculate_coherence(self, field: np.ndarray) -> float:
        """
        Calcula coerência do campo
        
        Args:
            field: Campo quântico
            
        Returns:
            Valor da coerência
        """
        try:
            # Calcular matriz densidade
            rho = np.outer(field, field.conj())
            
            # Calcular traço
            coherence = np.abs(np.trace(np.matmul(rho, rho)))
            
            return float(coherence)
            
        except Exception as e:
            logger.error(f"Erro no cálculo: {str(e)}")
            return 0.0
    
    def _generate_recommendation(self,
                               state: Dict[str, Any],
                               similar_states: List[Dict[str, Any]],
                               morphic_patterns: Dict[str, Any],
                               consciousness_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gera recomendação de trading
        
        Args:
            state: Estado atual
            similar_states: Estados similares
            morphic_patterns: Padrões mórficos
            consciousness_state: Estado da consciência
            
        Returns:
            Recomendação
        """
        try:
            # Extrair métricas
            current_price = state['price']
            current_entropy = state['entropy']
            current_coherence = state['coherence']
            
            # Analisar estados similares
            if similar_states:
                # Calcular tendência de preço
                prices = [s['state']['price'] for s in similar_states]
                price_trend = np.mean(prices) - current_price
                
                # Calcular tendência de entropia
                entropies = [s['state']['entropy'] for s in similar_states]
                entropy_trend = np.mean(entropies) - current_entropy
                
                # Calcular tendência de coerência
                coherences = [s['state']['coherence'] for s in similar_states]
                coherence_trend = np.mean(coherences) - current_coherence
                
                # Integrar padrões mórficos
                morphic_strength = morphic_patterns.get('strength', 0.5)
                morphic_direction = morphic_patterns.get('direction', 0.0)
                
                # Integrar consciência
                consciousness_confidence = consciousness_state.get('confidence', 0.5)
                consciousness_direction = consciousness_state.get('direction', 0.0)
                
                # Gerar sinal
                if (price_trend > 0 and
                    entropy_trend < 0 and
                    coherence_trend > 0 and
                    current_coherence > self.coherence_threshold and
                    morphic_direction > 0 and
                    consciousness_direction > 0):
                    signal = 'buy'
                    confidence = min(
                        current_coherence,
                        np.mean([s['similarity'] for s in similar_states]),
                        morphic_strength,
                        consciousness_confidence
                    )
                    
                elif (price_trend < 0 and
                      entropy_trend > 0 and
                      coherence_trend < 0 and
                      morphic_direction < 0 and
                      consciousness_direction < 0):
                    signal = 'sell'
                    confidence = min(
                        current_coherence,
                        np.mean([s['similarity'] for s in similar_states]),
                        morphic_strength,
                        consciousness_confidence
                    )
                    
                else:
                    signal = 'hold'
                    confidence = current_coherence
                    
            else:
                signal = 'hold'
                confidence = current_coherence
                price_trend = 0.0
                entropy_trend = 0.0
                coherence_trend = 0.0
                morphic_strength = 0.5
                consciousness_confidence = 0.5
            
            return {
                'signal': signal,
                'confidence': float(confidence),
                'metrics': {
                    'price_trend': float(price_trend) if similar_states else 0.0,
                    'entropy_trend': float(entropy_trend) if similar_states else 0.0,
                    'coherence_trend': float(coherence_trend) if similar_states else 0.0,
                    'morphic_strength': float(morphic_strength),
                    'consciousness_confidence': float(consciousness_confidence)
                }
            }
            
        except Exception as e:
            logger.error(f"Erro na recomendação: {str(e)}")
            return {
                'signal': 'hold',
                'confidence': 0.0,
                'metrics': {
                    'price_trend': 0.0,
                    'entropy_trend': 0.0,
                    'coherence_trend': 0.0,
                    'morphic_strength': 0.5,
                    'consciousness_confidence': 0.5
                }
            }
    
    def _get_default_result(self) -> Dict[str, Any]:
        """
        Retorna resultado padrão
        
        Returns:
            Resultado padrão
        """
        return {
            'state': None,
            'similar_states': [],
            'morphic_patterns': {},
            'consciousness': {},
            'metrics': {
                'entropy': 0.0,
                'coherence': 0.0,
                'total_patterns': 0
            },
            'recommendation': {
                'signal': 'hold',
                'confidence': 0.0,
                'metrics': {
                    'price_trend': 0.0,
                    'entropy_trend': 0.0,
                    'coherence_trend': 0.0,
                    'morphic_strength': 0.5,
                    'consciousness_confidence': 0.5
                }
            }
        } 