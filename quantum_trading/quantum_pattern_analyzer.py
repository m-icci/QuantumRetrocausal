"""
Sistema Integrado de Análise de Padrões Quânticos
Integra Memória Holográfica, Campo Mórfico e Proteção Quântica
"""

import numpy as np
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from .holographic_memory import HolographicMemory, HolographicPattern
from .morphic import MorphicField
from .consciousness import MarketConsciousness
from collections import deque

logger = logging.getLogger("quantum_pattern_analyzer")

class QuantumPatternAnalyzer:
    """
    Analisador de Padrões Quânticos que integra memória holográfica e campo mórfico
    """
    def __init__(
            self,
            max_memory_capacity: int = 1000,
            resonance_threshold: float = 0.7,
            protection_threshold: float = 0.8
        ):
        # Inicializar componentes
        self.memory = HolographicMemory(
            max_capacity=max_memory_capacity,
            resonance_threshold=resonance_threshold
        )
        self.morphic_field = MorphicField()
        self.consciousness = MarketConsciousness()
        self.protection_threshold = protection_threshold
        self.pattern_cache = {}
        self.pattern_buffer = deque(maxlen=max_memory_capacity)
        
        logger.info("Inicializando QuantumPatternAnalyzer com Consciência de Mercado")
        
        # Estado atual
        self.current_state = {
            'entropy': 0.0,
            'coherence': 1.0,
            'field': np.zeros(max_memory_capacity),
            'last_update': None
        }
        
        logger.info(f"Analisador inicializado com capacidade {max_memory_capacity}")
        
    def analyze_market_pattern(
            self,
            symbol: str,
            data: pd.DataFrame,
            window_size: int = 24
        ) -> Dict[str, Any]:
        """
        Analisa padrões de mercado usando memória holográfica e campo mórfico
        
        Args:
            symbol: Símbolo do par de trading
            data: DataFrame com dados OHLCV
            window_size: Tamanho da janela de análise
        """
        try:
            # Preparar dados para análise
            pattern_data = self._prepare_pattern_data(data, window_size)
            
            # Detectar padrões usando campo mórfico
            morphic_patterns = self.morphic_field.detect_patterns()
            
            # Buscar padrões similares na memória
            similar_patterns = self.memory.find_similar_patterns(
                pattern=pattern_data,
                threshold=self.memory.resonance_threshold
            )
            
            # Calcular métricas quânticas
            quantum_metrics = self._calculate_quantum_metrics(
                pattern_data,
                morphic_patterns,
                similar_patterns
            )
            
            # Armazenar novo padrão com metadados
            pattern_id = self.memory.store_pattern(
                pattern=pattern_data,
                metadata={
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat(),
                    'metrics': quantum_metrics,
                    'morphic_patterns': morphic_patterns
                }
            )
            
            # Calcular proteção quântica
            protection = self._calculate_quantum_protection(
                symbol,
                quantum_metrics,
                similar_patterns
            )
            
            return {
                'pattern_id': pattern_id,
                'quantum_metrics': quantum_metrics,
                'protection': protection,
                'similar_patterns': similar_patterns,
                'morphic_analysis': morphic_patterns,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erro na análise de padrões: {e}")
            return {
                'pattern_id': None,
                'quantum_metrics': {},
                'protection': {},
                'similar_patterns': [],
                'morphic_analysis': {},
                'timestamp': datetime.now().isoformat()
            }
            
    def _prepare_pattern_data(
            self,
            data: pd.DataFrame,
            window_size: int
        ) -> np.ndarray:
        """
        Prepara dados para análise de padrões
        """
        try:
            # Extrair features relevantes
            features = []
            
            # Preços normalizados
            close_prices = data['close'].values[-window_size:]
            normalized_prices = (close_prices - np.mean(close_prices)) / np.std(close_prices)
            features.append(normalized_prices)
            
            # Volumes normalizados
            volumes = data['volume'].values[-window_size:]
            normalized_volumes = (volumes - np.mean(volumes)) / np.std(volumes)
            features.append(normalized_volumes)
            
            # Volatilidade
            returns = np.diff(np.log(close_prices))
            volatility = np.std(returns) * np.sqrt(252)
            features.append(np.full(window_size, volatility))
            
            # Combinar features
            pattern_data = np.vstack(features)
            
            return pattern_data
            
        except Exception as e:
            logger.error(f"Erro ao preparar dados: {e}")
            return np.array([])
            
    def _integrate_consciousness(self, data: pd.DataFrame) -> Dict[str, float]:
        """Integra análise de consciência de mercado"""
        try:
            consciousness_field = self.consciousness.calculate_consciousness_field(
                data['close'].values
            )
            
            return {
                'entropy': consciousness_field['quantum_entropy'],
                'coherence': consciousness_field['coherence'],
                'entanglement': consciousness_field['entanglement'],
                'perception': consciousness_field['perception_field']
            }
        except Exception as e:
            logger.error(f"Erro na integração da consciência: {e}")
            return {
                'entropy': 0.5,
                'coherence': 0.5,
                'entanglement': 0.5,
                'perception': 0.5
            }

    def _calculate_quantum_metrics(
            self,
            pattern_data: np.ndarray,
            morphic_patterns: Dict[str, float],
            similar_patterns: List[Dict[str, Any]]
        ) -> Dict[str, float]:
        """
        Calcula métricas quânticas baseadas nos padrões
        """
        try:
            # Métricas base do campo mórfico
            metrics = {
                'trend_strength': morphic_patterns.get('trend_strength', 0),
                'coherence': morphic_patterns.get('pattern_coherence', 0),
                'resonance': morphic_patterns.get('resonance', 0)
            }
            
            # Integrar consciência de mercado
            consciousness_metrics = self._integrate_consciousness(pattern_data)
            metrics.update(consciousness_metrics)
            
            # Calcular força do campo quântico ajustada
            field_strength = np.mean([
                metrics['trend_strength'],
                metrics['coherence'],
                metrics['resonance'],
                consciousness_metrics['entropy'],
                consciousness_metrics['perception']
            ])
            
            # Calcular estabilidade baseada em padrões similares
            if similar_patterns:
                similarities = [p['similarity'] for p in similar_patterns]
                stability = np.mean(similarities) * consciousness_metrics['coherence']
            else:
                stability = 0.5 * consciousness_metrics['coherence']
            
            # Métricas finais ajustadas
            metrics.update({
                'field_strength': field_strength,
                'stability': stability,
                'quantum_confidence': field_strength * stability * consciousness_metrics['entanglement']
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erro ao calcular métricas quânticas: {e}")
            return {
                'trend_strength': 0,
                'coherence': 0,
                'resonance': 0,
                'field_strength': 0,
                'stability': 0,
                'quantum_confidence': 0
            }
            
    def _calculate_quantum_protection(
            self,
            symbol: str,
            quantum_metrics: Dict[str, float],
            similar_patterns: List[Dict[str, Any]]
        ) -> Dict[str, Any]:
        """
        Calcula níveis de proteção baseados em análise quântica
        """
        try:
            # Calcular nível de risco base
            base_risk = 1 - quantum_metrics['quantum_confidence']
            
            # Ajustar risco baseado em padrões similares
            if similar_patterns:
                pattern_risks = []
                for pattern in similar_patterns:
                    metadata = pattern.get('metadata', {})
                    metrics = metadata.get('metrics', {})
                    if metrics:
                        pattern_risk = 1 - metrics.get('quantum_confidence', 0.5)
                        pattern_risks.append(pattern_risk)
                
                if pattern_risks:
                    historical_risk = np.mean(pattern_risks)
                    risk_level = (base_risk + historical_risk) / 2
                else:
                    risk_level = base_risk
            else:
                risk_level = base_risk
            
            # Calcular níveis de proteção
            protection_level = 1 - risk_level
            
            # Definir multiplicadores de proteção
            if protection_level > self.protection_threshold:
                stop_multiplier = 0.8  # Mais próximo
                take_multiplier = 1.2  # Mais distante
            else:
                stop_multiplier = 1.2  # Mais distante
                take_multiplier = 0.8  # Mais próximo
            
            return {
                'risk_level': risk_level,
                'protection_level': protection_level,
                'stop_multiplier': stop_multiplier,
                'take_multiplier': take_multiplier,
                'confidence': quantum_metrics['quantum_confidence'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erro ao calcular proteção quântica: {e}")
            return {
                'risk_level': 0.5,
                'protection_level': 0.5,
                'stop_multiplier': 1.0,
                'take_multiplier': 1.0,
                'confidence': 0.5,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_trading_signals(
            self,
            symbol: str,
            data: pd.DataFrame
        ) -> Dict[str, Any]:
        """
        Gera sinais de trading baseados na análise quântica
        """
        try:
            # Analisar padrões atuais
            analysis = self.analyze_market_pattern(symbol, data)
            
            # Extrair métricas relevantes
            metrics = analysis['quantum_metrics']
            protection = analysis['protection']
            
            # Definir thresholds
            CONFIDENCE_THRESHOLD = 0.7
            STRENGTH_THRESHOLD = 0.6
            
            # Calcular sinais
            confidence = metrics['quantum_confidence']
            trend_strength = metrics['trend_strength']
            field_strength = metrics['field_strength']
            
            # Gerar sinal
            signal = None
            if confidence > CONFIDENCE_THRESHOLD:
                if trend_strength > STRENGTH_THRESHOLD:
                    signal = 'buy' if field_strength > 0 else 'sell'
                elif trend_strength < -STRENGTH_THRESHOLD:
                    signal = 'sell' if field_strength < 0 else 'buy'
            
            return {
                'signal': signal,
                'confidence': confidence,
                'metrics': metrics,
                'protection': protection,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erro ao gerar sinais de trading: {e}")
            return {
                'signal': None,
                'confidence': 0,
                'metrics': {},
                'protection': {},
                'timestamp': datetime.now().isoformat()
            }
    
    def optimize_position_sizing(
            self,
            symbol: str,
            available_balance: float,
            current_price: float,
            analysis: Dict[str, Any]
        ) -> Dict[str, Any]:
        """
        Otimiza o tamanho da posição baseado na análise quântica
        """
        try:
            metrics = analysis['quantum_metrics']
            protection = analysis['protection']
            
            # Calcular confiança geral
            confidence = metrics['quantum_confidence']
            
            # Base size (1-5% do balanço disponível)
            base_size = available_balance * 0.01
            
            # Ajustar tamanho baseado na confiança
            if confidence > 0.8:
                size_multiplier = 5.0  # Máximo 5%
            elif confidence > 0.7:
                size_multiplier = 3.0  # 3%
            elif confidence > 0.6:
                size_multiplier = 2.0  # 2%
            else:
                size_multiplier = 1.0  # Mínimo 1%
            
            # Ajustar baseado no nível de proteção
            protection_level = protection['protection_level']
            size_multiplier *= protection_level
            
            # Calcular tamanho final
            position_size = base_size * size_multiplier
            
            # Calcular quantidade
            quantity = position_size / current_price
            
            return {
                'position_size': position_size,
                'quantity': quantity,
                'size_multiplier': size_multiplier,
                'confidence': confidence,
                'protection_level': protection_level,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erro ao otimizar tamanho da posição: {e}")
            return {
                'position_size': 0,
                'quantity': 0,
                'size_multiplier': 1.0,
                'confidence': 0,
                'protection_level': 0,
                'timestamp': datetime.now().isoformat()
            }

    def analyze_pattern(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisa padrões nos dados de mercado
        
        Args:
            data: Dados do mercado
            
        Returns:
            Análise do padrão
        """
        try:
            # Validar dados
            if not self._validate_data(data):
                return {
                    'error': 'Dados inválidos',
                    'confidence': 0.0,
                    'pattern': None
                }
            
            # Extrair preço
            price = float(data['price'])
            
            # Atualizar buffer
            self.pattern_buffer.append(price)
            
            # Calcular campo quântico
            field = self._calculate_quantum_field(
                np.array(list(self.pattern_buffer))
            )
            
            # Calcular métricas
            entropy = self._calculate_entropy(field)
            coherence = self._calculate_coherence(field)
            
            # Detectar padrões
            patterns = self._detect_patterns(field)
            
            # Calcular confiança
            confidence = self._calculate_confidence(
                entropy,
                coherence,
                patterns['strength']
            )
            
            # Atualizar estado
            self.current_state.update({
                'entropy': entropy,
                'coherence': coherence,
                'field': field,
                'last_update': datetime.now()
            })
            
            # Armazenar na memória
            self.memory.store_state({
                'price': price,
                'field': field.tolist(),
                'entropy': entropy,
                'coherence': coherence,
                'patterns': patterns,
                'confidence': confidence,
                'timestamp': datetime.now().timestamp()
            })
            
            return {
                'confidence': confidence,
                'entropy': entropy,
                'coherence': coherence,
                'field': field.tolist(),
                'patterns': patterns,
                'trend': patterns['trend']
            }
            
        except Exception as e:
            logger.error(f"Erro na análise de padrões: {str(e)}")
            return {
                'error': str(e),
                'confidence': 0.0,
                'pattern': None
            }
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Retorna estado atual do analisador
        
        Returns:
            Estado atual
        """
        return {
            'entropy': float(self.current_state['entropy']),
            'coherence': float(self.current_state['coherence']),
            'last_update': self.current_state['last_update']
        }
    
    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """
        Valida dados de entrada
        
        Args:
            data: Dados a validar
            
        Returns:
            Se os dados são válidos
        """
        try:
            # Verificar preço
            if 'price' not in data:
                logger.warning("Preço não encontrado nos dados")
                return False
            
            # Validar tipo
            if not isinstance(data['price'], (int, float)):
                logger.warning(f"Preço inválido: {data['price']}")
                return False
            
            # Validar valor
            if float(data['price']) <= 0:
                logger.warning(f"Preço não positivo: {data['price']}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Erro na validação: {str(e)}")
            return False
    
    def _calculate_quantum_field(self, data: np.ndarray) -> np.ndarray:
        """
        Calcula campo quântico dos dados
        
        Args:
            data: Array de dados
            
        Returns:
            Campo quântico
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
            
            # Criar campo quântico
            field = amplitudes * np.exp(1j * phases)
            
            # Normalizar campo
            field = field / np.linalg.norm(field)
            
            return field
            
        except Exception as e:
            logger.error(f"Erro no cálculo do campo: {str(e)}")
            return np.zeros(len(data))
    
    def _calculate_entropy(self, field: np.ndarray) -> float:
        """
        Calcula entropia do campo quântico
        
        Args:
            field: Campo quântico
            
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
    
    def _calculate_coherence(self, field: np.ndarray) -> float:
        """
        Calcula coerência do campo quântico
        
        Args:
            field: Campo quântico
            
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
    
    def _detect_patterns(self, field: np.ndarray) -> Dict[str, Any]:
        """
        Detecta padrões no campo quântico
        
        Args:
            field: Campo quântico
            
        Returns:
            Padrões detectados
        """
        try:
            # Calcular espectro de potência
            power = np.abs(field) ** 2
            
            # Detectar picos
            peaks = self._find_peaks(power)
            
            # Calcular tendência
            trend = self._calculate_trend(field)
            
            # Calcular força do padrão
            strength = self._calculate_pattern_strength(peaks)
            
            return {
                'peaks': peaks.tolist(),
                'trend': float(trend),
                'strength': float(strength)
            }
            
        except Exception as e:
            logger.error(f"Erro na detecção de padrões: {str(e)}")
            return {
                'peaks': [],
                'trend': 0.0,
                'strength': 0.0
            }
    
    def _find_peaks(self, power: np.ndarray) -> np.ndarray:
        """
        Encontra picos no espectro de potência
        
        Args:
            power: Espectro de potência
            
        Returns:
            Array com picos
        """
        try:
            # Calcular diferenças
            diff = np.diff(power)
            
            # Encontrar pontos de inflexão
            peaks = np.where(
                (diff[:-1] > 0) & (diff[1:] < 0)
            )[0] + 1
            
            # Filtrar picos significativos
            threshold = np.mean(power) + np.std(power)
            peaks = peaks[power[peaks] > threshold]
            
            return peaks
            
        except Exception as e:
            logger.error(f"Erro na busca de picos: {str(e)}")
            return np.array([])
    
    def _calculate_trend(self, field: np.ndarray) -> float:
        """
        Calcula tendência do campo quântico
        
        Args:
            field: Campo quântico
            
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
    
    def _calculate_pattern_strength(self, peaks: np.ndarray) -> float:
        """
        Calcula força do padrão
        
        Args:
            peaks: Array com picos
            
        Returns:
            Força do padrão
        """
        try:
            if len(peaks) == 0:
                return 0.0
            
            # Calcular distâncias entre picos
            distances = np.diff(peaks)
            
            # Calcular regularidade
            regularity = 1.0 - np.std(distances) / (np.mean(distances) + 1e-8)
            
            # Calcular densidade
            density = len(peaks) / self.max_memory_capacity
            
            # Combinar métricas
            strength = (regularity + density) / 2
            
            return float(strength)
            
        except Exception as e:
            logger.error(f"Erro no cálculo da força do padrão: {str(e)}")
            return 0.0
    
    def _calculate_confidence(self, entropy: float, coherence: float,
                            pattern_strength: float) -> float:
        """
        Calcula confiança da análise
        
        Args:
            entropy: Entropia do campo
            coherence: Coerência do campo
            pattern_strength: Força do padrão
            
        Returns:
            Valor da confiança
        """
        try:
            # Combinar métricas
            confidence = np.mean([
                1.0 - entropy,  # Baixa entropia = alta confiança
                coherence,      # Alta coerência = alta confiança
                pattern_strength  # Padrão forte = alta confiança
            ])
            
            return float(confidence)
            
        except Exception as e:
            logger.error(f"Erro no cálculo da confiança: {str(e)}")
            return 0.0 