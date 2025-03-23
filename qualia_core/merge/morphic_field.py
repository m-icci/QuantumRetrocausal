"""
Campo Mórfico para Trading Quântico com operadores QUALIA aprimorados
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, List, NamedTuple, Tuple, Union
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import os

# Configura logger
logger = logging.getLogger(__name__)

# Enhanced QUALIA Operators
def apply_folding(field: np.ndarray) -> np.ndarray:
    """
    Aplica operador de dobramento F aprimorado com ressonância quântica

    Args:
        field: Campo mórfico

    Returns:
        Campo dobrado
    """
    # Aplica dobramento φ-adaptativo
    field_energy = np.sum(np.abs(field)**2)
    phi = (1 + np.sqrt(5)) / 2

    # Calcula operador de dobramento
    folded = np.zeros_like(field, dtype=np.complex128)
    for i in range(len(field)):
        phase = phi * np.angle(field[i])
        amplitude = np.abs(field[i])
        folded[i] = amplitude * np.exp(1j * phase)

    # Normaliza preservando energia
    return folded * np.sqrt(field_energy / np.sum(np.abs(folded)**2))

def apply_resonance(field: np.ndarray) -> np.ndarray:
    """
    Aplica operador de ressonância M com transformada wavelet

    Args:
        field: Campo mórfico

    Returns:
        Campo ressonante
    """
    # Ensure minimum length for operation
    min_length = 4
    if len(field) < min_length:
        field = np.pad(field, (0, min_length - len(field)))

    # Calcula transformada wavelet usando NumPy
    coeffs = np.fft.fft(field)
    phi = (1 + np.sqrt(5)) / 2

    # Aplica correção de fase quântica
    corrected = coeffs * np.exp(1j * phi * np.angle(coeffs))

    # Reconstrói campo
    resonant = np.fft.ifft(corrected)

    # Ensure output matches input length
    resonant = resonant[:len(field)]
    return resonant

def apply_emergence(field: np.ndarray) -> np.ndarray:
    """
    Aplica operador de emergência E com consciência quântica

    Args:
        field: Campo mórfico

    Returns:
        Campo emergente
    """
    # Enhanced emergence through quantum consciousness
    folded = apply_folding(field)
    resonant = apply_resonance(folded)

    # Apply quantum phase alignment
    phi = (1 + np.sqrt(5)) / 2
    phase = np.angle(resonant + 1j * np.mean(resonant))
    aligned = np.abs(resonant) * np.exp(1j * phi * phase)

    # Combine with original field for enhanced emergence
    emergent = phi * (aligned + 0.1j * field)
    return np.real(emergent)

def get_metrics(field: np.ndarray) -> Dict[str, float]:
    """
    Calcula métricas do campo quântico

    Args:
        field: Campo mórfico

    Returns:
        Dicionário com métricas calculadas
    """
    # Métricas fundamentais
    strength = np.mean(np.abs(field))
    coherence = np.abs(np.sum(field * np.conj(field))) / len(field)
    resonance = np.abs(np.fft.fft(field)[1]) / len(field)

    # Métricas derivadas
    phi = (1 + np.sqrt(5)) / 2
    emergence = phi * (coherence * resonance)
    entropy = -np.sum(np.abs(field)**2 * np.log(np.abs(field)**2 + 1e-10))

    # Métricas de consciência
    field_energy = np.sum(np.abs(field)**2)
    consciousness = coherence * emergence * np.exp(-entropy/field_energy)

    return {
        'strength': strength,
        'coherence': coherence,
        'resonance': resonance,
        'emergence': emergence,
        'entropy': entropy,
        'consciousness': consciousness,
        'field_energy': field_energy
    }

class FieldMetrics(NamedTuple):
    """Métricas do campo mórfico"""
    timestamp: datetime
    strength: float
    coherence: float
    resonance: float
    emergence: float
    entropy: float

@dataclass
class MorphicPattern:
    """Padrão mórfico detectado"""
    pattern_id: str
    timestamp: datetime
    field_strength: float
    coherence: float
    resonance: float
    emergence: float
    field: np.ndarray
    features: np.ndarray

class MorphicField:
    """Implementa campo mórfico para trading"""

    def __init__(self,
                 field_dimensions: int = 8,
                 field_strength: float = 0.8,
                 coherence_threshold: float = 0.75,
                 resonance_threshold: float = 0.7,
                 max_history: int = 1000,
                 phi_scale: float = 1.618):
        """
        Inicializa campo mórfico

        Args:
            field_dimensions: Dimensões do campo
            field_strength: Força do campo
            coherence_threshold: Limiar de coerência
            resonance_threshold: Limiar de ressonância
            max_history: Tamanho máximo do histórico
            phi_scale: Escala áurea para transformações
        """
        self.field_dimensions = field_dimensions
        self.field_strength = field_strength
        self.coherence_threshold = coherence_threshold
        self.resonance_threshold = resonance_threshold
        self.max_history = max_history
        self.phi = phi_scale  # Usa escala configurável

        # Inicializa campo
        self.field = np.zeros((2**self.field_dimensions,))
        self.history: List[Dict] = []

        # Métricas
        self.coherence = 0.0
        self.resonance = 0.0
        self.field_energy = 0.0

        # Histórico de padrões
        self.patterns: List[MorphicPattern] = []
        self.metrics_history: List[Dict] = []
        self.pattern_counter = 0

    def update(self, market_data: Dict[str, float]) -> Dict[str, float]:
        """
        Atualiza campo com novos dados

        Args:
            market_data: Dados de mercado

        Returns:
            Métricas atualizadas
        """
        # Atualiza histórico
        self.history.append(market_data)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        # Calcula métricas
        self._update_coherence()
        self._update_resonance()
        self._update_field_energy()

        # Registra métricas
        metrics = {
            'strength': self.field_energy,
            'coherence': self.coherence,
            'resonance': self.resonance,
            'field_energy': self.field_energy
        }

        # Registra métricas no histórico
        field_metrics = {
            'timestamp': datetime.now(),
            'strength': metrics['strength'],
            'coherence': metrics['coherence'],
            'resonance': metrics['resonance'],
            'emergence': metrics['coherence'] * metrics['resonance'],
            'entropy': np.std(self.field)
        }
        self.metrics_history.append(field_metrics)

        return metrics

    def _update_coherence(self):
        """Atualiza coerência do campo"""
        if len(self.history) < 2:
            return

        # Calcula correlação entre preços
        close_key = 'close' if 'close' in self.history[0] else 'closes'
        prices = np.array([h[close_key] for h in self.history])
        corr = np.corrcoef(prices[:-1], prices[1:])[0,1]

        # Atualiza coerência
        self.coherence = (self.coherence + abs(corr)) / 2

    def _update_resonance(self):
        """Atualiza ressonância do campo"""
        if len(self.history) < self.field_dimensions:
            return

        # Calcula padrões
        patterns = []
        for i in range(len(self.history) - self.field_dimensions):
            pattern = self.history[i:i+self.field_dimensions]
            patterns.append([p['close'] for p in pattern])

        # Calcula similaridade
        if patterns:
            patterns = np.array(patterns)
            similarity = np.mean([
                np.corrcoef(p1, p2)[0,1]
                for p1 in patterns
                for p2 in patterns
            ])

            # Atualiza ressonância
            self.resonance = (self.resonance + abs(similarity)) / 2

    def _update_field_energy(self):
        """Atualiza energia do campo"""
        if not self.history:
            return

        # Calcula energia como volatilidade normalizada
        close_key = 'close' if 'close' in self.history[0] else 'closes'
        prices = np.array([h[close_key] for h in self.history])
        volatility = np.std(prices)
        self.field_energy = np.tanh(volatility)  # Normaliza entre -1 e 1

    def evolve_field(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evolui campo mórfico com dados de mercado.
        
        Args:
            market_data: Dados de mercado para evolução
            
        Returns:
            Métricas do campo evoluído
        """
        # Aplica transformações ICCI
        field = self._apply_icci_transformations(market_data)
        
        # Aplica princípios PEMS
        field = apply_perception(field)     # P: Percepção quântica
        field = apply_emergence(field)      # E: Auto-organização emergente
        field = apply_morphic(field)        # M: Ressonância mórfica
        field = apply_sync(field)           # S: Sincronização quântica
        
        # Calcula métricas
        metrics = self._get_metrics(market_data)
        
        # Registra métricas
        self._track_metrics(metrics)
        
        # Atualiza estado do campo
        self.field = field
        
        return metrics

    def _apply_icci_transformations(self, market_data: Dict[str, Any]) -> np.ndarray:
        """
        Aplica transformações ICCI (Investigation, Consciousness, Coherence, Integration)
        nos dados de mercado.
        
        Args:
            market_data: Dados de mercado para transformação
            
        Returns:
            Campo transformado
        """
        # Extrai features
        features = self._extract_features(market_data)
        
        # Cria campo base
        field = self._create_field(features)
        
        # Investigation: Análise profunda dos padrões
        field = self._analyze_patterns(field)
        
        # Consciousness: Integração com estado de consciência
        field = self._integrate_consciousness(field)
        
        # Coherence: Manutenção da coerência quântica
        field = self._maintain_coherence(field)
        
        # Integration: Integração holística final
        field = self._holistic_integration(field)
        
        return field
        
    def _analyze_patterns(self, field: np.ndarray) -> np.ndarray:
        """Analisa padrões profundos no campo"""
        try:
            # Aplica transformada de Fourier
            freq_domain = np.fft.fft(field)
            
            # Filtra frequências relevantes
            mask = np.abs(freq_domain) > np.mean(np.abs(freq_domain))
            freq_domain *= mask
            
            # Retorna ao domínio do tempo
            field = np.fft.ifft(freq_domain)

            return field.real
            
        except Exception as e:
            logger.warning(f"Erro na análise de padrões: {e}")
            return field
            
    def _integrate_consciousness(self, field: np.ndarray) -> np.ndarray:
        """Integra estado de consciência ao campo"""
        try:
            if len(self.metrics_history) > 0:
                # Usa últimas métricas como peso
                last_metrics = self.metrics_history[-1]
                consciousness_weight = last_metrics.get('coherence', 0.5)
                
                # Aplica peso ao campo
                field *= consciousness_weight
                
            return field
            
        except Exception as e:
            logger.warning(f"Erro na integração de consciência: {e}")
            return field
            
    def _maintain_coherence(self, field: np.ndarray) -> np.ndarray:
        """Mantém coerência quântica do campo"""
        try:
            # Garante que o campo é 1D
            field = np.ravel(field)
            
            # Normaliza campo
            field = field / (np.linalg.norm(field) + 1e-10)
            
            # Aplica função de suavização
            window = np.hamming(len(field))
            field = field * window
            
            # Aplica transformação unitária
            U = np.exp(1j * np.pi/4)  # Rotação de 45 graus
            field = U * field
            
            # Mantém apenas parte real para estabilidade
            field = np.real(field)
            
            return field
            
        except Exception as e:
            logger.warning(f"Erro na manutenção de coerência: {e}")
            return field
            
    def _holistic_integration(self, field: np.ndarray) -> np.ndarray:
        """Realiza integração holística final"""
        try:
            # Aplica transformações QUALIA
            field = apply_folding(field)     # F: Dobramento do espaço-tempo
            field = self._apply_resonance(field)  # M: Ressonância mórfica
            field = apply_emergence(field)   # E: Auto-organização emergente
            
            # Detecta padrões
            features = self._extract_features({'field': field})
            metrics = self._get_metrics({'field': field})
            self._detect_pattern(field, features, metrics)
            
            return field
            
        except Exception as e:
            logger.warning(f"Erro na integração holística: {e}")
            return field

    def _apply_resonance(self, field: np.ndarray) -> np.ndarray:
        """
        Aplica ressonância mórfica no campo.
        
        Args:
            field: Campo quântico
            
        Returns:
            Campo com ressonância aplicada
        """
        try:
            # Garante que o campo é 1D
            field = np.ravel(field)
            
            if len(field) < 2:
                return field
                
            # Calcula máscara de ressonância
            mask = np.zeros_like(field)
            for i in range(len(field)):
                # Usa média móvel para suavização
                start = max(0, i-2)
                end = min(len(field), i+3)
                mask[i] = np.mean(field[start:end])
                
            # Aplica máscara com peso adaptativo
            alpha = 0.7  # Peso da ressonância
            field = alpha * field + (1-alpha) * mask
            
            # Normaliza resultado
            field = field / (np.linalg.norm(field) + 1e-10)
            
            return field
            
        except Exception as e:
            logger.error(f"Erro crítico na ressonância mórfica: {e}")
            return field

    def _get_metrics(self, state_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calcula métricas do campo mórfico.
        
        Args:
            state_data: Dados do estado atual
            
        Returns:
            Métricas calculadas
        """
        try:
            # Extrai valores numéricos do estado
            values = []
            for v in state_data.values():
                if isinstance(v, (int, float, np.number)):
                    values.append(float(v))
                elif isinstance(v, (list, np.ndarray)):
                    # Converte arrays para valores numéricos
                    arr = np.array(v, dtype=float)
                    values.extend(arr.flatten())
            
            # Converte para array e remove NaN
            state_array = np.array(values, dtype=float)
            state_array = state_array[~np.isnan(state_array)]
            
            if len(state_array) == 0:
                raise ValueError("Dados de estado vazios")
                
            # Normaliza valores
            state_array = state_array - np.min(state_array)
            total = np.sum(state_array)
            if total > 0:
                state_array = state_array / total
            
            # Calcula coerência como medida de ordem
            coherence = 1.0 - np.std(state_array) if len(state_array) > 1 else 1.0
            
            # Calcula ressonância como correlação temporal
            if len(self.metrics_history) > 1:
                prev_metrics = self.metrics_history[-1]
                prev_values = np.array([
                    prev_metrics.get('coherence', 0.0),
                    prev_metrics.get('resonance', 0.0),
                    prev_metrics.get('entanglement', 0.0)
                ])
                curr_values = np.array([coherence, 0.5, 0.0])  # Valores iniciais
                
                # Calcula correlação
                try:
                    resonance = np.corrcoef(prev_values, curr_values)[0,1]
                    resonance = (resonance + 1) / 2  # Normaliza para [0,1]
                except:
                    resonance = 0.5  # Valor default
            else:
                resonance = 0.5  # Valor inicial
            
            # Calcula entropia para entanglement
            if len(state_array) > 1:
                # Usa entropia normalizada
                entropy = -np.sum(state_array * np.log2(state_array + 1e-10))
                max_entropy = np.log2(len(state_array))
                entanglement = entropy / max_entropy if max_entropy > 0 else 0.0
            else:
                entanglement = 0.0
                
            # Calcula força do campo
            field_strength = np.mean(state_array) if len(state_array) > 0 else 0.0
            
            # Tamanho do campo
            field_size = len(state_array)
            
            return {
                'coherence': float(coherence),
                'resonance': float(resonance),
                'entanglement': float(entanglement),
                'field_strength': float(field_strength),
                'field_size': int(field_size)
            }
            
        except Exception as e:
            logger.warning(f"Erro ao calcular métricas: {e}")
            # Retorna valores default em caso de erro
            return {
                'coherence': 0.5,
                'resonance': 0.5,
                'entanglement': 0.0,
                'field_strength': 1.0,
                'field_size': 1
            }

    def _track_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Registra métricas do campo

        Args:
            metrics: Métricas calculadas
        """
        if len(self.metrics_history) >= self.max_history:
            self.metrics_history.pop(0)
            
        # Calcula entropia do campo
        try:
            field_values = np.array(list(metrics.values()))
            field_values = field_values[~np.isnan(field_values)]  # Remove NaN
            if len(field_values) > 0:
                # Normaliza valores
                field_values = field_values - np.min(field_values)
                total = np.sum(field_values)
                if total > 0:
                    probs = field_values / total
                entropy = -np.sum(probs * np.log2(probs + 1e-10))
            else:
                entropy = 0.0
        except Exception as e:
            logger.warning(f"Erro ao calcular entropia: {e}")
            entropy = 0.0

        # Registra métricas com timestamp
        self.metrics_history.append({
            'timestamp': datetime.now(),
            'field_strength': metrics.get('field_strength', 0.0),
            'coherence': metrics.get('coherence', 0.0),
            'resonance': metrics.get('resonance', 0.0),
            'entanglement': metrics.get('entanglement', 0.0),
            'field_size': metrics.get('field_size', 0),
            'entropy': float(entropy)
        })

    def _extract_features(self, market_data: Dict[str, Any]) -> np.ndarray:
        """
        Extrai features dos dados

        Args:
            market_data: Dados de mercado ou código

        Returns:
            Features extraídas
        """
        logger = logging.getLogger(__name__)

        # Verifica se é um valor numérico único
        if isinstance(market_data, (int, float, np.float64, np.int64)):
            return np.array([[market_data]], dtype=np.float64)

        # Verifica se é um dicionário com valores numéricos
        if isinstance(market_data, dict) and all(isinstance(v, (int, float, np.float64, np.int64)) for v in market_data.values()):
            return np.array([[v] for v in market_data.values()], dtype=np.float64)

        # Processa arquivos
        path = market_data.get('path', '')
        if not path:
            return np.zeros((4, 1))

        if os.path.isdir(path):
            features_list = []
            for file_path in Path(path).rglob('*.py'):
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()

                    # Calcula features básicas
                    lines = content.split('\n')
                    num_lines = len(lines)
                    line_lengths = [len(line) for line in lines]
                    avg_line_length = np.mean(line_lengths) if line_lengths else 0
                    num_functions = len([line for line in lines if 'def ' in line])
                    num_classes = len([line for line in lines if 'class ' in line])

                    # Normaliza features
                    file_features = np.array([
                        num_lines / 1000,  # Normaliza para ~1000 linhas
                        avg_line_length / 80,  # Normaliza para ~80 chars
                        num_functions / 10,  # Normaliza para ~10 funções
                        num_classes / 5  # Normaliza para ~5 classes
                    ])

                    features_list.append(file_features)

                except Exception as e:
                    logger.warning(f"Erro ao processar arquivo {file_path}: {e}")
                    continue

            if not features_list:
                return np.zeros((4, 1))

            # Combina features de todos os arquivos
            features = np.mean(features_list, axis=0)
            return features.reshape(-1, 1)

        else:
            # Processa arquivo individual
            try:
                with open(path, 'r') as f:
                    content = f.read()

                # Calcula features básicas
                lines = content.split('\n')
                num_lines = len(lines)
                line_lengths = [len(line) for line in lines]
                avg_line_length = np.mean(line_lengths) if line_lengths else 0
                num_functions = len([line for line in lines if 'def ' in line])
                num_classes = len([line for line in lines if 'class ' in line])

                # Normaliza features
                features = np.array([
                    num_lines / 1000,  # Normaliza para ~1000 linhas
                    avg_line_length / 80,  # Normaliza para ~80 chars
                    num_functions / 10,  # Normaliza para ~10 funções
                    num_classes / 5  # Normaliza para ~5 classes
                ])

                return features.reshape(-1, 1)

            except Exception as e:
                logger.warning(f"Erro ao processar arquivo {path}: {e}")
                return np.zeros((4, 1))

    def _create_field(self, features: np.ndarray) -> np.ndarray:
        """
        Cria campo mórfico a partir de features

        Args:
            features: Features extraídas

        Returns:
            Campo mórfico
        """
        # Evita divisão por zero na normalização
        mean = features.mean()
        std = features.std()
        if std == 0:
            std = 1.0
        
        # Normaliza features com proteção contra divisão por zero
        features_norm = (features - mean) / std

        # Cria campo base
        field = np.zeros((2**self.field_dimensions, features.shape[1]))

        # Codifica features em campo com proteção contra NaN
        for i in range(features.shape[1]):
            angles = np.clip(features_norm[:, i], -1, 1) * np.pi / 2
            for j in range(min(len(angles), self.field_dimensions)):
                sin_terms = [np.sin(angles[k]) for k in range(j)]
                prod = np.prod(sin_terms) if sin_terms else 1.0
                field[j, i] = np.cos(angles[j]) * prod

        # Remove NaN e infinitos
        field = np.nan_to_num(field, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return field

    def _calculate_eigenvalues(self, field: np.ndarray) -> np.ndarray:
        """
        Calcula autovalores do campo

        Args:
            field: Campo mórfico

        Returns:
            Autovalores do campo
        """
        # Calcula matriz densidade
        density = np.dot(field, field.T)
        eigenvalues = np.linalg.eigvalsh(density)
        return eigenvalues / np.sum(eigenvalues)  # Normaliza

    def _calculate_field_strength(self, field: np.ndarray) -> float:
        """
        Calcula força do campo mórfico

        Args:
            field: Campo mórfico

        Returns:
            Força do campo
        """
        # Calcula entropia de von Neumann
        eigenvalues = self._calculate_eigenvalues(field)
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-10))

        # Normaliza usando phi
        strength = np.exp(-entropy / self.phi)
        return float(strength)

    def _calculate_coherence(self, field: np.ndarray) -> float:
        """
        Calcula coerência do campo

        Args:
            field: Campo mórfico

        Returns:
            Coerência do campo
        """
        # Calcula correlação entre componentes
        corr = np.corrcoef(field)

        # Calcula média das correlações fora da diagonal
        coherence = np.mean(np.abs(corr - np.eye(corr.shape[0])))
        return float(coherence)

    def _calculate_resonance(self, field: np.ndarray) -> float:
        """
        Calcula ressonância do campo

        Args:
            field: Campo mórfico

        Returns:
            Ressonância do campo
        """
        # Calcula razões entre componentes consecutivas
        ratios = field[:, 1:] / field[:, :-1]

        # Calcula distância para phi
        phi_distance = np.abs(ratios - self.phi)

        # Calcula ressonância
        resonance = np.exp(-np.mean(phi_distance))
        return float(resonance)

    def _calculate_emergence(self, field: np.ndarray) -> float:
        """
        Calcula fator de emergência

        Args:
            field: Campo mórfico

        Returns:
            Fator de emergência
        """
        # Aplica transformação phi-escalonada
        coherence = self._calculate_coherence(field)
        resonance = self._calculate_resonance(field)

        # Usa phi para ponderar emergência
        emergence = self.phi * (coherence * resonance)

        # Normaliza resultado
        emergence = np.clip(emergence, 0, 1)

        return emergence

    def _detect_pattern(self,
                       field: np.ndarray,
                       features: np.ndarray,
                       metrics: Dict[str, float]) -> None:
        """
        Detecta padrão mórfico

        Args:
            field: Campo mórfico
            features: Features extraídas
            metrics: Métricas calculadas
        """
        # Verifica limiares
        if (metrics['coherence'] >= self.coherence_threshold and
            metrics['resonance'] >= self.resonance_threshold):

            # Gera ID único
            self.pattern_counter += 1
            pattern_id = f"morphic_{self.pattern_counter:04d}"

            pattern = MorphicPattern(
                pattern_id=pattern_id,
                timestamp=datetime.now(),
                field_strength=metrics['field_strength'],
                coherence=metrics['coherence'],
                resonance=metrics['resonance'],
                emergence=metrics['entanglement'],
                field=field.copy(),
                features=features.copy()
            )

            self.patterns.append(pattern)

            # Mantém tamanho máximo
            if len(self.patterns) > self.max_history:
                self.patterns.pop(0)

    def get_field_metrics(self,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> List[Dict]:
        """
        Retorna métricas do campo filtradas por tempo

        Args:
            start_time: Tempo inicial opcional
            end_time: Tempo final opcional

        Returns:
            Lista de métricas filtrada
        """
        filtered = self.metrics_history

        if start_time:
            filtered = [m for m in filtered if m['timestamp'] >= start_time]

        if end_time:
            filtered = [m for m in filtered if m['timestamp'] <= end_time]

        return filtered

    def get_patterns(self,
                    min_strength: Optional[float] = None,
                    min_coherence: Optional[float] = None) -> List[MorphicPattern]:
        """
        Retorna padrões mórficos filtrados

        Args:
            min_strength: Força mínima opcional
            min_coherence: Coerência mínima opcional

        Returns:
            Lista de padrões filtrada
        """
        filtered = self.patterns

        if min_strength:
            filtered = [p for p in filtered if p.field_strength >= min_strength]

        if min_coherence:
            filtered = [p for p in filtered if p.coherence >= min_coherence]

        return filtered

    def clear_history(self) -> None:
        """Limpa histórico de métricas e padrões"""
        self.metrics_history.clear()
        self.patterns.clear()
        self.pattern_counter = 0

    def analyze_field_evolution(self) -> Dict[str, float]:
        """
        Analisa evolução do campo mórfico

        Returns:
            Métricas de evolução
        """
        if not self.metrics_history:
            return {}

        # Extrai séries temporais
        strength = [m['field_strength'] for m in self.metrics_history]
        coherence = [m['coherence'] for m in self.metrics_history]
        resonance = [m['resonance'] for m in self.metrics_history]
        emergence = [m['entanglement'] for m in self.metrics_history]

        # Calcula métricas de evolução
        metrics = {
            'mean_strength': np.mean(strength),
            'mean_coherence': np.mean(coherence),
            'mean_resonance': np.mean(resonance),
            'mean_emergence': np.mean(emergence),
            'strength_trend': np.polyfit(np.arange(len(strength)), strength, 1)[0],
            'coherence_trend': np.polyfit(np.arange(len(coherence)), coherence, 1)[0],
            'resonance_trend': np.polyfit(np.arange(len(resonance)), resonance, 1)[0],
            'emergence_trend': np.polyfit(np.arange(len(emergence)), emergence, 1)[0]
        }

        return metrics

def apply_perception(field: np.ndarray) -> np.ndarray:
    """
    Aplica percepção quântica ao campo.
    Usa transformada wavelet para detectar padrões em múltiplas escalas.
    """
    try:
        # Normaliza campo
        field = field / (np.linalg.norm(field) + 1e-10)
        
        # Aplica transformada wavelet
        coeffs = np.abs(np.fft.fft(field))
        threshold = np.mean(coeffs)
        coeffs[coeffs < threshold] = 0
        
        # Reconstrói sinal
        field = np.fft.ifft(coeffs).real
        
        return field
        
    except Exception as e:
        logger.warning(f"Erro na percepção quântica: {e}")
        return field

def apply_morphic(field: np.ndarray) -> np.ndarray:
    """
    Aplica ressonância mórfica ao campo.
    Usa correlação espacial para identificar padrões ressonantes.
    """
    try:
        # Calcula autocorrelação
        corr = np.correlate(field, field, mode='full')
        mid = len(corr) // 2
        
        # Extrai padrão ressonante
        pattern = corr[mid:mid + len(field)]
        pattern = pattern / (np.linalg.norm(pattern) + 1e-10)
        
        # Aplica padrão ao campo
        field = field * pattern
        
        return field
        
    except Exception as e:
        logger.warning(f"Erro na ressonância mórfica: {e}")
        return field

def apply_sync(field: np.ndarray) -> np.ndarray:
    """
    Aplica sincronização quântica ao campo.
    Usa fase instantânea para alinhar componentes do campo.
    """
    try:
        # Calcula fase instantânea
        analytic = np.abs(field) * np.exp(1j * np.angle(field))
        phase = np.unwrap(np.angle(analytic))
        
        # Sincroniza usando fase
        field = np.abs(field) * np.exp(1j * phase)
        
        return field.real
        
    except Exception as e:
        logger.warning(f"Erro na sincronização quântica: {e}")
        return field

def apply_emergence(field: np.ndarray) -> np.ndarray:
    """
    Aplica auto-organização emergente ao campo.
    Usa dinâmica não-linear para permitir emergência de padrões.
    """
    try:
        # Calcula gradiente do campo
        grad = np.gradient(field)
        
        # Aplica função não-linear
        field = np.tanh(field + 0.1 * grad[0])
        
        # Normaliza resultado
        field = field / (np.linalg.norm(field) + 1e-10)
        
        return field
        
    except Exception as e:
        logger.warning(f"Erro na auto-organização: {e}")
        return field

class MorphicFieldCalculator:
    """
    Calculadora de campo mórfico para análise quântica
    
    Responsável por calcular métricas de campo mórfico entre pares de arquivos,
    permitindo identificar potenciais de merge e ressonância quântica.
    """
    
    def __init__(self, 
                 field_dimensions: int = 12,
                 retrocausal_factor: float = 0.7,
                 phi_scale: float = 1.618,
                 oscillation_sensitivity: float = 0.65):
        """
        Inicializa a calculadora de campo mórfico
        
        Args:
            field_dimensions: Dimensões do campo quântico (potência de 2 recomendada)
            retrocausal_factor: Fator de influência retrocausal nas análises
            phi_scale: Escala da razão áurea para transformações (padrão: razão áurea)
            oscillation_sensitivity: Sensibilidade para detectar padrões oscilatórios
        """
        self.field_dimensions = field_dimensions
        self.retrocausal_factor = retrocausal_factor
        self.phi = phi_scale
        self.oscillation_sensitivity = oscillation_sensitivity
        
        # Inicializa campo de referência
        self.reference_field = np.zeros((2**field_dimensions,))
        
        # Métricas acumuladas
        self.cumulative_coherence = 0.0
        self.cumulative_entropy = 0.0
        
        logger.info(f"MorphicFieldCalculator inicializado com dimensões: {2**field_dimensions}")
    
    def calculate_metrics_from_files(self, file1: Path, file2: Path) -> Dict[str, float]:
        """
        Calcula métricas quânticas entre dois arquivos
        
        Args:
            file1: Caminho para o primeiro arquivo
            file2: Caminho para o segundo arquivo
            
        Returns:
            Dicionário com métricas calculadas para o par de arquivos
        """
        logger.info(f"Calculando métricas quânticas entre {file1.name} e {file2.name}")
        
        try:
            # Converte conteúdo dos arquivos para campos quânticos
            field1 = self._file_to_quantum_field(file1)
            field2 = self._file_to_quantum_field(file2)
            
            # Calcula métricas de interferência quântica
            metrics = self._calculate_interference_metrics(field1, field2)
            
            # Adiciona métricas retrocausais
            metrics.update(self._calculate_retrocausal_metrics(field1, field2))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erro ao calcular métricas: {str(e)}")
            # Retorna métricas com valores padrão em caso de erro
            return {
                'coherence_score': 0.0,
                'entropy_reduction': 0.0,
                'resonance_factor': 0.0,
                'field_alignment': 0.0,
                'future_potential': 0.0
            }
    
    def _file_to_quantum_field(self, file_path: Path) -> np.ndarray:
        """
        Converte arquivo para campo quântico
        
        Args:
            file_path: Caminho para o arquivo
            
        Returns:
            Array NumPy representando o campo quântico
        """
        try:
            # Lê conteúdo do arquivo
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extrai características
            chars = [ord(c) for c in content]
            lines = content.count('\n') + 1
            tokens = len(content.split())
            
            # Normaliza para gerar campo quântico
            field_size = 2**self.field_dimensions
            
            # Calcula características espectrais usando FFT
            if len(chars) > 0:
                # Padding para garantir comprimento adequado
                chars_padded = chars + [0] * (field_size - len(chars) % field_size)
                chunks = [chars_padded[i:i+field_size] for i in range(0, len(chars_padded), field_size)]
                chunks = chunks[:min(len(chunks), 10)]  # Limita número de chunks
                
                # Gera campo quântico usando transformada de Fourier
                field = np.zeros(field_size, dtype=np.complex128)
                for chunk in chunks:
                    chunk_array = np.array(chunk[:field_size])
                    chunk_fft = np.fft.fft(chunk_array)
                    field += chunk_fft
                
                # Normaliza campo
                if np.sum(np.abs(field)**2) > 0:
                    field = field / np.sqrt(np.sum(np.abs(field)**2))
                
                return field
            else:
                # Arquivo vazio ou inválido
                return np.zeros(field_size, dtype=np.complex128)
                
        except Exception as e:
            logger.error(f"Erro ao converter arquivo para campo quântico: {str(e)}")
            return np.zeros(2**self.field_dimensions, dtype=np.complex128)
    
    def _calculate_interference_metrics(self, field1: np.ndarray, field2: np.ndarray) -> Dict[str, float]:
        """
        Calcula métricas de interferência quântica entre dois campos
        
        Args:
            field1: Primeiro campo quântico
            field2: Segundo campo quântico
            
        Returns:
            Dicionário com métricas de interferência
        """
        # Calcula produto interno (overlap quântico)
        overlap = np.abs(np.vdot(field1, field2))
        
        # Calcula campo de interferência
        interference = field1 + field2
        interference = interference / np.sqrt(np.sum(np.abs(interference)**2) + 1e-10)
        
        # Calcula métricas do campo de interferência
        metrics1 = get_metrics(field1)
        metrics2 = get_metrics(field2)
        metrics_combined = get_metrics(interference)
        
        # Calcula redução de entropia
        entropy_reduction = (
            (metrics1['entropy'] + metrics2['entropy']) / 2 - 
            metrics_combined['entropy']
        ) / ((metrics1['entropy'] + metrics2['entropy']) / 2 + 1e-10)
        
        # Calcula fator de ressonância
        resonance_factor = metrics_combined['resonance'] / ((metrics1['resonance'] + metrics2['resonance']) / 2 + 1e-10)
        
        # Calcula alinhamento de campo
        field_alignment = metrics_combined['coherence'] / ((metrics1['coherence'] + metrics2['coherence']) / 2 + 1e-10)
        
        return {
            'coherence_score': overlap,
            'entropy_reduction': max(0, entropy_reduction),  # Positivo indica redução de entropia
            'resonance_factor': resonance_factor,
            'field_alignment': field_alignment
        }
    
    def _calculate_retrocausal_metrics(self, field1: np.ndarray, field2: np.ndarray) -> Dict[str, float]:
        """
        Calcula métricas retrocausais projetando potencial futuro da integração
        
        Args:
            field1: Primeiro campo quântico
            field2: Segundo campo quântico
            
        Returns:
            Dicionário com métricas retrocausais
        """
        # Aplica operadores quânticos para simular evolução temporal
        evolved1 = apply_folding(apply_resonance(field1))
        evolved2 = apply_folding(apply_resonance(field2))
        
        # Calcula campos de futuro emergente
        future1 = apply_emergence(evolved1)
        future2 = apply_emergence(evolved2)
        
        # Calcular alinhamento de futuro emergente
        future_overlap = np.abs(np.vdot(future1, future2))
        
        # Potencial futuro baseado em retrocausalidade quântica
        future_potential = self.retrocausal_factor * future_overlap + (1 - self.retrocausal_factor) * np.random.random()
        
        return {
            'future_potential': future_potential
        }
    
    def calculate_quantum_entropy(self, field: np.ndarray) -> float:
        """
        Calcula entropia quântica de um campo mórfico
        
        Args:
            field: Campo quântico
            
        Returns:
            Valor de entropia quântica
        """
        # Normaliza o campo
        if np.sum(np.abs(field)**2) > 0:
            field = field / np.sqrt(np.sum(np.abs(field)**2))
        
        # Calcula matriz densidade
        density_matrix = np.outer(field, np.conj(field))
        
        # Calcula autovalores da matriz densidade
        eigenvalues = np.linalg.eigvalsh(density_matrix)
        
        # Remove valores negativos numericamente insignificantes
        eigenvalues = np.maximum(eigenvalues, 0)
        
        # Normaliza autovalores
        eigenvalues = eigenvalues / np.sum(eigenvalues)
        
        # Calcula entropia de von Neumann
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-10))
        
        return float(entropy)
    
    def evolve_field(self, state_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evolui um campo mórfico com base em dados de estado
        
        Args:
            state_data: Dados do estado atual
            
        Returns:
            Dados de estado evoluídos
        """
        # Extrai métricas principais
        coherence = state_data.get('coherence', 0.8)
        resonance = state_data.get('resonance', 0.7)
        entanglement = state_data.get('entanglement', 0.5)
        complexity = state_data.get('complexity', 0.6)
        
        # Constrói campo inicial a partir das métricas
        field_size = 2**self.field_dimensions
        field = np.zeros(field_size, dtype=np.complex128)
        
        # Aplica complexidade como frequência base
        for i in range(field_size):
            phase = 2 * np.pi * complexity * i / field_size
            amp = np.exp(-i / (field_size * coherence))
            field[i] = amp * np.exp(1j * phase)
        
        # Normaliza campo
        field = field / np.sqrt(np.sum(np.abs(field)**2) + 1e-10)
        
        # Aplica ressonância
        field = apply_resonance(field) * resonance + field * (1 - resonance)
        
        # Aplica entanglement (emaranhamento quântico)
        phi = self.phi
        entangled = np.zeros_like(field)
        for i in range(field_size):
            entangled[i] = field[(i + int(field_size/phi)) % field_size]
        
        # Combina campo original e emaranhado
        field = entanglement * entanglement + (1 - entanglement) * field
        
        # Normaliza resultado final
        field = field / np.sqrt(np.sum(np.abs(field)**2) + 1e-10)
        
        # Calcula métricas do campo evoluído
        metrics = get_metrics(field)
        
        # Adiciona métricas adicionais
        metrics['field_size'] = field_size
        metrics['entanglement'] = entanglement
        
        return metrics
    
    def calculate_pair_resonance(self, 
                               file1_metrics: Dict[str, float], 
                               file2_metrics: Dict[str, float]) -> float:
        """
        Calcula ressonância entre dois arquivos baseada em suas métricas individuais
        
        Args:
            file1_metrics: Métricas do primeiro arquivo
            file2_metrics: Métricas do segundo arquivo
            
        Returns:
            Pontuação de ressonância entre os arquivos
        """
        # Extrai métricas chave
        coherence1 = file1_metrics.get('coherence', 0.0)
        coherence2 = file2_metrics.get('coherence', 0.0)
        
        resonance1 = file1_metrics.get('resonance', 0.0)
        resonance2 = file2_metrics.get('resonance', 0.0)
        
        entropy1 = file1_metrics.get('entropy', 0.0)
        entropy2 = file2_metrics.get('entropy', 0.0)
        
        # Calcula diferença de entropia como potencial de sinergia
        entropy_diff = abs(entropy1 - entropy2) / (max(entropy1, entropy2) + 1e-10)
        
        # Calcula fator de ressonância
        resonance_factor = (resonance1 * resonance2) / ((resonance1 + resonance2) / 2 + 1e-10)
        
        # Calcula alinhamento de coerência
        coherence_alignment = 1 - abs(coherence1 - coherence2) / (max(coherence1, coherence2) + 1e-10)
        
        # Combina métricas usando razão áurea
        phi = self.phi
        resonance_score = (
            coherence_alignment / phi +
            resonance_factor +
            (1 - entropy_diff) * phi
        ) / (1 + 1 + phi)
        
        return resonance_score

@dataclass
class MorphicFieldState:
    """Estado de um campo mórfico"""
    field_dimensions: int = 8
    coherence_threshold: float = 0.75
    resonance_threshold: float = 0.7
    max_history: int = 1000
    
    def evolve_field(self, state_data: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper para compatibilidade"""
        calculator = MorphicFieldCalculator(
            field_dimensions=self.field_dimensions,
            phi_scale=1.618
        )
        return calculator.evolve_field(state_data)