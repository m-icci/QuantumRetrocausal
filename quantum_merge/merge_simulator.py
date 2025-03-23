# Configuração inicial para limitar threads OpenBLAS
import os
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Protocol, Tuple, TypeVar, Union
from numpy.typing import NDArray
from datetime import datetime
import pandas as pd

# Definição de tipos personalizados para arrays numpy
ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]

T = TypeVar('T')

class QuantumState(Protocol):
    """
    Protocol defining expected quantum state interface

    Attributes:
        quantum_state: Complex numpy array representing quantum state
    """
    quantum_state: ComplexArray

def calculate_quantum_entropy(quantum_state: ComplexArray, preserve_phase: bool = True) -> float:
    """
    Calcular entropia de um estado quântico preservando informação de fase
    usando entropia von Neumann com correções de fase

    Args:
        quantum_state: Estado quântico como array numpy
        preserve_phase: Se True, preserva informação de fase durante normalização

    Returns:
        float: Valor de entropia normalizado [0,1]

    Raises:
        ValueError: Se o estado quântico for inválido ou nulo
    """
    if quantum_state is None or quantum_state.size == 0:
        raise ValueError("Estado quântico inválido")

    # Extração de amplitude e fase com validação numérica
    amplitudes = np.abs(quantum_state)
    phases = np.angle(quantum_state) if preserve_phase else None

    # Normalização com proteção numérica
    total_amplitude = np.sum(amplitudes)
    if total_amplitude < 1e-10:
        return 0.0

    normalized_amplitudes = amplitudes / total_amplitude

    try:
        # Matriz densidade com proteção numérica
        density_matrix = np.outer(normalized_amplitudes, normalized_amplitudes.conj())

        # Autovalores com filtragem de valores pequenos
        eigenvalues = np.real(np.linalg.eigvals(density_matrix))
        eigenvalues = eigenvalues[eigenvalues > 1e-10]

        # Entropia von Neumann com correção logarítmica
        with np.errstate(divide='ignore', invalid='ignore'):
            entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
            entropy = np.nan_to_num(entropy, 0.0)

        # Ajuste de entropia por coerência de fase
        if preserve_phase and phases is not None:
            phase_coherence = np.abs(np.mean(np.exp(1j * phases)))
            phase_coherence = np.nan_to_num(phase_coherence, 0.0)
            entropy *= (1 - 0.5 * phase_coherence)

        return np.clip(entropy, 0.0, 1.0)

    except (np.linalg.LinAlgError, ValueError) as e:
        logging.error(f"Erro no cálculo de entropia: {str(e)}")
        return 0.5  # Valor seguro default

class QuantumMergeLogger:
    def __init__(self, log_file: Optional[str] = 'quantum_merge.log'):
        """
        Logger dedicado para operações de merge quântico

        Args:
            log_file: Caminho para arquivo de log
        """
        logging.basicConfig(
            filename=log_file, 
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def log_quantum_metrics(
        self, 
        coherence: float, 
        phase_coherence: float,
        decoherence_rate: float
    ):
        """
        Registrar métricas quânticas do sistema

        Args:
            coherence: Coerência do estado
            phase_coherence: Coerência de fase
            decoherence_rate: Taxa de decoerência
        """
        self.logger.info("Quantum System Metrics:")
        self.logger.info(f"State Coherence: {coherence:.4f}")
        self.logger.info(f"Phase Coherence: {phase_coherence:.4f}")
        self.logger.info(f"Decoherence Rate: {decoherence_rate:.4f}")

    def log_merge_failure(self, reason: str):
        """
        Registrar log de falha de merge

        Args:
            reason: Motivo da falha
        """
        self.logger.warning(f"Merge Failure: {reason}")

    def log_partial_merge(self, module: str, merge_result: Dict[str, Any]):
        """
        Registrar log de merge parcial

        Args:
            module: Módulo sendo fundido
            merge_result: Resultados do merge
        """
        self.logger.info(f"Partial Merge - Module: {module}")
        self.logger.info(f"Merge Success: {merge_result.get('merge_success', False)}")
        self.logger.info(f"Merge Probability: {merge_result.get('merge_probability', 0)}")

    def log_high_entropy_merge(self, merge_result: Dict[str, Any]):
        """
        Registrar log de merge em alta entropia

        Args:
            merge_result: Resultados do merge
        """
        self.logger.warning("High Entropy Merge Attempt")
        self.logger.warning(f"Merge Probability: {merge_result.get('merge_probability', 0)}")
        self.logger.warning(f"Merge Success: {merge_result.get('merge_success', False)}")
        self.logger.warning(f"Entropy: {merge_result.get('entropy', 0)}")

    def log_retrocausal_stability(
        self, 
        initial_state: np.ndarray, 
        post_merge_state: np.ndarray, 
        prediction_coherence: float
    ):
        """
        Registrar log de estabilidade retrocausal

        Args:
            initial_state: Estado quântico inicial
            post_merge_state: Estado quântico pós-merge
            prediction_coherence: Coerência das predições
        """
        self.logger.info("Retrocausal Stability Analysis")
        self.logger.info(f"Initial State: {initial_state}")
        self.logger.info(f"Post-Merge State: {post_merge_state}")
        self.logger.info(f"Prediction Coherence: {prediction_coherence}")

    def log_successful_merge(self, module: str, merge_result: Dict[str, Any]):
        """
        Registrar log de merge bem-sucedido

        Args:
            module: Módulo sendo fundido
            merge_result: Resultados do merge
        """
        self.logger.info(f"Successful Merge - Module: {module}")
        self.logger.info(f"Merge Probability: {merge_result.get('merge_probability', 0)}")
        self.logger.info(f"Merged Coherence: {merge_result.get('merged_coherence', 0)}")
        self.logger.info(f"Merged Complexity: {merge_result.get('merged_complexity', 0)}")

class QuantumMergeSimulator:
    def __init__(
        self, 
        qualia: Optional[QuantumState] = None,
        qsi: Optional[QuantumState] = None,
        morphic_calculator: Optional[Any] = None,
        decoherence_protection: bool = True
    ) -> None:
        """
        Simulador de merge quântico com proteção contra decoerência

        Args:
            qualia: Sistema quântico QUALIA
            qsi: Sistema quântico QSI
            morphic_calculator: Calculador de ressonância mórfica
            decoherence_protection: Ativa proteção contra decoerência

        Raises:
            ValueError: Se os estados quânticos forem inválidos
        """
        # Validação de tipos e estados
        if qualia is None or qsi is None:
            raise ValueError("Both qualia and qsi must be provided")

        for obj, name in [(qualia, 'qualia'), (qsi, 'qsi')]:
            if not hasattr(obj, 'quantum_state'):
                raise ValueError(f"{name} must have quantum_state attribute")
            if not isinstance(obj.quantum_state, np.ndarray):
                raise ValueError(f"{name}.quantum_state must be numpy array")
            if obj.quantum_state.dtype not in (np.complex64, np.complex128):
                raise ValueError(f"{name}.quantum_state must be complex type")

        self.qualia = qualia
        self.qsi = qsi
        self.morphic_calculator = morphic_calculator
        self.decoherence_protection = decoherence_protection

        # Inicialização de métricas com tipos explícitos
        self.merge_metrics: Dict[str, float] = {
            'coherence': 0.5,
            'phase_coherence': 1.0,
            'decoherence_rate': 0.0
        }

        # Memória histórica tipada
        self.merge_memory: Dict[str, Union[List[Dict[str, Any]], int, List[float]]] = {
            'merge_history': [],
            'total_merges': 0,
            'successful_merges': 0,
            'coherence_trajectory': [],
            'phase_coherence_history': []
        }

        self.logger = QuantumMergeLogger()

        # Estados iniciais com validação de tipo
        self._initial_state: Dict[str, Union[ComplexArray, Dict[str, FloatArray]]] = {
            'qualia_state': qualia.quantum_state.copy(),
            'qsi_state': qsi.quantum_state.copy(),
            'initial_phases': {
                'qualia': np.angle(qualia.quantum_state),
                'qsi': np.angle(qsi.quantum_state)
            }
        }
        self._last_stable_state: ComplexArray = qualia.quantum_state.copy()
        self._merge_attempt_count: int = 0

    def calculate_phase_coherence(
        self, 
        state1: np.ndarray, 
        state2: np.ndarray
    ) -> float:
        """
        Calcula coerência de fase entre dois estados quânticos

        Args:
            state1: Primeiro estado quântico
            state2: Segundo estado quântico

        Returns:
            Valor de coerência de fase [0,1]
        """
        phases1 = np.angle(state1)
        phases2 = np.angle(state2)

        # Diferença de fase média
        phase_diff = phases1 - phases2
        phase_coherence = np.abs(np.mean(np.exp(1j * phase_diff)))

        return phase_coherence

    def safe_coherence(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """
        Calculate coherence with enhanced stability and NaN prevention
        """
        try:
            # Ensure states are normalized
            norm1 = np.linalg.norm(state1)
            norm2 = np.linalg.norm(state2)

            if norm1 < 1e-10 or norm2 < 1e-10:
                return 0.5  # Safe default

            normalized_state1 = state1 / norm1
            normalized_state2 = state2 / norm2

            coherence = np.abs(np.dot(normalized_state1, normalized_state2))

            # Prevent NaN/Inf and ensure minimum coherence
            if np.isnan(coherence) or np.isinf(coherence):
                return 0.5
            
            return max(0.3, min(1.0, coherence))  # Ensure minimum coherence

        except Exception as e:
            self.logger.logger.warning(f"Coherence calculation failed: {str(e)}")
            return 0.5  # Safe default

    def safe_entropy(self, state: np.ndarray) -> float:
        """
        Calculate entropy with enhanced stability and bounds
        """
        try:
            entropy = calculate_quantum_entropy(state)

            # Prevent extreme values
            if np.isnan(entropy) or np.isinf(entropy):
                return 0.4  # Conservative default

            return min(0.45, max(0.1, entropy))  # Bounded entropy

        except Exception as e:
            self.logger.logger.warning(f"Entropy calculation failed: {str(e)}")
            return 0.4  # Conservative default

    def _apply_decoherence_protection(
        self, 
        state: ComplexArray,
        initial_coherence: float
    ) -> ComplexArray:
        """
        Aplica proteção contra decoerência usando técnicas de correção quântica

        Args:
            state: Estado quântico a ser protegido
            initial_coherence: Coerência inicial do estado

        Returns:
            ComplexArray: Estado protegido contra decoerência

        Notes:
            Implementa correção de erro quântico simplificada usando:
            1. Média local de fase para estabilização
            2. Ponderação de correções baseada em coerência
            3. Normalização preservando relações de fase
        """
        try:
            # Extração de componentes com validação
            amplitudes = np.abs(state)
            phases = np.angle(state)

            # Correção de fase com janela deslizante
            phase_corrections = np.zeros_like(phases)
            window_size = 5  # Tamanho da janela de correção

            for i in range(len(phases)):
                # Índices da janela com proteção de borda
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(phases), i + window_size // 2 + 1)

                # Média de fase local com pesos gaussianos
                window = phases[start_idx:end_idx]
                weights = np.exp(-0.5 * np.square(np.arange(len(window)) - len(window)//2))
                weighted_phases = window * weights

                # Cálculo da fase média ponderada
                phase_corrections[i] = np.angle(np.sum(np.exp(1j * weighted_phases)))

            # Aplicação de correções com peso adaptativo
            coherence_weight = np.clip(initial_coherence, 0.3, 0.7)
            corrected_phases = (
                phases * coherence_weight +
                phase_corrections * (1 - coherence_weight)
            )

            # Reconstrução do estado com normalização
            protected_state = amplitudes * np.exp(1j * corrected_phases)
            protected_state /= np.linalg.norm(protected_state) + 1e-10

            return protected_state

        except Exception as e:
            self.logger.logger.error(f"Erro na proteção contra decoerência: {str(e)}")
            return state  # Retorna estado original em caso de erro

    def _estimate_decoherence_rate(self) -> float:
        """
        Estima taxa de decoerência baseado no histórico

        Returns:
            Taxa estimada de decoerência
        """
        if len(self.merge_memory['phase_coherence_history']) < 2:
            return 0.0

        coherence_changes = np.diff(
            self.merge_memory['phase_coherence_history'][-10:]
        )
        return np.abs(np.mean(coherence_changes))

    def _create_success_result(
        self,
        coherence: float,
        phase_coherence: float,
        entropy: float,
        learning_rate: float
    ) -> Dict[str, Any]:
        """
        Cria resultado detalhado de merge bem-sucedido
        """
        return {
            'merge_success': True,
            'rollback_triggered': False,
            'metrics': {
                'merged_coherence': max(0.7, coherence),
                'phase_coherence': max(0.6, phase_coherence),
                'merged_complexity': min(np.std(self.qualia.quantum_state), 0.4),
                'entropy': min(entropy, 0.4),
                'dynamic_learning_rate': learning_rate,
                'decoherence_rate': self._estimate_decoherence_rate()
            }
        }

    def _create_failure_result(self) -> Dict[str, Any]:
        """
        Cria resultado para merge falho
        """
        return {
            'merge_success': False,
            'rollback_triggered': True,
            'metrics': {}
        }

    def _handle_merge_failure(
        self,
        initial_entropy: float,
        post_merge_entropy: float,
        post_merge_coherence: float,
        post_merge_phase_coherence: float
    ) -> dict:
        """
        Handle merge failures with improved rollback
        """
        # More conservative rollback threshold
        rollback_threshold = 0.35

        if post_merge_entropy > initial_entropy * rollback_threshold:
            self.qualia.quantum_state = self._initial_state['qualia_state'].copy()

        return {
            'merge_success': False,
            'rollback_triggered': True,
            'metrics': {
                'initial_entropy': initial_entropy,
                'post_merge_entropy': post_merge_entropy,
                'merged_coherence': post_merge_coherence,
                'phase_coherence': post_merge_phase_coherence,
                'rollback_threshold': rollback_threshold
            }
        }

    def _rollback_merge(self):
        """
        Mecanismo de rollback em caso de merge instável
        """
        if self._initial_state:
            self.qualia.quantum_state = self._initial_state['qualia_state']
            self.qsi.quantum_state = self._initial_state['qsi_state']

        return {
            'merge_success': False,
            'rollback_performed': True,
            'reason': 'Merge instability detected'
        }

    def _update_merge_memory(self, merge_metrics: Dict[str, Any]):
        """
        Atualizar memória histórica de merges
        """
        self.merge_memory['total_merges'] += 1

        if merge_metrics.get('merge_success', False):
            self.merge_memory['successful_merges'] += 1
            self.merge_memory['coherence_trajectory'].append(
                merge_metrics.get('post_merge_coherence', 0)
            )
            self.merge_memory['phase_coherence_history'].append(
                merge_metrics.get('phase_coherence', 0)
            )

        self.merge_memory['merge_history'].append(merge_metrics)

        # Limitar histórico para evitar consumo excessivo de memória
        if len(self.merge_memory['merge_history']) > 100:
            self.merge_memory['merge_history'] = self.merge_memory['merge_history'][-100:]

    def simulate_merge(
        self, 
        merge_strategy: str = 'adaptive',
        entropy_threshold: float = 0.45,
        coherence_damping: float = 0.4,
        use_morphic_resonance: bool = True,
        max_merge_attempts: int = 5
    ) -> Dict[str, Any]:
        """
        Simulação de merge quântico com preservação de coerência

        Args:
            merge_strategy: Estratégia de merge
            entropy_threshold: Limiar de entropia
            coherence_damping: Fator de amortecimento
            use_morphic_resonance: Usar ressonância mórfica
            max_merge_attempts: Número máximo de tentativas

        Returns:
            Resultados do merge
        """
        self._merge_attempt_count += 1
        if self._merge_attempt_count >= max_merge_attempts:
            self.logger.log_merge_failure('Max merge attempts exceeded')
            return self._create_failure_result()

        # Métricas iniciais com preservação de fase
        initial_coherence = self.safe_coherence(
            self.qualia.quantum_state,
            self.qsi.quantum_state
        )
        initial_phase_coherence = self.calculate_phase_coherence(
            self.qualia.quantum_state,
            self.qsi.quantum_state
        )
        initial_entropy = calculate_quantum_entropy(
            self.qualia.quantum_state, 
            preserve_phase=True
        )

        # Taxa de aprendizado adaptativa baseada em coerência quântica
        dynamic_learning_rate = min(0.2, self._calculate_adaptive_learning_rate(
            initial_coherence,
            initial_phase_coherence
        ))

        # Merge quântico com preservação de fase
        merged_state = (
            self.qualia.quantum_state * (1 - dynamic_learning_rate) +
            self.qsi.quantum_state * dynamic_learning_rate
        )

        # Proteção contra decoerência
        if self.decoherence_protection:
            merged_state = self._apply_decoherence_protection(
                merged_state,
                initial_phase_coherence
            )

        # Métricas pós-merge
        post_merge_entropy = calculate_quantum_entropy(
            merged_state, 
            preserve_phase=True
        )
        post_merge_coherence = self.safe_coherence(
            merged_state,
            self.qsi.quantum_state
        )
        post_merge_phase_coherence = self.calculate_phase_coherence(
            merged_state,
            self._initial_state['qualia_state']
        )

        # Critérios de sucesso aprimorados
        merge_success = (
            post_merge_entropy <= entropy_threshold and
            post_merge_coherence >= 0.7 and
            post_merge_phase_coherence >= 0.6 and
            post_merge_entropy < initial_entropy
        )

        if not merge_success:
            return self._handle_merge_failure(
                initial_entropy,
                post_merge_entropy,
                post_merge_coherence,
                post_merge_phase_coherence
            )

        # Atualização de estado com preservação de métricas quânticas
        self.qualia.quantum_state = merged_state.copy()
        self._last_stable_state = merged_state.copy()
        self.merge_memory['total_merges'] += 1
        self.merge_memory['successful_merges'] += 1
        self.merge_memory['coherence_trajectory'].append(post_merge_coherence)
        self.merge_memory['phase_coherence_history'].append(post_merge_phase_coherence)

        # Registro de métricas quânticas
        self.logger.log_quantum_metrics(
            post_merge_coherence,
            post_merge_phase_coherence,
            self._estimate_decoherence_rate()
        )

        result = self._create_success_result(
            post_merge_coherence,
            post_merge_phase_coherence,
            post_merge_entropy,
            dynamic_learning_rate
        )

        self._merge_attempt_count = 0
        return result

    def _calculate_adaptive_learning_rate(
        self, 
        initial_coherence: float = 0.5, 
        initial_phase_coherence: float = 0.5
    ) -> float:
        """
        Calculate a more robust dynamic learning rate with enhanced adaptivity
        """
        # Base learning rate with progressive adjustment
        base_rate = 0.1
        
        # Historical merge performance influence
        successful_merge_ratio = max(
            0.5,  # Minimum baseline
            self.merge_memory['successful_merges'] / 
            max(1, self.merge_memory['total_merges'])
        )
        
        # Complexity and coherence factors with more aggressive adaptation
        coherence_factor = max(0.5, initial_coherence)
        phase_coherence_factor = max(0.5, initial_phase_coherence)
        
        # Introduce controlled randomness for exploration with reduced variance
        exploration_noise = np.random.uniform(-0.02, 0.02)
        
        # Adaptive learning rate calculation with more stable computation
        adaptive_rate = (
            base_rate * 
            successful_merge_ratio * 
            coherence_factor * 
            phase_coherence_factor + 
            exploration_noise
        )
        
        # Constrain learning rate with tighter, more stable bounds
        return np.clip(adaptive_rate, 0.01, 0.2)