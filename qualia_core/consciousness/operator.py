"""
Operador de consciência quântica
"""
from .metrics import ConsciousnessMetrics
import numpy as np
from core.logging.quantum_logger import quantum_logger

class QuantumVoid:
    """Representa o vácuo quântico base para operações de consciência"""
    def __init__(self):
        self._entanglement = 0.0
        self._superposition = 0.0
        self._state = np.zeros(64, dtype=np.complex64)

    def measure_entanglement(self) -> float:
        return self._entanglement

    @property
    def superposition_state(self) -> float:
        return self._superposition

    def update_state(self, new_state: np.ndarray) -> None:
        """Atualiza estado do vácuo"""
        if new_state.shape != self._state.shape:
            quantum_logger.warning(
                "Dimensões incompatíveis no update do vácuo quântico",
                {
                    "current_shape": self._state.shape,
                    "new_shape": new_state.shape
                }
            )
            return
        self._state = new_state.copy()
        self._calculate_metrics()

    def _calculate_metrics(self) -> None:
        """Calcula métricas internas do vácuo"""
        try:
            # Calcula emaranhamento
            if len(self._state) > 1:
                self._entanglement = float(np.abs(
                    np.corrcoef(self._state.real, np.roll(self._state.real, 1))[0,1]
                ))
            else:
                self._entanglement = 0.0

            # Calcula superposição
            self._superposition = float(np.std(self._state.real))

        except Exception as e:
            quantum_logger.error(
                "Erro no cálculo de métricas do vácuo",
                {"error": str(e)}
            )
            self._entanglement = 0.0
            self._superposition = 0.0

    def __str__(self) -> str:
        return f"QuantumVoid(entanglement={self._entanglement}, superposition={self._superposition})"

class QuantumConsciousnessOperator:
    def __init__(self, quantum_void: QuantumVoid = None):
        self.quantum_void = quantum_void or QuantumVoid()
        self._state = np.zeros(64, dtype=np.complex64)
        quantum_logger.info(
            "Operador de consciência quântica iniciado",
            {"void_state": str(self.quantum_void)}
        )

    def calculate_metrics(self, wave_function: np.ndarray) -> ConsciousnessMetrics:
        """
        Calcula métricas de consciência a partir da função de onda
        """
        quantum_logger.debug(
            "Calculando métricas de consciência",
            {"wave_function_shape": wave_function.shape}
        )
        metrics = self._calculate_metrics(wave_function)

        quantum_logger.info(
            "Métricas calculadas",
            {
                "coherence": metrics.coherence,
                "entanglement": metrics.entanglement,
                "superposition": metrics.superposition
            }
        )
        return metrics

    def _calculate_metrics(self, wave_function: np.ndarray) -> ConsciousnessMetrics:
        coherence = self._calculate_coherence(wave_function)
        entanglement = self._calculate_entanglement()
        superposition = self._calculate_superposition()

        return ConsciousnessMetrics(coherence, entanglement, superposition)

    def _calculate_coherence(self, wave_function: np.ndarray) -> float:
        """Calcula coerência como a norma da função de onda normalizada"""
        quantum_logger.debug(
            "Calculando coerência",
            {"wave_function_norm": np.linalg.norm(wave_function)}
        )

        norm = np.linalg.norm(wave_function)
        if norm == 0:
            quantum_logger.warning("Função de onda com norma zero detectada")
            return 0

        normalized = wave_function / norm
        coherence = float(np.abs(np.vdot(normalized, normalized)))

        quantum_logger.debug(
            "Coerência calculada",
            {"coherence_value": coherence}
        )
        return coherence

    def _calculate_entanglement(self) -> float:
        """Obtém medida de emaranhamento do quantum void"""
        entanglement = self.quantum_void.measure_entanglement()
        quantum_logger.debug(
            "Emaranhamento calculado",
            {"entanglement_value": entanglement}
        )
        return entanglement

    def _calculate_superposition(self) -> float:
        """Obtém estado de superposição do quantum void"""
        superposition = self.quantum_void.superposition_state
        quantum_logger.debug(
            "Superposição calculada",
            {"superposition_value": superposition}
        )
        return superposition

    def get_state(self) -> np.ndarray:
        """Retorna o estado quântico atual do operador"""
        return self._state.copy()

    def analyze_state(self, state: dict) -> dict:
        """Analisa o estado quântico fornecido"""
        quantum_logger.debug(
            "Analisando estado quântico",
            {"state_keys": list(state.keys())}
        )

        # Atualiza estado interno
        if isinstance(state, dict) and "quantum_state" in state:
            self._state = state["quantum_state"]

        # Calcula métricas
        metrics = self._calculate_metrics(self._state)

        return {
            "coherence": metrics.coherence,
            "entanglement": metrics.entanglement,
            "superposition": metrics.superposition,
            "state_norm": float(np.linalg.norm(self._state))
        }

    def synchronize(self, other_state: np.ndarray) -> None:
        """
        Sincroniza o estado do operador com outro estado quântico
        """
        quantum_logger.debug(
            "Sincronizando estados",
            {
                "current_norm": float(np.linalg.norm(self._state)),
                "other_norm": float(np.linalg.norm(other_state))
            }
        )

        # Normaliza e atualiza estado
        norm = np.linalg.norm(other_state)
        if norm > 0:
            self._state = other_state / norm
        else:
            self._state = np.zeros_like(other_state)

        quantum_logger.info("Estados sincronizados")