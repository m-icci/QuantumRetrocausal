"""
Implementação do módulo Dark Finance para análise quântica de mercado
"""

from typing import Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
import threading
from core.logging.quantum_logger import quantum_logger

@dataclass
class DarkMetrics:
    """Métricas do Dark Finance"""
    field_strength: float
    dark_risk: float
    growth_potential: float
    coherence: float
    retrocausality: float

class DarkPortfolioIntegrator:
    """Integrador de portfólio com fatores ocultos"""
    _instance: Optional['DarkPortfolioIntegrator'] = None
    _initialized: bool = False
    _lock = threading.Lock()
    _field_coupling: float = 0.818  # Aumentado de 0.618 para 0.818

    def __new__(cls, field_coupling: float = 0.818) -> 'DarkPortfolioIntegrator':
        """Create or return singleton instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    instance._field_coupling = field_coupling
                    instance._quantum_state = np.zeros(64, dtype=np.complex64)
                    cls._instance = instance
                    quantum_logger.info(
                        "Dark Portfolio Singleton instance created",
                        {"field_coupling": field_coupling}
                    )
        return cls._instance

    def __init__(self, field_coupling: float = 0.818):
        """
        Inicializa integrador com acoplamento φ aumentado
        para maior influência nas decisões

        Args:
            field_coupling: Força de acoplamento (padrão: 0.818)
        """
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._initialized = True
                    self._quantum_state = np.zeros(64, dtype=np.complex64)
                    quantum_logger.info(
                        "Dark Portfolio inicializado com acoplamento aumentado",
                        {"field_coupling": self._field_coupling}
                    )

    # Add missing quantum state methods
    def get_quantum_state(self) -> np.ndarray:
        """
        Retorna o estado quântico atual do portfólio
        """
        with self._lock:
            return self._quantum_state.copy()

    def update_quantum_state(self, new_state: np.ndarray) -> None:
        """
        Atualiza o estado quântico do portfólio

        Args:
            new_state: Novo estado quântico
        """
        with self._lock:
            if new_state.shape != self._quantum_state.shape:
                quantum_logger.warning(
                    "Dimensões incompatíveis no update do estado quântico",
                    {
                        "current_shape": self._quantum_state.shape,
                        "new_shape": new_state.shape
                    }
                )
                return

            self._quantum_state = new_state.copy()
            quantum_logger.info(
                "Estado quântico atualizado",
                {"state_norm": float(np.linalg.norm(self._quantum_state))}
            )

    def synchronize_state(self, other_state: np.ndarray) -> None:
        """
        Sincroniza estado quântico com outro estado

        Args:
            other_state: Estado para sincronizar
        """
        with self._lock:
            if other_state.shape != self._quantum_state.shape:
                quantum_logger.warning(
                    "Dimensões incompatíveis na sincronização",
                    {
                        "current_shape": self._quantum_state.shape,
                        "other_shape": other_state.shape
                    }
                )
                return

            # Normalização e sincronização
            norm = np.linalg.norm(other_state)
            if norm > 0:
                self._quantum_state = other_state / norm
            else:
                self._quantum_state = np.zeros_like(other_state)

            quantum_logger.info(
                "Estados sincronizados",
                {"new_state_norm": float(np.linalg.norm(self._quantum_state))}
            )

    @property
    def field_coupling(self) -> float:
        """Get field coupling value with thread safety"""
        try:
            with self._lock:
                if not hasattr(self, '_field_coupling'):
                    self._field_coupling = 0.818  # Valor padrão
                value = self._field_coupling
                quantum_logger.debug(
                    "Field coupling accessed",
                    {
                        "current_value": value,
                        "thread_id": threading.get_ident()
                    }
                )
                return value
        except Exception as e:
            quantum_logger.error(
                "Erro acessando field_coupling",
                {"error": str(e), "thread_id": threading.get_ident()}
            )
            return 0.818  # Retorna valor padrão em caso de erro

    @field_coupling.setter 
    def field_coupling(self, value: float):
        """Set field coupling value with thread safety"""
        try:
            with self._lock:
                old_value = getattr(self, '_field_coupling', 0.818)
                self._field_coupling = float(value)  # Garante que é float
                quantum_logger.info(
                    "Field coupling updated",
                    {
                        "old_value": float(old_value),
                        "new_value": float(value),
                        "thread_id": threading.get_ident()
                    }
                )
        except Exception as e:
            quantum_logger.error(
                "Erro atualizando field_coupling",
                {
                    "intended_value": value,
                    "error": str(e),
                    "thread_id": threading.get_ident()
                }
            )

    def _calculate_field_strength(self, returns: list, consciousness: float) -> float:
        """Calcula força do campo mórfico com peso aumentado"""
        try:
            # Verifica se há dados suficientes
            if not returns:
                quantum_logger.warning(
                    "Sem dados de retorno para calcular field strength",
                    {"consciousness": consciousness}
                )
                return 0.0

            # Aumenta peso dos retornos recentes
            weighted_returns = returns[-10:] if len(returns) > 10 else returns
            field_strength = np.mean(weighted_returns) * consciousness * self.field_coupling * 1.5  # Peso aumentado

            quantum_logger.info(
                "Field strength calculada",
                {
                    "mean_returns": float(np.mean(returns)),
                    "weighted_returns_mean": float(np.mean(weighted_returns)),
                    "consciousness": consciousness,
                    "field_coupling": self.field_coupling,
                    "field_strength": float(field_strength)
                }
            )
            return abs(field_strength)
        except Exception as e:
            quantum_logger.error(
                "Erro calculando field strength",
                {"error": str(e)}
            )
            return 0.0  # Retorna valor padrão seguro em caso de erro

    def _calculate_dark_risk(self, returns: list, consciousness: float) -> float:
        """Calcula risco oculto com sensibilidade aumentada"""
        try:
            # Usa volatilidade dos últimos N períodos para maior sensibilidade
            recent_returns = returns[-20:] if len(returns) > 20 else returns
            volatility = np.std(recent_returns) if len(recent_returns) > 1 else 0
            dark_risk = volatility * (1 - consciousness) * self.field_coupling * 1.2  # Sensibilidade aumentada

            quantum_logger.info(
                "Dark risk calculado",
                {
                    "volatility": float(volatility),
                    "consciousness": consciousness,
                    "field_coupling": self.field_coupling,
                    "dark_risk": float(dark_risk),
                    "recent_periods": len(recent_returns)
                }
            )
            return abs(dark_risk)
        except Exception as e:
            quantum_logger.error(
                "Erro calculando dark risk",
                {"error": str(e)}
            )
            raise

    def _calculate_growth_potential(self, volumes: list) -> float:
        """Calcula potencial de crescimento com momentum exponencial"""
        try:
            if not volumes:
                return 0.0

            # Análise do momentum do volume com peso exponencial
            vol_changes = np.diff(volumes) if len(volumes) > 1 else [0]
            weights = np.exp(np.linspace(0, 1, len(vol_changes)))
            weighted_changes = vol_changes * weights
            growth_potential = np.mean(weighted_changes) * self.field_coupling

            quantum_logger.info(
                "Growth potential calculado",
                {
                    "volume_changes": [float(x) for x in vol_changes[-5:]],  # Mostra últimas 5 mudanças
                    "weighted_mean": float(np.mean(weighted_changes)),
                    "field_coupling": self.field_coupling,
                    "growth_potential": float(growth_potential),
                    "total_changes": len(vol_changes)
                }
            )
            return abs(growth_potential)
        except Exception as e:
            quantum_logger.error(
                "Erro calculando growth potential",
                {"error": str(e)}
            )
            raise

    def calculate_dark_metrics(
        self,
        returns: list,
        volumes: list,
        consciousness: float
    ) -> DarkMetrics:
        """
        Calcula métricas ocultas do Dark Finance com maior sensibilidade

        Args:
            returns: Lista de retornos históricos
            volumes: Lista de volumes históricos
            consciousness: Nível de consciência (0-1)

        Returns:
            DarkMetrics com métricas calculadas
        """
        try:
            quantum_logger.info(
                "Iniciando cálculo de métricas Dark Finance",
                {
                    "returns_count": len(returns),
                    "volumes_count": len(volumes),
                    "consciousness": consciousness,
                    "field_coupling": self.field_coupling
                }
            )

            # Verifica dados mínimos necessários
            if not returns or not volumes:
                quantum_logger.warning(
                    "Dados insuficientes para cálculo de métricas",
                    {
                        "returns_available": bool(returns),
                        "volumes_available": bool(volumes)
                    }
                )
                return DarkMetrics(
                    field_strength=0.0,
                    dark_risk=0.0,
                    growth_potential=0.0,
                    coherence=1.0,  # Valor neutro
                    retrocausality=1.0  # Valor neutro
                )

            # Calcula métricas base com maior sensibilidade
            field_strength = self._calculate_field_strength(returns, consciousness)
            dark_risk = self._calculate_dark_risk(returns, consciousness)
            growth_potential = self._calculate_growth_potential(volumes)

            # Calcula métricas quânticas com ajuste dinâmico
            field_factor = np.clip(field_strength * 1.2, 0, np.pi/2)  # Aumenta influência mas limita range
            coherence = abs(np.cos(field_factor))
            retrocausality = 1 - (dark_risk * 1.1)  # Aumenta impacto do risco

            metrics = DarkMetrics(
                field_strength=field_strength,
                dark_risk=dark_risk,
                growth_potential=growth_potential,
                coherence=coherence,
                retrocausality=retrocausality
            )

            quantum_logger.info(
                "Métricas Dark Finance calculadas",
                {
                    "field_strength": float(field_strength),
                    "dark_risk": float(dark_risk),
                    "growth_potential": float(growth_potential),
                    "coherence": float(coherence),
                    "retrocausality": float(retrocausality)
                }
            )

            return metrics

        except Exception as e:
            quantum_logger.error(
                "Erro calculando métricas Dark Finance",
                {
                    "error": str(e),
                    "returns_sample": str(returns[:5]) if returns else "[]",
                    "volumes_sample": str(volumes[:5]) if volumes else "[]",
                    "consciousness": consciousness
                }
            )
            raise

    @classmethod
    def reset(cls):
        """Reset singleton state - only for testing"""
        with cls._lock:
            cls._instance = None
            cls._initialized = False
            cls._field_coupling = 0.818