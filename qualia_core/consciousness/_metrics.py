"""
Métricas de consciência quântica para sistema unificado
"""
from typing import Dict, Any

class ConsciousnessMetrics:
    def __init__(self, coherence: float, entanglement: float, superposition: float):
        """
        Inicializa métricas de consciência
        
        Args:
            coherence: Nível de coerência [0,1]
            entanglement: Nível de emaranhamento [0,1]
            superposition: Nível de superposição [0,1]
        """
        self._validate_metric(coherence, "coherence")
        self._validate_metric(entanglement, "entanglement")
        self._validate_metric(superposition, "superposition")
        
        self.coherence = coherence
        self.entanglement = entanglement
        self.superposition = superposition
        
    @staticmethod
    def _validate_metric(value: float, name: str) -> None:
        """Valida valor de métrica"""
        if not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric, got {type(value)}")
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in range [0,1], got {value}")
            
    def normalize(self) -> 'ConsciousnessMetrics':
        """Normaliza métricas para somarem 1.0"""
        total = self.coherence + self.entanglement + self.superposition
        if total == 0:
            return ConsciousnessMetrics(1/3, 1/3, 1/3)
            
        return ConsciousnessMetrics(
            coherence=self.coherence / total,
            entanglement=self.entanglement / total,
            superposition=self.superposition / total
        )
        
    def to_dict(self) -> Dict[str, float]:
        """Converte para dicionário"""
        return {
            'coherence': self.coherence,
            'entanglement': self.entanglement,
            'superposition': self.superposition
        }
        
    def __str__(self) -> str:
        return (
            f"ConsciousnessMetrics("
            f"coherence={self.coherence:.3f}, "
            f"entanglement={self.entanglement:.3f}, "
            f"superposition={self.superposition:.3f})"
        )
