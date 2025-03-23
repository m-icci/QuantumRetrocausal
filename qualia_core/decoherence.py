import numpy as np
from qutip import *
from .sacred_geometry import generate_holonomic_pattern
from datetime import datetime
from typing import Optional, Dict, List, Union

class QuantumDecoherenceModel:
    def __init__(self, temperature: int = 310, bath_modes: int = 3):
        self.temperature = temperature  # Em Kelvin
        self.bath_modes = bath_modes
        self.H = self.build_hamiltonian()
        self.decoherence_history: List[Dict] = []

    def build_hamiltonian(self):
        # Hamiltonian do Sistema
        states = basis(2,0), basis(2,1)
        epsilon = [1.0, -1.0]
        V_kl = 0.15

        H_sys = sum(epsilon[k] * states[k] * states[k].dag() for k in [0,1])
        H_sys += V_kl * (states[0] * states[1].dag() + states[1] * states[0].dag())

        # Hamiltonian do Banho Termal
        H_bath = sum((np.sqrt(0.1*(self.temperature/300)) * 
                     tensor(qeye(2), destroy(self.bath_modes))**2) 
                    for _ in range(self.bath_modes))

        # Acoplamento não-Markoviano
        H_int = tensor(sigmaz(), qeye(self.bath_modes)) * 0.05 * (
            sum(tensor(qeye(2), destroy(self.bath_modes)) for _ in range(self.bath_modes))
        )

        return H_sys + H_bath + H_int

    def apply_zurek_correction(self, curvature: float = 0.021):
        """Aplica correção de holonomia quântica"""
        pattern = generate_holonomic_pattern(curvature)
        self.H += pattern * tensor(sigmay(), qeye(self.bath_modes))
        self._record_correction(curvature)

    def apply_suppression(self, state: np.ndarray, linguistic_pattern: str) -> np.ndarray:
        """Aplica supressão ativa de decoerência usando padrões linguísticos"""
        # Calcula matriz de densidade
        rho = Qobj(state)

        # Aplica operador de Lindblad modificado
        lindblad_ops = self._generate_lindblad_ops(linguistic_pattern)

        # Evolui estado com correção de decoerência
        result = mesolve(self.H, rho, np.linspace(0, 0.1, 2), lindblad_ops)

        # Registra histórico
        self._record_suppression(linguistic_pattern)

        return result.states[-1].full()

    def _generate_lindblad_ops(self, pattern: str) -> List:
        """Gera operadores de Lindblad baseados no padrão linguístico"""
        base_ops = [destroy(self.bath_modes) for _ in range(2)]

        # Modifica operadores baseado no padrão
        pattern_factor = len(pattern) / 100  # Normaliza influência do padrão
        modified_ops = [op * np.exp(-pattern_factor) for op in base_ops]

        return modified_ops

    def _record_correction(self, curvature: float) -> None:
        """Registra correção de holonomia aplicada"""
        self.decoherence_history.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'holonomic_correction',
            'curvature': curvature,
            'temperature': self.temperature
        })

    def _record_suppression(self, pattern: str) -> None:
        """Registra supressão de decoerência aplicada"""
        self.decoherence_history.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'decoherence_suppression',
            'pattern': pattern,
            'bath_modes': self.bath_modes
        })

    def get_decoherence_metrics(self) -> Dict[str, Union[float, str]]:
        """Retorna métricas do sistema de decoerência"""
        if not self.decoherence_history:
            return {
                'stability': 0.0,
                'last_correction': None,
                'suppression_count': 0
            }

        recent_events = self.decoherence_history[-10:]
        corrections = [e for e in recent_events if e['type'] == 'holonomic_correction']
        suppressions = [e for e in recent_events if e['type'] == 'decoherence_suppression']

        return {
            'stability': self._calculate_stability(recent_events),
            'last_correction': corrections[-1]['timestamp'] if corrections else None,
            'suppression_count': len(suppressions)
        }

    def _calculate_stability(self, events: List[Dict]) -> float:
        """Calcula estabilidade do sistema baseado no histórico"""
        if not events:
            return 1.0

        # Análise temporal das correções
        correction_times = [
            datetime.fromisoformat(e['timestamp']) 
            for e in events 
            if e['type'] == 'holonomic_correction'
        ]

        if len(correction_times) < 2:
            return 1.0

        # Calcula variância dos intervalos entre correções
        intervals = np.diff([t.timestamp() for t in correction_times])
        stability = 1.0 / (1.0 + np.std(intervals))

        return float(stability)