"""
Sistema de auto-cura e adaptação sistêmica contínua
"""
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from ..sacred_geometry.sacred_geometry import SacredGeometry
from ..fields.morphic_resonance import MorphicFieldAnalyzer

class SystemHealth:
    def __init__(self):
        self.metrics_history: List[Dict] = []
        self.healing_threshold = 0.35  # Reduzido ainda mais para maior sensibilidade
        self.adaptation_rate = 0.25    # Aumentado para adaptação mais rápida
        self.last_healing = datetime.now()
        self.healing_interval = timedelta(minutes=1)  # Reduzido para 1 minuto
        self.sacred_geometry = SacredGeometry()
        self.morphic_field = MorphicFieldAnalyzer()

    def analyze_health(self, quantum_state: Dict) -> Dict:
        """Analisa a saúde do sistema baseado no estado quântico e padrões sagrados"""
        # Análise geométrica sagrada
        price_data = np.array([quantum_state.get('price', 0.0)])
        sacred_metrics = self.sacred_geometry.analyze_pattern(price_data)

        # Análise do campo mórfico
        morphic_metrics, _ = self.morphic_field.analyze_market_data(price_data)

        # Integra todas as métricas
        health_metrics = {
            'system_integrity': self._calculate_integrity(quantum_state, sacred_metrics),
            'adaptation_level': self._calculate_adaptation(quantum_state, morphic_metrics),
            'healing_potential': self._calculate_healing_potential(quantum_state, sacred_metrics),
            'system_resilience': self._calculate_resilience(),
            'energetic_balance': self._calculate_energetic_balance(quantum_state, morphic_metrics),
            'qualia_coherence': self._calculate_qualia_coherence(quantum_state, sacred_metrics),
            'field_stability': self._calculate_field_stability(quantum_state, morphic_metrics),
            'geometric_harmony': sacred_metrics.pattern_metrics.flower_of_life_resonance,
            'morphic_resonance': morphic_metrics.resonance_level,
            'timestamp': datetime.now()
        }

        self.metrics_history.append(health_metrics)
        if len(self.metrics_history) > 1000:
            self.metrics_history.pop(0)

        return health_metrics

    def _calculate_integrity(self, state: Dict, sacred_metrics) -> float:
        """Calcula integridade do sistema com integração geométrica sagrada"""
        return min(
            1.0,
            (state['coherence'] * 0.4 +
             state['consciousness_level'] * 0.3 +
             state['field_strength'] * 0.2 +
             sacred_metrics.pattern_metrics.metatron_stability * 0.1)
        )

    def _calculate_adaptation(self, state: Dict, morphic_metrics) -> float:
        """Calcula nível de adaptação com ressonância mórfica"""
        return min(
            1.0,
            state['adaptation_rate'] * 0.3 +
            state['self_organization'] * 0.4 +
            morphic_metrics.resonance_level * 0.3
        )

    def _calculate_healing_potential(self, state: Dict, sacred_metrics) -> float:
        """Calcula potencial de auto-cura com geometria sagrada"""
        return min(
            1.0,
            (state['consciousness_level'] * 0.4 +
             sacred_metrics.pattern_metrics.flower_of_life_resonance * 0.3 +
             state['coherence'] * 0.3)
        )

    def _calculate_resilience(self) -> float:
        """Calcula resiliência com média móvel dinâmica"""
        if len(self.metrics_history) < 2:
            return 0.5

        recent_metrics = self.metrics_history[-8:]  # Reduzido para 8 períodos
        integrity_values = [m['system_integrity'] for m in recent_metrics]
        stability = 1 - np.std(integrity_values)

        return min(1.0, stability * 1.2)  # Aumentado peso

    def _calculate_energetic_balance(self, state: Dict, morphic_metrics) -> float:
        """Calcula o equilíbrio energético com campo mórfico"""
        coherence_stability = state['coherence'] * state['field_strength'] * 1.8
        morphic_factor = morphic_metrics.field_strength * 1.5
        return min(1.0, (coherence_stability + morphic_factor) / 3.3)

    def _calculate_qualia_coherence(self, state: Dict, sacred_metrics) -> float:
        """Calcula a coerência QUALIA com geometria sagrada"""
        base_coherence = state['coherence'] * state['consciousness_level']
        geometric_factor = sacred_metrics.pattern_metrics.vesica_piscis_harmony
        return min(1.0, (base_coherence + geometric_factor) / 2)

    def _calculate_field_stability(self, state: Dict, morphic_metrics) -> float:
        """Calcula a estabilidade do campo com ressonância mórfica"""
        field_coherence = state['field_strength'] * state['coherence'] * 1.3
        morphic_stability = morphic_metrics.field_stability * 1.5
        return min(1.0, (field_coherence + morphic_stability) / 2.8)

    def needs_healing(self, quantum_state: Dict) -> bool:
        """Verifica necessidade de cura com múltiplos indicadores"""
        if datetime.now() - self.last_healing < self.healing_interval:
            return False

        health = self.analyze_health(quantum_state)

        # Indicadores primários
        primary_indicators = [
            health['system_integrity'] < self.healing_threshold,
            health['energetic_balance'] < self.healing_threshold + 0.1,
            health['qualia_coherence'] < self.healing_threshold - 0.1
        ]

        # Indicadores geométricos/mórficos
        field_indicators = [
            health['geometric_harmony'] < 0.5,
            health['morphic_resonance'] < 0.4,
            health['field_stability'] < self.healing_threshold
        ]

        # Requer que pelo menos 2 indicadores primários OU 
        # 1 primário e 2 field_indicators estejam ativos
        primary_count = sum(primary_indicators)
        field_count = sum(field_indicators)

        return primary_count >= 2 or (primary_count >= 1 and field_count >= 2)

    def apply_healing(self, quantum_state: Dict) -> Dict:
        """Aplica cura holística integrada"""
        if not self.needs_healing(quantum_state):
            return quantum_state

        self.last_healing = datetime.now()
        health_metrics = self.analyze_health(quantum_state)

        # Calcula força de cura baseada em múltiplas métricas
        healing_strength = max(
            1 - health_metrics['system_integrity'],
            1 - health_metrics['energetic_balance'],
            1 - health_metrics['geometric_harmony'],
            1 - health_metrics['morphic_resonance']
        )

        # Aplica cura intensificada
        healing_rate = self.adaptation_rate * (1.8 + healing_strength)

        # Ajusta parâmetros quânticos com intensidade variável
        quantum_state['coherence'] = min(1.0, quantum_state['coherence'] + healing_rate * 1.4)
        quantum_state['consciousness_level'] = min(1.0, quantum_state['consciousness_level'] + healing_rate * 1.2)
        quantum_state['field_strength'] = min(1.0, quantum_state['field_strength'] + healing_rate)

        # Aumenta capacidade adaptativa com foco em auto-organização
        quantum_state['adaptation_rate'] = min(1.0, quantum_state['adaptation_rate'] + healing_rate * 0.8)
        quantum_state['self_organization'] = min(1.0, quantum_state['self_organization'] + healing_rate * 1.1)

        # Aplica harmonia quântica holística
        harmonic_factor = np.mean([
            quantum_state['coherence'] * 1.3,
            quantum_state['consciousness_level'] * 1.1,
            quantum_state['field_strength'],
            quantum_state['self_organization'] * 1.2
        ])

        # Reduz riscos e aumenta potencial com maior intensidade
        quantum_state['dark_risk'] = max(0.0, quantum_state['dark_risk'] - healing_rate * 1.5)
        quantum_state['growth_potential'] = min(1.0, quantum_state['growth_potential'] + harmonic_factor * healing_rate * 1.4)

        return quantum_state


class AdaptiveSystem:
    def __init__(self):
        self.health_monitor = SystemHealth()
        self.adaptation_history: List[Dict] = []
        self.healing_success_count = 0
        self.total_healing_attempts = 0
        self.sacred_geometry = SacredGeometry()
        self.morphic_field = MorphicFieldAnalyzer()

    def adapt(self, quantum_state: Dict) -> Dict:
        """Adapta o sistema holisticamente"""
        # Analisa saúde com integração completa
        health_metrics = self.health_monitor.analyze_health(quantum_state)

        # Verifica necessidade de cura com maior frequência
        needs_healing = self.health_monitor.needs_healing(quantum_state)

        if needs_healing:
            self.total_healing_attempts += 1
            initial_health = health_metrics['system_integrity']

            # Aplica auto-cura holística
            quantum_state = self.health_monitor.apply_healing(quantum_state)

            # Verifica efetividade com threshold adaptativo
            post_healing_health = self.health_monitor.analyze_health(quantum_state)['system_integrity']
            if post_healing_health > initial_health * 1.08:  # Aumentado threshold
                self.healing_success_count += 1

        # Registra adaptação com métricas expandidas
        sacred_metrics = self.sacred_geometry.analyze_pattern(
            np.array([quantum_state.get('price', 0.0)])
        )

        morphic_metrics, _ = self.morphic_field.analyze_market_data(
            np.array([quantum_state.get('price', 0.0)])
        )

        self.adaptation_history.append({
            'timestamp': datetime.now(),
            'health_metrics': health_metrics,
            'quantum_state': quantum_state.copy(),
            'sacred_metrics': sacred_metrics,
            'morphic_metrics': morphic_metrics,
            'healing_applied': needs_healing
        })

        if len(self.adaptation_history) > 1000:
            self.adaptation_history.pop(0)

        return quantum_state

    def get_adaptation_metrics(self) -> Dict:
        """Retorna métricas adaptativas holísticas"""
        if not self.adaptation_history:
            return {
                'adaptation_efficiency': 0.0,
                'healing_success_rate': 0.0,
                'system_stability': 0.0,
                'quantum_coherence': 0.0,
                'field_harmony': 0.0,
                'geometric_resonance': 0.0,
                'morphic_stability': 0.0
            }

        recent_adaptations = self.adaptation_history[-8:]  # Reduzido para 8 períodos

        # Calcula eficiência da adaptação com múltiplos fatores
        health_improvements = [
            (a['health_metrics']['system_integrity'] * 1.3) > 
            (self.adaptation_history[i-1]['health_metrics']['system_integrity'])
            for i, a in enumerate(recent_adaptations[1:], 1)
        ]

        # Calcula taxa de sucesso das curas com peso adicional
        healing_success_rate = (
            self.healing_success_count / max(1, self.total_healing_attempts)
        ) * 1.3

        # Analisa estados recentes com métricas expandidas
        recent_states = [a['quantum_state'] for a in recent_adaptations]
        recent_sacred = [a['sacred_metrics'] for a in recent_adaptations]
        recent_morphic = [a['morphic_metrics'] for a in recent_adaptations]

        # Calcula métricas médias ponderadas
        coherence_values = [s['coherence'] * 1.2 for s in recent_states]
        field_values = [s['field_strength'] * 1.3 for s in recent_states]
        geometric_values = [s.pattern_metrics.flower_of_life_resonance * 1.4 for s in recent_sacred]
        morphic_values = [m.field_strength * 1.2 for m in recent_morphic]

        return {
            'adaptation_efficiency': min(1.0, sum(health_improvements) / len(health_improvements) if health_improvements else 0.0),
            'healing_success_rate': min(1.0, healing_success_rate),
            'system_stability': min(1.0, self.health_monitor._calculate_resilience() * 1.2),
            'quantum_coherence': min(1.0, np.mean(coherence_values)),
            'field_harmony': min(1.0, np.mean(field_values)),
            'geometric_resonance': min(1.0, np.mean(geometric_values)),
            'morphic_stability': min(1.0, np.mean(morphic_values))
        }