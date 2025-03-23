#!/usr/bin/env python3
"""
Helix Controller: Integra o módulo Helix com o QUALIA Core Engine.
Traduz métricas quânticas e retrocausais em parâmetros adaptativos para otimização de trading.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from qualia_core.helix_analysis.helix_analyzer import HelixAnalyzer
from qualia_core.helix_analysis.helix_config import HelixConfig
from qualia_core.helix_analysis.quantum_pattern_recognizer import QuantumPatternRecognizer
from qualia_core.helix_analysis.retrocausal_integrator import RetrocausalIntegrator

logger = logging.getLogger("helix_controller")

class HelixController:
    """
    Controlador para integração do módulo Helix com QUALIA.
    Transforma análises quânticas em parâmetros adaptativos de trading.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa o controlador Helix
        
        Args:
            config: Configurações opcionais
        """
        self.config = config or {}
        
        # Configuração do Helix
        helix_config = HelixConfig(
            dimensions=self.config.get('dimensions', 256),
            num_qubits=self.config.get('num_qubits', 8),
            phi=self.config.get('phi', 0.618),
            temperature=self.config.get('temperature', 0.1),
            batch_size=self.config.get('batch_size', 1024),
            max_field_size=self.config.get('max_field_size', 1024)
        )
        
        # Inicializar componentes Helix
        self.helix_analyzer = HelixAnalyzer(helix_config)
        self.pattern_recognizer = QuantumPatternRecognizer(
            dimensions=helix_config.dimensions,
            num_qubits=helix_config.num_qubits,
            batch_size=helix_config.batch_size
        )
        self.retrocausal_integrator = RetrocausalIntegrator(
            tau=self.config.get('tau', 10),
            lambda_coupling=self.config.get('lambda_coupling', 0.5)
        )
        
        # Inicializar o campo da hélice
        self.helix_analyzer.initialize_helix()
        
        # Estado atual
        self.current_step = 0
        self.quantum_metrics = {}
        self.fractal_metrics = {}
        self.retrocausal_metrics = {}
        
        logger.info("Helix Controller inicializado")
    
    def evolve_and_analyze(self, steps: int = 1) -> Dict[str, Any]:
        """
        Evolui o campo da hélice e analisa padrões emergentes
        
        Args:
            steps: Número de passos de evolução
            
        Returns:
            Métricas e insights da evolução
        """
        # Evolução do campo
        evolution_results = self.helix_analyzer.evolve_helix(steps)
        self.current_step += steps
        
        if not evolution_results:
            logger.error("Falha na evolução do campo da hélice")
            return {}
        
        # Obter padrões quânticos
        field = self.helix_analyzer.helix_field
        self.quantum_metrics = self.pattern_recognizer.recognize_patterns(field)
        
        # Obter métricas fractais
        self.fractal_metrics = evolution_results.get('fractal_analysis', [])[-1] if evolution_results.get('fractal_analysis') else {}
        
        # Aplicar e analisar feedback retrocausal
        if self.current_step > self.retrocausal_integrator.tau:
            self.retrocausal_integrator.apply_feedback(field, self.current_step)
            self.retrocausal_metrics = self.retrocausal_integrator.get_feedback_metrics(field, self.current_step)
        
        # Consolidar resultados
        results = {
            'timestamp': datetime.now().isoformat(),
            'step': self.current_step,
            'quantum_metrics': self.quantum_metrics,
            'fractal_metrics': self.fractal_metrics,
            'retrocausal_metrics': self.retrocausal_metrics
        }
        
        logger.debug(f"Helix evoluído e analisado no passo {self.current_step}")
        return results
    
    def derive_trading_parameters(self) -> Dict[str, Any]:
        """
        Deriva parâmetros de trading a partir das métricas quânticas
        
        Returns:
            Parâmetros adaptativos para o QUALIA Core Engine
        """
        # Derivar threshold de confiança do LSTM
        lstm_threshold = self._calculate_lstm_threshold()
        
        # Derivar parâmetros quânticos
        quantum_coherence = self._calculate_quantum_coherence()
        quantum_complexity = self._calculate_quantum_complexity()
        
        # Outros parâmetros adaptativos
        adaptation_rate = self._calculate_adaptation_rate()
        risk_factor = self._calculate_risk_factor()
        
        return {
            'lstm_threshold': lstm_threshold,
            'quantum_coherence': quantum_coherence,
            'quantum_complexity': quantum_complexity,
            'adaptation_rate': adaptation_rate,
            'risk_factor': risk_factor
        }
    
    def get_quantum_state(self) -> np.ndarray:
        """
        Retorna o estado quântico atual para uso no QCNN
        
        Returns:
            Array do estado quântico
        """
        return self.helix_analyzer.quantum_state.flatten()
    
    def _calculate_lstm_threshold(self) -> float:
        """Calcula o threshold ideal para o LSTM baseado nas métricas quânticas"""
        base_threshold = 0.7  # Valor padrão
        
        # Ajuste baseado na coerência quântica
        coherence = self.quantum_metrics.get('coherence', 0.5)
        
        # Ajuste baseado na estabilidade do feedback retrocausal
        stability = self.retrocausal_metrics.get('stability', 0.5)
        
        # Maior coerência e estabilidade permitem thresholds mais baixos (mais oportunidades)
        threshold = base_threshold - (coherence * 0.2) - (stability * 0.1)
        
        # Limitar entre 0.6 e 0.9
        return min(0.9, max(0.6, threshold))
    
    def _calculate_quantum_coherence(self) -> float:
        """Calcula a coerência quântica para o estado de consciência do QCNN"""
        base_coherence = 0.5  # Valor padrão
        
        # Usar diretamente a coerência se disponível
        if 'coherence' in self.quantum_metrics:
            return min(0.9, max(0.1, self.quantum_metrics['coherence']))
        
        return base_coherence
    
    def _calculate_quantum_complexity(self) -> float:
        """Calcula a complexidade quântica para o estado de consciência do QCNN"""
        base_complexity = 0.3  # Valor padrão
        
        # Usar diretamente a complexidade se disponível
        if 'quantum_complexity' in self.quantum_metrics:
            return min(0.9, max(0.1, self.quantum_metrics['quantum_complexity']))
        
        # Alternativa: calcular a partir da decoerência e entropia
        decoherence = self.quantum_metrics.get('decoherence', 0.3)
        entropy = self.fractal_metrics.get('field_energy', 0.5) / 10.0
        
        return min(0.9, max(0.1, (decoherence + entropy) / 2))
    
    def _calculate_adaptation_rate(self) -> float:
        """Calcula a taxa de adaptação baseada nas métricas de feedback retrocausal"""
        base_rate = 0.01  # Valor padrão
        
        # Ajuste baseado na força do feedback retrocausal
        feedback_strength = self.retrocausal_metrics.get('feedback_strength', 0.5)
        
        # Maior feedback permite adaptação mais rápida
        rate = base_rate * (1 + feedback_strength)
        
        # Limitar entre 0.005 e 0.03
        return min(0.03, max(0.005, rate))
    
    def _calculate_risk_factor(self) -> float:
        """Calcula o fator de risco baseado nas métricas fractais"""
        base_risk = 0.5  # Valor padrão
        
        # Ajuste baseado no fator fractal
        fractal_factor = self.fractal_metrics.get('fractal_factor', 1.0)
        
        # Maior complexidade fractal sugere maior cautela
        risk = base_risk * fractal_factor
        
        # Limitar entre 0.3 e 0.8
        return min(0.8, max(0.3, risk)) 