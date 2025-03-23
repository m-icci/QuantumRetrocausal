"""
Testa inovações no sistema de ressonância mórfica.
Segue mantra: INVESTIGAR → INTEGRAR → INOVAR
"""

import numpy as np
import logging
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import json
from quantum.core.utils.morphic_resonance import MorphicResonance

# Configura logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QuantumState:
    """Estado quântico simplificado para testes"""
    amplitudes: np.ndarray
    
    def __post_init__(self):
        if not isinstance(self.amplitudes, np.ndarray):
            self.amplitudes = np.array(self.amplitudes)
        # Normaliza
        self.amplitudes = self.amplitudes / np.linalg.norm(self.amplitudes)

@dataclass
class QuantumCoherenceValidator:
    """Validador de coerência quântica"""
    threshold: float = 0.6
    
    def validate_quantum_state(self, state: np.ndarray) -> bool:
        """Valida coerência do estado"""
        # Verifica norma
        norm = np.linalg.norm(state)
        if not np.isclose(norm, 1.0, atol=1e-6):
            return False
            
        # Verifica valores complexos válidos
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            return False
            
        return True

def test_innovations():
    """Testa inovações do sistema mórfico"""
    logger.info("\nINVESTIGAR: Iniciando testes de inovação...")
    
    # Cria sistema mórfico
    resonance = MorphicResonance(
        field_strength=0.8,
        coherence_threshold=0.6,
        influence_radius=0.7,
        similarity_threshold=0.85,
        adaptation_rate=0.2
    )
    
    # 1. Teste de Detecção de Padrões
    logger.info("\nINTEGRAR: Testando detecção de padrões...")
    
    base = np.array([0.7, 0.5, 0.3, 0.2], dtype=complex)
    base = base / np.linalg.norm(base)
    
    # Cria padrão oscilante
    states = []
    for i in range(5):
        modified = base.copy()
        modified[0] += 0.1 * (-1)**i  # Alterna +/- 0.1
        modified[1] -= 0.1 * (-1)**i  # Alterna -/+ 0.1
        modified = modified / np.linalg.norm(modified)
        states.append(QuantumState(modified))
    
    # Aplica estados
    field_id = None
    for i, state in enumerate(states):
        field_id = resonance.create_field(state, f"test_{i}")
        
    # Verifica padrões
    patterns = resonance.get_emergent_patterns()
    if field_id in patterns:
        logger.info("Padrões emergentes detectados:")
        for pattern, freq in patterns[field_id]:
            logger.info(f"  {pattern}: {freq}x")
    
    # 2. Teste de Merge Adaptativo
    logger.info("\nINOVAR: Testando merge adaptativo...")
    
    # Cria estados similares
    states = []
    for i in range(10):
        noise = np.random.normal(0, 0.05, size=4)
        modified = base + noise
        modified = modified / np.linalg.norm(modified)
        states.append(QuantumState(modified))
    
    # Tenta merges
    for i, state in enumerate(states):
        field_id = resonance.create_field(state, f"merge_{i}")
    
    # Verifica estatísticas
    stats = resonance.get_merge_statistics()
    logger.info("\nEstatísticas de merge:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Resumo
    success = (
        len(patterns[field_id]) > 0 and  # Detectou padrões
        stats['success_rate'] > 0.5       # Merge eficiente
    )
    
    if success:
        logger.info("\n✨ Inovações validadas com sucesso!")
    else:
        logger.error("\n❌ Falha na validação das inovações")
        
    return success

if __name__ == "__main__":
    test_innovations()
