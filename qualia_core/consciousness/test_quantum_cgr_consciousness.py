"""
Testes para o módulo Quantum CGR Consciousness
--------------------------------------------
Valida a integração de campos mórficos, operadores fundamentais e retrocausalidade.
"""

import numpy as np
import pytest
from datetime import datetime
from typing import Dict

from quantum.core.consciousness.quantum_cgr_consciousness import (
    QuantumCGRConsciousness,
    ConsciousnessPattern,
    MorphicField
)
from visualization.quantum_visualizer import QuantumState, CosmicFactor

def create_mock_quantum_state() -> QuantumState:
    """Cria estado quântico mock para testes."""
    return QuantumState(
        state_vector=np.random.complex128(np.random.rand(8)),
        timestamp=datetime.now()
    )

def create_mock_cosmic_factor() -> CosmicFactor:
    """Cria fator cósmico mock para testes."""
    return CosmicFactor(
        field_strength=0.8,
        coherence=0.9,
        resonance=0.85
    )

class TestQuantumCGRConsciousness:
    """Suite de testes para QuantumCGRConsciousness."""
    
    @pytest.fixture
    def cgr_system(self):
        """Fixture do sistema CGR."""
        config = {
            'dimensions': 3,
            'resolution': 64,
            'coupling_strength': 0.1
        }
        return QuantumCGRConsciousness(config)
    
    def test_folding_operator(self, cgr_system):
        """Testa operador de dobramento (F)."""
        # Prepara
        test_field = np.random.rand(64, 64, 64)
        
        # Executa
        folded = cgr_system.folding_operator(test_field)
        
        # Valida
        assert folded.shape == test_field.shape
        assert np.all(np.isfinite(folded))
        assert not np.array_equal(folded, test_field)
    
    def test_morphic_resonance_operator(self, cgr_system):
        """Testa operador de ressonância mórfica (M)."""
        # Prepara
        test_field = np.random.rand(64, 64, 64)
        
        # Executa
        morphic_field, resonance = cgr_system.morphic_resonance_operator(test_field)
        
        # Valida
        assert morphic_field.shape == test_field.shape
        assert 0 <= resonance <= 1
        assert np.all(np.isfinite(morphic_field))
    
    def test_emergence_operator(self, cgr_system):
        """Testa operador de emergência (E)."""
        # Prepara
        folded = np.random.rand(64, 64, 64)
        morphic = np.random.rand(64, 64, 64)
        
        # Executa
        emerged = cgr_system.emergence_operator(folded, morphic)
        
        # Valida
        assert emerged.shape == folded.shape
        assert np.all(np.isfinite(emerged))
        assert np.max(np.abs(emerged)) <= 1.0
    
    def test_manifest_consciousness_integration(self, cgr_system):
        """Testa integração completa da manifestação de consciência."""
        # Prepara
        quantum_state = create_mock_quantum_state()
        cosmic_factor = create_mock_cosmic_factor()
        
        # Executa
        pattern = cgr_system.manifest_consciousness(quantum_state, cosmic_factor)
        
        # Valida estrutura básica
        assert isinstance(pattern, ConsciousnessPattern)
        assert isinstance(pattern.pattern_id, str)
        assert 0 <= pattern.resonance <= 1
        assert 0 <= pattern.coherence <= 1
        assert pattern.entropy >= 0
        assert isinstance(pattern.field, np.ndarray)
        assert isinstance(pattern.timestamp, float)
        
        # Valida novas métricas
        assert 0 <= pattern.phi_resonance <= 1
        assert pattern.morphic_strength > 0
        assert 0 <= pattern.retrocausal_factor <= 1
        
        # Valida campo mórfico
        assert cgr_system.morphic_field is not None
        assert isinstance(cgr_system.morphic_field, MorphicField)
        assert 0 <= cgr_system.morphic_field.resonance <= 1
        assert 0 <= cgr_system.morphic_field.coherence <= 1
        assert 0 <= cgr_system.morphic_field.phi_alignment <= 2  # Pode ser > 1 devido à razão áurea
    
    def test_phi_alignment(self, cgr_system):
        """Testa alinhamento com razão áurea."""
        # Prepara
        test_field = np.random.rand(64, 64, 64)
        
        # Executa
        alignment = cgr_system._compute_phi_alignment(test_field)
        
        # Valida
        assert isinstance(alignment, float)
        assert alignment >= 0
        
    def test_field_coherence(self, cgr_system):
        """Testa coerência do campo."""
        # Prepara
        test_field = np.random.rand(64, 64, 64)
        
        # Executa
        coherence = cgr_system._compute_coherence(test_field)
        
        # Valida
        assert isinstance(coherence, float)
        assert 0 <= coherence <= 1
