"""
Test suite for Quantum Refactoring Simulator
"""
import numpy as np
import pytest
from typing import List, Dict, Any
from datetime import datetime

from quantum.core.state.quantum_state import QuantumState, ComplexAmplitude
from quantum.core.qtypes.quantum_pattern import QuantumPattern
from quantum.core.qtypes.pattern_types import PatternType
from quantum.core.qtypes.quantum_metrics import QuantumMetrics
from quantum.core.integration.simulations.quantum_refactoring_simulator import QuantumRefactoringSimulator

def create_test_quantum_state(dimension: int = 64) -> QuantumState:
    """Create a test quantum state"""
    # Create a normalized state vector
    vector = np.random.normal(0, 1, dimension) + 1j * np.random.normal(0, 1, dimension)
    vector = vector / np.linalg.norm(vector)
    return QuantumState(state_vector=vector)

def create_test_patterns(num_patterns: int = 3, dimension: int = 64) -> List[QuantumPattern]:
    """Create test quantum patterns"""
    patterns = []
    for i in range(num_patterns):
        state = create_test_quantum_state(dimension)
        pattern = QuantumPattern(
            pattern_id=f"test_pattern_{i}",
            pattern_type=PatternType.QUANTUM,
            state=state,
            confidence=0.9,
            timestamp=datetime.now()
        )
        patterns.append(pattern)
    return patterns

class TestQuantumRefactoringSimulator:
    """Test cases for QuantumRefactoringSimulator"""
    
    @pytest.fixture
    def simulator(self) -> QuantumRefactoringSimulator:
        """Create a simulator instance"""
        config = {
            'dimension': 64,
            'resonance_threshold': 0.7,
            'coherence_threshold': 0.6,
            'stability_threshold': 1e-10
        }
        return QuantumRefactoringSimulator(config)
    
    def test_simulator_initialization(self, simulator: QuantumRefactoringSimulator):
        """Test simulator initialization"""
        assert simulator is not None
        assert simulator.resonance_threshold == 0.7
        assert simulator.epsilon == 1e-10
        assert simulator.morphic_field is not None
        assert simulator.quantum_field is not None
        assert simulator.pattern_analyzer is not None
    
    def test_simulate_refactoring(self, simulator: QuantumRefactoringSimulator):
        """Test refactoring simulation"""
        # Create test data
        code_state = create_test_quantum_state()
        target_patterns = create_test_patterns()
        
        # Run simulation
        result = simulator.simulate_refactoring(code_state, target_patterns)
        
        # Verify result structure
        assert isinstance(result, dict)
        assert 'success' in result
        assert 'metrics' in result
        assert 'stability' in result
        
        # Verify metrics
        metrics = result['metrics']
        assert 'resonance' in metrics
        assert 'field' in metrics
        assert 'organic' in metrics
        assert 'morphic' in metrics
        assert 'framework' in metrics
        
        # Verify resonance metrics
        resonance = metrics['resonance']
        assert 0 <= resonance['average'] <= 1
        assert 0 <= resonance['maximum'] <= 1
        assert len(resonance['values']) == len(target_patterns)
        
        # Verify field metrics
        field = metrics['field']
        assert 0 <= field['coherence'] <= 1
        assert len(field['metrics']) == len(target_patterns)
        
        # Verify stability
        stability = result['stability']
        assert stability['valid_states'] <= stability['total_states']
        assert 0 <= stability['coherence_stability'] <= 1
    
    def test_analyze_code_patterns(self, simulator: QuantumRefactoringSimulator):
        """Test code pattern analysis"""
        # Create test state
        code_state = create_test_quantum_state()
        
        # Analyze patterns
        results = simulator.analyze_code_patterns(code_state)
        
        # Verify results
        assert isinstance(results, list)
        if results:  # If patterns were detected
            pattern = results[0]
            assert 'pattern_id' in pattern
            assert 'strength' in pattern
            assert 0 <= pattern['strength'] <= 1
    
    def test_error_handling(self, simulator: QuantumRefactoringSimulator):
        """Test error handling with invalid input"""
        # Create invalid state (zero vector)
        invalid_state = QuantumState(state_vector=np.zeros(64))
        target_patterns = create_test_patterns()
        
        # Run simulation with invalid state
        result = simulator.simulate_refactoring(invalid_state, target_patterns)
        
        # Verify error handling
        assert result['success'] is False
        assert 'error' in result
        assert result['metrics'] is None
