"""
Quantum Integration Simulations Package
"""
from .quantum_merge_simulator import QuantumMergeSimulator
from .quantum_refactoring_simulator import QuantumRefactoringSimulator
from .consciousness_integration_simulator import ConsciousnessIntegrationSimulator

class UnifiedSimulationExecutor:
    """Executes all simulations in a unified manner"""

    def __init__(self, config=None):
        """Initialize unified executor"""
        self.merge_sim = QuantumMergeSimulator(config)
        self.refactoring_sim = QuantumRefactoringSimulator(config)
        self.consciousness_sim = ConsciousnessIntegrationSimulator(config)

    def run_unified_simulation(self, **states):
        """Run all simulations with provided states"""
        results = {}

        if 'source_state' in states and 'target_state' in states:
            # Run merge simulation with optional git context
            results['merge'] = self.merge_sim.simulate_merge(
                states['source_state'],
                states['target_state'],
                states.get('git_context')
            )

            # Run merge stability estimation if requested
            if states.get('estimate_stability'):
                results['merge_stability'] = self.merge_sim.estimate_merge_stability(
                    states['source_state'],
                    states['target_state'],
                    git_context=states.get('git_context')
                )

            # Run merge tests if test cases provided
            if states.get('merge_test_cases'):
                results['merge_tests'] = self.merge_sim.run_merge_test(
                    states['merge_test_cases'],
                    git_context=states.get('git_context')
                )

        if 'code_state' in states and 'target_patterns' in states:
            results['refactoring'] = self.refactoring_sim.simulate_refactoring(
                states['code_state'],
                states['target_patterns']
            )

        if 'consciousness_states' in states:
            results['consciousness'] = self.consciousness_sim.simulate_integration(
                states['consciousness_states']
            )

        return results

__all__ = [
    'QuantumMergeSimulator',
    'QuantumRefactoringSimulator',
    'ConsciousnessIntegrationSimulator',
    'UnifiedSimulationExecutor'
]