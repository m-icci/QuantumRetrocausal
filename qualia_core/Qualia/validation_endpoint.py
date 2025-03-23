"""
REST endpoint for QUALIA validation results
"""
from flask import jsonify
from .test_validation import run_validation_suite
import numpy as np

def register_validation_routes(app):
    @app.route('/api/qualia/validation', methods=['GET'])
    def get_validation_results():
        """Run validation suite and return results with analysis"""
        try:
            results = run_validation_suite()

            # Analyze coherence test results
            trace_values = results['coherence_test']['trace']
            min_eigenvals = results['coherence_test']['min_eigenvalue']
            entropy_history = results['coherence_test']['relative_entropy']
            entanglement_history = results['coherence_test']['entanglement']

            coherence_analysis = {
                'trace_preserved': np.allclose(trace_values, 1.0, atol=1e-6),
                'positive_definite': all(val >= 0 for val in min_eigenvals),
                'stability_metric': np.std(trace_values),
                'entanglement_rate': np.mean(np.diff(entanglement_history)),
                'entropy_flow': np.mean(np.diff(entropy_history))
            }

            # Analyze emergence patterns
            emergence_strength = results['emergence_test']['emergence_strength']
            pattern_stability = results['emergence_test']['pattern_stability']

            emergence_analysis = {
                'pattern_formation_rate': np.mean(np.diff(emergence_strength)),
                'stability_metric': np.mean(pattern_stability),
                'organization_trend': np.polyfit(
                    np.arange(len(emergence_strength)), 
                    emergence_strength, 
                    1
                )[0]
            }

            # Analyze retrocausality
            temporal_corr = results['retrocausality_test']['temporal_correlation']
            causal_strength = results['retrocausality_test']['causal_strength']

            retrocausal_analysis = {
                'temporal_coherence': np.mean(temporal_corr),
                'causal_strength_avg': np.mean(causal_strength),
                'temporal_stability': 1.0 - np.std(temporal_corr),
                'retrocausal_trend': np.polyfit(
                    np.arange(len(causal_strength)),
                    causal_strength,
                    1
                )[0]
            }

            return jsonify({
                'status': 'success',
                'data': results,
                'analysis': {
                    'coherence': coherence_analysis,
                    'emergence': emergence_analysis,
                    'retrocausality': retrocausal_analysis,
                    'summary': {
                        'quantum_stability': coherence_analysis['stability_metric'],
                        'pattern_strength': emergence_analysis['organization_trend'],
                        'temporal_influence': retrocausal_analysis['retrocausal_trend']
                    }
                }
            })

        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500