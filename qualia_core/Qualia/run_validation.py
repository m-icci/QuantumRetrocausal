"""
QUALIA Validation Test Runner
Executes validation suite and generates comprehensive report
"""
import numpy as np
from validation_report import ValidationReportGenerator
from operators.meta_validation import MetaOperatorValidator

def run_qualia_validation(dim: int = 8):
    """
    Run complete QUALIA validation suite
    
    Args:
        dim (int): Dimension of quantum system
        
    Returns:
        Dict containing validation results and analysis
    """
    print("\n=== Starting QUALIA Validation ===\n")
    
    # Initialize test quantum field
    field = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    field = (field + field.conj().T) / 2  # Ensure Hermitian
    field = field / np.trace(field @ field.conj().T)  # Normalize
    
    # Generate comprehensive report
    report = ValidationReportGenerator.generate_full_report(field)
    
    # Print summary
    print("\n=== Validation Results ===\n")
    
    print("Test 1: Coerência e Normalização")
    coherence = report['tests']['coherence']
    for key, value in coherence['observacoes'].items():
        if isinstance(value, dict) and 'status' in value:
            print(f"- {key}: {value['status']}")
    
    print("\nTest 2: Emergência de Padrões")
    emergence = report['tests']['emergence']
    for key, value in emergence['observacoes'].items():
        print(f"- {key}: {value['description']}")
    
    print("\nTest 3: Retrocausalidade")
    retrocausal = report['tests']['retrocausality']
    for key, value in retrocausal['observacoes'].items():
        print(f"- {key}: {value['description']}")
    
    print("\n=== Meta-Operators Analysis ===\n")
    meta_ops = report['tests']['meta_operators']
    for op_name, metrics in meta_ops['resultados'].items():
        print(f"{op_name.capitalize()}:")
        for metric, value in metrics.items():
            if metric != 'status':
                print(f"- {metric}: {value:.4f}")
        print(f"Status: {metrics['status']}\n")
    
    print("\n=== Validation Complete ===\n")
    return report

if __name__ == "__main__":
    run_qualia_validation()
