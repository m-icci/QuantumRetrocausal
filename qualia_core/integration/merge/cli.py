"""
Command-line interface for quantum merge operations
Following the mantra: INVESTIGAR → INTEGRAR → INOVAR
"""

from pathlib import Path
import logging
import click
from typing import Dict, Any

from .unified_quantum_merge import UnifiedQuantumMerge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_default_config() -> Dict[str, Any]:
    """Get optimized configuration with golden ratio"""
    return {
        'field_dimensions': 12,  # Balances pattern capture and noise
        'coherence_threshold': 0.75,  # Higher threshold for quality
        'resonance_threshold': 0.7,  # Increased resonance threshold
        'max_history': 1000,
        'phi_scale': 1.618  # Explicit golden ratio scale
    }

@click.command()
@click.argument('source', type=click.Path(exists=True))
@click.argument('target', type=click.Path())
@click.option('--config-file', '-c', type=click.Path(exists=True), help='Optional config file path')
def main(source: str, target: str, config_file: str = None):
    """Execute quantum merge between repositories"""
    config = get_default_config()
    
    if config_file:
        import json
        with open(config_file) as f:
            config.update(json.load(f))
    
    # Initialize merge system
    merge = UnifiedQuantumMerge(config)
    
    # INVESTIGAR: Analyze codebase
    logger.info('\n🔍 INVESTIGAR: Analyzing codebase...')
    result = merge.analyze_codebase(Path(source))
    
    logger.info('\nMetrics found:')
    logger.info(f"- Coherence: {result.get('coherence', 0):.2f}")
    logger.info(f"- Resonance: {result.get('resonance', 0):.2f}")
    
    # Verify metrics meet requirements
    coherence_ok = result.get('coherence', 0) >= config['coherence_threshold']
    resonance_ok = result.get('resonance', 0) >= config['resonance_threshold']
    
    if not (coherence_ok and resonance_ok):
        logger.warning('\n⚠️ WARNING: Metrics below threshold')
        logger.warning('Adjust parameters or review code before merge')
        return
    
    # INTEGRAR: Execute merge
    logger.info('\n🔄 INTEGRAR: Starting quantum merge...')
    merge.merge_repositories(Path(source), Path(target))
    logger.info('Merge completed!')
    
    # INOVAR: Verify result
    logger.info('\n💫 INOVAR: Verifying result...')
    final_result = merge.analyze_codebase(Path(target))
    
    logger.info('\nFinal metrics:')
    logger.info(f"- Coherence: {final_result.get('coherence', 0):.2f}")
    logger.info(f"- Resonance: {final_result.get('resonance', 0):.2f}")
    
    # Calculate improvements
    coherence_delta = final_result.get('coherence', 0) - result.get('coherence', 0)
    resonance_delta = final_result.get('resonance', 0) - result.get('resonance', 0)
    
    logger.info('\nMetric improvements:')
    logger.info(f"- Coherence: {'↑' if coherence_delta > 0 else '↓'} {abs(coherence_delta):.2f}")
    logger.info(f"- Resonance: {'↑' if resonance_delta > 0 else '↓'} {abs(resonance_delta):.2f}")

if __name__ == '__main__':
    main()
