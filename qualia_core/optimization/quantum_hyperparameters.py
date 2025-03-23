from typing import Dict


def optimize_consciousness_factor(orchestrator) -> float:
    current_factor = orchestrator.black_hole_field.consciousness_factor
    hashrate = orchestrator.mining_metrics.get('hashrate', 0.0)
    new_factor = current_factor * (1 + (hashrate / 1000))
    return max(0.5, min(new_factor, 1.2))
