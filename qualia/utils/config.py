"""
Configuration management utilities for QUALIA Trading System
Implements quantum-aware configuration handling with validation
"""
import os
import json
import numpy as np
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from .logging import setup_logger

logger = setup_logger(__name__)

class QuantumValidationError(Exception):
    """Custom exception for quantum parameter validation"""
    pass

class QuantumConfig:
    """Configuration manager for quantum parameters with validation"""

    VALID_RANGES = {
        "dimension": (8, 512),  # Power of 2 recommended
        "planck_constant": (0.1, 10.0),
        "temperature": (1.0, 1000.0),
        "phi": (1.5, 1.7),  # Around golden ratio
        "coherence_threshold": (0.0, 1.0),
        "memory_depth": (10, 10000),
        "hawking_temperature": (0.1, 100.0),
        "resonance_threshold": (0.0, 1.0),
        "morphic_field_strength": (0.0, 2.0),
        "entanglement_weight": (0.0, 1.0),
        "consciousness_coupling": (0.0, 1.0),
        "quantum_decay_rate": (0.0, 0.1)
    }

    REQUIRED_POWER_OF_TWO = ["dimension"]
    NORMALIZED_PARAMETERS = ["coherence_threshold", "resonance_threshold", "entanglement_weight"]
    TEMPERATURE_DEPENDENT = ["hawking_temperature", "temperature"]

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.quantum_config_path = self.config_dir / "quantum_config.json"
        self.config = self._load_config()
        self._validate_all_parameters()

    def _load_config(self) -> Dict[str, Any]:
        """Load quantum configuration with validated defaults"""
        default_config = {
            "dimension": 64,  # Optimal for quantum market analysis
            "planck_constant": 1.0,  # Normalized units
            "temperature": 310.0,  # Biological consciousness temperature
            "phi": 1.618033988749895,  # Golden ratio for quantum resonance
            "coherence_threshold": 0.9,
            "memory_depth": 100,
            "hawking_temperature": 2.725,  # Cosmic background radiation temperature
            "resonance_threshold": 0.8,
            "morphic_field_strength": 1.0,
            "quantum_metrics": {
                "entanglement_weight": 0.7,
                "coherence_decay": 0.05,
                "consciousness_coupling": 0.8,
                "quantum_decay_rate": 0.01
            }
        }

        if self.quantum_config_path.exists():
            try:
                with open(self.quantum_config_path) as f:
                    loaded_config = json.load(f)
                    merged_config = {**default_config, **loaded_config}
                    return self._validate_config(merged_config)
            except Exception as e:
                logger.error(f"Failed to load quantum config: {e}")
                return default_config
        return default_config

    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration values against theoretical bounds"""
        validated = {}
        for key, value in config.items():
            if isinstance(value, dict):
                validated[key] = self._validate_config(value)
                continue

            if key in self.VALID_RANGES:
                min_val, max_val = self.VALID_RANGES[key]
                if not isinstance(value, (int, float)):
                    raise QuantumValidationError(f"Invalid type for {key}: expected number")
                if value < min_val or value > max_val:
                    logger.warning(
                        f"Value for {key} ({value}) outside valid range "
                        f"[{min_val}, {max_val}]. Using default."
                    )
                    value = self._load_config()[key]
            validated[key] = value
        return validated

    def _validate_all_parameters(self) -> None:
        """Validate all quantum parameters for consistency"""
        try:
            # Check power of 2 requirements
            for param in self.REQUIRED_POWER_OF_TWO:
                value = self.config.get(param)
                if value and value & (value - 1) != 0:
                    logger.warning(f"{param} should be power of 2 for optimal FFT")

            # Validate temperature-dependent parameters
            for param in self.TEMPERATURE_DEPENDENT:
                value = self.config.get(param)
                if value and value < 100:
                    logger.warning(f"{param} might be too low for quantum effects")

            # Validate normalized parameters
            for param in self.NORMALIZED_PARAMETERS:
                value = self.config.get(param)
                if value and (value < 0 or value > 1):
                    logger.warning(f"{param} should be normalized between 0 and 1")

            # Check quantum metric weights sum to 1
            metrics = self.config.get("quantum_metrics", {})
            weight_sum = sum(
                value for key, value in metrics.items() 
                if key.endswith('_weight')
            )
            if abs(weight_sum - 1.0) > 0.001:
                logger.warning("Quantum metric weights should sum to 1.0")

            # Validate quantum decay rates
            decay_rate = metrics.get("quantum_decay_rate", 0)
            if decay_rate > 0.05:
                logger.warning("High quantum decay rate may lead to instability")

        except Exception as e:
            logger.error(f"Parameter validation error: {e}")

    def save(self) -> bool:
        """Save current configuration with validation"""
        try:
            validated_config = self._validate_config(self.config)
            with open(self.quantum_config_path, 'w') as f:
                json.dump(validated_config, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save quantum config: {e}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with validation"""
        try:
            keys = key.split('.')
            value = self.config
            for k in keys:
                value = value.get(k, default)
                if value is None:
                    return default

            if keys[-1] in self.VALID_RANGES:
                min_val, max_val = self.VALID_RANGES[keys[-1]]
                if not isinstance(value, (int, float)):
                    logger.error(f"Invalid type for {key}")
                    return default
                if value < min_val or value > max_val:
                    logger.warning(f"Value for {key} outside valid range")
                    return default
            return value
        except Exception as e:
            logger.error(f"Error getting config value: {e}")
            return default

    def set(self, key: str, value: Any) -> None:
        """Set configuration value with validation"""
        try:
            keys = key.split('.')
            if len(keys) > 1:
                # Handle nested configuration
                current = self.config
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                final_key = keys[-1]
            else:
                final_key = key
                current = self.config

            if final_key in self.VALID_RANGES:
                min_val, max_val = self.VALID_RANGES[final_key]
                if not isinstance(value, (int, float)):
                    raise QuantumValidationError(f"Invalid type for {final_key}")
                if value < min_val or value > max_val:
                    raise QuantumValidationError(
                        f"Value for {final_key} outside valid range [{min_val}, {max_val}]"
                    )

            current[final_key] = value
            self.save()

        except Exception as e:
            logger.error(f"Error setting config value: {e}")
            raise

    def update(self, updates: Dict[str, Any]) -> None:
        """Batch update configuration with validation"""
        for key, value in updates.items():
            self.set(key, value)

    def reset_defaults(self) -> None:
        """Reset to default configuration"""
        self.config = self._load_config()
        self.save()