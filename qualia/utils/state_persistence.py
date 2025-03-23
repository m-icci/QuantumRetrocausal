"""
State persistence utilities for QUALIA Trading System
Implements quantum state serialization and holographic pattern backup
"""
import os
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from .logging import setup_logger
import json

logger = setup_logger(__name__)

class QuantumStateValidationError(Exception):
    """Custom exception for quantum state validation errors"""
    pass

class QuantumStatePersistence:
    """Handles persistence of quantum states and holographic patterns"""

    def __init__(self, state_dir: str = "quantum_states"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for different types of data
        self.holographic_dir = self.state_dir / "holographic_patterns"
        self.holographic_dir.mkdir(exist_ok=True)

        self.metrics_dir = self.state_dir / "state_metrics"
        self.metrics_dir.mkdir(exist_ok=True)

        self.morphic_dir = self.state_dir / "morphic_fields"
        self.morphic_dir.mkdir(exist_ok=True)

    def validate_quantum_state(self, state: np.ndarray) -> bool:
        """
        Validate quantum state properties

        Args:
            state: Quantum state matrix

        Returns:
            True if valid, raises QuantumStateValidationError otherwise
        """
        try:
            # Check if state is a complex matrix
            if not isinstance(state, np.ndarray) or state.dtype.kind != 'c':
                raise QuantumStateValidationError("State must be a complex matrix")

            # Check normalization
            trace_norm = np.abs(np.trace(state @ state.conj().T))
            if not np.isclose(trace_norm, 1.0, atol=1e-6):
                raise QuantumStateValidationError(
                    f"State not properly normalized. Trace norm: {trace_norm}"
                )

            # Check Hermiticity
            if not np.allclose(state, state.conj().T):
                raise QuantumStateValidationError("State matrix not Hermitian")

            # Check positive semi-definiteness
            eigenvals = np.linalg.eigvalsh(state)
            if np.any(eigenvals < -1e-10):  # Allow for numerical errors
                raise QuantumStateValidationError("State matrix not positive semi-definite")

            return True

        except Exception as e:
            logger.error(f"State validation error: {str(e)}")
            raise QuantumStateValidationError(str(e))

    def _process_state_data(self, state_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process state data for serialization with validation"""
        processed = {}
        for key, value in state_data.items():
            if isinstance(value, np.ndarray):
                # Validate quantum states before processing
                if key.endswith('_state'):
                    self.validate_quantum_state(value)
                processed[key] = {
                    'type': 'ndarray',
                    'data': value.tolist(),
                    'dtype': str(value.dtype),
                    'shape': value.shape
                }
            else:
                processed[key] = value
        return processed

    def save_state(
        self,
        state_data: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None,
        patterns: Optional[List[Tuple[np.ndarray, float]]] = None,
        morphic_field: Optional[np.ndarray] = None,
        name: str = "quantum_state"
    ) -> Optional[str]:
        """
        Save quantum state with associated metrics and patterns

        Args:
            state_data: Dictionary containing quantum state and metrics
            metrics: Optional performance metrics
            patterns: Optional list of (pattern, resonance) tuples
            morphic_field: Optional morphic field matrix
            name: Identifier for the state

        Returns:
            Path to saved state file or None if save failed
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{name}_{timestamp}.qstate"
            filepath = self.state_dir / filename

            # Process and validate main state data
            processed_data = self._process_state_data(state_data)

            # Save metrics if provided
            if metrics:
                metrics_file = self.metrics_dir / f"{name}_{timestamp}_metrics.json"
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=2)
                processed_data['metrics_file'] = str(metrics_file)

            # Save holographic patterns if provided
            if patterns:
                patterns_file = self.holographic_dir / f"{name}_{timestamp}_patterns.npz"
                patterns_array = np.array([p[0] for p in patterns])
                resonances = np.array([p[1] for p in patterns])
                np.savez(
                    patterns_file,
                    patterns=patterns_array,
                    resonances=resonances,
                    timestamp=timestamp
                )
                processed_data['patterns_file'] = str(patterns_file)

            # Save morphic field if provided
            if morphic_field is not None:
                morphic_file = self.morphic_dir / f"{name}_{timestamp}_morphic.npy"
                np.save(morphic_file, morphic_field)
                processed_data['morphic_file'] = str(morphic_file)

            # Save main state file
            with open(filepath, 'wb') as f:
                pickle.dump(processed_data, f)

            logger.info(f"Saved quantum state to {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to save quantum state: {e}")
            return None

    def load_state(
        self,
        filepath: str,
        load_patterns: bool = True,
        load_morphic: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Load quantum state with associated data

        Args:
            filepath: Path to state file
            load_patterns: Whether to load associated holographic patterns
            load_morphic: Whether to load morphic field data

        Returns:
            Dictionary containing loaded state or None if load failed
        """
        try:
            with open(filepath, 'rb') as f:
                processed_data = pickle.load(f)

            # Convert lists back to numpy arrays
            state_data = {}
            for key, value in processed_data.items():
                if isinstance(value, dict) and value.get('type') == 'ndarray':
                    state_data[key] = np.array(
                        value['data'],
                        dtype=value['dtype']
                    )
                    # Validate quantum states after loading
                    if key.endswith('_state'):
                        self.validate_quantum_state(state_data[key])
                else:
                    state_data[key] = value

            # Load metrics if available
            metrics_file = processed_data.get('metrics_file')
            if metrics_file and Path(metrics_file).exists():
                with open(metrics_file) as f:
                    state_data['metrics'] = json.load(f)

            # Load patterns if requested and available
            patterns_file = processed_data.get('patterns_file')
            if load_patterns and patterns_file and Path(patterns_file).exists():
                with np.load(patterns_file) as data:
                    patterns = data['patterns']
                    resonances = data['resonances']
                    state_data['patterns'] = list(zip(patterns, resonances))
                    state_data['patterns_timestamp'] = data.get('timestamp')

            # Load morphic field if requested and available
            morphic_file = processed_data.get('morphic_file')
            if load_morphic and morphic_file and Path(morphic_file).exists():
                state_data['morphic_field'] = np.load(morphic_file)

            return state_data

        except Exception as e:
            logger.error(f"Failed to load quantum state: {e}")
            return None

    def compute_resonance_metrics(
        self, 
        patterns: List[Tuple[np.ndarray, float]]
    ) -> Dict[str, float]:
        """
        Compute resonance metrics for holographic patterns

        Args:
            patterns: List of (pattern, resonance) tuples

        Returns:
            Dictionary of resonance metrics
        """
        try:
            resonances = np.array([r for _, r in patterns])
            return {
                'mean_resonance': float(np.mean(resonances)),
                'std_resonance': float(np.std(resonances)),
                'max_resonance': float(np.max(resonances)),
                'min_resonance': float(np.min(resonances)),
                'pattern_count': len(patterns)
            }
        except Exception as e:
            logger.error(f"Failed to compute resonance metrics: {e}")
            return {}

    def list_states(
        self,
        pattern: Optional[str] = None,
        include_metrics: bool = False,
        include_patterns: bool = False
    ) -> List[Dict[str, Any]]:
        """List saved quantum states with optional filtering"""
        try:
            files = list(self.state_dir.glob(f"{pattern or '*'}.qstate"))
            states = []

            for file in sorted(files, key=os.path.getmtime, reverse=True):
                state_info = {
                    'filepath': str(file),
                    'timestamp': datetime.fromtimestamp(file.stat().st_mtime)
                }

                if include_metrics or include_patterns:
                    state_data = self.load_state(
                        str(file),
                        load_patterns=include_patterns
                    )
                    if state_data:
                        if include_metrics and 'metrics' in state_data:
                            state_info['metrics'] = state_data['metrics']
                        if include_patterns and 'patterns' in state_data:
                            state_info['pattern_count'] = len(state_data['patterns'])

                states.append(state_info)

            return states

        except Exception as e:
            logger.error(f"Failed to list quantum states: {e}")
            return []

    def cleanup_old_states(
        self,
        max_age_days: int = 30,
        keep_patterns: bool = True,
        keep_morphic: bool = True
    ) -> int:
        """
        Remove states older than specified age

        Args:
            max_age_days: Maximum age in days
            keep_patterns: Whether to keep holographic patterns
            keep_morphic: Whether to keep morphic fields

        Returns:
            Number of removed states
        """
        try:
            cutoff = datetime.now().timestamp() - (max_age_days * 86400)
            removed = 0

            for filepath in self.state_dir.glob("*.qstate"):
                if filepath.stat().st_mtime < cutoff:
                    # Load state to check for associated files
                    state_data = self.load_state(str(filepath), False, False)

                    if state_data:
                        # Remove metrics file
                        metrics_file = state_data.get('metrics_file')
                        if metrics_file:
                            Path(metrics_file).unlink(missing_ok=True)

                        # Remove patterns file if not keeping
                        if not keep_patterns:
                            patterns_file = state_data.get('patterns_file')
                            if patterns_file:
                                Path(patterns_file).unlink(missing_ok=True)

                        # Remove morphic field file if not keeping
                        if not keep_morphic:
                            morphic_file = state_data.get('morphic_file')
                            if morphic_file:
                                Path(morphic_file).unlink(missing_ok=True)

                    filepath.unlink()
                    removed += 1

            logger.info(f"Removed {removed} old quantum states")
            return removed

        except Exception as e:
            logger.error(f"Failed to cleanup old states: {e}")
            return 0