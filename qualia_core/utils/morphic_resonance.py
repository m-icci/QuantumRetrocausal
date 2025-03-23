"""
Morphic Field System Implementation
Implements Sheldrake's morphic resonance theory in quantum context.
"""
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import logging
from pathlib import Path
import json
import hashlib
from datetime import datetime
from dataclasses import dataclass

from quantum.core.QUALIA.base_types import QuantumState

logger = logging.getLogger(__name__)

@dataclass
class MorphicField:
    """Campo mórfico que preserva e propaga padrões"""
    field_id: str
    quantum_signature: np.ndarray
    strength: float = 1.0
    coherence: float = 1.0
    stability: float = 1.0
    influence_radius: float = 0.8
    creation_time: datetime = datetime.now()
    last_update: datetime = datetime.now()
    affected_systems: List[str] = None
    evolution_history: List[Tuple[datetime, float]] = None
    merge_success_rate: float = 1.0  # Taxa de sucesso em merges
    pattern_frequency: Dict[str, int] = None  # Frequência de padrões

    def __post_init__(self):
        self.affected_systems = self.affected_systems or []
        self.evolution_history = self.evolution_history or [(self.creation_time, self.strength)]
        self.pattern_frequency = self.pattern_frequency or {}

    def update(self, state: QuantumState, interaction_strength: float = 0.1, stability_threshold: float = 1e-10):
        """
        Atualiza campo com novo estado

        Args:
            state: Estado quântico
            interaction_strength: Força da interação
            stability_threshold: Threshold for numerical stability
        """
        # Atualiza assinatura quântica
        old_signature = self.quantum_signature.copy()

        # Apply interaction with stability check
        interaction = np.clip(interaction_strength, stability_threshold, 1.0)
        self.quantum_signature = (1 - interaction) * self.quantum_signature + \
                               interaction * state.amplitudes

        # Normalize with stability threshold
        norm = np.linalg.norm(self.quantum_signature)
        if norm > stability_threshold:
            self.quantum_signature /= norm
        else:
            self.quantum_signature = np.zeros_like(self.quantum_signature)
            logger.warning(f"Field strength below stability threshold: {norm}")

        # Update metrics with stability checks
        overlap = np.vdot(self.quantum_signature, state.amplitudes)
        self.strength = np.clip(np.abs(overlap), stability_threshold, 1.0)

        self_overlap = np.vdot(self.quantum_signature, self.quantum_signature)
        self.coherence = np.clip(np.abs(self_overlap), stability_threshold, 1.0)

        # Detect patterns with stability
        pattern = self._extract_pattern(old_signature, self.quantum_signature, stability_threshold)
        if pattern:
            self.pattern_frequency[pattern] = self.pattern_frequency.get(pattern, 0) + 1

        self.last_update = datetime.now()
        self.evolution_history.append((datetime.now(), self.strength))

    def _extract_pattern(self, old_state: np.ndarray, new_state: np.ndarray, 
                        stability_threshold: float = 1e-10) -> Optional[str]:
        """Extracts pattern from state transition with stability checks"""
        try:
            # Calculate state difference
            delta = new_state - old_state

            # Find significant components with stability threshold
            significant = np.where(np.abs(delta) > stability_threshold)[0]

            if len(significant) > 0:
                # Create stable pattern signature
                pattern_components = []
                for idx in significant:
                    value = delta[idx]
                    if np.abs(value) > stability_threshold:
                        pattern_components.append(f"{idx}:{value:.2f}")

                if pattern_components:
                    return "_".join(pattern_components)

            return None

        except Exception as e:
            logger.error(f"Error extracting pattern: {str(e)}")
            return None

    def apply(self, state: QuantumState, stability_threshold: float = 1e-10) -> QuantumState:
        """
        Applies field to state with stability checks

        Args:
            state: State to modify
            stability_threshold: Numerical stability threshold

        Returns:
            Modified state
        """
        # Calculate influence with stability
        influence = np.clip(
            self.strength * self.stability * self.influence_radius,
            stability_threshold,
            1.0
        )

        # Apply field with stability check
        modified = (1 - influence) * state.amplitudes + \
                  influence * self.quantum_signature

        # Normalize output state
        norm = np.linalg.norm(modified)
        if norm > stability_threshold:
            modified /= norm
        else:
            logger.warning(f"Output state magnitude below threshold: {norm}")
            modified = np.zeros_like(modified)

        return QuantumState(modified)

class MorphicResonance:
    """
    Sistema de Ressonância Mórfica.
    Permite que padrões bem-sucedidos influenciem evolução futura.
    """

    def __init__(
        self,
        field_strength: float = 0.7,
        coherence_threshold: float = 0.6,
        influence_radius: float = 0.8,
        similarity_threshold: float = 0.9,
        adaptation_rate: float = 0.1,
        stability_threshold: float = 1e-10
    ):
        self.field_strength = field_strength
        self.coherence_threshold = coherence_threshold
        self.influence_radius = influence_radius
        self.similarity_threshold = similarity_threshold
        self.adaptation_rate = adaptation_rate
        self.stability_threshold = stability_threshold

        # Estado do sistema
        self.active_fields: Dict[str, MorphicField] = {}
        self.field_interactions: Dict[str, Dict[str, float]] = {}
        self.system_influences: Dict[str, List[str]] = {}
        self.merge_history: List[Tuple[str, str, bool]] = []

        # Configuração
        self.memory_dir = Path.home() / '.quantum' / 'morphic_memory'
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # Carrega campos existentes
        self._load_fields()

    def _update_interactions(self, field_id: str) -> None:
        """Update field interactions with stability"""
        if field_id not in self.field_interactions:
            self.field_interactions[field_id] = {}

        field = self.active_fields[field_id]

        for other_id, other_field in self.active_fields.items():
            if other_id != field_id:
                interaction = np.abs(np.vdot(
                    field.quantum_signature,
                    other_field.quantum_signature
                ))
                # Apply stability threshold
                interaction = max(interaction, self.stability_threshold)
                self.field_interactions[field_id][other_id] = interaction
                self.field_interactions[other_id][field_id] = interaction

    def _register_influence(self, field_id: str, system: str) -> None:
        """Register field influence on system"""
        if system not in self.system_influences:
            self.system_influences[system] = []

        if field_id not in self.system_influences[system]:
            self.system_influences[system].append(field_id)

    def create_field(
        self,
        quantum_state: QuantumState,
        source_system: str
    ) -> Optional[str]:
        """
        Creates new morphic field with stability

        Args:
            quantum_state: Base quantum state
            source_system: Source system

        Returns:
            Field ID if created
        """
        try:
            # Generate unique ID
            field_id = self._generate_field_id(quantum_state.amplitudes, source_system)

            # Check similar fields using adaptive threshold
            for existing_id, field in self.active_fields.items():
                similarity = self._calculate_field_similarity(
                    field.quantum_signature,
                    quantum_state.amplitudes
                )

                # Adjust threshold based on success history
                adjusted_threshold = max(
                    self.similarity_threshold * field.merge_success_rate,
                    self.stability_threshold
                )

                if similarity > adjusted_threshold:
                    success = self._merge_fields(existing_id, quantum_state, source_system)
                    self.merge_history.append((existing_id, field_id, success))

                    # Update success rate with stability
                    field.merge_success_rate = max(
                        (1 - self.adaptation_rate) * field.merge_success_rate +
                        self.adaptation_rate * (1.0 if success else 0.0),
                        self.stability_threshold
                    )

                    return existing_id

            # Create new field
            field = MorphicField(
                field_id=field_id,
                quantum_signature=quantum_state.amplitudes.copy(),
                influence_radius=self.influence_radius
            )

            self.active_fields[field_id] = field
            self._update_interactions(field_id)
            self._register_influence(field_id, source_system)

            # Save field
            self._save_field(field_id)

            return field_id

        except Exception as e:
            logger.error(f"Error creating field: {str(e)}")
            return None

    def apply_resonance(
        self,
        quantum_state: QuantumState,
        target_system: str,
        organic_patterns: Optional[List] = None,
        stability_threshold: Optional[float] = None
    ) -> QuantumState:
        """
        Applies morphic resonance to state with stability

        Args:
            quantum_state: State to modify
            target_system: Target system
            organic_patterns: Optional organic patterns
            stability_threshold: Optional stability override

        Returns:
            Resonant state
        """
        if stability_threshold is None:
            stability_threshold = self.stability_threshold

        try:
            if target_system not in self.system_influences:
                return quantum_state

            # Get relevant fields
            relevant_fields = [
                self.active_fields[field_id]
                for field_id in self.system_influences[target_system]
                if field_id in self.active_fields
            ]

            if not relevant_fields:
                return quantum_state

            # Apply fields in order of influence
            resonant_state = quantum_state
            for field in sorted(
                relevant_fields,
                key=lambda f: max(f.strength * f.stability, stability_threshold),
                reverse=True
            ):
                resonant_state = field.apply(resonant_state, stability_threshold)

            return resonant_state

        except Exception as e:
            logger.error(f"Error applying resonance: {str(e)}")
            return quantum_state

    def _generate_field_id(self, state: np.ndarray, source: str) -> str:
        """Generate unique field ID"""
        content = f"{state.tobytes()}{source}{datetime.now().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _calculate_field_similarity(
        self,
        state1: np.ndarray,
        state2: np.ndarray
    ) -> float:
        """Calculate similarity between states with stability"""
        try:
            similarity = abs(np.vdot(state1, state2))
            return max(similarity, self.stability_threshold)
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return self.stability_threshold

    def _merge_fields(
        self,
        field_id: str,
        new_state: QuantumState,
        source_system: str
    ) -> bool:
        """
        Combines new state into existing field with stability

        Returns:
            True if merge successful
        """
        try:
            field = self.active_fields[field_id]

            # Save previous state
            old_coherence = field.coherence
            old_strength = field.strength

            # Attempt merge with stability
            field.update(new_state, stability_threshold=self.stability_threshold)

            if source_system not in field.affected_systems:
                field.affected_systems.append(source_system)

            # Check merge success with stability thresholds
            success = (
                field.coherence >= old_coherence * 0.9 and
                field.coherence > self.stability_threshold and
                field.strength >= old_strength * 0.9 and
                field.strength > self.stability_threshold
            )

            self._update_interactions(field_id)
            self._register_influence(field_id, source_system)
            self._save_field(field_id)

            return success

        except Exception as e:
            logger.error(f"Error merging fields: {str(e)}")
            return False

    def _save_fields(self):
        """Save all fields"""
        for field_id in self.active_fields:
            self._save_field(field_id)

    def _save_field(self, field_id: str):
        """Save specific field"""
        try:
            field = self.active_fields[field_id]

            data = {
                'field_id': field.field_id,
                'quantum_signature': field.quantum_signature.tolist(),
                'strength': float(field.strength),
                'coherence': float(field.coherence),
                'stability': float(field.stability),
                'influence_radius': float(field.influence_radius),
                'creation_time': field.creation_time.isoformat(),
                'last_update': field.last_update.isoformat(),
                'affected_systems': field.affected_systems,
                'evolution_history': [
                    (t.isoformat(), float(s))
                    for t, s in field.evolution_history
                ],
                'merge_success_rate': float(field.merge_success_rate),
                'pattern_frequency': field.pattern_frequency
            }

            field_file = self.memory_dir / f"field_{field_id}.json"
            field_file.write_text(json.dumps(data, indent=2))

        except Exception as e:
            logger.error(f"Error saving field {field_id}: {str(e)}")

    def _load_fields(self):
        """Load saved fields"""
        field_files = list(self.memory_dir.glob('field_*.json'))

        for file in field_files:
            try:
                data = json.loads(file.read_text())

                field = MorphicField(
                    field_id=data['field_id'],
                    quantum_signature=np.array(data['quantum_signature']),
                    strength=float(data['strength']),
                    coherence=float(data['coherence']),
                    stability=float(data['stability']),
                    influence_radius=float(data['influence_radius']),
                    creation_time=datetime.fromisoformat(data['creation_time']),
                    last_update=datetime.fromisoformat(data['last_update']),
                    affected_systems=data['affected_systems'],
                    evolution_history=[
                        (datetime.fromisoformat(t), float(s))
                        for t, s in data['evolution_history']
                    ],
                    merge_success_rate=float(data.get('merge_success_rate', 1.0)),
                    pattern_frequency=data.get('pattern_frequency', {})
                )

                self.active_fields[field.field_id] = field
                logger.info(f"Loaded morphic field: {field.field_id}")

            except Exception as e:
                logger.error(f"Error loading field {file}: {str(e)}")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get system metrics with stability

        Returns:
            System metrics
        """
        try:
            total_strength = 0.0
            total_coherence = 0.0
            total_fields = len(self.active_fields)

            for field in self.active_fields.values():
                total_strength += max(field.strength, self.stability_threshold)
                total_coherence += max(field.coherence, self.stability_threshold)

            avg_strength = total_strength / total_fields if total_fields > 0 else 0.0
            avg_coherence = total_coherence / total_fields if total_fields > 0 else 0.0

            return {
                'num_fields': total_fields,
                'avg_strength': float(avg_strength),
                'avg_coherence': float(avg_coherence),
                'stability_threshold': float(self.stability_threshold)
            }

        except Exception as e:
            logger.error(f"Error getting metrics: {str(e)}")
            return {
                'error': str(e),
                'num_fields': 0,
                'avg_strength': 0.0,
                'avg_coherence': 0.0,
                'stability_threshold': float(self.stability_threshold)
            }

    def get_field_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get metrics for individual fields"""
        try:
            return {
                field_id: {
                    'strength': float(max(field.strength, self.stability_threshold)),
                    'coherence': float(max(field.coherence, self.stability_threshold)),
                    'stability': float(max(field.stability, self.stability_threshold))
                }
                for field_id, field in self.active_fields.items()
            }
        except Exception as e:
            logger.error(f"Error getting field metrics: {str(e)}")
            return {}

    def get_emergent_patterns(self) -> Dict[str, List[Tuple[str, int]]]:
        """Get emergent patterns with stability filtering"""
        try:
            patterns = {}
            for field_id, field in self.active_fields.items():
                # Filter patterns by frequency threshold
                significant_patterns = [
                    (pattern, count)
                    for pattern, count in field.pattern_frequency.items()
                    if count > 1  # Basic significance threshold
                ]
                if significant_patterns:
                    patterns[field_id] = significant_patterns
            return patterns
        except Exception as e:
            logger.error(f"Error getting emergent patterns: {str(e)}")
            return {}

    def update_evolution(self):
        """Atualiza evolução dos campos mórficos"""
        now = datetime.now()
        fields_to_remove = []

        for field_id, field in self.active_fields.items():
            # Calcula idade do campo
            age = (now - field.creation_time).total_seconds() / 3600  # Em horas

            # Atualiza força baseado em uso
            usage_factor = len(field.affected_systems) / 10
            strength_delta = 0.1 * usage_factor * np.exp(-age/24)

            field.strength = min(1.0, field.strength + strength_delta)
            field.stability *= 0.99  # Decai levemente

            # Remove campos fracos
            if field.strength < 0.1 or field.stability < 0.1:
                fields_to_remove.append(field_id)

        # Remove campos obsoletos
        for field_id in fields_to_remove:
            self._remove_field(field_id)

        # Salva campos atualizados
        self._save_fields()

    def get_field_metrics(self) -> Dict[str, Any]:
        """
        Retorna métricas dos campos

        Returns:
            Métricas por campo
        """
        metrics = {}

        for field_id, field in self.active_fields.items():
            metrics[field_id] = {
                'strength': field.strength,
                'coherence': field.coherence,
                'stability': field.stability,
                'age': (datetime.now() - field.creation_time).total_seconds(),
                'num_affected': len(field.affected_systems),
                'last_update': field.last_update.isoformat()
            }

        return metrics

    def get_emergent_patterns(self) -> Dict[str, List[Tuple[str, int]]]:
        """Retorna padrões emergentes no sistema"""
        patterns = {}

        for field_id, field in self.active_fields.items():
            patterns[field_id] = field.get_dominant_patterns()

        return patterns

    def get_merge_statistics(self) -> Dict[str, float]:
        """Retorna estatísticas de merge"""
        if not self.merge_history:
            return {}

        total_merges = len(self.merge_history)
        successful_merges = sum(1 for _, _, success in self.merge_history if success)

        return {
            'total_merges': total_merges,
            'successful_merges': successful_merges,
            'success_rate': successful_merges / total_merges,
            'average_threshold': np.mean([
                field.merge_success_rate * self.similarity_threshold
                for field in self.active_fields.values()
            ])
        }
    def _remove_field(self, field_id: str):
        """Remove campo mórfico"""
        if field_id in self.active_fields:
            del self.active_fields[field_id]

            if field_id in self.field_interactions:
                del self.field_interactions[field_id]
            for interactions in self.field_interactions.values():
                if field_id in interactions:
                    del interactions[field_id]

            for influences in self.system_influences.values():
                if field_id in influences:
                    influences.remove(field_id)

            # Remove arquivo do campo
            field_file = self.memory_dir / f"field_{field_id}.json"
            if field_file.exists():
                field_file.unlink()