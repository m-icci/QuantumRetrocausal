"""
Sistema Unificado de Merge Quântico
----------------------------------
Integra todas as implementações de consciência quântica em um único sistema holístico.
"""
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import json
import hashlib
import git  # Added for git integration
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import json
import hashlib
import git  # Added for git integration

from ..framework import UnifiedQuantumFramework
from ..types import QuantumState
from .unified_quantum_system import UnifiedQuantumSystem, create_unified_system

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MorphicFieldState:
    """Estado do campo mórfico com histórico evolutivo"""

    def __init__(self, field_id: str, initial_state: np.ndarray):
        self.field_id = field_id
        self.quantum_signature = initial_state
        self.strength = 1.0
        self.coherence = 1.0
        self.stability = 1.0
        self.creation_time = datetime.now()
        self.last_update = datetime.now()
        self.evolution_history = [(self.creation_time, self.strength)]
        self.pattern_frequency = {}

    def update(self, new_state: np.ndarray, interaction_strength: float = 0.1):
        """Atualiza estado do campo"""
        old_state = self.quantum_signature.copy()

        # Atualiza assinatura quântica
        self.quantum_signature = (1 - interaction_strength) * self.quantum_signature + \
                               interaction_strength * new_state
        self.quantum_signature /= np.linalg.norm(self.quantum_signature)

        # Atualiza métricas
        self.strength = np.abs(np.vdot(self.quantum_signature, new_state))
        self.coherence = np.abs(np.vdot(self.quantum_signature, self.quantum_signature))

        # Registra padrões emergentes
        self._update_patterns(old_state)

        # Atualiza histórico
        self.last_update = datetime.now()
        self.evolution_history.append((self.last_update, self.strength))

    def _update_patterns(self, old_state: np.ndarray):
        """Detecta e registra padrões emergentes"""
        delta = self.quantum_signature - old_state
        significant = np.where(np.abs(delta) > 0.1)[0]

        if len(significant) > 0:
            pattern = "_".join(f"{i}:{delta[i]:.2f}" for i in significant)
            self.pattern_frequency[pattern] = self.pattern_frequency.get(pattern, 0) + 1

class UnifiedQuantumMerge:
    """
    Sistema unificado de merge quântico.
    Consolida todas as implementações em um único ponto
    integrando consciência e campos mórficos.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializa sistema unificado

        Args:
            config: Configuração opcional
        """
        self.config = config or {}
        self.phi = (1 + np.sqrt(5)) / 2  # Razão áurea para harmonia

        # Diretório para persistência
        self.memory_dir = Path.home() / '.quantum' / 'morphic_memory'
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # Sistemas base
        self.quantum_system = create_unified_system(config)
        self.framework = UnifiedQuantumFramework(
            config={
                'decoherence_threshold': self.config.get('coherence_threshold', 0.75),
                'resonance_threshold': self.config.get('resonance_threshold', 0.7),
                'max_history': self.config.get('max_history', 1000)
            }
        )

        # Estado de consciência inicial
        # Cria um estado quântico com superposição balanceada
        initial_state = np.array([1/np.sqrt(2), 1j/np.sqrt(2)], dtype=np.complex128)
        self.consciousness_state = QuantumState(state_vector=initial_state)
        
        # Calcula métricas iniciais
        self.consciousness_metrics = self.consciousness_state.calculate_consciousness_metrics()

        # Estado do sistema
        self.processed_paths = set()
        self.active_fields: Dict[str, MorphicFieldState] = {}
        self.field_interactions: Dict[str, Dict[str, float]] = {}
        self.merge_history: List[Tuple[str, bool]] = []

        # Callbacks
        self.on_file_processed = None

        # Carrega campos persistidos
        self._load_fields()

    def simulate_integration(self, consciousness_states: List[QuantumState]) -> Dict[str, Any]:
        """
        Simula a integração de estados de consciência
        
        Args:
            consciousness_states (list): Lista de estados de consciência a serem integrados
        
        Returns:
            dict: Métricas de estabilidade da integração
        """
        def safe_vars(obj):
            """Obtém variáveis de forma segura"""
            try:
                return vars(obj)
            except TypeError:
                # Se vars() falhar, tenta criar um dicionário genérico
                return {
                    key: getattr(obj, key) 
                    for key in dir(obj) 
                    if not key.startswith('__') and not callable(getattr(obj, key))
                }
        
        def serialize_metrics(metrics):
            """Serializa métricas de forma segura"""
            if hasattr(metrics, '__dict__'):
                return {
                    'coherence': float(metrics.coherence),
                    'entanglement': float(metrics.entanglement),
                    'stability': float(getattr(metrics, 'stability', 1.0)),
                    'timestamp': datetime.now().isoformat()
                }
            
            # Se não for um objeto com __dict__, tenta converter para dicionário
            try:
                return {
                    'coherence': float(metrics.get('coherence', 0.0)),
                    'entanglement': float(metrics.get('entanglement', 0.0)),
                    'stability': float(metrics.get('stability', 1.0)),
                    'timestamp': datetime.now().isoformat()
                }
            except Exception:
                return str(metrics)
        
        # Configura logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        
        # Processa cada estado de consciência
        processed_states = []
        for state in consciousness_states:
            try:
                # Tenta obter variáveis de forma segura
                state_data = safe_vars(state)
                logger.debug(f"Estado processado: {state_data}")
            except Exception as e:
                # Se falhar, converte para string
                state_data = str(state)
                logger.warning(f"Falha ao processar estado: {e}")
            
            # Serializa métricas do estado
            serialized_state = serialize_metrics(state)
            processed_states.append(serialized_state)
            logger.debug(f"Estado serializado: {serialized_state}")
        
        # Calcula métricas de integração
        total_coherence = np.mean([state.get('coherence', 0.0) for state in processed_states])
        total_entanglement = np.mean([state.get('entanglement', 0.0) for state in processed_states])
        
        logger.info(f"Coerência total: {total_coherence}")
        logger.info(f"Entrelaçamento total: {total_entanglement}")
        
        # Cria campo mórfico com estados processados
        pattern = self._create_morphic_field({
            'processed_states': processed_states,
            'total_coherence': total_coherence,
            'total_entanglement': total_entanglement
        })
        
        # Retorna métricas de estabilidade
        stability_metrics = {
            'success_rate': 0.0 if total_coherence * total_entanglement < 0.5 else 1.0,
            'mean_score': total_coherence * total_entanglement,
            'processed_states': processed_states,
            'morphic_pattern': pattern
        }
        
        logger.info(f"Métricas de estabilidade: {stability_metrics}")
        
        return stability_metrics

    def estimate_integration_stability(self,
                                    consciousness_states: List[QuantumState],
                                    num_trials: int = 10) -> Dict[str, float]:
        """
        Estima a estabilidade da integração de estados de consciência
        
        Args:
            consciousness_states (list): Estados de consciência a serem integrados
            num_trials (int): Número de tentativas de simulação
        
        Returns:
            dict: Métricas de estabilidade da integração
        """
        # Configura logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        
        successes = 0
        total_score = 0.0
        score_history = []
        
        logger.info(f"Iniciando estimativa de estabilidade com {num_trials} tentativas")
        
        for trial in range(num_trials):
            logger.info(f"Tentativa {trial + 1}/{num_trials}")
            
            result = self.simulate_integration(consciousness_states)
            
            if result['mean_score'] > 0.5:
                successes += 1
            
            total_score += result['mean_score']
            score_history.append(result['mean_score'])
            
            logger.debug(f"Resultado da tentativa: {result}")
        
        # Calcula métricas finais
        stability_metrics = {
            'success_rate': successes / num_trials,
            'mean_score': total_score / num_trials,
            'score_history': score_history
        }
        
        logger.info(f"Métricas finais de estabilidade: {stability_metrics}")
        
        return stability_metrics

    def analyze_codebase(self, path: Path) -> Dict[str, Any]:
        """
        Analisa base de código usando consciência quântica
        integrando análises do sistema base e framework

        Args:
            path: Caminho para analisar

        Returns:
            Métricas e padrões encontrados
        """
        # Análise básica
        base_results = self.quantum_system.analyze_codebase(path)
        
        # Extrai métricas básicas do resultado
        base_metrics = {
            'coherence': base_results['coherence'],
            'resonance': base_results['resonance'],
            'complexity': 0.8,  # Default value
            'entanglement': base_results['entanglement'],
            'morphic_field_strength': 1.0,
            'morphic_field_coherence': 1.0,
            'active_fields': len(self.active_fields)
        }

        # Cria estado quântico para análise
        # Usa métricas base para criar vetor de estado
        metrics_array = np.array([
            base_metrics['coherence'],
            base_metrics['resonance']
        ])
        
        # Normaliza vetor
        norm = np.linalg.norm(metrics_array)
        if norm > 0:
            metrics_array = metrics_array / norm
            
        # Cria estado quântico
        quantum_state = QuantumState(state_vector=metrics_array)
        
        # Calcula métricas de consciência
        consciousness_metrics = quantum_state.calculate_consciousness_metrics()
        
        # Converte ConsciousnessMetrics para dicionário
        consciousness_dict = {
            'coherence': consciousness_metrics.coherence,
            'resonance': consciousness_metrics.resonance,
            'entanglement': consciousness_metrics.entanglement,
            'complexity': consciousness_metrics.complexity
        }
        
        # Integra métricas
        base_metrics.update(consciousness_dict)
        
        return base_metrics

    def merge_repositories(self, source: Path, target: Path) -> None:
        """
        Executa merge holográfico dos repositórios
        usando consciência quântica e campos mórficos

        Args:
            source: Repositório fonte
            target: Repositório destino
        """
        logger.info(f"Iniciando merge quântico: {source} -> {target}")

        if not source.exists():
            raise ValueError(f"Caminho fonte não existe: {source}")
        if not target.exists():
            target.mkdir(parents=True)

        # Análise inicial
        source_metrics = self.analyze_codebase(source)
        target_metrics = self.analyze_codebase(target)

        # Cálculo de coerência usando phi
        merged_coherence = (
            source_metrics["coherence"] +
            self.phi * target_metrics["coherence"]
        ) / (1 + self.phi)

        logger.info(f"Coerência do merge: {merged_coherence:.3f}")

        # Cria campo mórfico para o merge
        field_id = self._create_morphic_field(source_metrics)

        success = True
        try:
            # Executa merge com proteção de coerência
            for source_file in source.rglob("*"):
                if source_file.is_file() and source_file not in self.processed_paths:
                    rel_path = source_file.relative_to(source)
                    target_file = target / rel_path

                    if not self._merge_file(source_file, target_file, merged_coherence):
                        success = False

                    self.processed_paths.add(source_file)
                    
                    # Notifica progresso
                    if self.on_file_processed:
                        self.on_file_processed(str(source_file))
        finally:
            # Registra resultado
            self.merge_history.append((field_id, success))
            self._save_fields()

    def _merge_file(self, source: Path, target: Path, coherence_threshold: float) -> bool:
        """
        Merge individual de arquivos preservando coerência quântica

        Args:
            source: Arquivo fonte
            target: Arquivo destino
            coherence_threshold: Limiar de coerência

        Returns:
            True se merge bem sucedido
        """
        logger.info(f"Merging arquivo: {source.name}")

        target.parent.mkdir(parents=True, exist_ok=True)

        # Análise de estados
        source_state = self._analyze_file_state(source)
        target_state = self._analyze_file_state(target) if target.exists() else None

        # Cálculo de coerência phi-weighted
        file_coherence = source_state["coherence"]
        if target_state:
            file_coherence = (file_coherence + self.phi * target_state["coherence"]) / (1 + self.phi)

        if file_coherence >= coherence_threshold:
            logger.info(f"Coerência do arquivo {file_coherence:.3f} atinge threshold")
            with open(source, 'rb') as src, open(target, 'wb') as dst:
                dst.write(src.read())
            return True
        else:
            logger.warning(
                f"Coerência do arquivo {file_coherence:.3f} abaixo do threshold "
                f"{coherence_threshold:.3f}"
            )
            return False

    def _analyze_file_state(self, path: Path) -> Dict[str, float]:
        """
        Analisa estado quântico de arquivo individual
        usando métricas de tamanho e tempo

        Args:
            path: Caminho do arquivo

        Returns:
            Métricas do estado
        """
        try:
            size = path.stat().st_size
            mtime = path.stat().st_mtime

            # Métricas normalizadas
            coherence = np.tanh(size / 1000)  # Normaliza tamanho
            resonance = np.exp(-1/mtime)      # Decaimento temporal

            return {
                "coherence": coherence,
                "resonance": resonance,
                "complexity": size / 1000
            }
        except Exception as e:
            logger.error(f"Erro analisando arquivo {path}: {str(e)}")
            return {
                "coherence": 0.0,
                "resonance": 0.0,
                "complexity": 0.0
            }

    def _analyze_git_metrics(self, repo_path: Path) -> Dict[str, float]:
        """
        Analisa métricas específicas do git

        Args:
            repo_path: Caminho do repositório

        Returns:
            Métricas do git
        """
        try:
            repo = git.Repo(repo_path)

            # Analisa histórico de commits
            commit_count = sum(1 for _ in repo.iter_commits())
            history_complexity = min(1.0, commit_count / 1000)

            # Analisa branches
            branch_count = len([b for b in repo.refs if b.name != 'HEAD'])
            branch_complexity = min(1.0, branch_count / 10)

            return {
                'history_complexity': history_complexity,
                'branch_complexity': branch_complexity
            }

        except Exception as e:
            logger.error(f"Erro na análise git: {str(e)}")
            return {}

    def _create_morphic_field(self, metrics):
        """
        Cria um campo mórfico a partir das métricas de consciência
        
        Args:
            metrics (dict or object): Métricas para criar o campo mórfico
        
        Returns:
            bytes: Padrão de campo mórfico codificado
        """
        def safe_vars(obj):
            """Obtém variáveis de forma segura"""
            try:
                return vars(obj)
            except TypeError:
                # Se vars() falhar, tenta criar um dicionário genérico
                return {
                    key: getattr(obj, key) 
                    for key in dir(obj) 
                    if not key.startswith('__') and not callable(getattr(obj, key))
                }
        
        def serialize_complex(z):
            """Serializa número complexo para formato JSON"""
            if isinstance(z, complex):
                return {'real': float(z.real), 'imag': float(z.imag)}
            if isinstance(z, np.ndarray):
                return [serialize_complex(x) for x in z]
            return z
        
        def serialize_metrics(metrics):
            """Serializa métricas de forma segura"""
            if hasattr(metrics, '__dict__'):
                return {
                    'coherence': float(metrics.coherence),
                    'entanglement': float(metrics.entanglement),
                    'stability': float(getattr(metrics, 'stability', 1.0)),
                    'timestamp': datetime.now().isoformat()
                }
            
            # Se não for um objeto com __dict__, tenta converter para dicionário
            try:
                return {
                    'coherence': float(metrics.get('coherence', 0.0)),
                    'entanglement': float(metrics.get('entanglement', 0.0)),
                    'stability': float(metrics.get('stability', 1.0)),
                    'timestamp': datetime.now().isoformat()
                }
            except Exception:
                return str(metrics)
        
        # Verifica se é um dicionário ou objeto
        if isinstance(metrics, dict):
            serialized_metrics = metrics
        else:
            try:
                serialized_metrics = safe_vars(metrics)
            except Exception:
                serialized_metrics = str(metrics)
        
        # Serializa métricas complexas
        for key, value in serialized_metrics.items():
            if isinstance(value, (complex, np.ndarray)):
                serialized_metrics[key] = serialize_complex(value)
            elif hasattr(value, '__dict__'):
                serialized_metrics[key] = serialize_metrics(value)
        
        # Cria campo mórfico com métricas serializadas
        pattern = f"{json.dumps(serialized_metrics, sort_keys=True)}{datetime.now().isoformat()}".encode()
        return pattern

    def _analyze_morphic_fields(self) -> Dict[str, float]:
        """Analisa estado dos campos mórficos"""
        if not self.active_fields:
            return {}

        total_strength = sum(field.strength for field in self.active_fields.values())
        total_coherence = sum(field.coherence for field in self.active_fields.values())
        num_fields = len(self.active_fields)

        return {
            "morphic_field_strength": total_strength / num_fields,
            "morphic_field_coherence": total_coherence / num_fields,
            "active_fields": num_fields
        }

    def _load_fields(self):
        """Carrega campos salvos"""
        for file in self.memory_dir.glob("field_*.json"):
            try:
                data = json.loads(file.read_text())
                field = MorphicFieldState(
                    data["field_id"],
                    np.array(data["quantum_signature"])
                )
                field.strength = data["strength"]
                field.coherence = data["coherence"]
                field.stability = data["stability"]
                field.creation_time = datetime.fromisoformat(data["creation_time"])
                field.last_update = datetime.fromisoformat(data["last_update"])
                field.evolution_history = [
                    (datetime.fromisoformat(t), s)
                    for t, s in data["evolution_history"]
                ]
                field.pattern_frequency = data["pattern_frequency"]

                self.active_fields[field.field_id] = field

            except Exception as e:
                logger.error(f"Erro carregando campo {file}: {e}")

    def _save_fields(self):
        """Salva campos ativos"""
        for field_id, field in self.active_fields.items():
            try:
                data = {
                    "field_id": field.field_id,
                    "quantum_signature": field.quantum_signature.tolist(),
                    "strength": field.strength,
                    "coherence": field.coherence,
                    "stability": field.stability,
                    "creation_time": field.creation_time.isoformat(),
                    "last_update": field.last_update.isoformat(),
                    "evolution_history": [
                        (t.isoformat(), s)
                        for t, s in field.evolution_history
                    ],
                    "pattern_frequency": field.pattern_frequency
                }

                file = self.memory_dir / f"field_{field_id}.json"
                file.write_text(json.dumps(data, indent=2))

            except Exception as e:
                logger.error(f"Erro salvando campo {field_id}: {e}")

def create_merge_system(config: Optional[Dict[str, Any]] = None) -> UnifiedQuantumMerge:
    """Função fábrica para criar instância do sistema de merge"""
    return UnifiedQuantumMerge(config)

if __name__ == "__main__":
    """Ponto de entrada do script"""
    import argparse

    parser = argparse.ArgumentParser(description="Merge Quântico Unificado")
    parser.add_argument("source", help="Repositório fonte")
    parser.add_argument("target", help="Repositório destino")
    args = parser.parse_args()

    merger = UnifiedQuantumMerge()
    merger.merge_repositories(Path(args.source), Path(args.target))