#!/usr/bin/env python3
"""
StateSync Service
================
Microserviço assíncrono que sincroniza o estado entre QUALIAEngine, 
HelixController e outros módulos externos (como analistas humanos ou bots de hedge).

Este serviço permite que múltiplas instâncias do QUALIA colaborem
compartilhando estados, insights e decisões em tempo real.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Callable, Union, Tuple

import numpy as np

logger = logging.getLogger("state_sync")

class ComponentType(Enum):
    """Tipos de componentes que podem participar da sincronização de estado."""
    QUALIA_ENGINE = "qualia_engine"
    HELIX_CONTROLLER = "helix_controller"
    SPECTRA_STRATEGY = "spectra_strategy"
    EXTERNAL_ANALYST = "external_analyst"
    HEDGE_BOT = "hedge_bot"
    SEMANTIC_INTERPRETER = "semantic_interpreter"
    WAVE_STRATEGY = "wave_strategy"
    LSTM_PREDICTOR = "lstm_predictor"
    CUSTOM = "custom"

class StateType(Enum):
    """Tipos de estados que podem ser sincronizados."""
    MARKET_DATA = "market_data"
    TRADING_SIGNAL = "trading_signal"
    QUANTUM_INSIGHT = "quantum_insight"
    FRACTAL_PATTERN = "fractal_pattern"
    DECISION = "decision"
    PERFORMANCE_METRIC = "performance_metric"
    EXTERNAL_FEEDBACK = "external_feedback"
    PREDICTION = "prediction"
    CONFIGURATION = "configuration"
    CUSTOM = "custom"

class StateEntry:
    """Entrada de estado para sincronização."""
    
    def __init__(
        self, 
        component_type: ComponentType,
        state_type: StateType,
        data: Any,
        source_id: str,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None  # Tempo de vida em segundos
    ):
        self.id = str(uuid.uuid4())
        self.component_type = component_type
        self.state_type = state_type
        self.data = data
        self.source_id = source_id
        self.timestamp = timestamp or time.time()
        self.metadata = metadata or {}
        self.ttl = ttl
        self.expiry = (self.timestamp + ttl) if ttl else None
    
    def is_expired(self) -> bool:
        """Verifica se a entrada de estado expirou."""
        if self.expiry is None:
            return False
        return time.time() > self.expiry
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte a entrada de estado para um dicionário."""
        result = {
            "id": self.id,
            "component_type": self.component_type.value,
            "state_type": self.state_type.value,
            "source_id": self.source_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
        
        # Tenta serializar os dados da forma mais adequada
        if isinstance(self.data, (dict, list, str, int, float, bool)) or self.data is None:
            result["data"] = self.data
        elif isinstance(self.data, np.ndarray):
            result["data"] = self.data.tolist()
            result["metadata"]["data_type"] = "numpy.ndarray"
            result["metadata"]["dtype"] = str(self.data.dtype)
            result["metadata"]["shape"] = list(self.data.shape)
        else:
            # Para tipos complexos, armazena uma representação em string
            result["data"] = str(self.data)
            result["metadata"]["data_type"] = type(self.data).__name__
        
        if self.expiry:
            result["expiry"] = self.expiry
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateEntry':
        """Cria uma entrada de estado a partir de um dicionário."""
        # Reconstruir objetos especiais se necessário
        parsed_data = data["data"]
        metadata = data.get("metadata", {})
        
        if metadata.get("data_type") == "numpy.ndarray":
            parsed_data = np.array(parsed_data, dtype=metadata.get("dtype", None))
        
        instance = cls(
            component_type=ComponentType(data["component_type"]),
            state_type=StateType(data["state_type"]),
            data=parsed_data,
            source_id=data["source_id"],
            timestamp=data["timestamp"],
            metadata=metadata
        )
        
        instance.id = data["id"]
        if "expiry" in data:
            instance.expiry = data["expiry"]
            instance.ttl = data["expiry"] - data["timestamp"]
            
        return instance

class StateChangeCallback:
    """Callback para notificação de mudanças de estado."""
    
    def __init__(
        self,
        callback: Callable[[StateEntry], None],
        component_types: Optional[Set[ComponentType]] = None,
        state_types: Optional[Set[StateType]] = None,
        source_ids: Optional[Set[str]] = None,
        one_time: bool = False
    ):
        self.id = str(uuid.uuid4())
        self.callback = callback
        self.component_types = component_types
        self.state_types = state_types
        self.source_ids = source_ids
        self.one_time = one_time
        self.triggered = False
    
    def matches(self, state_entry: StateEntry) -> bool:
        """Verifica se a entrada de estado corresponde aos critérios do callback."""
        if self.triggered and self.one_time:
            return False
            
        matches = True
        
        if self.component_types and state_entry.component_type not in self.component_types:
            matches = False
            
        if self.state_types and state_entry.state_type not in self.state_types:
            matches = False
            
        if self.source_ids and state_entry.source_id not in self.source_ids:
            matches = False
            
        return matches
    
    def execute(self, state_entry: StateEntry) -> None:
        """Executa o callback."""
        if self.matches(state_entry):
            self.callback(state_entry)
            if self.one_time:
                self.triggered = True

class StateSyncService:
    """
    Serviço para sincronização de estado entre componentes do QUALIA.
    
    Implementa um barramento de eventos assíncrono que permite que diferentes
    componentes compartilhem estados e se comuniquem.
    """
    
    def __init__(
        self,
        instance_id: Optional[str] = None,
        storage_path: Optional[str] = None,
        enable_persistence: bool = True,
        cleanup_interval: int = 300  # 5 minutos
    ):
        self.instance_id = instance_id or str(uuid.uuid4())
        self.storage_path = Path(storage_path) if storage_path else Path("./state_storage")
        self.enable_persistence = enable_persistence
        self.cleanup_interval = cleanup_interval
        
        # Dicionário de estados, indexado por component_type > state_type > id
        self._state_store: Dict[str, Dict[str, Dict[str, StateEntry]]] = {}
        self._callbacks: List[StateChangeCallback] = []
        
        # Lock para sincronização de acesso ao state_store
        self._lock = asyncio.Lock()
        
        # Flag para controle da tarefa de limpeza
        self._running = False
        self._cleanup_task = None
        
        if enable_persistence:
            self._ensure_storage_path()
    
    def _ensure_storage_path(self) -> None:
        """Garante que o diretório de armazenamento existe."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    async def start(self) -> None:
        """Inicia o serviço de sincronização."""
        self._running = True
        
        # Carrega estados persistidos, se disponíveis
        if self.enable_persistence:
            await self._load_persisted_states()
        
        # Inicia tarefa de limpeza de estados expirados
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info(f"StateSyncService iniciado com ID {self.instance_id}")
    
    async def stop(self) -> None:
        """Para o serviço de sincronização."""
        self._running = False
        
        if self._cleanup_task:
            await self._cleanup_task
            self._cleanup_task = None
        
        # Persiste estados antes de parar
        if self.enable_persistence:
            await self._persist_states()
        
        logger.info(f"StateSyncService {self.instance_id} parado.")
    
    async def publish_state(
        self,
        component_type: Union[ComponentType, str],
        state_type: Union[StateType, str],
        data: Any,
        source_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None
    ) -> StateEntry:
        """
        Publica uma entrada de estado para sincronização.
        
        Args:
            component_type: Tipo do componente
            state_type: Tipo do estado
            data: Dados a serem publicados
            source_id: ID da fonte dos dados
            metadata: Metadados associados aos dados
            ttl: Tempo de vida em segundos
            
        Returns:
            Entrada de estado publicada
        """
        # Converte strings para enums se necessário
        if isinstance(component_type, str):
            component_type = ComponentType(component_type)
        
        if isinstance(state_type, str):
            state_type = StateType(state_type)
        
        # Cria a entrada de estado
        state_entry = StateEntry(
            component_type=component_type,
            state_type=state_type,
            data=data,
            source_id=source_id,
            metadata=metadata,
            ttl=ttl
        )
        
        # Armazena a entrada no state_store
        async with self._lock:
            comp_type = component_type.value
            st_type = state_type.value
            
            if comp_type not in self._state_store:
                self._state_store[comp_type] = {}
                
            if st_type not in self._state_store[comp_type]:
                self._state_store[comp_type][st_type] = {}
                
            self._state_store[comp_type][st_type][state_entry.id] = state_entry
        
        # Notifica callbacks
        await self._notify_callbacks(state_entry)
        
        # Persiste o estado se necessário
        if self.enable_persistence:
            await self._persist_single_state(state_entry)
        
        return state_entry
    
    async def get_state(
        self,
        component_type: Union[ComponentType, str],
        state_type: Union[StateType, str],
        state_id: Optional[str] = None,
        source_id: Optional[str] = None,
        latest_only: bool = True
    ) -> Union[Optional[StateEntry], List[StateEntry]]:
        """
        Obtém estados do state_store.
        
        Args:
            component_type: Tipo do componente
            state_type: Tipo do estado
            state_id: ID específico do estado (opcional)
            source_id: ID da fonte (opcional)
            latest_only: Se True, retorna apenas a entrada mais recente
            
        Returns:
            Entrada de estado ou lista de entradas de estado
        """
        # Converte strings para enums se necessário
        if isinstance(component_type, str):
            component_type = ComponentType(component_type)
        
        if isinstance(state_type, str):
            state_type = StateType(state_type)
        
        comp_type = component_type.value
        st_type = state_type.value
        
        async with self._lock:
            # Verifica se os níveis existem no state_store
            if comp_type not in self._state_store:
                return None if latest_only else []
                
            if st_type not in self._state_store[comp_type]:
                return None if latest_only else []
            
            # Caso específico: busca por ID
            if state_id:
                if state_id in self._state_store[comp_type][st_type]:
                    entry = self._state_store[comp_type][st_type][state_id]
                    if not entry.is_expired():
                        return entry
                return None
            
            # Filtra entradas por source_id e não expiradas
            entries = [
                entry for entry in self._state_store[comp_type][st_type].values()
                if not entry.is_expired() and (source_id is None or entry.source_id == source_id)
            ]
            
            if not entries:
                return None if latest_only else []
            
            # Ordenação por timestamp (mais recente primeiro)
            entries.sort(key=lambda x: x.timestamp, reverse=True)
            
            return entries[0] if latest_only else entries
    
    def register_callback(
        self,
        callback: Callable[[StateEntry], None],
        component_types: Optional[List[Union[ComponentType, str]]] = None,
        state_types: Optional[List[Union[StateType, str]]] = None,
        source_ids: Optional[List[str]] = None,
        one_time: bool = False
    ) -> str:
        """
        Registra um callback para mudanças de estado.
        
        Args:
            callback: Função a ser chamada quando o estado mudar
            component_types: Lista de tipos de componentes para filtrar
            state_types: Lista de tipos de estados para filtrar
            source_ids: Lista de IDs de fonte para filtrar
            one_time: Se True, o callback será removido após a primeira execução
            
        Returns:
            ID do callback registrado
        """
        # Converte strings para enums se necessário
        if component_types:
            component_types_set = set()
            for ct in component_types:
                if isinstance(ct, str):
                    component_types_set.add(ComponentType(ct))
                else:
                    component_types_set.add(ct)
        else:
            component_types_set = None
            
        if state_types:
            state_types_set = set()
            for st in state_types:
                if isinstance(st, str):
                    state_types_set.add(StateType(st))
                else:
                    state_types_set.add(st)
        else:
            state_types_set = None
            
        source_ids_set = set(source_ids) if source_ids else None
        
        # Cria o objeto de callback
        cb = StateChangeCallback(
            callback=callback,
            component_types=component_types_set,
            state_types=state_types_set,
            source_ids=source_ids_set,
            one_time=one_time
        )
        
        self._callbacks.append(cb)
        return cb.id
    
    def unregister_callback(self, callback_id: str) -> bool:
        """
        Remove um callback registrado.
        
        Args:
            callback_id: ID do callback a ser removido
            
        Returns:
            True se o callback foi removido, False se não foi encontrado
        """
        for i, cb in enumerate(self._callbacks):
            if cb.id == callback_id:
                self._callbacks.pop(i)
                return True
        return False
    
    async def _notify_callbacks(self, state_entry: StateEntry) -> None:
        """Notifica todos os callbacks registrados sobre a mudança de estado."""
        for cb in list(self._callbacks):  # Lista para permitir modificação durante iteração
            if cb.matches(state_entry):
                try:
                    # Executa o callback - que pode ser síncrono ou assíncrono
                    result = cb.callback(state_entry)
                    if asyncio.iscoroutine(result):
                        await result
                    
                    # Remove o callback se for one_time
                    if cb.one_time:
                        self._callbacks.remove(cb)
                except Exception as e:
                    logger.error(f"Erro ao executar callback {cb.id}: {e}")
    
    async def _cleanup_loop(self) -> None:
        """Loop para limpeza de estados expirados."""
        try:
            while self._running:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired_states()
        except asyncio.CancelledError:
            logger.debug("Tarefa de limpeza de estados cancelada.")
        except Exception as e:
            logger.error(f"Erro no loop de limpeza de estados: {e}")
    
    async def _cleanup_expired_states(self) -> None:
        """Remove estados expirados do state_store."""
        async with self._lock:
            for comp_type in list(self._state_store.keys()):
                for st_type in list(self._state_store[comp_type].keys()):
                    for state_id in list(self._state_store[comp_type][st_type].keys()):
                        state_entry = self._state_store[comp_type][st_type][state_id]
                        if state_entry.is_expired():
                            del self._state_store[comp_type][st_type][state_id]
                            
                            # Remove arquivos persistidos
                            if self.enable_persistence:
                                file_path = self._get_state_file_path(state_entry)
                                if file_path.exists():
                                    file_path.unlink()
                    
                    # Remove dicionários vazios
                    if not self._state_store[comp_type][st_type]:
                        del self._state_store[comp_type][st_type]
                
                if not self._state_store[comp_type]:
                    del self._state_store[comp_type]
    
    def _get_state_file_path(self, state_entry: StateEntry) -> Path:
        """Obtém o caminho do arquivo para persistência do estado."""
        comp_dir = self.storage_path / state_entry.component_type.value
        state_dir = comp_dir / state_entry.state_type.value
        state_dir.mkdir(parents=True, exist_ok=True)
        
        return state_dir / f"{state_entry.id}.json"
    
    async def _persist_single_state(self, state_entry: StateEntry) -> None:
        """Persiste um único estado em disco."""
        if not self.enable_persistence:
            return
            
        file_path = self._get_state_file_path(state_entry)
        
        try:
            # Se o estado já expirou, não persiste
            if state_entry.is_expired():
                return
                
            state_dict = state_entry.to_dict()
            
            # Persiste de forma atômica
            temp_path = file_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(state_dict, f)
            
            temp_path.replace(file_path)
        except Exception as e:
            logger.error(f"Erro ao persistir estado {state_entry.id}: {e}")
    
    async def _persist_states(self) -> None:
        """Persiste todos os estados em disco."""
        if not self.enable_persistence:
            return
            
        async with self._lock:
            for comp_type in self._state_store:
                for st_type in self._state_store[comp_type]:
                    for state_id, state_entry in self._state_store[comp_type][st_type].items():
                        await self._persist_single_state(state_entry)
    
    async def _load_persisted_states(self) -> None:
        """Carrega estados persistidos em disco."""
        if not self.enable_persistence or not self.storage_path.exists():
            return
            
        try:
            # Itera sobre todos os arquivos .json na estrutura de diretórios
            for comp_dir in self.storage_path.glob('*'):
                if not comp_dir.is_dir():
                    continue
                    
                for state_dir in comp_dir.glob('*'):
                    if not state_dir.is_dir():
                        continue
                        
                    for state_file in state_dir.glob('*.json'):
                        try:
                            with open(state_file, 'r') as f:
                                state_dict = json.load(f)
                                
                            state_entry = StateEntry.from_dict(state_dict)
                            
                            # Ignora estados expirados
                            if state_entry.is_expired():
                                state_file.unlink()
                                continue
                                
                            # Adiciona ao state_store
                            comp_type = state_entry.component_type.value
                            st_type = state_entry.state_type.value
                            
                            if comp_type not in self._state_store:
                                self._state_store[comp_type] = {}
                                
                            if st_type not in self._state_store[comp_type]:
                                self._state_store[comp_type][st_type] = {}
                                
                            self._state_store[comp_type][st_type][state_entry.id] = state_entry
                        except Exception as e:
                            logger.error(f"Erro ao carregar estado do arquivo {state_file}: {e}")
        except Exception as e:
            logger.error(f"Erro ao carregar estados persistidos: {e}")
    
    async def get_all_states(
        self,
        component_type: Optional[Union[ComponentType, str]] = None,
        state_type: Optional[Union[StateType, str]] = None,
        source_id: Optional[str] = None
    ) -> Dict[str, Dict[str, List[StateEntry]]]:
        """
        Obtém todos os estados, opcionalmente filtrados.
        
        Args:
            component_type: Filtro por tipo de componente
            state_type: Filtro por tipo de estado
            source_id: Filtro por ID de fonte
        
        Returns:
            Dicionário de estados, indexado por component_type > state_type
        """
        # Converte strings para enums se necessário
        if isinstance(component_type, str) and component_type:
            component_type = ComponentType(component_type)
        
        if isinstance(state_type, str) and state_type:
            state_type = StateType(state_type)
        
        result = {}
        async with self._lock:
            # Determina quais component_types incluir
            if component_type:
                comp_types = [component_type.value]
            else:
                comp_types = list(self._state_store.keys())
            
            for comp_type in comp_types:
                if comp_type not in self._state_store:
                    continue
                    
                result[comp_type] = {}
                
                # Determina quais state_types incluir
                if state_type:
                    st_types = [state_type.value]
                else:
                    st_types = list(self._state_store[comp_type].keys())
                
                for st_type in st_types:
                    if st_type not in self._state_store[comp_type]:
                        continue
                        
                    # Filtra por source_id e não expirados
                    entries = [
                        entry for entry in self._state_store[comp_type][st_type].values()
                        if not entry.is_expired() and (source_id is None or entry.source_id == source_id)
                    ]
                    
                    if entries:
                        # Ordenação por timestamp (mais recente primeiro)
                        entries.sort(key=lambda x: x.timestamp, reverse=True)
                        result[comp_type][st_type] = entries
        
        return result
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Obtém estatísticas sobre o estado atual do serviço.
        
        Returns:
            Dicionário com estatísticas
        """
        stats = {
            "instance_id": self.instance_id,
            "timestamp": datetime.now().isoformat(),
            "running": self._running,
            "components": {},
            "total_states": 0,
            "total_callbacks": len(self._callbacks),
            "expired_states_count": 0
        }
        
        async with self._lock:
            for comp_type in self._state_store:
                comp_stats = {
                    "state_types": {},
                    "total_states": 0
                }
                
                for st_type in self._state_store[comp_type]:
                    states = self._state_store[comp_type][st_type]
                    expired_count = sum(1 for s in states.values() if s.is_expired())
                    
                    type_stats = {
                        "count": len(states),
                        "expired_count": expired_count,
                        "sources": set(s.source_id for s in states.values())
                    }
                    
                    comp_stats["state_types"][st_type] = type_stats
                    comp_stats["total_states"] += len(states)
                    stats["expired_states_count"] += expired_count
                
                stats["components"][comp_type] = comp_stats
                stats["total_states"] += comp_stats["total_states"]
        
        return stats 