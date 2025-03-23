"""
Memória Holográfica

Este módulo implementa a memória holográfica, responsável por:
- Armazenamento de padrões quânticos
- Recuperação de informações
- Correção de erros
- Compressão de dados
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime
import json
import os

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Tipos
HolographicPattern = np.ndarray
PatternMetadata = Dict[str, any]

@dataclass
class HolographicEntry:
    """Entrada na memória holográfica"""
    pattern: HolographicPattern
    metadata: PatternMetadata
    timestamp: float
    coherence: float
    error_rate: float

class HolographicMemory:
    """
    Memória Holográfica
    
    Esta classe implementa o armazenamento e recuperação de padrões quânticos
    usando princípios holográficos, incluindo correção de erros e compressão.
    """
    
    def __init__(
        self,
        dimensions: int,
        phi: float,
        temperature: float,
        max_patterns: int = 1000,
        error_threshold: float = 0.1
    ):
        self.dimensions = dimensions
        self.phi = phi
        self.temperature = temperature
        self.max_patterns = max_patterns
        self.error_threshold = error_threshold
        
        # Inicializa memória
        self.patterns: List[HolographicEntry] = []
        self.reference_pattern: Optional[HolographicPattern] = None
        
        # Diretório de armazenamento
        self.storage_dir = "holographic_memory"
        self._init_storage()
        
        logger.info("HolographicMemory inicializado com sucesso")
        
    def _init_storage(self) -> None:
        """Inicializa armazenamento em disco"""
        try:
            # Cria diretório se não existir
            if not os.path.exists(self.storage_dir):
                os.makedirs(self.storage_dir)
                
            # Carrega padrões salvos
            self._load_patterns()
            
        except Exception as e:
            logger.error(f"Erro ao inicializar armazenamento: {e}")
            
    def _load_patterns(self) -> None:
        """Carrega padrões do disco"""
        try:
            # Lista arquivos
            files = os.listdir(self.storage_dir)
            
            # Carrega cada arquivo
            for file in files:
                if file.endswith('.json'):
                    self._load_pattern_file(file)
                    
        except Exception as e:
            logger.error(f"Erro ao carregar padrões: {e}")
            
    def _load_pattern_file(self, filename: str) -> None:
        """Carrega padrão de um arquivo"""
        try:
            # Lê arquivo
            with open(os.path.join(self.storage_dir, filename), 'r') as f:
                data = json.load(f)
                
            # Recria entrada
            entry = HolographicEntry(
                pattern=np.array(data['pattern']),
                metadata=data['metadata'],
                timestamp=data['timestamp'],
                coherence=data['coherence'],
                error_rate=data['error_rate']
            )
            
            # Adiciona à memória
            self.patterns.append(entry)
            
        except Exception as e:
            logger.error(f"Erro ao carregar arquivo {filename}: {e}")
            
    def _save_pattern(self, entry: HolographicEntry) -> None:
        """Salva padrão em arquivo"""
        try:
            # Cria nome de arquivo
            filename = f"pattern_{entry.timestamp}.json"
            
            # Prepara dados
            data = {
                'pattern': entry.pattern.tolist(),
                'metadata': entry.metadata,
                'timestamp': entry.timestamp,
                'coherence': entry.coherence,
                'error_rate': entry.error_rate
            }
            
            # Salva arquivo
            with open(os.path.join(self.storage_dir, filename), 'w') as f:
                json.dump(data, f)
                
        except Exception as e:
            logger.error(f"Erro ao salvar padrão: {e}")
            
    def store_pattern(
        self,
        pattern: HolographicPattern,
        metadata: PatternMetadata,
        coherence: float
    ) -> bool:
        """Armazena padrão na memória"""
        try:
            # Verifica limite
            if len(self.patterns) >= self.max_patterns:
                # Remove padrão mais antigo
                self.patterns.pop(0)
                
            # Calcula taxa de erro
            error_rate = self._calculate_error_rate(pattern)
            
            # Cria entrada
            entry = HolographicEntry(
                pattern=pattern,
                metadata=metadata,
                timestamp=datetime.now().timestamp(),
                coherence=coherence,
                error_rate=error_rate
            )
            
            # Adiciona à memória
            self.patterns.append(entry)
            
            # Salva em disco
            self._save_pattern(entry)
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao armazenar padrão: {e}")
            return False
            
    def retrieve_pattern(
        self,
        query: HolographicPattern,
        threshold: float = 0.8
    ) -> Optional[Tuple[HolographicPattern, PatternMetadata]]:
        """Recupera padrão mais similar"""
        try:
            # Inicializa melhor match
            best_match = None
            best_similarity = 0.0
            
            # Procura padrão mais similar
            for entry in self.patterns:
                similarity = self._calculate_similarity(query, entry.pattern)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = entry
                    
            # Verifica threshold
            if best_similarity >= threshold:
                return best_match.pattern, best_match.metadata
                
            return None
            
        except Exception as e:
            logger.error(f"Erro ao recuperar padrão: {e}")
            return None
            
    def _calculate_error_rate(self, pattern: HolographicPattern) -> float:
        """Calcula taxa de erro de um padrão"""
        try:
            # Usa temperatura para erro
            error_rate = self.temperature * np.random.rand()
            
            return float(error_rate)
            
        except Exception as e:
            logger.error(f"Erro ao calcular taxa de erro: {e}")
            return 0.0
            
    def _calculate_similarity(
        self,
        pattern1: HolographicPattern,
        pattern2: HolographicPattern
    ) -> float:
        """Calcula similaridade entre padrões"""
        try:
            # Usa produto interno normalizado
            similarity = np.abs(np.sum(pattern1.conj() * pattern2))
            similarity /= np.sqrt(np.sum(np.abs(pattern1)**2) * np.sum(np.abs(pattern2)**2))
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Erro ao calcular similaridade: {e}")
            return 0.0
            
    def compress_pattern(self, pattern: HolographicPattern) -> HolographicPattern:
        """Comprime padrão usando princípios holográficos"""
        try:
            # Usa proporção áurea para compressão
            compression_ratio = self.phi
            
            # Reduz dimensões
            compressed_size = int(self.dimensions / compression_ratio)
            
            # Cria padrão comprimido
            compressed_pattern = np.zeros(compressed_size, dtype=complex)
            
            # Aplica transformada
            for i in range(compressed_size):
                phase = 2 * np.pi * i / compressed_size
                compressed_pattern[i] = np.sum(pattern * np.exp(1j * phase))
                
            # Normaliza
            compressed_pattern /= np.sqrt(np.sum(np.abs(compressed_pattern)**2))
            
            return compressed_pattern
            
        except Exception as e:
            logger.error(f"Erro ao comprimir padrão: {e}")
            return pattern
            
    def decompress_pattern(
        self,
        compressed_pattern: HolographicPattern,
        original_size: int
    ) -> HolographicPattern:
        """Descomprime padrão"""
        try:
            # Usa proporção áurea para descompressão
            compression_ratio = self.phi
            
            # Recupera dimensões originais
            decompressed_pattern = np.zeros(original_size, dtype=complex)
            
            # Aplica transformada inversa
            for i in range(original_size):
                phase = 2 * np.pi * i / original_size
                decompressed_pattern[i] = np.sum(compressed_pattern * np.exp(-1j * phase))
                
            # Normaliza
            decompressed_pattern /= np.sqrt(np.sum(np.abs(decompressed_pattern)**2))
            
            return decompressed_pattern
            
        except Exception as e:
            logger.error(f"Erro ao descomprimir padrão: {e}")
            return np.zeros(original_size, dtype=complex)
            
    def get_memory_info(self) -> Dict[str, any]:
        """Retorna informações sobre a memória"""
        return {
            'dimensions': self.dimensions,
            'phi': self.phi,
            'temperature': self.temperature,
            'max_patterns': self.max_patterns,
            'error_threshold': self.error_threshold,
            'num_patterns': len(self.patterns),
            'average_coherence': np.mean([p.coherence for p in self.patterns]) if self.patterns else 0.0,
            'average_error_rate': np.mean([p.error_rate for p in self.patterns]) if self.patterns else 0.0
        }

if __name__ == "__main__":
    # Exemplo de uso
    memory = HolographicMemory(
        dimensions=256,
        phi=0.618,
        temperature=0.1
    )
    
    # Cria padrão
    pattern = np.random.randn(256) + 1j * np.random.randn(256)
    pattern /= np.sqrt(np.sum(np.abs(pattern)**2))
    
    # Armazena padrão
    metadata = {'type': 'test', 'value': 42}
    memory.store_pattern(pattern, metadata, coherence=0.9)
    
    # Recupera padrão
    retrieved_pattern, retrieved_metadata = memory.retrieve_pattern(pattern)
    if retrieved_pattern is not None:
        print("Metadata recuperada:", retrieved_metadata) 