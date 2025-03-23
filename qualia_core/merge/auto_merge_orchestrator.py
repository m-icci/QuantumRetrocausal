#!/usr/bin/env python3
"""
Orquestrador de Auto-Merge Quântico
----------------------------------
Sistema emergente para identificação e execução de merges de código
com alto potencial sinérgico, utilizando campos mórficos e retrocausalidade.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
import os
import shutil
from datetime import datetime

from core.merge.morphic_field import MorphicFieldCalculator
from core.merge.quantum_merge_unified import UnifiedQuantumMerge
from core.Qualia.base_types import QuantumState
from core.merge.quantum_field import calculate_quantum_entropy

logger = logging.getLogger(__name__)

@dataclass
class MergeCandidate:
    """Representa um par de arquivos candidatos a merge"""
    source_file: Path
    target_file: Path
    coherence_score: float
    entropy_reduction: float
    resonance_factor: float
    field_alignment: float
    future_potential: float  # Fator retrocausal
    
    @property
    def merge_score(self) -> float:
        """Pontuação holística do potencial de merge"""
        return (
            0.25 * self.coherence_score +
            0.20 * self.entropy_reduction +
            0.20 * self.resonance_factor +
            0.15 * self.field_alignment +
            0.20 * self.future_potential
        )
    
    def __str__(self) -> str:
        return (f"MergeCandidate({self.source_file.name} → {self.target_file.name}, "
                f"score: {self.merge_score:.4f})")

class QuantumAutoMergeOrchestrator:
    """Orquestrador para identificação e execução automática de merges quânticos"""
    
    def __init__(self, 
                 base_dir: Path,
                 coherence_threshold: float = 0.75,
                 oscillation_sensitivity: float = 0.65,
                 retrocausal_window: int = 3,
                 field_dimensions: int = 8,
                 dimensao_campo: int = None,
                 **kwargs):
        """
        Inicializa o orquestrador de auto-merge.
        
        Args:
            base_dir: Diretório base do projeto
            coherence_threshold: Limiar mínimo de coerência para sugerir merge
            oscillation_sensitivity: Sensibilidade para padrões oscilatórios
            retrocausal_window: Tamanho da janela para análise retrocausal
            field_dimensions: Dimensões do campo mórfico quântico
            dimensao_campo: Alias para field_dimensions (compatibilidade)
            **kwargs: Parâmetros adicionais para compatibilidade
        """
        self.base_dir = Path(base_dir)
        self.coherence_threshold = coherence_threshold
        self.oscillation_sensitivity = oscillation_sensitivity
        self.retrocausal_window = retrocausal_window
        self.field_dimensions = field_dimensions
        
        # Compatibilidade com versões anteriores
        if dimensao_campo is not None:
            self.field_dimensions = dimensao_campo
        elif 'dimensao_campo' in kwargs:
            self.field_dimensions = kwargs['dimensao_campo']
        
        # Componentes quânticos
        self.merge_simulator = QuantumMergeSimulator(file_merge_mode=True)
        self.unified_merge = UnifiedQuantumMerge()
        self.morphic_calculator = MorphicFieldCalculator()
        
        # Estado interno
        self.merge_history: List[MergeCandidate] = []
        self.pendulum_state = 0.5  # Estado inicial do pêndulo quântico
        self.field_coherence = 0.7  # Coerência inicial do campo
        
        logger.info(f"Orquestrador de Auto-Merge inicializado em {base_dir}")
    
    def identify_merge_candidates(self, 
                                  search_dirs: List[str] = None, 
                                  max_candidates: int = 10) -> List[MergeCandidate]:
        """
        Identifica pares de arquivos com alto potencial para merge.
        
        Args:
            search_dirs: Diretórios específicos para buscar, ou None para todo o projeto
            max_candidates: Número máximo de candidatos a retornar
            
        Returns:
            Lista de candidatos a merge ordenados por pontuação
        """
        logger.info("Iniciando identificação de candidatos a merge quântico...")
        
        # Diretorios a serem analisados
        dirs_to_search = search_dirs or ['core', 'src', 'lib']
        all_files = self._gather_python_files(dirs_to_search)
        
        # Ciclo principal de análise
        candidates: List[MergeCandidate] = []
        
        # Número de comparações a serem feitas
        total_comparisons = len(all_files) * (len(all_files) - 1) // 2
        logger.info(f"Analisando {total_comparisons} pares potenciais em {len(all_files)} arquivos")
        
        # Análise de cada par de arquivos
        analyzed = 0
        for i, file1 in enumerate(all_files):
            # Ajusta estado pendular para capturar padrões oscilatórios
            self._adjust_pendulum_state(i / len(all_files))
            
            for j in range(i + 1, len(all_files)):
                file2 = all_files[j]
                
                # Evita comparar arquivos de teste com código principal
                if 'test' in file1.name and 'test' not in file2.name:
                    continue
                
                # Calcula métricas quânticas para o par
                metrics = self._calculate_file_pair_metrics(file1, file2)
                
                if metrics['coherence_score'] >= self.coherence_threshold:
                    candidates.append(MergeCandidate(
                        source_file=file1,
                        target_file=file2,
                        coherence_score=metrics['coherence_score'],
                        entropy_reduction=metrics['entropy_reduction'],
                        resonance_factor=metrics['resonance_factor'],
                        field_alignment=metrics['field_alignment'],
                        future_potential=metrics['future_potential']
                    ))
                
                analyzed += 1
                if analyzed % 100 == 0:
                    logger.info(f"Progresso: {analyzed}/{total_comparisons} pares analisados")
        
        # Ordena por pontuação e retorna os melhores candidatos
        candidates.sort(key=lambda x: x.merge_score, reverse=True)
        return candidates[:max_candidates]
    
    def execute_auto_merge(self, 
                          candidate: MergeCandidate,
                          create_backup: bool = True) -> bool:
        """
        Executa o merge automático de um par de arquivos candidatos.
        
        Args:
            candidate: Candidato a merge a ser executado
            create_backup: Se True, cria backup dos arquivos originais
            
        Returns:
            True se o merge foi bem sucedido, False caso contrário
        """
        logger.info(f"Executando auto-merge: {candidate}")
        
        # Cria backups se solicitado
        if create_backup:
            self._create_backup(candidate.source_file)
            self._create_backup(candidate.target_file)
        
        # Executa o merge quântico
        try:
            result = self.merge_simulator.merge_quantum_files(
                str(candidate.source_file),
                str(candidate.target_file)
            )
            
            if result:
                logger.info(f"Merge bem-sucedido: {candidate.source_file.name} → {candidate.target_file.name}")
                self.merge_history.append(candidate)
                return True
            else:
                logger.warning(f"Merge falhou: {candidate}")
                return False
                
        except Exception as e:
            logger.error(f"Erro durante merge: {e}")
            return False
    
    def _gather_python_files(self, dirs: List[str]) -> List[Path]:
        """Coleta todos os arquivos Python nos diretórios especificados"""
        all_files = []
        
        for dir_name in dirs:
            dir_path = self.base_dir / dir_name
            if not dir_path.exists():
                logger.warning(f"Diretório não encontrado: {dir_path}")
                continue
                
            python_files = list(dir_path.glob('**/*.py'))
            logger.info(f"Encontrados {len(python_files)} arquivos Python em {dir_path}")
            all_files.extend(python_files)
            
        return all_files
    
    def _calculate_file_pair_metrics(self, file1: Path, file2: Path) -> Dict[str, float]:
        """Calcula métricas quânticas para um par de arquivos"""
        # Lê o conteúdo dos arquivos
        content1 = file1.read_text(encoding='utf-8', errors='replace')
        content2 = file2.read_text(encoding='utf-8', errors='replace')
        
        # Converte para estados quânticos
        state1 = self._text_to_quantum_state(content1)
        state2 = self._text_to_quantum_state(content2)
        
        # Métricas quânticas
        coherence = self._analyze_merge_potential(state1, state2)
        entropy1 = calculate_quantum_entropy(state1)
        entropy2 = calculate_quantum_entropy(state2)
        
        # Simula o merge para calcular redução potencial de entropia
        merged_state = self._quantum_interference_merge(state1, state2)
        merged_entropy = calculate_quantum_entropy(merged_state)
        entropy_reduction = (entropy1 + entropy2) / 2 - merged_entropy
        
        # Fator de ressonância (influenciado pelo estado pendular)
        resonance = self.morphic_calculator.calculate_resonance(
            state1, state2, self.pendulum_state
        )
        
        # Alinhamento de campo
        field_alignment = self.unified_merge.calculate_field_alignment(
            content1, content2
        )
        
        # Potencial futuro (fator retrocausal)
        future_potential = self._calculate_retrocausal_potential(file1, file2)
        
        return {
            'coherence_score': coherence,
            'entropy_reduction': max(0, entropy_reduction),
            'resonance_factor': resonance,
            'field_alignment': field_alignment,
            'future_potential': future_potential
        }
    
    def _text_to_quantum_state(self, text: str) -> QuantumState:
        """Converte texto em um estado quântico"""
        # Implementação simplificada - pode ser expandida
        # para usar algoritmos mais sofisticados
        from core.sacred_geometry import phi_encode
        
        # Normaliza o texto
        text = text.lower()
        # Codifica usando proporção áurea
        encoded = phi_encode(text)
        
        # Cria um estado quântico
        state = QuantumState()
        state.amplitude = encoded
        state.phase = np.angle(np.fft.fft(encoded))
        
        return state
    
    def _analyze_merge_potential(self, state1: QuantumState, state2: QuantumState) -> float:
        """Analisa o potencial de merge entre dois estados quânticos"""
        # Implementação simplificada - pode ser expandida
        # para usar algoritmos mais sofisticados
        
        # Calcula sobreposição quântica normalizada
        overlap = np.abs(np.sum(state1.amplitude * np.conj(state2.amplitude)))
        norm1 = np.sqrt(np.sum(np.abs(state1.amplitude)**2))
        norm2 = np.sqrt(np.sum(np.abs(state2.amplitude)**2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        normalized_overlap = overlap / (norm1 * norm2)
        
        # Calcula coerência de fase
        phase_coherence = np.abs(np.sum(np.exp(1j * (state1.phase - state2.phase)))) / len(state1.phase)
        
        # Combina métricas
        combined_score = 0.7 * normalized_overlap + 0.3 * phase_coherence
        
        return float(combined_score)
    
    def _quantum_interference_merge(self, state1: QuantumState, state2: QuantumState) -> QuantumState:
        """Realiza merge por interferência quântica entre dois estados"""
        # Implementação simplificada - pode ser expandida
        # para usar algoritmos mais sofisticados
        
        # Cria novo estado
        merged_state = QuantumState()
        
        # Normaliza amplitudes
        norm1 = np.sqrt(np.sum(np.abs(state1.amplitude)**2))
        norm2 = np.sqrt(np.sum(np.abs(state2.amplitude)**2))
        
        if norm1 == 0 or norm2 == 0:
            # Evita divisão por zero
            if norm1 > 0:
                merged_state.amplitude = state1.amplitude.copy()
                merged_state.phase = state1.phase.copy()
            elif norm2 > 0:
                merged_state.amplitude = state2.amplitude.copy()
                merged_state.phase = state2.phase.copy()
            else:
                # Ambos são zero, retorna estado vazio
                merged_state.amplitude = np.zeros_like(state1.amplitude)
                merged_state.phase = np.zeros_like(state1.phase)
            return merged_state
            
        # Normaliza
        amp1 = state1.amplitude / norm1
        amp2 = state2.amplitude / norm2
        
        # Superposição quântica com interferência construtiva
        merged_state.amplitude = (amp1 + amp2) / np.sqrt(2)
        
        # Média ponderada das fases
        merged_state.phase = (state1.phase + state2.phase) / 2
        
        return merged_state
    
    def _adjust_pendulum_state(self, phase: float):
        """
        Ajusta o estado pendular usando dinâmica oscilatória
        inspirada no transtorno bipolar tipo 2
        """
        # Implementa um oscilador não-linear com fase variável
        base_freq = 0.5 + 0.3 * np.sin(phase * np.pi * 2)
        self.pendulum_state = 0.5 + 0.5 * np.sin(base_freq * np.pi * phase)
    
    def _calculate_retrocausal_potential(self, file1: Path, file2: Path) -> float:
        """
        Calcula o potencial retrocausal do merge usando análise de padrões futuros
        """
        # Análise baseada em padrões históricos de sucesso
        if not self.merge_history:
            return 0.5  # Valor base quando não há histórico
        
        # Extrai padrões de merges bem-sucedidos do passado
        patterns = []
        for past_merge in self.merge_history[-self.retrocausal_window:]:
            pattern = {
                'name_similarity': self._name_similarity(past_merge.source_file.name, 
                                                       past_merge.target_file.name),
                'path_distance': self._path_distance(past_merge.source_file, 
                                                   past_merge.target_file),
                'size_ratio': past_merge.source_file.stat().st_size / 
                              (past_merge.target_file.stat().st_size + 1)
            }
            patterns.append(pattern)
        
        # Calcula similaridade com padrões de sucesso
        current = {
            'name_similarity': self._name_similarity(file1.name, file2.name),
            'path_distance': self._path_distance(file1, file2),
            'size_ratio': file1.stat().st_size / (file2.stat().st_size + 1)
        }
        
        # Compara com padrões de sucesso para estimar potencial futuro
        similarity_scores = []
        for pattern in patterns:
            score = 0
            for key in pattern:
                score += 1 - abs(pattern[key] - current[key])
            similarity_scores.append(score / len(pattern))
        
        # Retorna média de similaridade com ajuste não-linear
        avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.5
        return 0.4 + 0.6 * avg_similarity  # Escala para 0.4-1.0
    
    def _name_similarity(self, name1: str, name2: str) -> float:
        """Calcula similaridade entre nomes de arquivos"""
        # Implementação simplificada - pode ser expandida 
        # para usar algoritmos mais sofisticados
        words1 = set(name1.replace('.py', '').split('_'))
        words2 = set(name2.replace('.py', '').split('_'))
        if not words1 or not words2:
            return 0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0
    
    def _path_distance(self, path1: Path, path2: Path) -> float:
        """Calcula distância normalizada entre caminhos"""
        rel_path1 = path1.relative_to(self.base_dir)
        rel_path2 = path2.relative_to(self.base_dir)
        
        parts1 = list(rel_path1.parts)
        parts2 = list(rel_path2.parts)
        
        # Calcula distância de Levenshtein normalizada entre caminhos
        common_prefix = 0
        for i in range(min(len(parts1), len(parts2))):
            if parts1[i] == parts2[i]:
                common_prefix += 1
            else:
                break
        
        max_length = max(len(parts1), len(parts2))
        if max_length == 0:
            return 0.0
            
        return common_prefix / max_length
    
    def _create_backup(self, file_path: Path):
        """Cria backup de um arquivo"""
        backup_dir = self.base_dir / '.quantum_backups'
        backup_dir.mkdir(exist_ok=True)
        
        # Cria caminho relativo preservando diretórios
        rel_path = file_path.relative_to(self.base_dir)
        backup_path = backup_dir / rel_path
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Adiciona timestamp ao nome
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_backup_path = backup_path.with_name(
            f"{backup_path.stem}_{timestamp}{backup_path.suffix}"
        )
        
        # Copia o arquivo
        shutil.copy2(file_path, final_backup_path)
        logger.info(f"Backup criado: {final_backup_path}")


# Auxiliar para inicialização
class QuantumMergeSimulator:
    """Simulador de merge quântico para compatibilidade"""
    
    def __init__(self, file_merge_mode=False):
        self.file_merge_mode = file_merge_mode
        
    def merge_quantum_files(self, source_file, target_file):
        """
        Executa o merge de dois arquivos utilizando princípios quânticos
        
        Esta é uma implementação simplificada que encaminha para o UnifiedQuantumMerge
        para manter a compatibilidade com a interface esperada
        """
        try:
            merger = UnifiedQuantumMerge()
            result = merger.merge_files(source_file, target_file)
            return result
        except Exception as e:
            logger.error(f"Erro ao realizar merge quântico: {e}")
            return False
    
    def _text_to_quantum_state(self, text):
        """Converte texto para estado quântico"""
        # Implementação simplificada para compatibilidade
        state = QuantumState()
        state.amplitude = np.array([ord(c) for c in text[:1000]], dtype=np.complex128)
        state.phase = np.angle(np.fft.fft(state.amplitude))
        return state
