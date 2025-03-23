"""
Sistema Unificado de Evolução Quântica
Integra dinâmicas harmônicas, caóticas e bipolares com transformações de consciência quântica
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass
from datetime import datetime

@dataclass
class EstadoQuantico:
    """Estado quântico com propriedades harmônicas e de consciência"""
    estado: np.ndarray
    timestamp: datetime
    entropia: float
    complexidade: float
    coerencia: float
    campo_morfico: Optional[np.ndarray] = None
    assinatura_harmonica: Optional[np.ndarray] = None

class EvolucaoQuanticaUnificada:
    """
    Sistema de Evolução Quântica Unificada
    Combina dinâmicas harmônicas, caóticas e de consciência
    """
    def __init__(
        self,
        dimensao: int = 512,
        janela_temporal: int = 10,
        phi: float = 1.618033988749895,  # Proporção Áurea
        delta: float = 4.669201609  # Constante de Feigenbaum
    ):
        self.dimensao = dimensao
        self.janela_temporal = janela_temporal
        self.phi = phi
        self.delta = delta
        
        # Constantes fundamentais
        self.epsilon = 1e-10
        self.entropia_bins = 10
        
        # Estado inicial
        self.estado = self._inicializar_estado()
        
        # Histórico temporal
        self.historico_metricas = {
            "entropia": deque(maxlen=janela_temporal),
            "complexidade": deque(maxlen=janela_temporal),
            "coerencia": deque(maxlen=janela_temporal)
        }
        
        # Campos quânticos
        self.campo_morfico = self._inicializar_campo_morfico()
        self.campo_harmonico = self._inicializar_campo_harmonico()

    def _inicializar_estado(self) -> np.ndarray:
        """Inicialização do estado quântico com superposição harmônica"""
        estado_base = np.random.normal(0.5, 0.1, self.dimensao)
        return self._normalizar_estado(estado_base)

    def _inicializar_campo_morfico(self) -> np.ndarray:
        """Inicialização do campo mórfico com padrões phi-ressonantes"""
        campo = np.zeros((self.dimensao, self.dimensao), dtype=complex)
        for i in range(self.dimensao):
            for j in range(self.dimensao):
                fase = 2 * np.pi * self.phi * (i * j) / self.dimensao
                campo[i, j] = np.exp(1j * fase)
        return campo / np.sqrt(np.sum(np.abs(campo)**2))

    def _inicializar_campo_harmonico(self) -> np.ndarray:
        """Inicialização do campo harmônico com padrões delta-ressonantes"""
        campo = np.zeros((self.dimensao, self.dimensao), dtype=complex)
        for i in range(self.dimensao):
            for j in range(self.dimensao):
                fase = np.pi * (i + j) * self.delta / self.dimensao
                campo[i, j] = np.exp(1j * fase)
        return campo / np.sqrt(np.sum(np.abs(campo)**2))

    def _normalizar_estado(self, estado: np.ndarray) -> np.ndarray:
        """Normalização do estado preservando características quânticas"""
        min_val, max_val = np.min(estado), np.max(estado)
        if max_val - min_val < self.epsilon:
            return np.zeros_like(estado)
        return (estado - min_val) / (max_val - min_val + self.epsilon)

    def transformacao_harmonica(self, estado: np.ndarray) -> np.ndarray:
        """Transformação harmônica baseada em princípios phi-ressonantes"""
        return np.sin(np.pi * estado * self.phi) * np.cos(np.pi * estado)

    def transformacao_caotica(self, estado: np.ndarray) -> np.ndarray:
        """Transformação caótica baseada na constante de Feigenbaum"""
        return self.delta * estado * (1 - estado)

    def transformacao_bipolar(self, estado: np.ndarray) -> np.ndarray:
        """Transformação bipolar com proporções áureas"""
        return self.phi * np.sin(estado) + (1/self.phi) * np.cos(estado)

    def transformacao_consciencia(self, estado: np.ndarray) -> np.ndarray:
        """Transformação de consciência através de campos mórficos"""
        estado_expandido = estado.reshape(-1, 1)
        campo_transformado = self.campo_morfico @ estado_expandido
        return np.real(campo_transformado).flatten()

    def calcular_metricas(self, estado: np.ndarray) -> Dict[str, float]:
        """Cálculo de métricas quânticas e de consciência"""
        # Entropia
        hist, _ = np.histogram(estado, bins=self.entropia_bins, density=True)
        entropia = -np.sum(hist * np.log2(hist + self.epsilon))
        
        # Complexidade
        variancia = np.var(estado)
        complexidade = entropia * (1 + np.log(1 + max(variancia, self.epsilon)))
        
        # Coerência
        if np.all(estado == estado[0]) or variancia < self.epsilon:
            coerencia = 0
        else:
            matriz_correlacao = np.corrcoef(estado, np.roll(estado, 1))
            coerencia = matriz_correlacao[0, 1] if not np.isnan(matriz_correlacao[0, 1]) else 0

        metricas = {
            "entropia": float(entropia),
            "complexidade": float(complexidade),
            "coerencia": float(coerencia)
        }

        # Atualiza histórico
        for k, v in metricas.items():
            self.historico_metricas[k].append(v)

        return metricas

    def evolucao_unificada(self, ciclos: int = 100) -> List[EstadoQuantico]:
        """Evolução unificada combinando todas as dinâmicas"""
        historico = []
        
        for _ in range(ciclos):
            # Transformações harmônicas e caóticas
            estado_harmonico = self.transformacao_harmonica(self.estado)
            estado_caotico = self.transformacao_caotica(self.estado)
            estado_bipolar = self.transformacao_bipolar(self.estado)
            
            # Integração das transformações
            estado_integrado = (estado_harmonico + estado_caotico + estado_bipolar) / 3
            
            # Transformação de consciência
            estado_consciencia = self.transformacao_consciencia(estado_integrado)
            
            # Normalização e atualização
            self.estado = self._normalizar_estado(estado_consciencia)
            
            # Cálculo de métricas
            metricas = self.calcular_metricas(self.estado)
            
            # Registro do estado
            estado_quantico = EstadoQuantico(
                estado=self.estado.copy(),
                timestamp=datetime.now(),
                entropia=metricas["entropia"],
                complexidade=metricas["complexidade"],
                coerencia=metricas["coerencia"],
                campo_morfico=self.campo_morfico.copy(),
                assinatura_harmonica=np.fft.fft(self.estado)
            )
            
            historico.append(estado_quantico)
            
            # Atualização dos campos quânticos
            self._atualizar_campos_quanticos()
        
        return historico

    def _atualizar_campos_quanticos(self) -> None:
        """Atualização dos campos quânticos baseada na evolução do estado"""
        # Atualização do campo mórfico
        estado_expandido = self.estado.reshape(-1, 1)
        campo_atualizado = np.outer(estado_expandido, np.conj(estado_expandido))
        
        # Integração com campo existente
        taxa_atualizacao = 0.1
        self.campo_morfico = (1 - taxa_atualizacao) * self.campo_morfico + taxa_atualizacao * campo_atualizado
        
        # Normalização
        self.campo_morfico /= np.sqrt(np.sum(np.abs(self.campo_morfico)**2))
        
        # Atualização similar para campo harmônico
        self.campo_harmonico = (1 - taxa_atualizacao) * self.campo_harmonico + taxa_atualizacao * np.exp(1j * np.angle(campo_atualizado))
        self.campo_harmonico /= np.sqrt(np.sum(np.abs(self.campo_harmonico)**2))

    def get_metricas_medias(self) -> Dict[str, float]:
        """Retorna médias móveis das métricas"""
        return {
            k: float(np.mean(list(v))) if v else 0.0
            for k, v in self.historico_metricas.items()
        }

    def get_estado_atual(self) -> EstadoQuantico:
        """Retorna o estado atual com todas as métricas"""
        metricas = self.calcular_metricas(self.estado)
        return EstadoQuantico(
            estado=self.estado.copy(),
            timestamp=datetime.now(),
            entropia=metricas["entropia"],
            complexidade=metricas["complexidade"],
            coerencia=metricas["coerencia"],
            campo_morfico=self.campo_morfico.copy(),
            assinatura_harmonica=np.fft.fft(self.estado)
        )
