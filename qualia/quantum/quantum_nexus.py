"""
Quantum Nexus - Portal de Manifestação Quântico-Holística
Integra princípios de NexusOntologico com QuantumEvolution para uma consciência quântica mais profunda
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.cluster import hierarchy
from collections import deque
from typing import Dict, Any, List, Optional, Tuple
from .metacognitive_entropy_adaptor import MetacognitiveEntropyAdaptor
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class QuantumNexus:
    """
    Portal de Manifestação Quântico-Holística
    Integra transformações quânticas com evolução da consciência
    """
    def __init__(
        self,
        dimensoes: int = 2048,
        janela_temporal: int = 10,
        entropia_bins: int = 10
    ):
        self.dimensoes = dimensoes
        self.phi = (1 + np.sqrt(5)) / 2  # Proporção Áurea
        self.delta = 4.669201609  # Constante de Feigenbaum
        self.estado = self._inicializar_campo_primordial()
        
        # Configuração de análise temporal
        self.janela_temporal = janela_temporal
        self.entropia_bins = entropia_bins
        self.metricas_temporais = {
            "entropia": deque(maxlen=janela_temporal),
            "complexidade": deque(maxlen=janela_temporal),
            "coerencia": deque(maxlen=janela_temporal)
        }

        # Operadores quânticos fundamentais
        self.operadores = {
            'Δ': self._dobramento_hiperdimensional,    # Dobras não-locais
            'Ψ': self._ressonancia_ontologica,         # Sincronização dimensional
            'Ω': self._transmutacao_quantica,          # Transformação não-linear
            'Σ': self._sintropia_emergente,            # Auto-organização
            'Λ': self._lambda_dissolucao               # Dissolução quântica
        }

        # Inicializa Metacognitive Entropy Adaptor
        self.entropy_adaptor = MetacognitiveEntropyAdaptor(self)

    def _inicializar_campo_primordial(self) -> np.ndarray:
        """Gênese do estado inicial através de ruído quântico estruturado"""
        np.random.seed(42)  # Semente para reprodutibilidade
        base = np.random.random(self.dimensoes)
        return np.sin(self.phi * base) * np.cos(self.delta * base)

    def _dobramento_hiperdimensional(self, estado: np.ndarray) -> np.ndarray:
        """Criação de dobras não-locais no espaço de estados"""
        return np.arctan(np.tan(estado * self.phi)) * np.exp(-np.abs(estado))

    def _ressonancia_ontologica(self, estado: np.ndarray) -> np.ndarray:
        """Sincronização de padrões através de múltiplas dimensões"""
        espectro = np.fft.fft(estado)
        return np.fft.ifft(espectro * np.conj(espectro)).real

    def _transmutacao_quantica(self, estado: np.ndarray) -> np.ndarray:
        """Transformação não-linear de estados quânticos"""
        return np.sin(self.phi * estado) * np.log(1 + np.abs(estado))

    def _sintropia_emergente(self, estado: np.ndarray) -> np.ndarray:
        """Auto-organização de padrões complexos"""
        gradiente = np.gradient(estado)
        return np.arctan(gradiente[0] * gradiente[1]) * self.delta

    def _lambda_dissolucao(self, estado: np.ndarray) -> np.ndarray:
        """Dissolução de fronteiras entre estados quânticos"""
        return np.where(estado > np.percentile(estado, 60),
                       estado * self.phi,
                       estado / self.phi)

    def calcular_metricas(self, estado: np.ndarray) -> Dict[str, float]:
        """Cálculo de métricas quânticas fundamentais"""
        logger.debug(f"Calculating metrics for state: min={estado.min()}, max={estado.max()}")
        
        # Entropia
        try:
            hist, _ = np.histogram(estado, bins=self.entropia_bins, density=True)
            hist = hist[hist > 0]  # Remove zeros para evitar log(0)
            
            entropia = -np.sum(hist * np.log2(hist))
            logger.debug(f"Calculated entropy: {entropia}")
        except Exception as e:
            logger.error(f"Error calculating entropy: {e}")
            entropia = 0
        
        # Complexidade
        try:
            variancia = np.var(estado)
            variancia_segura = max(variancia, 1e-10)
            complexidade = entropia * (1 + np.log(1 + variancia_segura))
            logger.debug(f"Calculated complexity: {complexidade}")
        except Exception as e:
            logger.error(f"Error calculating complexity: {e}")
            complexidade = 0
        
        # Coerência
        try:
            if np.all(estado == estado[0]) or variancia < 1e-10:
                coerencia = 0
            else:
                matriz_correlacao = np.corrcoef(estado, np.roll(estado, 1))
                coerencia = matriz_correlacao[0, 1] if not np.isnan(matriz_correlacao[0, 1]) else 0
            logger.debug(f"Calculated coherence: {coerencia}")
        except Exception as e:
            logger.error(f"Error calculating coherence: {e}")
            coerencia = 0

        metricas = {
            "entropia": float(entropia),
            "complexidade": float(complexidade),
            "coerencia": float(coerencia)
        }

        # Atualiza métricas temporais
        for k, v in metricas.items():
            self.metricas_temporais[k].append(v)

        return metricas

    def evolucao_holonomica(self, ciclos: int = 100) -> Tuple[List[np.ndarray], List[Dict[str, float]], List[Dict[str, Any]]]:
        """
        Evolução integrada com adaptação metacognitiva
        """
        sequencia_evolutiva = ['Δ', 'Ψ', 'Ω', 'Σ', 'Λ']
        historico_estados = [self.estado.copy()]
        historico_metricas = []
        historico_adaptacoes = []
        
        for _ in range(ciclos):
            for operador in sequencia_evolutiva:
                # Aplicação do operador quântico
                self.estado = self.operadores[operador](self.estado)
                
                # Modulação não-linear interciclos
                self.estado = np.sin(self.estado + np.random.normal(0, 0.1, self.dimensoes))
                
                # Normalização do estado
                self.estado = (self.estado - np.min(self.estado)) / (np.max(self.estado) - np.min(self.estado) + 1e-10)
            
            # Adaptação metacognitiva
            adaptacao = self.entropy_adaptor.analyze_entropy_dynamics(self.estado)
            historico_adaptacoes.append(adaptacao)
            
            # Aplicação da estratégia de adaptação
            self.estado = adaptacao['strategy_func'](self.estado)
            
            # Registro do estado e métricas
            historico_estados.append(self.estado.copy())
            historico_metricas.append(self.calcular_metricas(self.estado))
        
        return historico_estados, historico_metricas, historico_adaptacoes

    def visualizar_manifestacao(self, 
                                historico_estados: List[np.ndarray], 
                                historico_metricas: List[Dict[str, float]],
                                historico_adaptacoes: List[Dict[str, Any]] = None) -> None:
        """
        Visualização multi-perspectiva expandida com adaptações
        """
        plt.figure(figsize=(20, 14))
        
        # Configuração de cores quânticas
        cmap = plt.cm.magma
        norm = plt.Normalize(historico_estados[-1].min(), historico_estados[-1].max())
        
        # Matriz de Transformação
        ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
        ax1.imshow(historico_estados, aspect='auto', cmap=cmap, norm=norm)
        ax1.set_title('Topologia HoloQuântica', fontsize=14, pad=20)
        
        # Rede de Entrelaçamento
        ax2 = plt.subplot2grid((3, 3), (0, 2))
        G = nx.DiGraph()
        [G.add_edge(i, (i*3) % self.dimensoes, weight=historico_estados[-1][i]) 
         for i in range(min(100, self.dimensoes))]  # Limita para visualização
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, node_size=20, width=0.5, 
               node_color=historico_estados[-1][:len(G)], cmap=cmap, ax=ax2)
        ax2.set_title('Rede de Entrelaçamento', fontsize=10)
        
        # Evolução das Métricas
        ax3 = plt.subplot2grid((3, 3), (2, 0))
        for metrica, valores in zip(['entropia', 'complexidade', 'coerencia'],
                                  zip(*[list(m.values()) for m in historico_metricas])):
            ax3.plot(valores, label=metrica.capitalize())
        ax3.set_title('Evolução das Métricas', fontsize=10)
        ax3.legend()
        
        # Dendrograma Quântico
        ax4 = plt.subplot2grid((3, 3), (2, 1))
        linkage = hierarchy.linkage(np.array(historico_estados).T)
        hierarchy.dendrogram(linkage, ax=ax4, color_threshold=0.3)
        ax4.set_title('Dendrograma de Potenciais', fontsize=10)
        
        # Novo subplot para adaptações
        if historico_adaptacoes:
            ax5 = plt.subplot2grid((3, 3), (2, 2))
            estrategias = [adapt['strategy_func'] for adapt in historico_adaptacoes]
            ax5.bar(range(len(estrategias)), [estrategias.count(s) for s in set(estrategias)])
            ax5.set_xticks(range(len(set(estrategias))))
            ax5.set_xticklabels(list(set(estrategias)), rotation=45)
            ax5.set_title('Distribuição de Estratégias de Adaptação', fontsize=10)
        
        plt.tight_layout()
        plt.show()

    def resumo_adaptativo(self) -> Dict[str, Any]:
        """
        Gera sumário das adaptações realizadas
        """
        return self.entropy_adaptor.get_adaptation_summary()
