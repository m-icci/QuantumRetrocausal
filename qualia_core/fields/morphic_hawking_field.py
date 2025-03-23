"""
Campo Mórfico de Hawking
Uma implementação onde a radiação Hawking emerge da ressonância mórfica do mercado
"""
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
from scipy import stats
from scipy.signal import hilbert
import matplotlib.pyplot as plt

@dataclass
class MorphicState:
    """Estado do campo mórfico"""
    campo: np.ndarray
    entropia: float
    coerencia: float
    ressonancia: float
    timestamp: datetime

class MorphicMemory:
    """Memória mórfica adaptativa"""
    def __init__(self, size=144):  # 12 * 12
        self.size = size
        self.memory = []
        self.campo_morfico = self._inicializar_campo()
    
    def _inicializar_campo(self) -> np.ndarray:
        """Inicializa campo mórfico usando proporção áurea"""
        phi = (1 + np.sqrt(5)) / 2
        campo = np.zeros((12, 12), dtype=complex)
        
        for i in range(12):
            for j in range(12):
                theta = 2 * np.pi * (i + j) / (12 * phi)
                campo[i, j] = np.exp(1j * theta) / np.sqrt(144)
        
        return campo
    
    def add(self, state: MorphicState):
        """Adiciona estado à memória mórfica"""
        if len(self.memory) >= self.size:
            self.memory.pop(0)
        self.memory.append(state)
        self._atualizar_campo(state)
    
    def _atualizar_campo(self, state: MorphicState):
        """Atualiza campo mórfico via ressonância"""
        # Fase do campo atual
        phase_atual = np.angle(self.campo_morfico)
        
        # Fase do novo estado
        phase_novo = np.angle(state.campo)
        
        # Ressonância via interferência quântica
        interferencia = np.exp(1j * (phase_atual + phase_novo))
        
        # Atualiza campo usando proporção áurea
        phi = (1 + np.sqrt(5)) / 2
        self.campo_morfico = (self.campo_morfico * (1 - 1/phi) + 
                            interferencia * (1/phi))
    
    def get_similar_states(self, current: MorphicState, n=5) -> List[MorphicState]:
        """Recupera estados similares via ressonância mórfica"""
        if not self.memory:
            return []
        
        # Calcula ressonância via interferência quântica
        ressonancias = []
        for state in self.memory:
            # Interferência entre campos
            interferencia = np.sum(current.campo * np.conj(state.campo))
            
            # Força da ressonância
            ressonancia = np.abs(interferencia) * state.coerencia
            ressonancias.append(ressonancia)
        
        # Encontra estados mais ressonantes
        indices = np.argsort(ressonancias)[-n:]
        return [self.memory[i] for i in indices]

class MorphicHawkingField:
    def __init__(self):
        """Inicializa campo mórfico de Hawking"""
        self.memory = MorphicMemory()
        self.historico_estados = []
        self.historico_radiacao = []
        self.historico_coerencia = []
        self.historico_ressonancia = []
        self.historico_precos = []
        
    def _calcular_entropia(self, campo: np.ndarray) -> float:
        """Calcula entropia do campo"""
        # Usa amplitude como distribuição de probabilidade
        prob = np.abs(campo)**2
        prob /= np.sum(prob)
        return stats.entropy(prob.flatten())
    
    def _calcular_coerencia(self, campo: np.ndarray) -> float:
        """Calcula coerência do campo"""
        # Usa amplitude do campo
        amplitude = np.abs(campo)
        phase = np.angle(campo)
        return np.abs(np.mean(np.exp(1j * phase)))
    
    def _calcular_ressonancia(self, campo: np.ndarray, 
                             estados_similares: List[MorphicState]) -> float:
        """Calcula ressonância com estados similares"""
        if not estados_similares:
            return 0.0
        
        # Média das interferências quânticas
        ressonancias = []
        for state in estados_similares:
            interferencia = np.sum(campo * np.conj(state.campo))
            ressonancias.append(np.abs(interferencia))
        
        return np.mean(ressonancias)
    
    def _gerar_radiacao_hawking(self, entropia: float, 
                               coerencia: float) -> np.ndarray:
        """Gera radiação Hawking baseada em entropia e coerência"""
        # Temperatura Hawking
        temperatura = entropia * (1 - coerencia)
        
        # Espectro de radiação
        radiacao = np.random.exponential(scale=1/temperatura, size=(12, 12))
        
        # Modula com campo mórfico
        return radiacao * np.abs(self.memory.campo_morfico)
    
    def processar_estado(self, preco: float) -> Dict[str, float]:
        """Processa novo estado de mercado"""
        # 1. Gera campo inicial
        campo = np.full((12, 12), preco, dtype=complex)
        campo *= np.exp(1j * np.angle(self.memory.campo_morfico))
        
        # 2. Calcula métricas
        entropia = self._calcular_entropia(campo)
        coerencia = self._calcular_coerencia(campo)
        
        # 3. Recupera estados similares
        estado_atual = MorphicState(
            campo=campo,
            entropia=entropia,
            coerencia=coerencia,
            ressonancia=0.0,
            timestamp=datetime.now()
        )
        estados_similares = self.memory.get_similar_states(estado_atual)
        
        # 4. Calcula ressonância
        ressonancia = self._calcular_ressonancia(campo, estados_similares)
        estado_atual.ressonancia = ressonancia
        
        # 5. Gera radiação Hawking
        radiacao = self._gerar_radiacao_hawking(entropia, coerencia)
        
        # 6. Atualiza memória
        self.memory.add(estado_atual)
        
        # 7. Registra histórico
        self.historico_estados.append(estado_atual)
        self.historico_radiacao.append(np.mean(radiacao))
        self.historico_coerencia.append(coerencia)
        self.historico_ressonancia.append(ressonancia)
        self.historico_precos.append(preco)
        
        return {
            'entropia': entropia,
            'coerencia': coerencia,
            'ressonancia': ressonancia,
            'radiacao': np.mean(radiacao)
        }
    
    def decidir_trade(self, metricas: Dict[str, float]) -> str:
        """Decide trade baseado nas métricas do campo"""
        # 1. Verifica coerência mínima
        if metricas['coerencia'] < 0.21:  # Horizonte de eventos
            return 'hold'
        
        # 2. Força do sinal
        forca_sinal = metricas['ressonancia'] * metricas['coerencia']
        
        # 3. Direção via radiação
        if forca_sinal > 0.618:  # Proporção áurea
            return 'buy' if metricas['radiacao'] > 0 else 'sell'
        
        return 'hold'
    
    def plotar_campo(self):
        """Plota visualização do campo mórfico"""
        if not self.historico_estados:
            return
            
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Campo Mórfico
        ax1 = plt.subplot(221)
        campo = np.abs(self.memory.campo_morfico)
        im1 = ax1.imshow(campo, cmap='viridis')
        ax1.set_title('Campo Mórfico')
        plt.colorbar(im1, ax=ax1)
        
        # 2. Métricas Temporais
        ax2 = plt.subplot(222)
        t = range(len(self.historico_coerencia))
        ax2.plot(t, self.historico_coerencia, 'b-', label='Coerência')
        ax2.plot(t, self.historico_ressonancia, 'r-', label='Ressonância')
        ax2.axhline(y=0.21, color='k', linestyle='--', label='Horizonte')
        ax2.set_title('Métricas Temporais')
        ax2.legend()
        
        # 3. Radiação Hawking
        ax3 = plt.subplot(223)
        ax3.plot(t, self.historico_radiacao, 'g-', label='Radiação')
        ax3.fill_between(t, self.historico_radiacao, alpha=0.3)
        ax3.set_title('Radiação Hawking')
        
        # 4. Preços e Sinais
        ax4 = plt.subplot(224)
        ax4.plot(t, self.historico_precos, 'k-', label='Preço')
        
        # Marca sinais
        for i in range(len(t)):
            metricas = {
                'coerencia': self.historico_coerencia[i],
                'ressonancia': self.historico_ressonancia[i],
                'radiacao': self.historico_radiacao[i]
            }
            decisao = self.decidir_trade(metricas)
            if decisao != 'hold':
                color = 'g' if decisao == 'buy' else 'r'
                ax4.axvline(x=i, color=color, alpha=0.3)
        
        ax4.set_title('Preços e Sinais')
        
        plt.tight_layout()
        plt.savefig('morphic_hawking_field.png')
        plt.close()

def simular_campo():
    """Simula campo mórfico de Hawking"""
    campo = MorphicHawkingField()
    
    # Simula preços
    precos = np.random.normal(100, 10, 100)
    precos = np.cumsum(precos) / 100 + 100
    
    print("Iniciando simulação do campo mórfico de Hawking...")
    for preco in precos:
        # Processa estado
        metricas = campo.processar_estado(preco)
        
        # Decide trade
        decisao = campo.decidir_trade(metricas)
        
        # Log
        print(f"\nPreço: ${preco:.2f}")
        print(f"Coerência: {metricas['coerencia']:.4f}")
        print(f"Ressonância: {metricas['ressonancia']:.4f}")
        print(f"Radiação: {metricas['radiacao']:.4f}")
        print(f"Decisão: {decisao}")
    
    # Plota resultados
    campo.plotar_campo()

if __name__ == "__main__":
    simular_campo()
