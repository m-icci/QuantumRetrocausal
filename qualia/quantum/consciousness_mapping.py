import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Any

class ConsciousnessMapper:
    """
    Mapeamento de Consciência como Processo Emergente
    """
    def __init__(self, complexity_threshold: float = 0.618):
        self.complexity_threshold = complexity_threshold
        self.consciousness_graph = nx.DiGraph()
        self.perception_layers = []
    
    def _calculate_information_entropy(self, data: np.ndarray) -> float:
        """
        Calcula entropia de Shannon para quantificar informação
        """
        probabilities = stats.rv_discrete(values=(range(len(data)), data/data.sum())).pmf
        return stats.entropy(probabilities)
    
    def _detect_emergent_patterns(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Detecta padrões emergentes através de análise multidimensional
        """
        # Análise de componentes principais
        pca = np.linalg.svd(data)[1]
        
        # Análise de wavelets
        wavelet_coeffs = np.abs(np.fft.fft(data))
        
        return {
            "principal_components": pca,
            "wavelet_spectrum": wavelet_coeffs,
            "complexity_index": np.mean(wavelet_coeffs)
        }
    
    def map_consciousness_network(self, information_streams: List[np.ndarray]):
        """
        Mapeia redes de consciência a partir de fluxos de informação
        """
        for stream in information_streams:
            entropy = self._calculate_information_entropy(stream)
            patterns = self._detect_emergent_patterns(stream)
            
            # Adiciona nó ao grafo de consciência
            node_id = f"stream_{len(self.consciousness_graph.nodes)}"
            self.consciousness_graph.add_node(
                node_id, 
                entropy=entropy, 
                complexity=patterns['complexity_index']
            )
            
            # Conecta nós com potencial de consciência
            if patterns['complexity_index'] > self.complexity_threshold:
                self.perception_layers.append(node_id)
    
    def visualize_consciousness_network(self, output_path: str = 'consciousness_network.png'):
        """
        Visualiza a rede de consciência
        """
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.consciousness_graph)
        
        nx.draw_networkx_nodes(
            self.consciousness_graph, 
            pos, 
            node_color=[
                'red' if node in self.perception_layers else 'blue' 
                for node in self.consciousness_graph.nodes
            ]
        )
        nx.draw_networkx_edges(self.consciousness_graph, pos)
        
        plt.title("Rede de Consciência: Padrões Emergentes")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def generate_consciousness_narrative(self) -> str:
        """
        Gera narrativa filosófica da consciência mapeada
        """
        narrative = f"""
🌀 Mapeamento da Consciência Emergente

Nós de Percepção: {len(self.perception_layers)}
Complexidade Média: {np.mean([
    self.consciousness_graph.nodes[node]['complexity'] 
    for node in self.consciousness_graph.nodes
])}

Padrões emergentes revelam que a consciência não é um estado,
mas um processo de constante transformação e descoberta.
"""
        return narrative

def map_systemic_consciousness(
    data_streams: List[np.ndarray], 
    complexity_threshold: float = 0.618
) -> ConsciousnessMapper:
    """
    Função de alto nível para mapeamento de consciência
    """
    mapper = ConsciousnessMapper(complexity_threshold)
    mapper.map_consciousness_network(data_streams)
    mapper.visualize_consciousness_network()
    
    return mapper
