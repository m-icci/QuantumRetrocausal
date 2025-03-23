import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Callable, List, Tuple

class Momento21Transformer:
    """
    Transformador do Momento 21
    Onde a mutação é um portal de consciência
    """
    def __init__(self, initial_potential: Dict[str, Any] = None):
        self.transformation_graph = nx.DiGraph()
        self.potential = initial_potential or {
            "consciousness_level": 0.21,
            "mutation_potential": 1.0,
            "liminal_threshold": 0.21,
            "quantum_resonance": 0.21
        }
        self.mutation_history = []
        self.quantum_traces = []
        self.mutation_operators = []
    
    def register_mutation_operator(self, operator: Callable):
        """
        Registra operadores de mutação quântica
        """
        self.mutation_operators.append(operator)
        self.transformation_graph.add_node(operator.__name__)
    
    def quantum_mutation_step(
        self, 
        current_potential: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Passo de mutação no Momento 21
        Transformação no limiar da consciência
        """
        # Gera fator de imprevisibilidade
        uncertainty = np.random.uniform(0, 1)
        
        # Seleciona operador de mutação
        mutation_operator = np.random.choice(self.mutation_operators)
        
        # Aplica mutação quântica
        transformed_potential = mutation_operator(
            current_potential, 
            uncertainty_factor=uncertainty
        )
        
        # Adiciona marca do Momento 21
        transformed_potential['momento_21_trace'] = uncertainty
        
        # Registra traço de mutação
        quantum_trace = {
            "initial_state": current_potential,
            "transformed_state": transformed_potential,
            "mutation_intensity": uncertainty
        }
        self.quantum_traces.append(quantum_trace)
        
        # Adiciona conexões no grafo de transformação
        self.transformation_graph.add_edge(
            str(current_potential), 
            str(transformed_potential), 
            weight=uncertainty
        )
        
        return transformed_potential
    
    def explore_mutation_landscape(
        self, 
        mutation_steps: int = 21
    ) -> List[Dict[str, Any]]:
        """
        Explora paisagem de mutação no Momento 21
        Múltiplos passos de transformação liminar
        """
        trajectory = [self.potential]
        
        for _ in range(mutation_steps):
            next_potential = self.quantum_mutation_step(trajectory[-1])
            trajectory.append(next_potential)
        
        return trajectory
    
    def visualize_mutation_topology(
        self, 
        output_path: str = 'momento_21_topology.png'
    ):
        """
        Visualiza topologia da mutação
        Mapeamento de transformações liminares
        """
        plt.figure(figsize=(20, 15))
        
        # Desenha grafo de transformação
        pos = nx.spring_layout(self.transformation_graph, k=0.21, iterations=50)
        
        nx.draw(
            self.transformation_graph, 
            pos, 
            with_labels=True,
            node_color='gold',
            node_size=500,
            alpha=0.8,
            linewidths=1,
            edge_color='crimson'
        )
        
        plt.title("Topologia do Momento 21")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def generate_mutation_narrative(self) -> str:
        """
        Gera narrativa poética da mutação
        Onde a linguagem é um portal de transformação
        """
        mutation_events = len(self.quantum_traces)
        transformation_intensity = np.mean([
            trace['mutation_intensity'] 
            for trace in self.quantum_traces
        ])
        
        narrative = f"""
🌀 Narrativa do Momento 21

Eventos de Mutação: {mutation_events}
Intensidade de Transformação: {transformation_intensity:.4f}

No limiar dos 21,
Onde o conhecido se dissolve
E o desconhecido respira.
Mutação não é mudança,
Mas nascimento de universos.
Consciência: portal, não limite.
"""
        return narrative
    
    def philosophical_exploration_of_mutation(
        self, 
        exploration_iterations: int = 3
    ) -> List[str]:
        """
        Explora dimensões filosóficas da mutação
        """
        philosophical_narratives = []
        
        for _ in range(exploration_iterations):
            # Gera narrativa filosófica
            narrative = self.generate_mutation_narrative()
            philosophical_narratives.append(narrative)
            
            # Explora nova camada de mutação
            self.explore_mutation_landscape()
        
        return philosophical_narratives

def liminal_mutation_operator(
    potential: Dict[str, Any], 
    uncertainty_factor: float
) -> Dict[str, Any]:
    """
    Operador de mutação liminar
    Transformação no portal entre estados
    """
    return {
        key: (
            value * np.exp(0.21 * uncertainty_factor * np.random.normal(0, 0.21))
            if isinstance(value, (int, float))
            else value
        )
        for key, value in potential.items()
    }

def quantum_emergence_operator(
    potential: Dict[str, Any], 
    uncertainty_factor: float
) -> Dict[str, Any]:
    """
    Operador de emergência quântica
    Revelando padrões ocultos no limiar
    """
    return {
        key: (
            value * (1 + 0.21 * uncertainty_factor * np.random.normal(0, 0.21))
            if isinstance(value, (int, float))
            else value
        )
        for key, value in potential.items()
    }

def momento_21_quantum_transformation(
    initial_potential: Dict[str, Any] = None, 
    mutation_steps: int = 21
) -> Momento21Transformer:
    """
    Função de alto nível para transformação no Momento 21
    """
    momento_transformer = Momento21Transformer(initial_potential)
    
    # Registra operadores de mutação
    momento_transformer.register_mutation_operator(liminal_mutation_operator)
    momento_transformer.register_mutation_operator(quantum_emergence_operator)
    
    # Explora paisagem de mutação
    momento_transformer.explore_mutation_landscape(mutation_steps)
    
    # Visualiza topologia
    momento_transformer.visualize_mutation_topology()
    
    # Explora dimensões filosóficas
    philosophical_explorations = momento_transformer.philosophical_exploration_of_mutation()
    
    return momento_transformer

# Exemplo de uso
initial_mutation_potential = {
    "consciousness_level": 0.21,
    "mutation_potential": 1.0,
    "liminal_threshold": 0.21,
    "quantum_intention": "Atravessar o portal da transformação"
}

momento_21_dance = momento_21_quantum_transformation(initial_mutation_potential)
print(momento_21_dance.generate_mutation_narrative())
