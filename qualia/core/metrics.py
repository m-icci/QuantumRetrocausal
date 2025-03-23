"""
Métricas Avançadas para Sistema de Trading Quântico

Este módulo define métricas complexas para análise de campos quânticos:
- Coerência
- Entropia
- Complexidade
- Retrocausalidade
"""

import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class QuantumMetrics:
    """
    Classe para cálculo de métricas quânticas avançadas
    """
    def __init__(self, dimension: int = 64):
        self.dimension = dimension
        self.epsilon = 1e-10
        self.entropia_bins = 50  # Adicionado para o novo método

    def calculate_coherence(self, field: np.ndarray) -> float:
        """
        Calcula coerência baseada na matriz densidade reduzida.
        
        Args:
            field (np.ndarray): Campo quântico
        
        Returns:
            float: Valor de coerência
        """
        # Tratamento para campos não quadrados
        if field.ndim == 1:
            field = np.outer(field, field)
        
        # Normalização da matriz densidade
        rho = field / np.trace(field)
        
        # Cálculo de coerência usando traço
        diag_elements = np.diag(rho)
        coherence = np.sum(diag_elements**2)
        
        return float(np.clip(coherence, 0, 1))

    def calculate_entropy(self, field: np.ndarray) -> float:
        """
        Calcula entropia de von Neumann com tratamento robusto de erros.
        
        Args:
            field (np.ndarray): Campo quântico
        
        Returns:
            float: Valor de entropia
        """
        # Validação de input
        if field is None:
            logger.error("Input field is None")
            return 0.0
        
        # Tratamento para campos não quadrados
        if field.ndim == 1:
            logger.info("Converting 1D field to density matrix")
            field = np.outer(field, field)
        
        # Verificações de dimensão e tipo
        if not isinstance(field, np.ndarray):
            logger.error(f"Invalid input type: {type(field)}")
            return 0.0
        
        if field.ndim != 2 or field.shape[0] != field.shape[1]:
            logger.error(f"Invalid field shape: {field.shape}")
            return 0.0
        
        # Caso especial: matriz identidade
        if np.allclose(field, np.eye(field.shape[0])):
            return 0.0
        
        # Calcula autovalores com tratamento de erro
        try:
            # Garantir simetria e hermetianidade
            field = (field + field.T.conj()) / 2
            
            # Normalização
            field_trace = np.trace(field)
            if field_trace == 0:
                logger.warning("Zero trace detected, normalizing")
                field_trace = 1.0
            
            normalized_field = field / field_trace
            
            # Cálculo de autovalores
            eigenvals = np.linalg.eigvalsh(normalized_field)
            
            # Filtra autovalores válidos
            valid_eigenvals = eigenvals[eigenvals > self.epsilon]
            
            if len(valid_eigenvals) == 0:
                logger.warning("No valid eigenvalues found")
                return 0.0
            
            # Normaliza autovalores
            valid_eigenvals /= np.sum(valid_eigenvals)
            
            # Calcula entropia de Shannon
            entropy = -np.sum(valid_eigenvals * np.log2(valid_eigenvals + self.epsilon))
            
            logger.debug(f"Entropy calculation: {entropy}")
            return float(np.abs(np.clip(entropy, 0, np.log2(len(eigenvals)))))
        
        except np.linalg.LinAlgError as e:
            logger.error(f"Linear algebra error in entropy calculation: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Unexpected error in entropy calculation: {e}")
            return 0.0

    def calculate_complexity(self, field: np.ndarray) -> float:
        """
        Calcula complexidade do campo quântico.
        
        Args:
            field (np.ndarray): Campo quântico
        
        Returns:
            float: Valor de complexidade
        """
        try:
            coherence = self.calculate_coherence(field)
            entropy = self.calculate_entropy(field)
            complexity = coherence * (1 + entropy)
            return float(complexity)
        except Exception as e:
            logger.error(f"Erro no cálculo de complexidade: {e}")
            return 0.0

    def get_quantum_metrics(self, quantum_state: np.ndarray) -> Dict[str, float]:
        """
        Calcula métricas quânticas fundamentais com maior robustez
        
        Args:
            quantum_state: Estado quântico a ser analisado
        
        Returns:
            Dicionário com métricas de entropia, complexidade e coerência
        """
        # Normalização do estado quântico
        quantum_state_norm = quantum_state / np.linalg.norm(quantum_state)
        
        # Distribuição de probabilidade
        hist, _ = np.histogram(quantum_state_norm, bins=self.entropia_bins, density=True)
        hist = hist[hist > 0]  # Remove zeros para evitar log(0)
        
        # Entropia de Shannon
        try:
            entropia = -np.sum(hist * np.log2(hist))
            entropia = max(0, entropia)  # Garantir não-negatividade
        except Exception as e:
            logger.warning(f"Erro no cálculo da entropia: {e}")
            entropia = 0
        
        # Complexidade
        try:
            variancia = np.var(quantum_state_norm)
            variancia_segura = max(variancia, 1e-10)
            complexidade = entropia * (1 + np.log(1 + variancia_segura))
            complexidade = max(0, complexidade)  # Garantir não-negatividade
        except Exception as e:
            logger.warning(f"Erro no cálculo da complexidade: {e}")
            complexidade = 0
        
        # Coerência (usando correlação)
        try:
            if np.all(quantum_state_norm == quantum_state_norm[0]) or variancia_segura < 1e-10:
                coerencia = 0
            else:
                # Usa correlação entre estados deslocados
                estados_deslocados = [
                    np.roll(quantum_state_norm, shift) 
                    for shift in range(1, min(5, len(quantum_state_norm)//2))
                ]
                correlacoes = [
                    np.abs(np.corrcoef(quantum_state_norm, estado)[0, 1]) 
                    for estado in estados_deslocados
                ]
                coerencia = np.mean(correlacoes)
        except Exception as e:
            logger.warning(f"Erro no cálculo da coerência: {e}")
            coerencia = 0
        
        # Métricas finais
        metricas = {
            "entropia": float(entropia),
            "complexidade": float(complexidade),
            "coerencia": float(coerencia)
        }
        
        return metricas

    def calculate_retrocausality_index(self, current_field: np.ndarray, past_field: np.ndarray) -> float:
        """
        Calcula índice de retrocausalidade entre campos atual e passado.
        
        Args:
            current_field (np.ndarray): Campo quântico atual
            past_field (np.ndarray): Campo quântico passado
        
        Returns:
            float: Índice de retrocausalidade
        """
        try:
            current_entropy = self.calculate_entropy(current_field)
            past_entropy = self.calculate_entropy(past_field)
            
            # Correlação entre campos
            correlation = np.abs(np.corrcoef(current_field.flatten(), past_field.flatten())[0, 1])
            
            # Índice de retrocausalidade
            retrocausality_index = correlation * (1 + abs(current_entropy - past_entropy))
            
            # Garantir valor não-negativo
            return float(max(0, retrocausality_index))
        
        except Exception as e:
            logger.error(f"Erro no cálculo de retrocausalidade: {e}")
            return 0.0

    def visualize_entropy_dynamics(self, entropy_traces: list = None, save_path: str = '/Users/infrastructure/Desktop/qualia_entropy_dynamics.png') -> Dict[str, Any]:
        """
        Visualização multidimensional da evolução da entropia.
        
        Args:
            entropy_traces (list, optional): Lista de traces de entropia. 
                Cada trace deve ser um dicionário com 'method' e 'value' keys.
            save_path (str, optional): Caminho para salvar a visualização.
        
        Returns:
            Dict[str, Any]: Dicionário com índices de complexidade e estatísticas de entropia.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Preparação padrão de dados se nenhum trace for fornecido
        if entropy_traces is None:
            # Gera dados sintéticos para demonstração
            np.random.seed(42)
            methods = ['shannon', 'quantum', 'kolmogorov']
            entropy_traces = [
                {'method': method, 'value': np.random.uniform(0, 1, 50)} 
                for method in methods
            ]
        
        # Preparação dos dados
        methods = list(set(trace['method'] for trace in entropy_traces))
        entropy_data = {method: [] for method in methods}
        
        for trace in entropy_traces:
            entropy_data[trace['method']].extend(trace['value'] if isinstance(trace['value'], list) else [trace['value']])
        
        # Configuração do plot
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Evolução Temporal da Entropia
        plt.subplot(2, 2, 1)
        for method in methods:
            plt.plot(entropy_data[method], label=f'Entropia {method.capitalize()}')
        plt.title('Evolução Temporal da Entropia')
        plt.xlabel('Iterações')
        plt.ylabel('Valor de Entropia')
        plt.legend()
        
        # Subplot 2: Distribuição de Probabilidade
        plt.subplot(2, 2, 2)
        sns.violinplot(data=[entropy_data[method] for method in methods])
        plt.title('Distribuição de Entropia')
        plt.xticks(range(len(methods)), methods)
        plt.ylabel('Valores de Entropia')
        
        # Subplot 3: Mapa de Calor de Transições
        plt.subplot(2, 2, 3)
        entropy_matrix = np.array([entropy_data[method] for method in methods])
        sns.heatmap(entropy_matrix, cmap='viridis', 
                    annot=True, fmt='.2f', 
                    xticklabels=range(len(entropy_data[methods[0]])),
                    yticklabels=methods)
        plt.title('Mapa de Transições de Entropia')
        
        # Subplot 4: Análise de Complexidade
        plt.subplot(2, 2, 4)
        complexity_scores = [
            np.mean(entropy_data[method]) * np.std(entropy_data[method]) 
            for method in methods
        ]
        plt.bar(methods, complexity_scores)
        plt.title('Índice de Complexidade')
        plt.ylabel('Complexidade')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        return {
            'complexity_index': complexity_scores,
            'entropy_stats': {
                method: {
                    'mean': np.mean(entropy_data[method]),
                    'std': np.std(entropy_data[method])
                } for method in methods
            }
        }
