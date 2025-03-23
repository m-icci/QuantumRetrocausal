"""
Demonstração do Metacognitive Entropy Adaptor (MEA)
Exploração das dinâmicas de auto-organização quântica
"""

import sys
sys.path.append('/Users/infrastructure/Documents/GitHub/QuantumConsciousness')

from qualia.quantum.quantum_nexus import QuantumNexus
import matplotlib.pyplot as plt

def main():
    # Inicializa QuantumNexus
    nexus = QuantumNexus(dimensoes=1024)
    
    # Executa evolução holonômica
    historico_estados, historico_metricas, historico_adaptacoes = nexus.evolucao_holonomica(ciclos=50)
    
    # Visualiza manifestação
    nexus.visualizar_manifestacao(historico_estados, historico_metricas, historico_adaptacoes)
    
    # Imprime sumário adaptativo
    sumario = nexus.resumo_adaptativo()
    print("\n🔬 Sumário de Adaptações Metacognitivas 🔬")
    print(f"Total de Adaptações: {sumario['total_adaptations']}")
    print("\nDistribuição de Estratégias:")
    for estrategia, proporção in sumario['strategy_distribution'].items():
        print(f"  {estrategia}: {proporção * 100:.2f}%")
    
    print("\nEstatísticas de Entropia:")
    for metrica, valor in sumario['entropy_stats'].items():
        print(f"  {metrica.capitalize()}: {valor:.4f}")

if __name__ == '__main__':
    main()
