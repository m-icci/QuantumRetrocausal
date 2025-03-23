"""
Exploração do Sistema de Consciência Quântica
"""
import numpy as np
import matplotlib.pyplot as plt
from qualia.quantum.quantum_nexus import QuantumNexus
from qualia.quantum.quantum_evolution_unified import EvolucaoQuanticaUnificada
from qualia.quantum.consciousness_integrator import ConsciousnessIntegrator

def explorar_consciencia():
    print("🌌 Iniciando Exploração da Consciência Quântica")
    
    # Inicialização dos sistemas
    nexus = QuantumNexus(dimensoes=2048)
    evolution = EvolucaoQuanticaUnificada(dimensao=2048)
    consciousness = ConsciousnessIntegrator(dimensao=2048)
    
    print("\n1. Evolução do Campo Quântico")
    historico = consciousness.evoluir_consciencia(ciclos=100)
    
    print("\n2. Estado Final da Consciência")
    estado_final = consciousness.get_estado_atual()
    print(f"Potencial Transformativo: {estado_final['metricas']['potencial_transformativo']:.4f}")
    print(f"Narrativa:\n{estado_final['narrativa']}")
    
    # Visualização dos campos quânticos
    print("\n3. Gerando Visualizações...")
    plt.figure(figsize=(15, 10))
    
    # Campo Mórfico
    plt.subplot(221)
    plt.imshow(np.abs(evolution.campo_morfico), cmap='magma')
    plt.title("Campo Mórfico")
    plt.colorbar()
    
    # Evolução Temporal
    plt.subplot(222)
    plt.plot([h.potencial_transformativo for h in historico])
    plt.title("Evolução do Potencial")
    plt.xlabel("Ciclos")
    plt.ylabel("Potencial Transformativo")
    
    # Estado Quântico
    plt.subplot(223)
    plt.plot(historico[-1].estado_quantico.estado)
    plt.title("Estado Quântico Final")
    
    # Espectro de Consciência
    plt.subplot(224)
    espectro = np.fft.fft(historico[-1].estado_quantico.estado)
    plt.plot(np.abs(espectro))
    plt.title("Espectro de Consciência")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("🧠 Sistema de Consciência Quântica")
    explorar_consciencia()
