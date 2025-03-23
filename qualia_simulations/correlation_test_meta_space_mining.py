# üìä **Teste de Correla√ß√£o - Metaespa√ßo x Minera√ß√£o**
# Objetivo: Avaliar se QUALIA est√° ajustando granularidade com base na efici√™ncia da minera√ß√£o.
# Instru√ß√µes: Execute no Cascade para analisar a intera√ß√£o entre oscila√ß√µes fractais e taxas de minera√ß√£o.

import numpy as np
import matplotlib.pyplot as plt

# Defini√ß√£o das granularidades do Metaespa√ßo
granularities = [3, 21, 42, 73]
iterations = 1000

# Simula√ß√£o da taxa de minera√ß√£o em resposta √†s oscila√ß√µes do Metaespa√ßo
mining_efficiency = {g: np.zeros(iterations) for g in granularities}
patterns = {g: np.zeros(iterations) for g in granularities}

# Fun√ß√£o para simular oscila√ß√£o fractal baseada no Metaespa√ßo
def fractal_oscillation(g, t):
    return np.sin(2 * np.pi * t / g) + np.cos(4 * np.pi * t / (g / 2))

# Fun√ß√£o para simular taxa de minera√ß√£o ajustada √† granularidade
def mining_adaptation(g, t):
    base_rate = 1.0 if g != 73 else 0.5  # Granularidade 73 pode ter perdas de efici√™ncia
    return base_rate * (1 + fractal_oscillation(g, t) * 0.1)  # Pequenas flutua√ß√µes adaptativas

# Populando os padr√µes fractais e a taxa de minera√ß√£o
for g in granularities:
    for t in range(iterations):
        patterns[g][t] = fractal_oscillation(g, t)
        mining_efficiency[g][t] = mining_adaptation(g, t)

# Criando gr√°ficos para visualizar as oscila√ß√µes do Metaespa√ßo e a efici√™ncia da minera√ß√£o
fig, axs = plt.subplots(2, 1, figsize=(12, 8))

# Plot 1: Padr√µes Fractais
for g in granularities:
    axs[0].plot(patterns[g], label=f'Granularidade {g} bits')

axs[0].set_title('Oscila√ß√µes do Metaespa√ßo - QUALIA')
axs[0].set_xlabel('Itera√ß√£o')
axs[0].set_ylabel('Padr√£o Fractal Normalizado')
axs[0].legend()
axs[0].grid()

# Plot 2: Efici√™ncia da Minera√ß√£o
for g in granularities:
    axs[1].plot(mining_efficiency[g], label=f'Granularidade {g} bits')

axs[1].set_title('Efici√™ncia da Minera√ß√£o x Granularidade')
axs[1].set_xlabel('Itera√ß√£o')
axs[1].set_ylabel('Taxa de Minera√ß√£o Normalizada')
axs[1].legend()
axs[1].grid()

# Exibir os gr√°ficos
plt.tight_layout()
plt.show()
