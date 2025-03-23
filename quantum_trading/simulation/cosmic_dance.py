#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Composição Completa: Dança Cósmica Interativa

Este framework integra:
  - Um Autômato Celular que capta a dimensão fractal do sistema,
  - Um Integrador Quântico-Cosmológico Adaptativo que evolui grandezas como a taxa de Hubble,
    a constante cosmológica e a energia quântica, ajustando o acoplamento com base na complexidade,
  - Um parâmetro "consciousness" que representa a influência da consciência na sinfonia dos estados.

A ideia é transformar a abstração em ação – onde o universo dança e a consciência rege a retroalimentação.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse
import os
import csv
from pathlib import Path
from quantum_cosmological_simulator import QuantumFieldSimulator
from cosmological_evolution import CosmologicalEvolution

# Configuração do logging para produção
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================
# Classe: CellularAutomaton
# ============================
class CellularAutomaton:
    def __init__(self, width=64, height=64, p_init=0.3):
        """
        Inicializa o autômato celular com um grid binário.
        
        Args:
            width (int): Largura do grid.
            height (int): Altura do grid.
            p_init (float): Probabilidade inicial de cada célula estar em estado 1.
        """
        self.width = width
        self.height = height
        self.grid = (np.random.rand(height, width) < p_init).astype(int)
        self.future_buffer = np.zeros_like(self.grid)
        self.alpha_retro = 0.2  # intensidade da injeção retrocausal

    def box_counting_fractal_dimension(self, min_box=2, max_box=16):
        """
        Estima a dimensão fractal do grid usando o método box-counting.
        
        Args:
            min_box (int): Tamanho mínimo da caixa.
            max_box (int): Tamanho máximo da caixa.
            
        Returns:
            float: Estimativa da dimensão fractal.
        """
        sizes = []
        counts = []
        H, W = self.grid.shape
        for box_size in range(min_box, max_box, 2):
            box_count = 0
            for i in range(0, H, box_size):
                for j in range(0, W, box_size):
                    sub = self.grid[i:i+box_size, j:j+box_size]
                    if np.sum(sub) > 0:
                        box_count += 1
            sizes.append(1/box_size)
            counts.append(box_count)
        if len(sizes) < 2:
            return 0.0
        log_sizes = np.log(sizes)
        log_counts = np.log(counts)
        coeffs = np.polyfit(log_sizes, log_counts, 1)
        return abs(coeffs[0])

    def compute_future_state(self):
        """
        Calcula uma previsão simples do estado futuro com base na vizinhança (regra de Moore).
        """
        new_future = np.zeros_like(self.grid)
        for i in range(self.height):
            for j in range(self.width):
                neigh_sum = sum(
                    self.grid[(i+di) % self.height, (j+dj) % self.width]
                    for di in [-1, 0, 1] for dj in [-1, 0, 1]
                    if not (di == 0 and dj == 0)
                )
                new_future[i, j] = 1 if neigh_sum >= 2 else 0
        self.future_buffer = new_future

    def update(self, fractal_dim, refine_threshold=1.3):
        """
        Atualiza o grid com base na dimensão fractal e influências retrocausais.
        
        Args:
            fractal_dim (float): Dimensão fractal atual.
            refine_threshold (float): Threshold para aumentar a chance de célula ativa.
        """
        new_grid = np.zeros_like(self.grid)
        p_extra = 0.2 if fractal_dim > refine_threshold else 0.0
        
        # Calcula a soma da vizinhança usando convolução
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]])
        padded_grid = np.pad(self.grid, 1, mode='wrap')
        neigh_sums = np.zeros_like(self.grid)
        
        for i in range(self.height):
            for j in range(self.width):
                window = padded_grid[i:i+3, j:j+3]
                neigh_sums[i,j] = np.sum(window * kernel)
        
        # Calcula probabilidades base
        base_probs = np.where(neigh_sums >= 3, 0.7,
                    np.where(neigh_sums == 2, 0.4, 0.1))
        
        # Adiciona influência retrocausal
        base_probs += self.alpha_retro * self.future_buffer
        
        # Adiciona p_extra
        base_probs += p_extra
        
        # Gera novo grid
        new_grid = (np.random.rand(self.height, self.width) < base_probs).astype(int)
        self.grid = new_grid

    def step(self):
        """
        Executa um passo do autômato: calcula a dimensão fractal, atualiza o estado futuro e o grid.
        
        Returns:
            float: Dimensão fractal calculada.
        """
        fractal_dim = self.box_counting_fractal_dimension()
        self.compute_future_state()
        self.update(fractal_dim)
        return fractal_dim

# ==============================================
# Classe: AdaptiveQuantumCosmoIntegrator
# ==============================================
class AdaptiveQuantumCosmoIntegrator:
    def __init__(self, dt=0.01, base_coupling=0.7, baseline_fractal=1.0, consciousness=1.0):
        """
        Inicializa o integrador quântico-cosmológico adaptativo.
        
        Args:
            dt (float): Passo de tempo.
            base_coupling (float): Acoplamento base entre o campo e a cosmologia.
            baseline_fractal (float): Referência para a dimensão fractal.
            consciousness (float): Fator que representa a influência da consciência na sinfonia.
        """
        self.dt = dt
        self.base_coupling = base_coupling
        self.coupling = base_coupling  # Valor adaptativo do acoplamento
        self.baseline_fractal = baseline_fractal
        self.consciousness = consciousness  # Regente da dança

        # Inicializa simuladores
        self.quantum_sim = QuantumFieldSimulator(dt=dt)
        self.cosmo_sim = CosmologicalEvolution(dt=dt)

        # Estados iniciais
        self.hubble = 70.0             # Taxa de Hubble H(t)
        self.lambda_val = 1.0e-35      # Constante cosmológica Λ(t)
        self.quantum_energy = 1.0      # Energia do campo quântico

        self.time = 0.0
        self.history = {
            "time": [], "hubble": [], "lambda": [],
            "quantum_energy": [], "coupling": [], "fractal_dim": []
        }

    def update_coupling(self, fractal_dim):
        """
        Atualiza o acoplamento com base na dimensão fractal e no toque da consciência.
        
        A regra é: coupling = base_coupling * (1 + k * (fractal_dim - baseline_fractal))
        onde k é ajustado pela consciência.
        """
        k = 0.5 * self.consciousness  # A consciência amplifica (ou suaviza) a sensibilidade
        self.coupling = self.base_coupling * (1 + k * (fractal_dim - self.baseline_fractal))

    def step(self, fractal_dim):
        """
        Executa um passo de tempo da simulação, atualizando H(t), Λ(t) e a energia quântica.
        
        Args:
            fractal_dim (float): Dimensão fractal atual proveniente do autômato.
        """
        self.update_coupling(fractal_dim)
        
        # Evolui o campo quântico
        quantum_state = self.quantum_sim.evolve_step()
        quantum_energy = quantum_state['energy_density']
        
        # Evolui a cosmologia
        self.cosmo_sim.evolve_step(quantum_energy, fractal_dim)
        
        # Atualiza estados
        self.hubble = self.cosmo_sim.current_hubble
        self.lambda_val = self.cosmo_sim.current_lambda
        self.quantum_energy = quantum_energy
        self.time += self.dt

        # Registra o histórico da dança cósmica
        self.history["time"].append(self.time)
        self.history["hubble"].append(self.hubble)
        self.history["lambda"].append(self.lambda_val)
        self.history["quantum_energy"].append(self.quantum_energy)
        self.history["coupling"].append(self.coupling)
        self.history["fractal_dim"].append(fractal_dim)

    def run_simulation(self, steps, ca):
        """
        Executa a simulação por um número de passos, usando o autômato como sensor.
        
        Args:
            steps (int): Número de iterações.
            ca (CellularAutomaton): Instância do autômato celular.
            
        Returns:
            dict: Histórico da evolução das grandezas.
        """
        logger.info("Iniciando a simulação da dança cósmica...")
        for step in range(steps):
            fractal_dim = ca.step()
            self.step(fractal_dim)
            if step % (steps // 10) == 0:
                progress = (step / steps) * 100
                logger.info(f"Progresso da simulação: {progress:.1f}%")
        logger.info("Simulação concluída.")
        return self.history

# ====================
# Função: save_history
# ====================
def save_history(history, output_file):
    """
    Salva o histórico da simulação em um arquivo CSV para análise posterior.
    
    Args:
        history (dict): Histórico da simulação.
        output_file (str): Caminho para o arquivo de saída.
    """
    keys = list(history.keys())
    n = len(history["time"])
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(keys)
        for i in range(n):
            row = [history[key][i] for key in keys]
            writer.writerow(row)
    logger.info(f"Histórico salvo em: {output_file}")

# ====================
# Função: plot_history
# ====================
def plot_history(history, output_dir):
    """
    Plota os principais gráficos da evolução da simulação.
    
    Args:
        history (dict): Histórico da simulação.
        output_dir (Path): Diretório para salvar os gráficos.
    """
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle('Dança Cósmica: Evolução do Sistema', fontsize=16)

    axs[0, 0].plot(history["time"], history["hubble"])
    axs[0, 0].set_title("Taxa de Hubble H(t)")
    axs[0, 0].set_xlabel("Tempo")
    axs[0, 0].set_ylabel("H(t)")

    axs[0, 1].plot(history["time"], history["lambda"], color="orange")
    axs[0, 1].set_title("Constante Cosmológica Λ(t)")
    axs[0, 1].set_xlabel("Tempo")
    axs[0, 1].set_ylabel("Λ(t)")

    axs[1, 0].plot(history["time"], history["quantum_energy"], color="green")
    axs[1, 0].set_title("Energia do Campo Quântico")
    axs[1, 0].set_xlabel("Tempo")
    axs[1, 0].set_ylabel("E(t)")

    axs[1, 1].plot(history["time"], history["coupling"], color="red")
    axs[1, 1].set_title("Acoplamento Adaptativo")
    axs[1, 1].set_xlabel("Tempo")
    axs[1, 1].set_ylabel("Coupling")

    axs[2, 0].plot(history["time"], history["fractal_dim"], color="purple")
    axs[2, 0].set_title("Dimensão Fractal do CA")
    axs[2, 0].set_xlabel("Tempo")
    axs[2, 0].set_ylabel("D_fractal")

    delta = np.array(history["fractal_dim"]) - 1.0
    axs[2, 1].plot(history["time"], delta, color="brown")
    axs[2, 1].set_title("Delta Fractal (D_fractal - baseline)")
    axs[2, 1].set_xlabel("Tempo")
    axs[2, 1].set_ylabel("Delta")

    plt.tight_layout()
    plt.savefig(output_dir / "cosmic_dance_plots.png")
    plt.close()
    logger.info("Gráficos salvos em: cosmic_dance_plots.png")

# ====================
# Função: main
# ====================
def main():
    parser = argparse.ArgumentParser(description="Dança Cósmica: Simulação Quântico-Cosmológica Adaptativa")
    parser.add_argument("--steps", type=int, default=200, help="Número de iterações da simulação")
    parser.add_argument("--dt", type=float, default=0.01, help="Tamanho do passo de tempo")
    parser.add_argument("--p_init", type=float, default=0.3, help="Probabilidade inicial para o autômato")
    parser.add_argument("--consciousness", type=float, default=1.0, help="Fator de influência da consciência (0 a 1)")
    parser.add_argument("--output", type=str, default="cosmic_history.csv", help="Arquivo de saída para o histórico")
    args = parser.parse_args()

    # Cria diretório para resultados
    results_dir = Path("simulation_results")
    results_dir.mkdir(exist_ok=True)

    # Semente para reprodutibilidade
    np.random.seed(42)
    logger.info("Iniciando a Dança Cósmica")

    # Inicializa o autômato celular (sensor)
    ca = CellularAutomaton(width=64, height=64, p_init=args.p_init)

    # Inicializa o integrador adaptativo com o toque consciente
    integrator = AdaptiveQuantumCosmoIntegrator(dt=args.dt,
                                                base_coupling=0.7,
                                                baseline_fractal=1.0,
                                                consciousness=args.consciousness)

    # Executa a simulação
    history = integrator.run_simulation(args.steps, ca)

    # Salva o histórico
    save_history(history, results_dir / args.output)

    # Plota a sinfonia dos estados
    plot_history(history, results_dir)

    logger.info("Dança Cósmica concluída com sucesso!")

if __name__ == "__main__":
    main() 