#!/usr/bin/env python3
"""
Módulo de Simulação de Mineração para Comparação QUALIA vs ASIC

Este módulo implementa uma simulação comparativa avançada entre o minerador QUALIA e ASICs tradicionais,
analisando desempenho, consumo de energia e lucratividade em diferentes cenários.

Autor: QUALIA Monero Team
Data: 2025-03-16
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import json
from scipy.stats import norm
from matplotlib.ticker import FuncFormatter

# Configuração do estilo de visualização
plt.style.use('ggplot')
sns.set(font_scale=1.1)
sns.set_style("whitegrid")


class MiningSimulation:
    def __init__(self, scenario='default', simulation_days=365, log_scale=False, output_dir='simulation_results'):
        """
        Inicializa a simulação de mineração.
        
        Parâmetros:
            scenario (str): Cenário predefinido a ser utilizado
            simulation_days (int): Número de dias para simulação
            log_scale (bool): Usar escala logarítmica em alguns gráficos
            output_dir (str): Diretório para saída de arquivos
        """
        # Inicializar log de simulação
        self.simulation_log = []
        
        # Configurações básicas
        self.simulation_days = simulation_days
        self.log_scale = log_scale
        self.output_dir = output_dir
        
        # Parâmetros de mercado
        self.monero_price = 150  # USD
        self.price_volatility = 0.02  # Volatilidade diária
        self.price_trend = 1.0002  # Tendência de preço (0.02% ao dia)
        self.block_time = 2 * 60  # 2 minutos por bloco
        self.blocks_per_day = 24 * 60 * 60 / self.block_time
        self.difficulty = 300e9  # Dificuldade inicial da rede
        self.difficulty_adjustment = 1.0001  # Fator de ajuste de dificuldade
        self.difficulty_volatility = 0.005
        self.reward_per_block = 0.6  # XMR
        
        # Parâmetros ASIC
        self.asic_hashrate = 15000  # H/s
        self.asic_power = 1100  # Watts
        self.asic_hardware_cost = 4000  # USD inicial
        self.asic_maintenance_factor = 0.001  # 0.1% do custo por dia
        self.asic_failure_rate = 0.0005  # Probabilidade de falha diária
        self.asic_repair_cost_factor = 0.2  # 20% do custo do hardware
        self.asic_degradation_rate = 0.0002  # Degradação diária
        
        # Parâmetros QUALIA
        self.qualia_hashrate = 9500  # H/s
        self.qualia_power = 550  # Watts
        self.qualia_hardware_cost = 3000  # USD inicial
        self.qualia_maintenance_factor = 0.0005  # 0.05% do custo por dia
        self.qualia_failure_rate = 0.0002  # Probabilidade de falha diária
        self.qualia_repair_cost_factor = 0.15  # 15% do custo do hardware
        self.qualia_degradation_rate = 0.0001  # Degradação diária
        
        # Parâmetros específicos QUALIA
        self.qualia_resonance_factor = 1.1  # Fator de ressonância quântica
        self.qualia_adaptive_learning = True  # Habilitar aprendizado adaptativo
        self.qualia_learning_rate = 0.01  # Taxa de aprendizagem
        self.qualia_quantum_optimization = True  # Otimização quântica
        self.qualia_entanglement_boost = 1.05  # Fator de boost por entrelaçamento
        
        # Custos operacionais
        self.electricity_cost = 0.12  # USD por kWh
        self.hardware_lifespan = 730  # Dias (2 anos)
        
        # Inicializar estruturas de dados para resultados
        self.results_df = None
        self.summary = {}
        self.events = []
        
        # Inicializar valores atuais
        self.current_difficulty = self.difficulty
        self.current_monero_price = self.monero_price
        
        # Aplicar cenário se solicitado
        if scenario != 'default':
            self._apply_scenario(scenario)
            
        # Criar diretório de saída se necessário
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Registrar inicialização
        self._log_event("Simulação inicializada", {"parâmetros": self._get_params_dict()})
        
    def _get_params_dict(self):
        """Retorna um dicionário com todos os parâmetros da simulação"""
        params = {
            # Parâmetros de mercado
            "monero_price": self.monero_price,
            "price_volatility": self.price_volatility,
            "price_trend": self.price_trend,
            "difficulty_adjustment": self.difficulty_adjustment,
            "reward_per_block": self.reward_per_block,
            
            # Parâmetros ASIC
            "asic_hashrate": self.asic_hashrate,
            "asic_power": self.asic_power,
            "asic_hardware_cost": self.asic_hardware_cost,
            
            # Parâmetros QUALIA
            "qualia_hashrate": self.qualia_hashrate,
            "qualia_power": self.qualia_power,
            "qualia_hardware_cost": self.qualia_hardware_cost,
            "qualia_resonance_factor": self.qualia_resonance_factor,
        }
        return params
    
    def _log_event(self, event_type, details=None):
        """Adiciona um evento ao log de simulação"""
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event_type": event_type,
            "details": details or {}
        }
        self.simulation_log.append(log_entry)
        
    def _apply_scenario(self, scenario):
        """
        Aplica configurações predefinidas de cenários.
        
        Parâmetros:
            scenario (str): Nome do cenário (basico, conservador, realista, otimista)
        """
        if scenario.lower() == "basico" or scenario.lower() == "básico":
            # Cenário básico - usa valores padrão
            self._log_event("Aplicando cenário básico", {"status": "ok"})
            # Mantém configurações padrão
            
        elif scenario.lower() == "conservador":
            # Cenário conservador - mercado em baixa
            self._log_event("Aplicando cenário conservador", {"status": "ok"})
            self.monero_price = 120
            self.price_trend = 0.9999  # Tendência de queda
            self.difficulty_adjustment = 1.0002  # Aumenta gradualmente
            self.electricity_cost = 0.15
            self.qualia_resonance_factor = 1.08
            self.qualia_entanglement_boost = 1.03
            
        elif scenario.lower() == "realista":
            # Cenário realista - baseado em dados recentes
            self._log_event("Aplicando cenário realista", {"status": "ok"})
            self.monero_price = 160
            self.price_volatility = 0.025
            self.price_trend = 1.0003
            self.difficulty_adjustment = 1.0004
            self.difficulty_volatility = 0.007
            self.electricity_cost = 0.13
            self.asic_failure_rate = 0.0007
            self.qualia_failure_rate = 0.0003
            self.qualia_resonance_factor = 1.12
            self.qualia_learning_rate = 0.015
            self.qualia_entanglement_boost = 1.06
            
        elif scenario.lower() == "otimista":
            # Cenário otimista - mercado em alta
            self._log_event("Aplicando cenário otimista", {"status": "ok"})
            self.monero_price = 200
            self.price_volatility = 0.03
            self.price_trend = 1.0005
            self.difficulty_adjustment = 1.0006
            self.electricity_cost = 0.11
            self.qualia_resonance_factor = 1.15
            self.qualia_learning_rate = 0.02
            self.qualia_entanglement_boost = 1.08
            
        else:
            # Cenário desconhecido
            self._log_event(f"Cenário '{scenario}' não encontrado, usando padrão", {"status": "aviso"})
            
    def _calculate_block_time(self, hashrate, difficulty):
        """
        Calcula o tempo médio para minerar um bloco baseado no hashrate e dificuldade.
        
        Parâmetros:
            hashrate (float): Taxa de hash em H/s
            difficulty (float): Dificuldade da rede
            
        Retorna:
            float: Tempo em segundos para minerar um bloco
        """
        # Tempo médio (segundos) = dificuldade / hashrate
        if hashrate > 0:
            return difficulty / hashrate
        else:
            return float('inf')  # Prevenir divisão por zero
    
    def _calculate_energy_cost(self, power_consumption, time_seconds, electricity_cost):
        """
        Calcula o custo de energia para um período de mineração.
        
        Parâmetros:
            power_consumption (float): Consumo de energia em Watts
            time_seconds (float): Tempo de operação em segundos
            electricity_cost (float): Custo da eletricidade em USD/kWh
            
        Retorna:
            float: Custo de energia em USD
        """
        # kWh = (W * h) / 1000
        # Converter segundos para horas: h = s / 3600
        kwh = (power_consumption * (time_seconds / 3600)) / 1000
        return kwh * electricity_cost
    
    def _calculate_maintenance_cost(self, daily_maintenance_cost, time_seconds):
        """
        Calcula o custo de manutenção para um período de mineração.
        
        Parâmetros:
            daily_maintenance_cost (float): Custo diário de manutenção em USD
            time_seconds (float): Tempo de operação em segundos
            
        Retorna:
            float: Custo de manutenção em USD
        """
        # Converter segundos para dias e multiplicar pelo custo diário
        days = time_seconds / (24 * 60 * 60)
        return daily_maintenance_cost * days
    
    def _calculate_hardware_depreciation(self, hardware_cost, daily_rate, time_seconds):
        """
        Calcula a depreciação do hardware para um período de mineração.
        
        Parâmetros:
            hardware_cost (float): Custo do hardware em USD
            daily_rate (float): Taxa diária de depreciação
            time_seconds (float): Tempo de operação em segundos
            
        Retorna:
            float: Valor da depreciação em USD
        """
        # Converter segundos para dias e calcular depreciação
        days = time_seconds / (24 * 60 * 60)
        depreciation = hardware_cost * daily_rate * days
        return depreciation
    
    def _calculate_failure_cost(self, failure_rate, repair_cost, time_seconds):
        """
        Calcula o custo médio esperado devido a falhas de hardware.
        
        Parâmetros:
            failure_rate (float): Probabilidade diária de falha
            repair_cost (float): Custo de reparo em USD
            time_seconds (float): Tempo de operação em segundos
            
        Retorna:
            float: Custo esperado de falhas em USD
        """
        # Converter segundos para dias
        days = time_seconds / (24 * 60 * 60)
        
        # Probabilidade de falha durante o período
        failure_prob = failure_rate * days
        
        # Limitar a probabilidade a 1 (100%)
        failure_prob = min(failure_prob, 1.0)
        
        # Custo esperado = probabilidade de falha * custo de reparo
        return failure_prob * repair_cost

    def run_simulation(self):
        """
        Executa a simulação completa de mineração com modelo avançado.
        
        Retorna:
            pandas.DataFrame: DataFrame com os resultados da simulação
        """
        self._log_event("Simulação iniciada", {"tipo": "completa"})
        
        # Inicialização de variáveis
        data = []
        current_difficulty = self.difficulty
        current_price = self.monero_price
        
        # Estados iniciais
        asic_cumulative_profit = -self.asic_hardware_cost
        qualia_cumulative_profit = -self.qualia_hardware_cost
        
        asic_cumulative_energy = 0
        qualia_cumulative_energy = 0
        
        asic_cumulative_maintenance = 0
        qualia_cumulative_maintenance = 0
        
        asic_cumulative_repairs = 0
        qualia_cumulative_repairs = 0
        
        asic_cumulative_depreciation = 0
        qualia_cumulative_depreciation = 0
        
        asic_blocks_mined = 0
        qualia_blocks_mined = 0
        
        asic_total_time = 0
        qualia_total_time = 0
        
        # Estado atual dos mineradores
        asic_current_hashrate = self.asic_hashrate
        qualia_current_hashrate = self.qualia_hashrate
        
        # Estado da recompensa
        current_reward = self.reward_per_block
        
        # Registro de tempo de simulação
        start_date = datetime.now()
        current_date = start_date
        
        # Loop principal da simulação
        for day in range(self.simulation_days):
            # Simulação do minerador ASIC
            asic_mining_time = self._calculate_block_time(asic_current_hashrate, current_difficulty)
            asic_energy_cost = self._calculate_energy_cost(
                self.asic_power, 
                asic_mining_time, 
                self.electricity_cost
            )
            asic_maintenance = self._calculate_maintenance_cost(
                self.asic_maintenance_factor * self.asic_hardware_cost, 
                asic_mining_time
            )
            asic_depreciation = self._calculate_hardware_depreciation(
                self.asic_hardware_cost,
                self.asic_degradation_rate,
                asic_mining_time
            )
            asic_repair = self._calculate_failure_cost(
                self.asic_failure_rate,
                self.asic_repair_cost_factor * self.asic_hardware_cost,
                asic_mining_time
            )
            
            # Receita e lucro do ASIC
            asic_revenue = current_reward * current_price
            asic_expenses = asic_energy_cost + asic_maintenance + asic_repair + asic_depreciation
            asic_profit = asic_revenue - asic_expenses
            
            # Atualização dos valores cumulativos para ASIC
            asic_cumulative_profit += asic_profit
            asic_cumulative_energy += asic_energy_cost
            asic_cumulative_maintenance += asic_maintenance
            asic_cumulative_repairs += asic_repair
            asic_cumulative_depreciation += asic_depreciation
            asic_blocks_mined += 1
            asic_total_time += asic_mining_time
            
            # Simulação do minerador QUALIA
            qualia_mining_time = self._calculate_block_time(qualia_current_hashrate, current_difficulty)
            qualia_energy_cost = self._calculate_energy_cost(
                self.qualia_power, 
                qualia_mining_time, 
                self.electricity_cost
            )
            qualia_maintenance = self._calculate_maintenance_cost(
                self.qualia_maintenance_factor * self.qualia_hardware_cost, 
                qualia_mining_time
            )
            qualia_depreciation = self._calculate_hardware_depreciation(
                self.qualia_hardware_cost,
                self.qualia_degradation_rate,
                qualia_mining_time
            )
            qualia_repair = self._calculate_failure_cost(
                self.qualia_failure_rate,
                self.qualia_repair_cost_factor * self.qualia_hardware_cost,
                qualia_mining_time
            )
            
            # Receita e lucro do QUALIA
            qualia_revenue = current_reward * current_price
            qualia_expenses = qualia_energy_cost + qualia_maintenance + qualia_repair + qualia_depreciation
            qualia_profit = qualia_revenue - qualia_expenses
            
            # Atualização dos valores cumulativos para QUALIA
            qualia_cumulative_profit += qualia_profit
            qualia_cumulative_energy += qualia_energy_cost
            qualia_cumulative_maintenance += qualia_maintenance
            qualia_cumulative_repairs += qualia_repair
            qualia_cumulative_depreciation += qualia_depreciation
            qualia_blocks_mined += 1
            qualia_total_time += qualia_mining_time
            
            # Atualização da data (avanço no tempo da simulação)
            current_date += timedelta(days=1)
            
            # Cálculos de métricas de desempenho
            qualia_hashrate_advantage = (qualia_current_hashrate / asic_current_hashrate) - 1
            qualia_efficiency_ratio = (qualia_profit / qualia_energy_cost) / (asic_profit / asic_energy_cost) if asic_energy_cost > 0 and asic_profit > 0 else float('inf')
            qualia_profit_ratio = qualia_profit / asic_profit if asic_profit > 0 else float('inf')
            
            # Registro dos dados deste dia
            data.append({
                # Informações gerais
                'day': day + 1,
                'date': current_date,
                'difficulty': current_difficulty,
                'monero_price': current_price,
                'block_reward': current_reward,
                
                # Informações do ASIC
                'asic_hashrate': asic_current_hashrate,
                'asic_mining_time': asic_mining_time,
                'asic_energy_cost': asic_energy_cost,
                'asic_maintenance_cost': asic_maintenance,
                'asic_repair_cost': asic_repair,
                'asic_depreciation': asic_depreciation,
                'asic_total_cost': asic_expenses,
                'asic_revenue': asic_revenue,
                'asic_profit': asic_profit,
                'asic_cumulative_profit': asic_cumulative_profit,
                'asic_cumulative_energy': asic_cumulative_energy,
                
                # Informações do QUALIA
                'qualia_hashrate': qualia_current_hashrate,
                'qualia_mining_time': qualia_mining_time,
                'qualia_energy_cost': qualia_energy_cost,
                'qualia_maintenance_cost': qualia_maintenance,
                'qualia_repair_cost': qualia_repair,
                'qualia_depreciation': qualia_depreciation,
                'qualia_total_cost': qualia_expenses,
                'qualia_revenue': qualia_revenue,
                'qualia_profit': qualia_profit,
                'qualia_cumulative_profit': qualia_cumulative_profit,
                'qualia_cumulative_energy': qualia_cumulative_energy,
                
                # Métricas comparativas
                'hashrate_advantage': qualia_hashrate_advantage,
                'profit_difference': qualia_cumulative_profit - asic_cumulative_profit,
                'energy_difference': asic_cumulative_energy - qualia_cumulative_energy,
                'qualia_efficiency_ratio': qualia_efficiency_ratio,
                'qualia_profit_ratio': qualia_profit_ratio
            })
            
            # Log de eventos significativos
            if day % 10 == 0:
                self._log_event("Progresso da simulação", {
                    "dia": day,
                    "dificuldade": current_difficulty,
                    "preço": current_price,
                    "lucro_qualia": qualia_cumulative_profit,
                    "lucro_asic": asic_cumulative_profit
                })
        
        # Criação do DataFrame com os resultados
        self.results_df = pd.DataFrame(data)
        
        # Identificação dos dias de ROI (retorno do investimento)
        asic_roi_day = None
        qualia_roi_day = None
        
        for idx, row in self.results_df.iterrows():
            if asic_roi_day is None and row['asic_cumulative_profit'] > 0:
                asic_roi_day = row['day']
            if qualia_roi_day is None and row['qualia_cumulative_profit'] > 0:
                qualia_roi_day = row['day']
        
        # Cálculo de métricas finais
        final_row = self.results_df.iloc[-1]
        
        # Registro das métricas no sumário
        self.summary = {
            # Informações gerais
            'dias_simulados': self.simulation_days,
            'data_inicial': start_date.strftime('%d/%m/%Y'),
            'data_final': current_date.strftime('%d/%m/%Y'),
            
            # Informações de ROI
            'asic_roi_day': asic_roi_day,
            'qualia_roi_day': qualia_roi_day,
            
            # Métricas de desempenho
            'asic_total_time': asic_total_time,
            'qualia_total_time': qualia_total_time,
            'asic_avg_hashrate': self.results_df['asic_hashrate'].mean(),
            'qualia_avg_hashrate': self.results_df['qualia_hashrate'].mean(),
            
            # Métricas de custo e eficiência
            'asic_total_energy_cost': final_row['asic_cumulative_energy'],
            'qualia_total_energy_cost': final_row['qualia_cumulative_energy'],
            'asic_energy_efficiency': asic_blocks_mined / final_row['asic_cumulative_energy'] if final_row['asic_cumulative_energy'] > 0 else 0,
            'qualia_energy_efficiency': qualia_blocks_mined / final_row['qualia_cumulative_energy'] if final_row['qualia_cumulative_energy'] > 0 else 0,
            
            # Métricas de lucro
            'asic_final_profit': final_row['asic_cumulative_profit'],
            'qualia_final_profit': final_row['qualia_cumulative_profit'],
            'profit_difference': final_row['qualia_cumulative_profit'] - final_row['asic_cumulative_profit'],
            'profit_ratio': final_row['qualia_cumulative_profit'] / final_row['asic_cumulative_profit'] if final_row['asic_cumulative_profit'] > 0 else float('inf'),
            
            # Parâmetros finais da rede
            'final_difficulty': final_row['difficulty'],
            'final_price': final_row['monero_price']
        }
        
        self._log_event("Simulação concluída", {"métricas": {k: v for k, v in self.summary.items() if not isinstance(v, datetime)}})
        
        return self.results_df

    def generate_summary_plots(self, output_dir='results'):
        """
        Gera gráficos de resumo da simulação e salva no diretório especificado.
        
        Parâmetros:
            output_dir (str): Diretório para salvar os gráficos
        """
        if self.results_df is None:
            raise ValueError("Nenhum resultado disponível. Execute a simulação primeiro.")
        
        # Criar diretório se não existir
        os.makedirs(output_dir, exist_ok=True)
        
        self._log_event("Gerando gráficos", {"diretório": output_dir})
        
        # Configuração dos gráficos
        plt.style.use('ggplot')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        
        # 1. Gráfico de lucro cumulativo
        self._plot_cumulative_profit(output_dir)
        
        # 2. Gráfico de consumo de energia
        self._plot_energy_consumption(output_dir)
        
        # 3. Gráfico de hashrate efetivo
        self._plot_hashrate_comparison(output_dir)
        
        # 4. Gráfico de ROI (Return on Investment)
        self._plot_roi_analysis(output_dir)
        
        # 5. Gráfico de métricas de rede
        self._plot_network_metrics(output_dir)
        
        # 6. Resumo financeiro
        self._plot_financial_summary(output_dir)
        
        self._log_event("Gráficos gerados", {"quantidade": 6, "diretório": output_dir})
        
    def _plot_cumulative_profit(self, output_dir):
        """
        Gera gráfico comparativo de lucro cumulativo.
        """
        plt.figure()
        plt.plot(self.results_df['day'], self.results_df['asic_cumulative_profit'], 
                 label='ASIC', color='#FF5733', linewidth=2)
        plt.plot(self.results_df['day'], self.results_df['qualia_cumulative_profit'], 
                 label='QUALIA', color='#33A8FF', linewidth=2)
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, label='Break-even')
        
        # Marcar pontos de ROI
        if 'asic_roi_day' in self.summary and self.summary['asic_roi_day'] is not None:
            plt.axvline(x=self.summary['asic_roi_day'], color='#FF5733', linestyle='--', 
                       alpha=0.5, label='ASIC ROI')
        
        if 'qualia_roi_day' in self.summary and self.summary['qualia_roi_day'] is not None:
            plt.axvline(x=self.summary['qualia_roi_day'], color='#33A8FF', linestyle='--', 
                       alpha=0.5, label='QUALIA ROI')
        
        plt.xlabel('Dias')
        plt.ylabel('Lucro Cumulativo (USD)')
        plt.title('Comparação de Lucro Cumulativo: ASIC vs QUALIA')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Adicionar anotações
        plt.annotate(f"Lucro final ASIC: ${self.summary['asic_final_profit']:.2f}", 
                     xy=(0.02, 0.05), xycoords='axes fraction', fontsize=10)
        plt.annotate(f"Lucro final QUALIA: ${self.summary['qualia_final_profit']:.2f}", 
                     xy=(0.02, 0.10), xycoords='axes fraction', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cumulative_profit.png'), dpi=300)
        plt.close()
        
    def _plot_energy_consumption(self, output_dir):
        """
        Gera gráfico comparativo de consumo de energia.
        """
        plt.figure(figsize=(12, 8))
        
        # Gráfico de barras para consumo de energia cumulativo
        data = [self.summary['asic_total_energy_cost'], self.summary['qualia_total_energy_cost']]
        labels = ['ASIC', 'QUALIA']
        colors = ['#FF5733', '#33A8FF']
        
        plt.bar(labels, data, color=colors)
        
        # Adicionar valores acima das barras
        for i, v in enumerate(data):
            plt.text(i, v + 0.1, f"${v:.2f}", ha='center', fontsize=10)
        
        plt.xlabel('Tecnologia de Mineração')
        plt.ylabel('Custo Total de Energia (USD)')
        plt.title('Comparação de Consumo de Energia: ASIC vs QUALIA')
        
        # Adicionar economia percentual
        if data[0] > 0:
            energy_saving = (data[0] - data[1]) / data[0] * 100
            plt.annotate(f"Economia QUALIA: {energy_saving:.1f}%", 
                         xy=(0.5, 0.9), xycoords='axes fraction', 
                         ha='center', fontsize=12,
                         bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.6))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'energy_consumption.png'), dpi=300)
        plt.close()
        
    def _plot_hashrate_comparison(self, output_dir):
        """
        Gera gráfico comparativo de hashrate entre ASIC e QUALIA.
        """
        plt.figure(figsize=(12, 8))
        
        # Plotar hashrate base
        plt.plot(self.results_df['day'], self.results_df['asic_hashrate'], 
                 label='ASIC (Base)', color='#FF5733', linewidth=2)
        plt.plot(self.results_df['day'], self.results_df['qualia_hashrate'], 
                 label='QUALIA (Base)', color='#33A8FF', linewidth=2)
        
        # Configuração do gráfico
        plt.xlabel('Dias')
        plt.ylabel('Hashrate (H/s)')
        plt.title('Comparação de Hashrate: ASIC vs QUALIA')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        # Converter para escala logarítmica para melhor visualização
        plt.yscale('log')
        
        # Adicionar anotações
        plt.annotate(f"Hashrate médio ASIC: {self.summary['asic_avg_hashrate']:.2e} H/s", 
                     xy=(0.02, 0.15), xycoords='axes fraction', fontsize=10)
        plt.annotate(f"Hashrate médio QUALIA: {self.summary['qualia_avg_hashrate']:.2e} H/s", 
                     xy=(0.02, 0.10), xycoords='axes fraction', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'hashrate_comparison.png'), dpi=300)
        plt.close()
        
    def _plot_roi_analysis(self, output_dir):
        """
        Gera gráfico de análise de ROI (Return on Investment).
        """
        # Se não houver ROI para nenhum dos mineradores, não gera o gráfico
        if (self.summary['asic_roi_day'] is None and self.summary['qualia_roi_day'] is None):
            return
        
        plt.figure(figsize=(12, 8))
        
        # Criar dados para o gráfico
        tech_labels = []
        roi_days = []
        colors = []
        
        if self.summary['asic_roi_day'] is not None:
            tech_labels.append('ASIC')
            roi_days.append(self.summary['asic_roi_day'])
            colors.append('#FF5733')
        
        if self.summary['qualia_roi_day'] is not None:
            tech_labels.append('QUALIA')
            roi_days.append(self.summary['qualia_roi_day'])
            colors.append('#33A8FF')
        
        # Criar gráfico de barras para ROI
        plt.bar(tech_labels, roi_days, color=colors)
        
        # Adicionar valores acima das barras
        for i, v in enumerate(roi_days):
            plt.text(i, v + 0.5, f"{v:.1f} dias", ha='center', fontsize=10)
        
        plt.xlabel('Tecnologia de Mineração')
        plt.ylabel('Dias até ROI')
        plt.title('Análise de ROI: ASIC vs QUALIA')
        
        # Adicionar comparação percentual se ambos tiverem ROI
        if len(roi_days) == 2:
            roi_improvement = (roi_days[0] - roi_days[1]) / roi_days[0] * 100
            if roi_improvement > 0:
                message = f"ROI {roi_improvement:.1f}% mais rápido com QUALIA"
            else:
                message = f"ROI {-roi_improvement:.1f}% mais rápido com ASIC"
                
            plt.annotate(message, 
                        xy=(0.5, 0.9), xycoords='axes fraction', 
                        ha='center', fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", alpha=0.6))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roi_analysis.png'), dpi=300)
        plt.close()

    def _plot_network_metrics(self, output_dir):
        """
        Gera gráfico de evolução das métricas da rede ao longo do tempo.
        """
        plt.figure(figsize=(14, 10))
        
        # Criar quatro subplots (2x2)
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Evolução da dificuldade
        axs[0, 0].plot(self.results_df['day'], self.results_df['difficulty'], 
                    color='purple', linewidth=2)
        axs[0, 0].set_xlabel('Dias')
        axs[0, 0].set_ylabel('Dificuldade da Rede')
        axs[0, 0].set_title('Evolução da Dificuldade')
        axs[0, 0].grid(True, alpha=0.3)
        # Escala logarítmica para dificuldade
        axs[0, 0].set_yscale('log')
        
        # 2. Evolução do preço
        axs[0, 1].plot(self.results_df['day'], self.results_df['monero_price'], 
                    color='orange', linewidth=2)
        axs[0, 1].set_xlabel('Dias')
        axs[0, 1].set_ylabel('Preço (USD)')
        axs[0, 1].set_title('Evolução do Preço do Monero')
        axs[0, 1].grid(True, alpha=0.3)
        
        # 3. Evolução da recompensa por bloco
        axs[1, 0].plot(self.results_df['day'], self.results_df['block_reward'], 
                    color='red', linewidth=2)
        axs[1, 0].set_xlabel('Dias')
        axs[1, 0].set_ylabel('Recompensa por Bloco (XMR)')
        axs[1, 0].set_title('Evolução da Recompensa por Bloco')
        axs[1, 0].grid(True, alpha=0.3)
        
        # 4. Evolução do hashrate
        axs[1, 1].plot(self.results_df['day'], self.results_df['asic_hashrate'], 
                    color='#FF5733', linewidth=2, label='ASIC')
        axs[1, 1].plot(self.results_df['day'], self.results_df['qualia_hashrate'], 
                    color='#33A8FF', linewidth=2, label='QUALIA')
        axs[1, 1].set_xlabel('Dias')
        axs[1, 1].set_ylabel('Hashrate (H/s)')
        axs[1, 1].set_title('Evolução do Hashrate')
        axs[1, 1].legend()
        axs[1, 1].grid(True, alpha=0.3)
        
        # Ajustar layout
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'network_metrics.png'), dpi=300)
        plt.close()
        
    def _plot_financial_summary(self, output_dir):
        """
        Gera gráfico de resumo financeiro comparativo.
        """
        plt.figure(figsize=(12, 10))
        
        # Dados para gráficos de pizza com componentes de custo
        # 1. Componentes de custo ASIC
        asic_costs = {
            'Energia': self.summary['asic_total_energy_cost'],
            'Hardware (Depreciação)': self.results_df['asic_depreciation'].sum(),
            'Manutenção': self.results_df['asic_maintenance_cost'].sum(),
            'Reparos': self.results_df['asic_repair_cost'].sum()
        }
        
        # 2. Componentes de custo QUALIA
        qualia_costs = {
            'Energia': self.summary['qualia_total_energy_cost'],
            'Hardware (Depreciação)': self.results_df['qualia_depreciation'].sum(),
            'Manutenção': self.results_df['qualia_maintenance_cost'].sum(),
            'Reparos': self.results_df['qualia_repair_cost'].sum()
        }
        
        # Criar figura com subplots
        fig, axs = plt.subplots(2, 2, figsize=(14, 12))
        
        # Gráfico de pizza para componentes de custo ASIC
        asic_labels = list(asic_costs.keys())
        asic_values = list(asic_costs.values())
        axs[0, 0].pie(asic_values, labels=asic_labels, autopct='%1.1f%%', 
                   startangle=90, shadow=True, explode=[0.05, 0, 0, 0],
                   colors=['#FF5733', '#FFC300', '#C70039', '#900C3F'])
        axs[0, 0].set_title('Componentes de Custo - ASIC')
        
        # Gráfico de pizza para componentes de custo QUALIA
        qualia_labels = list(qualia_costs.keys())
        qualia_values = list(qualia_costs.values())
        axs[0, 1].pie(qualia_values, labels=qualia_labels, autopct='%1.1f%%', 
                   startangle=90, shadow=True, explode=[0.05, 0, 0, 0],
                   colors=['#33A8FF', '#8CDBF4', '#5E95D2', '#3A46A8'])
        axs[0, 1].set_title('Componentes de Custo - QUALIA')
        
        # Métricas comparativas em gráfico de barras
        metrics = {
            'Custo por H/s': [self.asic_hardware_cost / self.asic_hashrate, 
                             self.qualia_hardware_cost / self.qualia_hashrate],
            'Energia por H/s': [self.asic_power / self.asic_hashrate, 
                               self.qualia_power / self.qualia_hashrate],
            'Custo por Bloco': [self.results_df['asic_total_cost'].mean(), 
                               self.results_df['qualia_total_cost'].mean()],
            'Lucro por Bloco': [self.results_df['asic_profit'].mean(), 
                               self.results_df['qualia_profit'].mean()]
        }
        
        # Normalizar para comparação relativa (ASIC = 100%)
        normalized_metrics = {}
        for key, values in metrics.items():
            normalized_metrics[key] = [100, values[1]/values[0]*100]
        
        # Preparar dados para o gráfico de barras
        labels = list(normalized_metrics.keys())
        asic_data = [normalized_metrics[k][0] for k in labels]
        qualia_data = [normalized_metrics[k][1] for k in labels]
        
        x = np.arange(len(labels))
        width = 0.35
        
        # Gráfico de barras para métricas normalizadas
        axs[1, 0].bar(x - width/2, asic_data, width, label='ASIC', color='#FF5733')
        axs[1, 0].bar(x + width/2, qualia_data, width, label='QUALIA', color='#33A8FF')
        
        axs[1, 0].set_ylabel('Valor Relativo (%)')
        axs[1, 0].set_title('Métricas Comparativas (ASIC = 100%)')
        axs[1, 0].set_xticks(x)
        axs[1, 0].set_xticklabels(labels)
        axs[1, 0].legend()
        
        # Gráfico de barras para os custos totais por dia
        if self.summary['dias_simulados'] > 0:
            asic_daily_costs = sum(asic_values) / self.summary['dias_simulados']
            qualia_daily_costs = sum(qualia_values) / self.summary['dias_simulados']
            
            daily_costs = [asic_daily_costs, qualia_daily_costs]
            technologies = ['ASIC', 'QUALIA']
            
            axs[1, 1].bar(technologies, daily_costs, color=['#FF5733', '#33A8FF'])
            axs[1, 1].set_ylabel('Custo Diário (USD)')
            axs[1, 1].set_title('Custo Médio Diário')
            
            # Adicionar valores acima das barras
            for i, v in enumerate(daily_costs):
                axs[1, 1].text(i, v + 0.1, f"${v:.2f}/dia", ha='center')
            
            # Adicionar texto sobre economia diária
            if asic_daily_costs > 0:
                daily_saving = (asic_daily_costs - qualia_daily_costs) / asic_daily_costs * 100
                axs[1, 1].text(0.5, max(daily_costs) * 1.2, 
                           f"Economia diária com QUALIA: {daily_saving:.1f}%",
                           ha='center', fontsize=12,
                           bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.6))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'financial_summary.png'), dpi=300)
        plt.close()
        
    def run_monte_carlo_simulation(self, num_simulations=100, output_dir='monte_carlo_results'):
        """
        Executa uma simulação de Monte Carlo para avaliação de diferentes cenários.
        
        Parâmetros:
            num_simulations (int): Número de simulações a serem executadas
            output_dir (str): Diretório para salvar os resultados
            
        Retorna:
            pandas.DataFrame: DataFrame com os resultados consolidados das simulações
        """
        # Criar diretório se não existir
        os.makedirs(output_dir, exist_ok=True)
        
        self._log_event("Iniciando simulação Monte Carlo", {"num_simulações": num_simulations})
        
        # Armazenar configuração original para restaurar após a simulação
        original_config = {
            'monero_price': self.monero_price,
            'price_volatility': self.price_volatility,
            'price_trend': self.price_trend,
            'difficulty_adjustment': self.difficulty_adjustment,
            'qualia_resonance_factor': self.qualia_resonance_factor,
            'qualia_learning_rate': self.qualia_learning_rate,
            'qualia_entanglement_boost': self.qualia_entanglement_boost
        }
        
        # Resultados consolidados
        monte_carlo_results = []
        
        for sim in range(num_simulations):
            # Variações aleatórias para parâmetros-chave
            self.monero_price = original_config['monero_price'] * np.random.uniform(0.7, 1.3)
            self.price_volatility = original_config['price_volatility'] * np.random.uniform(0.5, 1.5)
            self.price_trend = original_config['price_trend'] * np.random.uniform(0.8, 1.2)
            self.difficulty_adjustment = original_config['difficulty_adjustment'] * np.random.uniform(0.8, 1.2)
            
            # Variações para parâmetros QUALIA
            self.qualia_resonance_factor = original_config['qualia_resonance_factor'] * np.random.uniform(0.9, 1.1)
            self.qualia_learning_rate = original_config['qualia_learning_rate'] * np.random.uniform(0.9, 1.1)
            
            if self.qualia_quantum_optimization:
                self.qualia_entanglement_boost = original_config['qualia_entanglement_boost'] * np.random.uniform(0.9, 1.1)
            
            # Executar simulação com parâmetros aleatorizados
            self._log_event(f"Executando simulação Monte Carlo #{sim+1}", {
                "preço": self.monero_price,
                "volatilidade": self.price_volatility,
                "ressonância": self.qualia_resonance_factor
            })
            
            # Executar simulação
            self.run_simulation()
            
            # Armazenar resultados principais
            mc_result = {
                'simulation': sim + 1,
                'monero_price': self.monero_price,
                'price_volatility': self.price_volatility,
                'price_trend': self.price_trend,
                'difficulty_adjustment': self.difficulty_adjustment,
                'qualia_resonance_factor': self.qualia_resonance_factor,
                'qualia_learning_rate': self.qualia_learning_rate,
                'qualia_entanglement_boost': self.qualia_entanglement_boost,
                'asic_final_profit': self.summary['asic_final_profit'],
                'qualia_final_profit': self.summary['qualia_final_profit'],
                'profit_difference': self.summary['profit_difference'],
                'asic_roi_day': self.summary['asic_roi_day'],
                'qualia_roi_day': self.summary['qualia_roi_day'],
                'asic_energy_efficiency': self.summary['asic_energy_efficiency'],
                'qualia_energy_efficiency': self.summary['qualia_energy_efficiency'],
            }
            
            monte_carlo_results.append(mc_result)
        
        # Restaurar configuração original
        for key, value in original_config.items():
            setattr(self, key, value)
        
        # Converter resultados para DataFrame
        mc_df = pd.DataFrame(monte_carlo_results)
        
        # Salvar resultados em CSV
        mc_df.to_csv(os.path.join(output_dir, 'monte_carlo_results.csv'), index=False)
        
        # Criar gráficos de distribuição dos resultados
        self._plot_monte_carlo_results(mc_df, output_dir)
        
        self._log_event("Simulação Monte Carlo concluída", {"num_simulações": num_simulations})
        
        return mc_df
    
    def _plot_monte_carlo_results(self, mc_df, output_dir):
        """
        Gera gráficos de análise dos resultados da simulação Monte Carlo.
        
        Parâmetros:
            mc_df (pandas.DataFrame): DataFrame com resultados da simulação
            output_dir (str): Diretório para salvar os gráficos
        """
        plt.figure(figsize=(14, 10))
        
        # Configurar subplots
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Distribuição de lucros ASIC vs QUALIA
        axs[0, 0].hist(mc_df['asic_final_profit'], bins=20, alpha=0.5, color='#FF5733', label='ASIC')
        axs[0, 0].hist(mc_df['qualia_final_profit'], bins=20, alpha=0.5, color='#33A8FF', label='QUALIA')
        axs[0, 0].set_xlabel('Lucro Final (USD)')
        axs[0, 0].set_ylabel('Frequência')
        axs[0, 0].set_title('Distribuição de Lucro Final')
        axs[0, 0].legend()
        axs[0, 0].grid(True, alpha=0.3)
        
        # 2. Distribuição da diferença de lucro
        axs[0, 1].hist(mc_df['profit_difference'], bins=20, color='green', alpha=0.7)
        axs[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        axs[0, 1].set_xlabel('Diferença de Lucro (QUALIA - ASIC)')
        axs[0, 1].set_ylabel('Frequência')
        axs[0, 1].set_title('Diferença de Lucro (QUALIA vs ASIC)')
        axs[0, 1].grid(True, alpha=0.3)
        
        # Adicionar anotação sobre a probabilidade de QUALIA ser mais lucrativo
        prob_qualia_better = (mc_df['profit_difference'] > 0).mean() * 100
        axs[0, 1].annotate(f"Prob. QUALIA mais lucrativo: {prob_qualia_better:.1f}%", 
                         xy=(0.05, 0.9), xycoords='axes fraction', fontsize=10,
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        # 3. ROI em dias
        valid_asic_roi = mc_df[mc_df['asic_roi_day'].notna()]['asic_roi_day']
        valid_qualia_roi = mc_df[mc_df['qualia_roi_day'].notna()]['qualia_roi_day']
        
        if len(valid_asic_roi) > 0 and len(valid_qualia_roi) > 0:
            axs[1, 0].hist(valid_asic_roi, bins=20, alpha=0.5, color='#FF5733', label='ASIC')
            axs[1, 0].hist(valid_qualia_roi, bins=20, alpha=0.5, color='#33A8FF', label='QUALIA')
            axs[1, 0].set_xlabel('Dias até ROI')
            axs[1, 0].set_ylabel('Frequência')
            axs[1, 0].set_title('Distribuição de Tempo até ROI')
            axs[1, 0].legend()
            axs[1, 0].grid(True, alpha=0.3)
            
            # Adicionar médias
            axs[1, 0].axvline(x=valid_asic_roi.mean(), color='#FF5733', linestyle='--', alpha=0.7)
            axs[1, 0].axvline(x=valid_qualia_roi.mean(), color='#33A8FF', linestyle='--', alpha=0.7)
            
            # Anotações sobre as médias
            axs[1, 0].annotate(f"Média ASIC: {valid_asic_roi.mean():.1f} dias", 
                            xy=(0.05, 0.85), xycoords='axes fraction', fontsize=10,
                            color='#FF5733')
            axs[1, 0].annotate(f"Média QUALIA: {valid_qualia_roi.mean():.1f} dias", 
                            xy=(0.05, 0.78), xycoords='axes fraction', fontsize=10,
                            color='#33A8FF')
        
        # 4. Eficiência energética
        axs[1, 1].hist(mc_df['asic_energy_efficiency'], bins=20, alpha=0.5, color='#FF5733', label='ASIC')
        axs[1, 1].hist(mc_df['qualia_energy_efficiency'], bins=20, alpha=0.5, color='#33A8FF', label='QUALIA')
        axs[1, 1].set_xlabel('Eficiência Energética (blocos/USD)')
        axs[1, 1].set_ylabel('Frequência')
        axs[1, 1].set_title('Distribuição de Eficiência Energética')
        axs[1, 1].legend()
        axs[1, 1].grid(True, alpha=0.3)
        
        # Ajustar layout
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'monte_carlo_analysis.png'), dpi=300)
        plt.close()
        
        # Gráfico de dispersão: Ressonância vs. Lucro
        plt.figure(figsize=(10, 6))
        plt.scatter(mc_df['qualia_resonance_factor'], mc_df['qualia_final_profit'], 
                   alpha=0.7, color='#33A8FF')
        plt.xlabel('Fator de Ressonância Quântica')
        plt.ylabel('Lucro Final QUALIA (USD)')
        plt.title('Impacto da Ressonância Quântica no Lucro Final')
        plt.grid(True, alpha=0.3)
        
        # Adicionar linha de tendência
        z = np.polyfit(mc_df['qualia_resonance_factor'], mc_df['qualia_final_profit'], 1)
        p = np.poly1d(z)
        plt.plot(mc_df['qualia_resonance_factor'], p(mc_df['qualia_resonance_factor']), 
                "r--", alpha=0.7)
        
        # Correlação entre ressonância e lucro
        corr = mc_df['qualia_resonance_factor'].corr(mc_df['qualia_final_profit'])
        plt.annotate(f"Correlação: {corr:.2f}", 
                    xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'resonance_profit_correlation.png'), dpi=300)
        plt.close()

    def run_sensitivity_analysis(self, parameter_ranges, output_dir='sensitivity_analysis'):
        """
        Executa uma análise de sensibilidade para entender o impacto de diferentes parâmetros nos resultados.
        
        Parâmetros:
            parameter_ranges (dict): Dicionário com os parâmetros a testar e seu intervalo (lista ou tupla)
            output_dir (str): Diretório para salvar os resultados
            
        Retorna:
            dict: Dicionário com os resultados da análise de sensibilidade para cada parâmetro
        """
        # Criar diretório se não existir
        os.makedirs(output_dir, exist_ok=True)
        
        self._log_event("Iniciando análise de sensibilidade", {"parâmetros": list(parameter_ranges.keys())})
        
        # Armazenar configuração original para restaurar após a análise
        original_config = {}
        for param in parameter_ranges.keys():
            original_config[param] = getattr(self, param)
        
        # Dicionário para armazenar resultados
        sensitivity_results = {}
        
        # Para cada parâmetro, executar simulações variando seu valor
        for param, value_range in parameter_ranges.items():
            self._log_event(f"Analisando sensibilidade do parâmetro: {param}", {"intervalo": value_range})
            
            # Resultados para este parâmetro
            param_results = []
            
            # Restaurar todos os parâmetros ao valor original antes de testar este parâmetro
            for p, v in original_config.items():
                setattr(self, p, v)
            
            # Executar simulações para cada valor do parâmetro
            for value in value_range:
                # Definir o novo valor do parâmetro
                setattr(self, param, value)
                
                # Executar simulação
                self.run_simulation()
                
                # Armazenar resultados principais
                result = {
                    'parameter': param,
                    'value': value,
                    'asic_final_profit': self.summary['asic_final_profit'],
                    'qualia_final_profit': self.summary['qualia_final_profit'],
                    'profit_difference': self.summary['profit_difference'],
                    'asic_roi_day': self.summary['asic_roi_day'],
                    'qualia_roi_day': self.summary['qualia_roi_day'],
                    'asic_energy_efficiency': self.summary['asic_energy_efficiency'],
                    'qualia_energy_efficiency': self.summary['qualia_energy_efficiency'],
                }
                
                param_results.append(result)
            
            # Converter para DataFrame
            param_df = pd.DataFrame(param_results)
            
            # Salvar resultados deste parâmetro em CSV
            param_df.to_csv(os.path.join(output_dir, f'sensitivity_{param}.csv'), index=False)
            
            # Criar gráfico para este parâmetro
            self._plot_sensitivity_analysis(param_df, param, output_dir)
            
            # Armazenar resultados
            sensitivity_results[param] = param_df
        
        # Restaurar configuração original
        for param, value in original_config.items():
            setattr(self, param, value)
        
        self._log_event("Análise de sensibilidade concluída", {"parâmetros": list(parameter_ranges.keys())})
        
        return sensitivity_results
    
    def _plot_sensitivity_analysis(self, param_df, param_name, output_dir):
        """
        Gera gráficos de análise de sensibilidade para um parâmetro específico.
        
        Parâmetros:
            param_df (pandas.DataFrame): DataFrame com resultados da análise
            param_name (str): Nome do parâmetro analisado
            output_dir (str): Diretório para salvar os gráficos
        """
        plt.figure(figsize=(14, 10))
        
        # Configurar subplots
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Impacto no lucro final
        axs[0, 0].plot(param_df['value'], param_df['asic_final_profit'], marker='o', 
                     color='#FF5733', linewidth=2, label='ASIC')
        axs[0, 0].plot(param_df['value'], param_df['qualia_final_profit'], marker='o', 
                     color='#33A8FF', linewidth=2, label='QUALIA')
        axs[0, 0].set_xlabel(f'Valor de {param_name}')
        axs[0, 0].set_ylabel('Lucro Final (USD)')
        axs[0, 0].set_title('Impacto no Lucro Final')
        axs[0, 0].legend()
        axs[0, 0].grid(True, alpha=0.3)
        
        # 2. Impacto na diferença de lucro
        axs[0, 1].plot(param_df['value'], param_df['profit_difference'], marker='o', 
                      color='green', linewidth=2)
        axs[0, 1].set_xlabel(f'Valor de {param_name}')
        axs[0, 1].set_ylabel('Diferença de Lucro (QUALIA - ASIC) em USD')
        axs[0, 1].set_title('Impacto na Diferença de Lucro')
        axs[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axs[0, 1].grid(True, alpha=0.3)
        
        # 3. Impacto no ROI
        if 'asic_roi_day' in param_df.columns and 'qualia_roi_day' in param_df.columns:
            # Remover valores None do ROI para plotagem
            valid_data = param_df.dropna(subset=['asic_roi_day', 'qualia_roi_day'])
            
            if not valid_data.empty:
                axs[1, 0].plot(valid_data['value'], valid_data['asic_roi_day'], marker='o', 
                              label='ASIC', color='#FF5733', linewidth=2)
                axs[1, 0].plot(valid_data['value'], valid_data['qualia_roi_day'], marker='o', 
                              label='QUALIA', color='#33A8FF', linewidth=2)
                axs[1, 0].set_xlabel(f'Valor de {param_name}')
                axs[1, 0].set_ylabel('Dias até ROI')
                axs[1, 0].set_title('Impacto no Tempo até ROI')
                axs[1, 0].legend()
                axs[1, 0].grid(True, alpha=0.3)
                
                # Adicionar médias
                if len(valid_data) > 0:
                    axs[1, 0].axhline(y=valid_data['asic_roi_day'].mean(), color='#FF5733', linestyle='--', alpha=0.7)
                    axs[1, 0].axhline(y=valid_data['qualia_roi_day'].mean(), color='#33A8FF', linestyle='--', alpha=0.7)
                    
                    # Anotações sobre as médias
                    axs[1, 0].annotate(f"Média ASIC: {valid_data['asic_roi_day'].mean():.1f} dias", 
                                    xy=(0.05, 0.85), xycoords='axes fraction', fontsize=10,
                                    color='#FF5733')
                    axs[1, 0].annotate(f"Média QUALIA: {valid_data['qualia_roi_day'].mean():.1f} dias", 
                                    xy=(0.05, 0.78), xycoords='axes fraction', fontsize=10,
                                    color='#33A8FF')
            else:
                axs[1, 0].text(0.5, 0.5, 'Dados de ROI não disponíveis para este parâmetro', 
                             ha='center', va='center', transform=axs[1, 0].transAxes)
        else:
            axs[1, 0].text(0.5, 0.5, 'Dados de ROI não disponíveis para este parâmetro', 
                         ha='center', va='center', transform=axs[1, 0].transAxes)
        
        # 4. Impacto na eficiência energética
        axs[1, 1].plot(param_df['value'], param_df['asic_energy_efficiency'], marker='o', 
                      label='ASIC', color='#FF5733', linewidth=2)
        axs[1, 1].plot(param_df['value'], param_df['qualia_energy_efficiency'], marker='o', 
                      label='QUALIA', color='#33A8FF', linewidth=2)
        axs[1, 1].set_xlabel(f'Valor de {param_name}')
        axs[1, 1].set_ylabel('Eficiência Energética (blocos/USD)')
        axs[1, 1].set_title('Impacto na Eficiência Energética')
        axs[1, 1].legend()
        axs[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sensitivity_{param_name}.png'), dpi=300)
        plt.close()
        
        # Gráfico adicional específico para ressonância QUALIA (se aplicável)
        if param_name == 'qualia_resonance_factor' and 'qualia_efficiency_ratio' in param_df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(param_df['value'], param_df['qualia_efficiency_ratio'], marker='o', 
                   color='purple', linewidth=2)
            plt.xlabel('Fator de Ressonância Quântica')
            plt.ylabel('Razão de Eficiência QUALIA/ASIC')
            plt.title('Impacto da Ressonância Quântica na Eficiência Relativa')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5, 
                       label='Paridade de Eficiência')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'resonance_impact_{param_name}.png'), dpi=300)
            plt.close()
        
    def generate_all_charts(self, output_dir='results'):
        """
        Gera todos os gráficos disponíveis na simulação.
        
        Parâmetros:
            output_dir (str): Diretório para salvar os gráficos
        """
        if self.results_df is None:
            raise ValueError("Nenhum resultado disponível. Execute a simulação primeiro.")
        
        # Criar diretório se não existir
        os.makedirs(output_dir, exist_ok=True)
        
        self._log_event("Gerando todos os gráficos", {"diretório": output_dir})
        
        # Gerar gráficos básicos
        self.generate_summary_plots(output_dir)
        
        # Gráficos adicionais específicos para análise QUALIA
        self._plot_qualia_advantage(output_dir)
        self._plot_comparative_metrics(output_dir)
        
        self._log_event("Todos os gráficos gerados", {"diretório": output_dir})
    
    def _plot_qualia_advantage(self, output_dir):
        """
        Gera gráfico destacando as vantagens específicas do QUALIA.
        """
        if 'hashrate_advantage' not in self.results_df.columns:
            return
            
        plt.figure(figsize=(12, 8))
        
        # Plotar vantagem de hashrate ao longo do tempo
        plt.plot(self.results_df['day'], self.results_df['hashrate_advantage'] * 100, 
                 label='Vantagem de Hashrate (%)', color='#33A8FF', linewidth=2)
        
        # Linha de referência em 0%
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Configuração do gráfico
        plt.xlabel('Dias')
        plt.ylabel('Vantagem do QUALIA (%)')
        plt.title('Análise de Vantagem do Paradigma QUALIA')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        # Adicionar anotações sobre fatores QUALIA
        plt.annotate(f"Fator de Ressonância: {self.qualia_resonance_factor:.2f}", 
                     xy=(0.02, 0.95), xycoords='axes fraction', fontsize=10)
        plt.annotate(f"Taxa de Aprendizado: {self.qualia_learning_rate:.2f}", 
                     xy=(0.02, 0.90), xycoords='axes fraction', fontsize=10)
        plt.annotate(f"Boost de Entrelaçamento: {self.qualia_entanglement_boost:.2f}", 
                     xy=(0.02, 0.85), xycoords='axes fraction', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'qualia_advantage.png'), dpi=300)
        plt.close()
    
    def _plot_comparative_metrics(self, output_dir):
        """
        Gera gráfico comparativo de várias métricas entre ASIC e QUALIA.
        """
        # Dados para o gráfico radar
        metrics = ['Custo inicial', 'Consumo de energia', 'Lucro final', 
                   'Manutenção', 'Vida útil', 'Adaptabilidade']
        
        # Normalizar valores (quanto maior, melhor)
        asic_values = [
            1 - (self.asic_hardware_cost / max(self.asic_hardware_cost, self.qualia_hardware_cost)),
            1 - (self.asic_power / max(self.asic_power, self.qualia_power)),
            max(0, self.summary.get('asic_final_profit', 0)) / 
                max(abs(self.summary.get('asic_final_profit', 0)), 
                    abs(self.summary.get('qualia_final_profit', 0)), 1),
            1 - (self.asic_maintenance_factor / max(self.asic_maintenance_factor, self.qualia_maintenance_factor)),
            (1 - self.asic_degradation_rate) / max(1 - self.asic_degradation_rate, 1 - self.qualia_degradation_rate),
            0.5  # ASIC tem menor adaptabilidade por definição
        ]
        
        qualia_values = [
            1 - (self.qualia_hardware_cost / max(self.asic_hardware_cost, self.qualia_hardware_cost)),
            1 - (self.qualia_power / max(self.asic_power, self.qualia_power)),
            max(0, self.summary.get('qualia_final_profit', 0)) / 
                max(abs(self.summary.get('asic_final_profit', 0)), 
                    abs(self.summary.get('qualia_final_profit', 0)), 1),
            1 - (self.qualia_maintenance_factor / max(self.asic_maintenance_factor, self.qualia_maintenance_factor)),
            (1 - self.qualia_degradation_rate) / max(1 - self.asic_degradation_rate, 1 - self.qualia_degradation_rate),
            0.9  # QUALIA tem maior adaptabilidade devido ao aprendizado quântico
        ]
        
        # Adicionar o primeiro valor novamente para fechar o gráfico
        metrics = np.array(metrics)
        asic_values = np.array(asic_values)
        qualia_values = np.array(qualia_values)
        
        # Converter para formato polar
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        
        # Fechar o gráfico circular
        metrics = np.append(metrics, metrics[0])
        asic_values = np.append(asic_values, asic_values[0])
        qualia_values = np.append(qualia_values, qualia_values[0])
        angles = np.append(angles, angles[0])
        
        # Criar figura
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, polar=True)
        
        # Plotar ASIC
        ax.plot(angles, asic_values, 'o-', linewidth=2, color='#FF5733', label='ASIC')
        ax.fill(angles, asic_values, alpha=0.1, color='#FF5733')
        
        # Plotar QUALIA
        ax.plot(angles, qualia_values, 'o-', linewidth=2, color='#33A8FF', label='QUALIA')
        ax.fill(angles, qualia_values, alpha=0.1, color='#33A8FF')
        
        # Configurar gráfico
        ax.set_thetagrids(np.degrees(angles[:-1]), metrics[:-1])
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.set_title('Comparação de Métricas: ASIC vs QUALIA', size=15, y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comparative_metrics.png'), dpi=300)
        plt.close()
