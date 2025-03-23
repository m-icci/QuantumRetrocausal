"""
Módulo de simulação para o projeto QUALIA.

Este módulo fornece componentes para simulação de mercados, operações quânticas,
ordens de negociação, avaliação de risco e um motor de simulação completo.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Callable, Optional, Union, Tuple
from datetime import datetime, timedelta
import json
import uuid
import os
from dataclasses import dataclass, field

from ..exceptions import SimulationError

class MarketSimulator:
    """Simulador de mercado financeiro com parâmetros configuráveis."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o simulador de mercado.
        
        Args:
            config: Configuração do simulador, com parâmetros de mercado.
        """
        self.config = config.get("market", {})
        self.initial_price = self.config.get("initial_price", 100.0)
        self.volatility = self.config.get("volatility", 0.2)
        self.drift = self.config.get("drift", 0.1)
        self.tick_size = self.config.get("tick_size", 0.01)
    
    def simulate(self, duration: timedelta, interval: str = "1min") -> pd.DataFrame:
        """
        Simula o mercado pelo período especificado.
        
        Args:
            duration: Duração da simulação.
            interval: Intervalo entre os pontos de dados.
            
        Returns:
            DataFrame com os dados de mercado simulados.
        """
        # Calcula número de pontos com base na duração e intervalo
        if interval == "1min":
            num_points = int(duration.total_seconds() / 60)
        elif interval == "1h":
            num_points = int(duration.total_seconds() / 3600)
        elif interval == "1d":
            num_points = int(duration.total_seconds() / 86400)
        else:
            raise SimulationError(f"Intervalo não suportado: {interval}")
        
        # Gera timestamps
        now = datetime.now()
        timestamps = [now + timedelta(minutes=i) for i in range(num_points)]
        
        # Simula preços com movimento browniano geométrico
        dt = 1.0 / 252.0  # Dia de negociação como fração do ano
        price_path = [self.initial_price]
        for _ in range(num_points - 1):
            price = price_path[-1]
            daily_return = np.random.normal(
                self.drift * dt, 
                self.volatility * np.sqrt(dt)
            )
            price = price * (1 + daily_return)
            price = round(price / self.tick_size) * self.tick_size  # Arredonda para o tick size
            price_path.append(max(price, self.tick_size))  # Preço não pode ser negativo
        
        # Gera OHLCV a partir dos preços de fechamento
        df = pd.DataFrame(index=range(num_points))
        df["timestamp"] = timestamps
        df["close"] = price_path
        
        # Simula alta, baixa e abertura com base no fechamento
        daily_vol = self.volatility / np.sqrt(252)
        df["open"] = df["close"].shift(1).fillna(self.initial_price)
        df["high"] = df[["open", "close"]].max(axis=1) * (1 + abs(np.random.normal(0, daily_vol, num_points)))
        df["low"] = df[["open", "close"]].min(axis=1) * (1 - abs(np.random.normal(0, daily_vol, num_points)))
        
        # Simula volume
        df["volume"] = np.random.exponential(1, num_points) * 1000
        
        return df

class QuantumSimulator:
    """Simulador de computação quântica para análise de mercado."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o simulador quântico.
        
        Args:
            config: Configuração do simulador, com parâmetros quânticos.
        """
        self.config = config.get("quantum", {})
        self.num_qubits = self.config.get("num_qubits", 4)
        self.shots = self.config.get("shots", 1000)
        self.noise_model = self.config.get("noise_model", "basic")
    
    def prepare_state(self, num_qubits: int, initial_state: List[float]) -> np.ndarray:
        """
        Prepara um estado quântico inicial.
        
        Args:
            num_qubits: Número de qubits.
            initial_state: Estado inicial como vetor de amplitudes.
            
        Returns:
            Vetor de estado quântico.
        """
        if len(initial_state) != num_qubits:
            raise SimulationError(f"Tamanho do estado inicial ({len(initial_state)}) não corresponde ao número de qubits ({num_qubits})")
        
        # Normaliza o estado inicial
        norm = np.sqrt(sum(x**2 for x in initial_state))
        normalized = [x / norm for x in initial_state]
        
        # Cria estado como vetor de tamanho 2^n
        state = np.zeros(2**num_qubits, dtype=complex)
        for i, amp in enumerate(normalized):
            state[i] = complex(amp, 0)
        
        return state
    
    def apply_operations(self, state: np.ndarray, operations: List[str]) -> Dict[str, Any]:
        """
        Aplica operações quânticas ao estado.
        
        Args:
            state: Estado quântico.
            operations: Lista de operações a serem aplicadas.
            
        Returns:
            Dicionário com o estado final e medições.
        """
        # Matrizes para operações básicas
        H = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex)  # Hadamard
        X = np.array([[0, 1], [1, 0]], dtype=complex)                  # Pauli-X
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)               # Pauli-Y
        Z = np.array([[1, 0], [0, -1]], dtype=complex)                 # Pauli-Z
        
        # Aplica as operações
        for op in operations:
            if op == "H":
                # Aplica H ao primeiro qubit
                state = np.kron(H, np.eye(len(state) // 2)) @ state
            elif op == "X":
                # Aplica X ao primeiro qubit
                state = np.kron(X, np.eye(len(state) // 2)) @ state
            elif op == "Y":
                # Aplica Y ao primeiro qubit
                state = np.kron(Y, np.eye(len(state) // 2)) @ state
            elif op == "Z":
                # Aplica Z ao primeiro qubit
                state = np.kron(Z, np.eye(len(state) // 2)) @ state
            elif op == "CNOT":
                # Implementação simples de CNOT entre os dois primeiros qubits
                dim = len(state)
                cnot = np.eye(dim, dtype=complex)
                half_dim = dim // 2
                for i in range(half_dim):
                    cnot[i + half_dim, i + half_dim] = 0
                    cnot[i + half_dim, i] = 0
                    cnot[i, i + half_dim] = 0
                    cnot[i, i] = 0
                    cnot[i, i + half_dim] = 1
                    cnot[i + half_dim, i] = 1
                state = cnot @ state
            else:
                raise SimulationError(f"Operação quântica não suportada: {op}")
        
        # Normaliza o estado final
        norm = np.sqrt(np.sum(np.abs(state)**2))
        if norm > 0:
            state = state / norm
        
        # Simula medições
        probabilities = np.abs(state)**2
        measurements = np.random.choice(
            range(len(state)), 
            size=self.shots, 
            p=probabilities
        )
        
        # Adiciona ruído se solicitado
        if self.noise_model != "none":
            noise_factor = 0.05  # 5% de ruído
            for i in range(len(state)):
                noise = np.random.normal(0, noise_factor)
                state[i] += complex(noise, noise)
            
            # Renormaliza após adicionar ruído
            norm = np.sqrt(np.sum(np.abs(state)**2))
            if norm > 0:
                state = state / norm
        
        return {
            "final_state": state,
            "measurements": measurements.tolist(),
            "probabilities": probabilities.tolist()
        }

class OrderSimulator:
    """Simulador de ordens de negociação com latência e slippage."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o simulador de ordens.
        
        Args:
            config: Configuração do simulador, com parâmetros de ordens.
        """
        self.config = config.get("order", {})
        self.latency = self.config.get("latency", 50)  # ms
        self.slippage = self.config.get("slippage", 0.001)  # 0.1%
        self.fees = self.config.get("fees", 0.001)  # 0.1%
    
    def simulate_execution(self, order: Dict[str, Any], market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Simula a execução de uma ordem no mercado.
        
        Args:
            order: Ordem a ser executada.
            market_data: Dados de mercado para a simulação.
            
        Returns:
            Resultado da execução da ordem.
        """
        # Encontra o preço na hora da ordem
        timestamp = order["timestamp"]
        
        # Encontra o índice mais próximo
        idx = market_data["timestamp"].searchsorted(timestamp)
        if idx >= len(market_data):
            idx = len(market_data) - 1
        
        # Obtém os dados do mercado no momento
        market_price = market_data.iloc[idx]["close"]
        
        # Simula latência
        latency_ms = np.random.normal(self.latency, self.latency * 0.2)
        latency_ms = max(1, latency_ms)
        
        # Simula slippage
        slippage_factor = 1.0
        if order["side"] == "buy":
            slippage_factor = 1.0 + abs(np.random.normal(self.slippage, self.slippage * 0.5))
        else:  # sell
            slippage_factor = 1.0 - abs(np.random.normal(self.slippage, self.slippage * 0.5))
        
        # Calcula preço de execução com slippage
        execution_price = market_price * slippage_factor
        
        # Calcula taxa
        fee_amount = order["amount"] * execution_price * self.fees
        
        # Prepara resultado da execução
        return {
            "order_id": str(uuid.uuid4()),
            "original_order": order,
            "filled_amount": order["amount"],
            "average_price": execution_price,
            "fees": fee_amount,
            "latency": latency_ms,
            "timestamp": timestamp + timedelta(milliseconds=latency_ms),
            "status": "filled"
        }

class RiskSimulator:
    """Simulador de risco para análise de posições."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o simulador de risco.
        
        Args:
            config: Configuração do simulador, com parâmetros de risco.
        """
        self.config = config.get("risk", {})
        self.initial_balance = self.config.get("initial_balance", 100000)
        self.max_position = self.config.get("max_position", 0.1)  # 10% do portfolio
        self.stop_loss = self.config.get("stop_loss", 0.02)  # 2% de perda máxima
    
    def simulate_risk(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simula métricas de risco para uma posição.
        
        Args:
            position: Posição atual.
            
        Returns:
            Métricas de risco.
        """
        # Extrai dados da posição
        entry_price = position["entry_price"]
        current_price = position["current_price"]
        amount = position["amount"]
        
        # Valor da posição
        position_value = amount * current_price
        
        # Calcula PnL
        pnl = (current_price - entry_price) * amount
        pnl_percent = pnl / (entry_price * amount)
        
        # Calcula Value at Risk (VaR) - simplificado usando distribuição normal
        daily_var_95 = position_value * 0.02  # Assume 2% VaR diário no nível de confiança de 95%
        
        # Calcula DrawDown máximo
        max_loss = entry_price * amount * self.stop_loss
        
        # Calcula preço de margin call (simplificado)
        margin_call_price = entry_price * (1 - 1 / self.max_position)
        
        return {
            "position_value": position_value,
            "pnl": pnl,
            "pnl_percent": pnl_percent,
            "value_at_risk": daily_var_95,
            "max_loss": max_loss,
            "margin_call_price": margin_call_price
        }

class SimulationEngine:
    """Motor de simulação que integra os diferentes simuladores."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o motor de simulação.
        
        Args:
            config: Configuração para todos os simuladores.
        """
        self.config = config
        self.market_simulator = MarketSimulator(config)
        self.quantum_simulator = QuantumSimulator(config)
        self.order_simulator = OrderSimulator(config)
        self.risk_simulator = RiskSimulator(config)
        
        # Estado da simulação
        self.balance = config.get("risk", {}).get("initial_balance", 100000)
        self.positions = {}
        self.trades = []
        self.history = []
    
    def run_simulation(self, strategy: Callable, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Executa uma simulação completa usando uma estratégia definida.
        
        Args:
            strategy: Função de estratégia que gera sinais de negociação.
            market_data: Dados de mercado para a simulação.
            
        Returns:
            Resultados da simulação.
        """
        # Reseta o estado da simulação
        self.balance = self.config.get("risk", {}).get("initial_balance", 100000)
        self.positions = {}
        self.trades = []
        self.history = []
        
        # Gera sinais de negociação
        signals = strategy(market_data)
        
        # Cria portfólio vazio
        portfolio = pd.DataFrame({
            "timestamp": market_data["timestamp"],
            "balance": self.balance,
            "equity": self.balance,
            "open_positions": 0
        })
        
        # Para cada ponto de dados
        for i in range(1, len(market_data)):
            data = market_data.iloc[i]
            signal = signals.iloc[i]
            timestamp = data["timestamp"]
            price = data["close"]
            
            # Processa o sinal
            if signal > 0 and "BTC" not in self.positions:  # Compra
                amount = (self.balance * 0.1) / price  # Usa 10% do balanço
                
                # Cria ordem
                order = {
                    "type": "market",
                    "side": "buy",
                    "amount": amount,
                    "timestamp": timestamp
                }
                
                # Simula execução
                execution = self.order_simulator.simulate_execution(order, market_data)
                
                # Atualiza balanço e posições
                cost = execution["filled_amount"] * execution["average_price"] + execution["fees"]
                self.balance -= cost
                
                # Registra posição
                self.positions["BTC"] = {
                    "amount": execution["filled_amount"],
                    "entry_price": execution["average_price"],
                    "timestamp": execution["timestamp"]
                }
                
                # Registra trade
                self.trades.append({
                    "type": "buy",
                    "amount": execution["filled_amount"],
                    "price": execution["average_price"],
                    "cost": cost,
                    "timestamp": execution["timestamp"]
                })
                
            elif signal < 0 and "BTC" in self.positions:  # Venda
                position = self.positions["BTC"]
                
                # Cria ordem
                order = {
                    "type": "market",
                    "side": "sell",
                    "amount": position["amount"],
                    "timestamp": timestamp
                }
                
                # Simula execução
                execution = self.order_simulator.simulate_execution(order, market_data)
                
                # Atualiza balanço
                revenue = execution["filled_amount"] * execution["average_price"] - execution["fees"]
                self.balance += revenue
                
                # Calcula P&L
                pnl = revenue - (position["amount"] * position["entry_price"])
                
                # Remove posição
                del self.positions["BTC"]
                
                # Registra trade
                self.trades.append({
                    "type": "sell",
                    "amount": execution["filled_amount"],
                    "price": execution["average_price"],
                    "revenue": revenue,
                    "pnl": pnl,
                    "timestamp": execution["timestamp"]
                })
            
            # Atualiza equity (balanço + valor das posições abertas)
            open_positions_value = sum(
                pos["amount"] * price 
                for asset, pos in self.positions.items()
            )
            equity = self.balance + open_positions_value
            
            # Atualiza história do portfólio
            portfolio.at[i, "balance"] = self.balance
            portfolio.at[i, "equity"] = equity
            portfolio.at[i, "open_positions"] = len(self.positions)
        
        # Calcula métricas de desempenho
        returns = portfolio["equity"].pct_change().dropna()
        
        # Calcula métricas
        total_return = (portfolio["equity"].iloc[-1] / portfolio["equity"].iloc[0]) - 1
        ann_return = (1 + total_return) ** (252 / len(portfolio)) - 1
        ann_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = ann_return / ann_volatility if ann_volatility > 0 else 0
        
        # Calcula drawdown
        rolling_max = portfolio["equity"].cummax()
        drawdown = (portfolio["equity"] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Resultados da simulação
        results = {
            "trades": self.trades,
            "portfolio": portfolio,
            "performance": {
                "total_return": total_return,
                "annual_return": ann_return,
                "annual_volatility": ann_volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "trade_count": len(self.trades)
            }
        }
        
        return results
    
    def run_monte_carlo(self, num_simulations: int, duration: timedelta) -> List[pd.DataFrame]:
        """
        Executa múltiplas simulações de Monte Carlo.
        
        Args:
            num_simulations: Número de simulações a executar.
            duration: Duração de cada simulação.
            
        Returns:
            Lista de DataFrames com os resultados das simulações.
        """
        simulations = []
        
        for _ in range(num_simulations):
            # Simula mercado
            market_data = self.market_simulator.simulate(duration)
            simulations.append(market_data)
            
        return simulations
    
    def calculate_monte_carlo_stats(self, simulations: List[pd.DataFrame]) -> Dict[str, Any]:
        """
        Calcula estatísticas de simulações Monte Carlo.
        
        Args:
            simulations: Lista de simulações.
            
        Returns:
            Estatísticas das simulações.
        """
        # Calcula retornos das simulações
        final_returns = []
        
        for sim in simulations:
            start_price = sim["close"].iloc[0]
            end_price = sim["close"].iloc[-1]
            ret = (end_price / start_price) - 1
            final_returns.append(ret)
        
        # Calcula estatísticas
        mean_return = np.mean(final_returns)
        std_return = np.std(final_returns)
        var_95 = np.percentile(final_returns, 5)  # VaR a 95%
        max_drawdown = min(final_returns)
        
        return {
            "mean_return": mean_return,
            "std_return": std_return,
            "var_95": var_95,
            "max_drawdown": max_drawdown
        }
    
    def run_stress_tests(self, scenarios: List[Dict[str, Any]], market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Executa testes de estresse com diferentes cenários.
        
        Args:
            scenarios: Lista de cenários para teste de estresse.
            market_data: Dados de mercado base.
            
        Returns:
            Resultados dos testes de estresse.
        """
        results = []
        
        for scenario in scenarios:
            # Cria configuração de cenário
            scenario_config = self.config.copy()
            for key, value in scenario.items():
                if key in scenario_config.get("market", {}):
                    scenario_config["market"][key] = value
            
            # Cria simulador para este cenário
            simulator = MarketSimulator(scenario_config)
            
            # Simula mercado para este cenário
            duration = timedelta(days=len(market_data) // 1440)  # Assumindo dados de 1 minuto
            scenario_data = simulator.simulate(duration)
            
            # Cria motor de simulação para este cenário
            engine = SimulationEngine(scenario_config)
            
            # Define estratégia simples para teste
            def simple_strategy(data):
                close = data["close"]
                sma = close.rolling(window=20).mean()
                return pd.Series(np.where(close > sma, 1, -1), index=data.index)
            
            # Executa simulação com a estratégia
            result = engine.run_simulation(simple_strategy, scenario_data)
            
            # Adiciona identificador do cenário
            result["scenario"] = scenario
            
            results.append(result)
        
        return results
    
    def analyze_scenarios(self, scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Analisa diferentes cenários de mercado.
        
        Args:
            scenarios: Dicionário de cenários para análise.
            
        Returns:
            Resultados da análise de cenários.
        """
        results = {}
        
        for name, scenario in scenarios.items():
            # Cria configuração específica para o cenário
            scenario_config = self.config.copy()
            for key, value in scenario.items():
                if key in scenario_config.get("market", {}):
                    scenario_config["market"][key] = value
            
            # Simula mercado para este cenário
            simulator = MarketSimulator(scenario_config)
            market_data = simulator.simulate(timedelta(days=30))
            
            # Calcula retornos
            returns = market_data["close"].pct_change().dropna()
            
            # Calcula métricas de risco
            volatility = returns.std() * np.sqrt(252)
            var_95 = np.percentile(returns, 5) * np.sqrt(252)
            
            # Calcula retorno esperado
            expected_return = (market_data["close"].iloc[-1] / market_data["close"].iloc[0]) - 1
            
            results[name] = {
                "expected_return": expected_return,
                "risk_metrics": {
                    "volatility": volatility,
                    "var_95": var_95
                }
            }
        
        return results

    def save_simulation(self, results: Dict[str, Any], filepath: str) -> None:
        """
        Salva os resultados da simulação em arquivo.
        
        Args:
            results: Resultados da simulação.
            filepath: Caminho do arquivo para salvar.
        """
        # Converte DataFrame para formato serializável
        serializable_results = results.copy()
        
        if "portfolio" in serializable_results:
            serializable_results["portfolio"] = serializable_results["portfolio"].to_dict(orient="records")
        
        # Salva em arquivo JSON
        with open(filepath, "w") as f:
            json.dump(serializable_results, f, indent=2, default=str)
    
    def load_simulation(self, filepath: str) -> Dict[str, Any]:
        """
        Carrega resultados de simulação de arquivo.
        
        Args:
            filepath: Caminho do arquivo para carregar.
            
        Returns:
            Resultados da simulação.
        """
        # Carrega de arquivo JSON
        with open(filepath, "r") as f:
            results = json.load(f)
        
        # Converte dicionários de volta para DataFrame
        if "portfolio" in results:
            results["portfolio"] = pd.DataFrame.from_records(results["portfolio"])
        
        return results

__all__ = [
    'MarketSimulator',
    'QuantumSimulator',
    'OrderSimulator',
    'RiskSimulator',
    'SimulationEngine'
] 