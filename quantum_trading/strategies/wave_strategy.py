"""
Estratégia WAVE: Weighted Adaptive Volatility Exploitation
Implementação de sistema adaptativo para arbitragem de criptomoedas
"""

import numpy as np
import pandas as pd
import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import json
import os
from ..market_api import MarketAPI
from ..exceptions import StrategyError
import math
import traceback

logger = logging.getLogger("wave_strategy")

class QuantumEntropyAnalyzer:
    """Analisador de entropia quântica para detecção de padrões de mercado"""
    
    def __init__(self, window_size: int = 24):
        self.window_size = window_size
        self.entropy_history = {}
    
    def calculate_quantum_entropy(self, price_data: List[float]) -> float:
        """Calcula entropia quântica dos preços"""
        if len(price_data) < 2:
            return 0.0
            
        try:
            # Converter para array numpy e garantir tipo float
            price_data = np.array([float(p) for p in price_data], dtype=np.float64)
            
            # Normalizar preços
            mean_price = np.mean(price_data)
            if mean_price == 0:
                return 0.0
                
            normalized_prices = price_data / mean_price
            
            # Calcular matriz de densidade
            density_matrix = np.outer(normalized_prices, normalized_prices)
            trace = np.trace(density_matrix)
            if trace == 0:
                return 0.0
                
            density_matrix = density_matrix / trace
            
            # Calcular entropia de von Neumann
            eigenvalues = np.linalg.eigvals(density_matrix)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Filtrar valores muito próximos de zero
            
            if len(eigenvalues) == 0:
                return 0.0
                
            entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
            
            return float(entropy)
            
        except Exception as e:
            logger.error(f"Erro ao calcular entropia quântica: {e}")
            return 0.0

class RetrocausalOptimizer:
    """Otimizador retrocausal para melhorar decisões de trading"""
    
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.trade_history = []
    
    def analyze_future_patterns(self, current_state: Dict[str, Any]) -> Dict[str, float]:
        """Analisa padrões futuros para otimizar decisões presentes"""
        if len(self.trade_history) < self.history_size:
            return {"confidence": 0.5, "adjustment_factor": 1.0}
            
        # Análise de padrões futuros
        future_patterns = self._extract_future_patterns()
        
        # Calcular fatores de ajuste
        confidence = self._calculate_confidence(future_patterns)
        adjustment_factor = self._calculate_adjustment(future_patterns)
        
        return {
            "confidence": confidence,
            "adjustment_factor": adjustment_factor
        }
    
    def _extract_future_patterns(self) -> List[Dict[str, Any]]:
        """Extrai padrões futuros do histórico"""
        patterns = []
        for i in range(len(self.trade_history) - 1):
            current = self.trade_history[i]
            future = self.trade_history[i + 1]
            patterns.append({
                "current_state": current,
                "future_outcome": future
            })
        return patterns
    
    def _calculate_confidence(self, patterns: List[Dict[str, Any]]) -> float:
        """Calcula nível de confiança baseado em padrões futuros"""
        if not patterns:
            return 0.5
            
        successful_patterns = sum(1 for p in patterns if p["future_outcome"]["success"])
        return successful_patterns / len(patterns)
    
    def _calculate_adjustment(self, patterns: List[Dict[str, Any]]) -> float:
        """Calcula fator de ajuste baseado em padrões futuros"""
        if not patterns:
            return 1.0
            
        try:
            # Extrair profits e converter para array numpy
            profits = np.array([float(p["future_outcome"]["profit"]) for p in patterns], dtype=np.float64)
            
            # Remover valores inválidos
            profits = profits[~np.isnan(profits)]
            profits = profits[~np.isinf(profits)]
            
            if len(profits) == 0:
                return 1.0
                
            # Calcular média dos lucros
            avg_profit = np.mean(profits)
            
            # Limitar ajuste entre 0.5 e 2.0
            adjustment = 1.0 + (avg_profit / 100.0)
            return float(np.clip(adjustment, 0.5, 2.0))
            
        except Exception as e:
            logger.error(f"Erro ao calcular ajuste retrocausal: {e}")
            return 1.0

class FractalAnalyzer:
    """Analisador fractal para compreensão de padrões de mercado"""
    
    def __init__(self, min_points: int = 10):
        self.min_points = min_points
    
    def calculate_fractal_dimension(self, data1: List[float], data2: List[float]) -> float:
        """
        Calcula a dimensão fractal entre duas séries de preços usando o método box-counting
        
        Args:
            data1: Série de preços da primeira exchange
            data2: Série de preços da segunda exchange
            
        Returns:
            Dimensão fractal (entre 1.0 e 2.0)
        """
        try:
            # Garantir que temos dados suficientes
            if len(data1) < 5 or len(data2) < 5:
                return 1.5  # Valor padrão
            
            # Calcular diferenças entre os preços
            diff = []
            for i in range(min(len(data1), len(data2))):
                if data1[i] > 0 and data2[i] > 0:
                    # Usar spread relativo ao invés da diferença absoluta
                    spread = abs(data1[i] - data2[i]) / max(data1[i], data2[i])
                    diff.append(spread)
            
            if not diff:
                return 1.5  # Valor padrão
            
            # Normalizar para intervalo [0, 1]
            max_diff = max(diff)
            if max_diff == 0:
                return 1.5  # Valor padrão
            
            norm_diff = [d / max_diff for d in diff]
            
            # Calcular dimensão fractal por box-counting
            # Número de boxes em diferentes escalas
            scales = [2, 4, 8, 16]
            counts = []
            
            for scale in scales:
                boxsize = 1.0 / scale
                count = 0
                for d in norm_diff:
                    box_idx = int(d / boxsize)
                    count += 1
                counts.append(count)
            
            # Calcular dimensão por regressão linear
            x = np.log(1.0 / np.array(scales))
            y = np.log(np.array(counts))
            
            # Usar regressão linear se temos pontos suficientes
            if len(x) > 1 and len(y) > 1:
                try:
                    slope, _, _, _, _ = np.polyfit(x, y, 1, full=True)
                    fractal_dim = slope[0]
                    
                    # Limitar ao intervalo [1.0, 2.0]
                    fractal_dim = max(1.0, min(2.0, fractal_dim))
                    return float(fractal_dim)
                except:
                    # Em caso de erro na regressão
                    return 1.5
            else:
                return 1.5
            
        except Exception as e:
            logger.warning(f"Erro ao calcular dimensão fractal: {e}")
            return 1.5  # Valor padrão em caso de erro

class AdaptiveSystem:
    """Sistema adaptativo baseado em biologia"""
    
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.adaptation_history = {}
    
    def adapt_parameters(self, 
                        current_state: Dict[str, Any],
                        performance_metrics: Dict[str, float]) -> Dict[str, float]:
        """Adapta parâmetros do sistema baseado em performance"""
        if not self.adaptation_history:
            return current_state
            
        # Análise de performance histórica
        success_rate = performance_metrics.get("success_rate", 0.5)
        avg_profit = performance_metrics.get("avg_profit", 0.0)
        
        # Calcular ajustes adaptativos
        adjustments = {
            "threshold": self._calculate_threshold_adjustment(success_rate),
            "position_size": self._calculate_position_adjustment(avg_profit),
            "rebalance_frequency": self._calculate_rebalance_adjustment(performance_metrics)
        }
        
        # Aplicar ajustes com learning rate
        adapted_state = current_state.copy()
        for param, adjustment in adjustments.items():
            if param in adapted_state:
                adapted_state[param] *= (1 + self.learning_rate * adjustment)
        
        return adapted_state
    
    def _calculate_threshold_adjustment(self, success_rate: float) -> float:
        """Calcula ajuste de threshold baseado em taxa de sucesso"""
        return success_rate - 0.5  # Ajuste entre -0.5 e 0.5
    
    def _calculate_position_adjustment(self, avg_profit: float) -> float:
        """Calcula ajuste de tamanho de posição baseado em lucro médio"""
        return np.tanh(avg_profit / 100.0)  # Ajuste entre -1 e 1
    
    def _calculate_rebalance_adjustment(self, metrics: Dict[str, float]) -> float:
        """Calcula ajuste de frequência de rebalanceamento"""
        volatility = metrics.get("volatility", 0.5)
        return volatility - 0.5  # Ajuste entre -0.5 e 0.5

class WAVEStrategy:
    """
    Implementação da estratégia WAVE (Weighted Adaptive Volatility Exploitation)
    
    A estratégia utiliza análise de volatilidade temporal para ajustar thresholds
    de spread, alocação dinâmica de capital e rebalanceamento adaptativo.
    """
    
    def __init__(self, exchanges: List[MarketAPI], pairs: List[str], config: Dict[str, Any]):
        """
        Inicializa a estratégia WAVE
        
        Args:
            exchanges: Lista de exchanges
            pairs: Lista de pares de trading
            config: Configuração da estratégia
        """
        self.exchanges = exchanges
        self.pairs = pairs
        self.config = config
        
        # Inicializar componentes
        self.quantum_entropy_analyzer = QuantumEntropyAnalyzer()
        self.retrocausal_optimizer = RetrocausalOptimizer()
        self.adaptive_system = AdaptiveSystem()
        self.fractal_analyzer = FractalAnalyzer()
        
        # Estado interno
        self.volatility_matrix = {}
        self.trade_history = []
        self.active_arbitrages = {}
        self.exchange_balances = {}
        
        # Métricas por par
        self.pair_metrics = self._initialize_metrics()
        
        # Estado da estratégia
        self.state = {
            "last_rebalance": None,
            "performance": {},
            "last_update": datetime.now().isoformat()
        }
        
        logger.info(f"Estratégia WAVE inicializada com {len(exchanges)} exchanges e {len(pairs)} pares")
    
    def _initialize_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Inicializa métricas para cada par"""
        metrics = {}
        for pair in self.pairs:
            metrics[pair] = {
                "total_trades": 0,
                "successful_trades": 0,
                "failed_trades": 0,
                "total_profit": 0.0,
                "success_rate": 0.0,
                "avg_profit": 0.0,
                "best_trade": 0.0,
                "worst_trade": 0.0,
                "volatility_index": 0.5,  # Índice inicial de volatilidade
                "opportunity_frequency": 0.0,
                "exchange_stats": {}
            }
            
            # Estatísticas por exchange
            for exchange in self.exchanges:
                ex_id = exchange.exchange_id
                metrics[pair]["exchange_stats"][ex_id] = {
                    "trades": 0,
                    "profit": 0.0,
                    "success_rate": 0.0,
                    "avg_execution_time": 0.0,
                    "slippage": 0.0,
                    "profit_ratio": 1.0  # Inicialmente igual
                }
        
        return metrics
    
    async def initialize(self) -> None:
        """Inicializa a estratégia com dados históricos"""
        try:
            # Obter balanços iniciais
            await self.update_balances()
            
            # Construir matriz de volatilidade
            await self.build_volatility_matrix()
            
            # Calcular alocação inicial
            target_allocation = self.calculate_allocation()
            
            # Executar rebalanceamento inicial
            if any(self.exchange_balances.values()):
                await self.rebalance_exchanges(target_allocation)
            
            self.state["last_rebalance"] = datetime.now().isoformat()
            logger.info("Inicialização completa da estratégia WAVE")
            
        except Exception as e:
            logger.error(f"Erro na inicialização da estratégia: {e}")
            raise StrategyError(f"Falha na inicialização: {str(e)}")
    
    async def update_balances(self) -> Dict[str, float]:
        """Atualiza balanços de todas as exchanges"""
        balances = {}
        
        for exchange in self.exchanges:
            try:
                ex_id = exchange.exchange_id
                # Usar await diretamente na chamada de get_balance, já que agora é assíncrona
                balance = await exchange.get_balance('USDT')
                balances[ex_id] = float(balance)
                logger.info(f"Balanço atualizado para {ex_id}: {balance} USDT")
            except Exception as e:
                logger.error(f"Erro ao obter balanço de {exchange.exchange_id}: {e}")
                balances[exchange.exchange_id] = 0.0
        
        self.exchange_balances = balances
        return balances
    
    async def build_volatility_matrix(self) -> Dict[str, List[float]]:
        """Constrói matriz de volatilidade para cada par/segmento de tempo"""
        matrix = {}
        
        for pair in self.pairs:
            matrix[pair] = []
            
            # Obter dados históricos de cada exchange
            all_data = []
            for exchange in self.exchanges:
                try:
                    # Obter dados das últimas 24 horas em intervalos de 1 hora
                    data = await exchange.get_ohlcv(
                        pair,
                        '1h',
                        self.config["history_window"]
                    )
                    
                    if isinstance(data, list) and data:
                        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
                        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
                        df["hour"] = df["datetime"].dt.hour
                        all_data.append(df)
                except Exception as e:
                    logger.error(f"Erro ao obter dados históricos de {exchange.exchange_id}: {e}")
            
            if not all_data:
                logger.warning(f"Sem dados históricos para {pair}, usando valores padrão")
                matrix[pair] = [0.005] * 24  # Valor padrão de volatilidade
                continue
            
            # Combinar dados de todas as exchanges
            try:
                combined_data = pd.concat(all_data)
                
                # Calcular volatilidade por hora do dia
                volatility_by_hour = []
                for hour in range(24):
                    hour_data = combined_data[combined_data["hour"] == hour]
                    
                    if len(hour_data) >= 2:
                        # Calcular volatilidade como desvio padrão normalizado
                        std = hour_data["close"].std()
                        mean = hour_data["close"].mean()
                        
                        if mean > 0:
                            volatility = std / mean
                        else:
                            volatility = 0.005  # Valor padrão
                    else:
                        volatility = 0.005  # Valor padrão para horas sem dados suficientes
                    
                    volatility_by_hour.append(float(volatility))
                
                matrix[pair] = volatility_by_hour
            except Exception as e:
                logger.error(f"Erro ao calcular matriz de volatilidade para {pair}: {e}")
                matrix[pair] = [0.005] * 24  # Valor padrão em caso de erro
        
        self.volatility_matrix = matrix
        logger.info(f"Matriz de volatilidade construída para {len(matrix)} pares")
        return matrix
    
    def calculate_allocation(self) -> Dict[str, float]:
        """Calcula alocação ótima de capital entre exchanges"""
        # Obter total de capital disponível
        total_capital = sum(self.exchange_balances.values())
        if total_capital <= 0:
            return {ex.exchange_id: 0.0 for ex in self.exchanges}
        
        # Inicializar scores
        allocation_scores = {}
        for exchange in self.exchanges:
            ex_id = exchange.exchange_id
            allocation_scores[ex_id] = 0.0
            
            # Calcular score baseado em métricas históricas
            for pair in self.pairs:
                if ex_id in self.pair_metrics[pair]["exchange_stats"]:
                    stats = self.pair_metrics[pair]["exchange_stats"][ex_id]
                    
                    # Combinar fatores para score
                    profit_factor = stats["profit_ratio"]
                    success_factor = stats["success_rate"] if stats["trades"] > 0 else 0.5
                    
                    # Adicionar ao score total
                    allocation_scores[ex_id] += profit_factor * (0.5 + 0.5 * success_factor)
        
        # Normalizar scores
        total_score = sum(allocation_scores.values()) if sum(allocation_scores.values()) > 0 else 1.0
        for ex_id in allocation_scores:
            normalized_score = allocation_scores[ex_id] / total_score
            allocation_scores[ex_id] = normalized_score * total_capital
        
        return allocation_scores
    
    def calculate_spread_threshold(self, pair: str, current_hour: int) -> float:
        """Calcula threshold adaptativo baseado na volatilidade"""
        base_threshold = self.config["base_threshold"]
        
        # Obter volatilidade para esta hora
        if pair in self.volatility_matrix and len(self.volatility_matrix[pair]) > current_hour:
            volatility_index = self.volatility_matrix[pair][current_hour]
        else:
            volatility_index = 0.02  # Valor padrão
        
        # Ajustar threshold baseado na volatilidade
        if volatility_index < 0.01:  # Baixa volatilidade
            return base_threshold * 0.8
        elif volatility_index < 0.03:  # Média volatilidade
            return base_threshold
        else:  # Alta volatilidade
            return base_threshold * 1.2
    
    async def _detect_opportunities(self) -> List[Dict]:
        """Detecta oportunidades de arbitragem entre exchanges"""
        opportunities = []
        
        # Configurar loggers específicos
        market_logger = logging.getLogger("market_data")
        spread_logger = logging.getLogger("spread_analysis")
        
        for pair in self.pairs:
            prices = {}
            volumes = {}
            price_history = {}
            
            for exchange in self.exchanges:
                retry_count = 0
                max_retries = 3
                
                while retry_count < max_retries:
                    try:
                        ex_id = exchange.exchange_id
                        
                        # Obter preço atual com retry
                        price = await exchange.get_price(pair)
                        prices[ex_id] = float(price)
                        market_logger.debug(f"Preço de {pair} em {ex_id}: {price}")
                        
                        # Obter volume em 24h
                        ticker = await exchange.get_ticker(pair)
                        volumes[ex_id] = float(ticker['volume'])
                        market_logger.debug(f"Volume de {pair} em {ex_id}: {ticker['volume']}")
                        
                        # Obter histórico de preços
                        history = await exchange.get_ohlcv(
                            pair,
                            '1h',
                            24  # Últimas 24 horas
                        )
                        
                        if isinstance(history, list) and len(history) > 0:
                            price_history[ex_id] = [float(candle[4]) for candle in history]  # Preços de fechamento
                            market_logger.debug(f"Histórico de {pair} em {ex_id}: Últimos preços = [{price_history[ex_id][-3:]}, ...]")
                        else:
                            price_history[ex_id] = []
                            market_logger.warning(f"Sem histórico para {pair} em {ex_id}")
                        
                        # Se chegamos aqui, tudo funcionou
                        break
                    except Exception as e:
                        logger.error(f"Erro ao obter dados de {exchange.exchange_id} para {pair}: {e}")
                        retry_count += 1
                        await asyncio.sleep(1)  # Esperar um pouco antes de tentar novamente
            
            # Verificar se temos dados suficientes (pelo menos 2 exchanges)
            if len(prices) < 2:
                logger.debug(f"Dados insuficientes para {pair} (menos de 2 exchanges)")
                continue
            
            market_logger.info(f"Preços de {pair}: {prices}")
            
            # Detectar arbitragem entre os pares de exchanges
            for buy_ex in self.exchanges:
                for sell_ex in self.exchanges:
                    buy_ex_id = buy_ex.exchange_id
                    sell_ex_id = sell_ex.exchange_id
                    
                    # Pular se for a mesma exchange
                    if buy_ex_id == sell_ex_id:
                        continue
                    
                    # Verificar se temos preços para ambas exchanges
                    if buy_ex_id not in prices or sell_ex_id not in prices:
                        continue
                    
                    # Verificar se temos histórico para ambas exchanges
                    if buy_ex_id not in price_history or sell_ex_id not in price_history:
                        continue
                    
                    buy_price = prices[buy_ex_id]
                    sell_price = prices[sell_ex_id]
                    
                    # Considerar apenas quando o preço de venda é maior que o de compra
                    if sell_price <= buy_price:
                        spread_logger.debug(f"{pair}: {sell_ex_id} ({sell_price}) <= {buy_ex_id} ({buy_price}) - Ignorando")
                        continue
                    
                    # Calcular spread
                    spread = (sell_price / buy_price) - 1
                    spread_logger.info(f"Spread para {pair}: {buy_ex_id} -> {sell_ex_id} = {spread:.6f} ({spread*100:.4f}%)")
                    
                    # Verificar spread mínimo com threshold mais baixo para detecção inicial
                    min_spread_threshold = self.config["min_spread"] * 0.9  # Reduzir em 10% para detecção
                    if spread <= min_spread_threshold:
                        spread_logger.debug(f"Spread {spread:.6f} abaixo do mínimo {min_spread_threshold:.6f}")
                        continue
                    
                    # Calcular volatilidade conjunta
                    hour = datetime.now().hour
                    volatility = self.volatility_matrix.get(pair, [0.005] * 24)[hour]
                    spread_logger.debug(f"Volatilidade de {pair} na hora {hour}: {volatility:.6f}")
                    
                    # Calcular threshold adaptativo com ajuste mais dinâmico
                    base_threshold = self.config["base_threshold"]
                    volatility_factor = 1 + volatility * 3  # Aumentar sensibilidade à volatilidade
                    threshold = base_threshold * volatility_factor
                    spread_logger.debug(f"Threshold adaptativo: base={base_threshold:.6f}, ajustado={threshold:.6f}")
                    
                    # Aplicar threshold
                    if spread <= threshold:
                        spread_logger.debug(f"Spread {spread:.6f} menor que threshold {threshold:.6f}")
                        continue
                    
                    # Calcular entropia quântica
                    entropy = self.quantum_entropy_analyzer.calculate_quantum_entropy(
                        price_history[buy_ex_id] + price_history[sell_ex_id]
                    )
                    spread_logger.debug(f"Entropia quântica para {pair}: {entropy:.6f}")
                    
                    # Calcular dimensão fractal
                    fractal_dimension = self.fractal_analyzer.calculate_fractal_dimension(
                        price_history[buy_ex_id],
                        price_history[sell_ex_id]
                    )
                    spread_logger.debug(f"Dimensão fractal para {pair}: {fractal_dimension:.6f}")
                    
                    # Calcular ajuste retrocausal
                    retrocausal_factors = self.retrocausal_optimizer.analyze_future_patterns({
                        "pair": pair,
                        "buy_ex": buy_ex_id,
                        "sell_ex": sell_ex_id,
                        "spread": spread,
                        "threshold": threshold,
                        "entropy": entropy,
                        "fractal_dimension": fractal_dimension
                    })
                    
                    # Aplicar ajuste retrocausal ao spread com peso maior para detecção inicial
                    adjustment_factor = retrocausal_factors["adjustment_factor"] * 1.1  # Aumentar em 10%
                    adjusted_spread = spread * adjustment_factor
                    confidence = retrocausal_factors["confidence"]
                    spread_logger.info(f"Ajuste retrocausal: fator={adjustment_factor:.4f}, spread={spread:.6f} -> ajustado={adjusted_spread:.6f}, confiança={confidence:.4f}")
                    
                    # Verificar se está próximo do threshold mesmo sendo menor
                    threshold_proximity_ratio = spread / threshold
                    if threshold_proximity_ratio > 0.9:  # Está 90% próximo do threshold
                        spread_logger.info(f"Spread próximo do threshold: {spread:.6f} vs {threshold:.6f} ({threshold_proximity_ratio:.4f})")
                    
                    # Cálculo de lucro esperado
                    expected_profit = adjusted_spread
                    
                    # Volume mínimo para execução - com ajuste dinâmico
                    min_volume = self.config["min_volume_24h"]
                    if entropy > 1.0:  # Alta entropia indica maior potencial
                        min_volume *= 0.8  # Reduzir requisito de volume
                    
                    volume_ok = all(v >= min_volume for v in volumes.values() if v > 0)
                    
                    if not volume_ok:
                        spread_logger.debug(f"Volume insuficiente para {pair}")
                        continue
                    
                    # Verificar se temos saldo para compra
                    buy_balance = self.exchange_balances.get(buy_ex_id, 0)
                    
                    if buy_balance <= 0:
                        spread_logger.debug(f"Sem saldo para compra em {buy_ex_id}")
                        continue
                    
                    # Adicionar oportunidade
                    opportunities.append({
                        "pair": pair,
                        "buy_exchange": buy_ex_id,
                        "sell_exchange": sell_ex_id,
                        "buy_price": buy_price,
                        "sell_price": sell_price,
                        "spread": spread,
                        "adjusted_spread": adjusted_spread,
                        "threshold": threshold,
                        "entropy": entropy,
                        "fractal_dimension": fractal_dimension,
                        "confidence": confidence,
                        "expected_profit": expected_profit,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    logger.info(f"Oportunidade detectada: {pair} - Spread: {spread:.4%}, Ajustado: {adjusted_spread:.4%}")
        
        # Ordenar oportunidades por lucro esperado
        opportunities.sort(key=lambda x: x["expected_profit"], reverse=True)
        
        if opportunities:
            logger.info(f"Detectadas {len(opportunities)} oportunidades. Melhor spread: {opportunities[0]['spread']:.4%}")
        
        return opportunities
    
    def calculate_trade_amount(self, opportunity: Dict[str, Any]) -> float:
        """Calcula o valor ótimo para o trade"""
        # Obter balanço na exchange de compra
        buy_exchange_id = opportunity['buy_exchange']
        buy_exchange_balance = self.exchange_balances.get(buy_exchange_id, 0)
        
        if buy_exchange_balance <= 0:
            return 0.0
        
        # Calcular valor máximo baseado na porcentagem configurada
        max_position = buy_exchange_balance * self.config["max_position_pct"]
        
        # Ajustar baseado na confiança (spread vs threshold)
        confidence = opportunity['spread'] / opportunity['threshold']
        confidence_factor = min(1.0, confidence)
        
        # Valor final do trade
        trade_amount = max_position * confidence_factor
        
        return trade_amount
    
    async def execute_arbitrage(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executa operação de arbitragem
        
        Args:
            opportunity: Oportunidade detectada
            
        Returns:
            Resultado da arbitragem
        """
        # Obtém logger de transações para registrar ações e mudanças no balanço
        trans_logger = logging.getLogger("transactions")
        
        # Registrar detecção de sinal de oportunidade
        trans_logger.info(f"🔍 SINAL DETECTADO: {opportunity['pair']} - Spread: {opportunity['spread']*100:.4f}%")
        trans_logger.info(f"   * Compra em {opportunity['buy_exchange']} a {opportunity['buy_price']:.6f}")
        trans_logger.info(f"   * Venda em {opportunity['sell_exchange']} a {opportunity['sell_price']:.6f}")
        trans_logger.info(f"   * Spread ajustado: {opportunity['adjusted_spread']*100:.4f}%")
        
        logger.info(f"Executando arbitragem: {opportunity['pair']} entre {opportunity['buy_exchange']} e {opportunity['sell_exchange']}")
        
        pair = opportunity['pair']
        buy_exchange = next((e for e in self.exchanges if e.exchange_id == opportunity['buy_exchange']), None)
        sell_exchange = next((e for e in self.exchanges if e.exchange_id == opportunity['sell_exchange']), None)
        
        if not buy_exchange or not sell_exchange:
            trans_logger.warning(f"❌ SINAL REJEITADO: Exchange não encontrada")
            return {'success': False, 'reason': 'exchange_not_found'}
        
        # Obter balanços atuais para registro de mudanças
        old_buy_balance = self.exchange_balances.get(opportunity['buy_exchange'], 0)
        old_sell_balance = self.exchange_balances.get(opportunity['sell_exchange'], 0)
        
        # Informar balanços iniciais
        trans_logger.info(f"💰 BALANÇO ANTES:")
        trans_logger.info(f"   * {opportunity['buy_exchange']}: {old_buy_balance:.4f} USDT")
        trans_logger.info(f"   * {opportunity['sell_exchange']}: {old_sell_balance:.4f} USDT")
        
        # Calcular quantidade baseada no spread e balanço disponível
        buy_price = opportunity['buy_price']
        sell_price = opportunity['sell_price']
        spread_pct = opportunity['spread']
        
        # Limitar quantidade a uma porcentagem do balanço
        max_amount = min(
            old_buy_balance * self.config["max_position_pct"],
            old_sell_balance * self.config["max_position_pct"] / sell_price
        )
        
        # Quantidade final
        quantity = max(0.001, min(max_amount, 0.01))  # Pelo menos 0.001, no máximo 0.01 ou max_amount
        trans_logger.info(f"🔢 QUANTIDADE CALCULADA: {quantity} {pair.split('/')[0]}")
        
        if quantity * buy_price < 10:  # Mínimo de 10 USDT por operação
            trans_logger.warning(f"❌ SINAL REJEITADO: Fundos insuficientes ({quantity * buy_price:.2f} USDT < 10 USDT mínimo)")
            return {'success': False, 'reason': 'insufficient_funds', 'min_required': 10}
        
        # Registrar início da arbitragem
        arb_id = f"{int(time.time())}_{pair}"
        self.active_arbitrages[arb_id] = {
            'pair': pair,
            'buy_exchange': opportunity['buy_exchange'],
            'sell_exchange': opportunity['sell_exchange'],
            'quantity': quantity,
            'buy_price': buy_price,
            'sell_price': sell_price,
            'start_time': datetime.now().isoformat(),
            'status': 'started'
        }
        
        # Executar compra
        try:
            trans_logger.info(f"🔄 EXECUTANDO COMPRA: {quantity} {pair.split('/')[0]} em {buy_exchange.exchange_id} a {buy_price}")
            logger.info(f"Comprando {quantity} {pair} em {buy_exchange.exchange_id} a {buy_price}")
            
            # Criar ordem de compra
            buy_order = await buy_exchange.create_market_buy_order(
                pair, 
                quantity
            )
            
            if not buy_order or not buy_order.get('orderId'):
                self.active_arbitrages[arb_id]['status'] = 'failed_buy'
                trans_logger.error(f"❌ COMPRA FALHOU: Não foi possível criar ordem")
                return {'success': False, 'reason': 'buy_order_failed'}
            
            trans_logger.info(f"✅ ORDEM DE COMPRA CRIADA: ID {buy_order.get('orderId')}")
            
            # Verificar status da ordem
            buy_order_id = buy_order['orderId']
            max_wait = 30  # Máximo 30 segundos de espera
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                buy_order_status = await buy_exchange.get_order_status(
                    buy_order_id
                )
                
                if buy_order_status['status'] == 'filled':
                    break
                    
                await asyncio.sleep(2)
                trans_logger.info(f"⏳ AGUARDANDO EXECUÇÃO DA COMPRA: {time.time() - start_time:.1f}s decorridos")
                buy_order_status = await buy_exchange.get_order_status(
                    buy_order_id
                )
                
                if buy_order_status['status'] == 'filled':
                    break
                elif buy_order_status['status'] in ['rejected', 'cancelled']:
                    self.active_arbitrages[arb_id]['status'] = 'buy_rejected'
                    trans_logger.error(f"❌ COMPRA REJEITADA: Status {buy_order_status['status']}")
                    return {'success': False, 'reason': 'buy_order_rejected'}
            
            # Verificar se a ordem foi concluída
            if buy_order_status['status'] != 'filled':
                # Tentar cancelar a ordem
                buy_exchange.cancel_order(buy_order_id)
                self.active_arbitrages[arb_id]['status'] = 'buy_timeout'
                trans_logger.error(f"❌ COMPRA TIMEOUT: {max_wait}s se passaram sem execução")
                return {'success': False, 'reason': 'buy_order_timeout'}
            
            # Atualizar estado da arbitragem
            actual_quantity = buy_order_status['filled']
            actual_price = buy_order_status['price']
            
            self.active_arbitrages[arb_id]['quantity'] = actual_quantity
            self.active_arbitrages[arb_id]['buy_price'] = actual_price
            self.active_arbitrages[arb_id]['status'] = 'bought'
            
            trans_logger.info(f"✅ COMPRA EXECUTADA: {actual_quantity} {pair.split('/')[0]} a {actual_price}")
            
            # Executar venda
            trans_logger.info(f"🔄 EXECUTANDO VENDA: {actual_quantity} {pair.split('/')[0]} em {sell_exchange.exchange_id} a {sell_price}")
            logger.info(f"Vendendo {actual_quantity} {pair} em {sell_exchange.exchange_id} a {sell_price}")
            
            sell_order = await sell_exchange.create_market_sell_order(
                pair, 
                actual_quantity
            )
            
            if not sell_order or not sell_order.get('orderId'):
                self.active_arbitrages[arb_id]['status'] = 'failed_sell'
                trans_logger.error(f"❌ VENDA FALHOU: Não foi possível criar ordem")
                return {'success': False, 'reason': 'sell_order_failed'}
            
            trans_logger.info(f"✅ ORDEM DE VENDA CRIADA: ID {sell_order.get('orderId')}")
            
            # Verificar status da ordem de venda
            sell_order_id = sell_order['orderId']
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                sell_order_status = await sell_exchange.get_order_status(
                    sell_order_id
                )
                
                if sell_order_status['status'] == 'filled':
                    break
                    
                await asyncio.sleep(2)
                trans_logger.info(f"⏳ AGUARDANDO EXECUÇÃO DA VENDA: {time.time() - start_time:.1f}s decorridos")
                sell_order_status = await sell_exchange.get_order_status(
                    sell_order_id
                )
                
                if sell_order_status['status'] == 'filled':
                    break
                elif sell_order_status['status'] in ['rejected', 'cancelled']:
                    self.active_arbitrages[arb_id]['status'] = 'sell_rejected'
                    trans_logger.error(f"❌ VENDA REJEITADA: Status {sell_order_status['status']}")
                    return {'success': False, 'reason': 'sell_order_rejected', 'buy_completed': True}
            
            # Verificar se a ordem foi concluída
            if sell_order_status['status'] != 'filled':
                # Tentar cancelar a ordem
                sell_exchange.cancel_order(sell_order_id)
                self.active_arbitrages[arb_id]['status'] = 'sell_timeout'
                trans_logger.error(f"❌ VENDA TIMEOUT: {max_wait}s se passaram sem execução")
                return {'success': False, 'reason': 'sell_order_timeout', 'buy_completed': True}
            
            # Calcular lucro
            sell_quantity = sell_order_status['filled']
            sell_price_actual = sell_order_status['price']
            
            buy_cost = actual_quantity * actual_price
            sell_revenue = sell_quantity * sell_price_actual
            
            profit = sell_revenue - buy_cost
            profit_pct = (profit / buy_cost) * 100 if buy_cost > 0 else 0
            
            trans_logger.info(f"✅ VENDA EXECUTADA: {sell_quantity} {pair.split('/')[0]} a {sell_price_actual}")
            
            # Atualizar estado da arbitragem
            self.active_arbitrages[arb_id]['sell_price'] = sell_price_actual
            self.active_arbitrages[arb_id]['profit'] = profit
            self.active_arbitrages[arb_id]['profit_pct'] = profit_pct
            self.active_arbitrages[arb_id]['status'] = 'completed'
            self.active_arbitrages[arb_id]['end_time'] = datetime.now().isoformat()
            
            # Atualizar métricas
            self.pair_metrics[pair]['total_trades'] += 1
            self.pair_metrics[pair]['total_profit'] += profit
            
            if profit > 0:
                self.pair_metrics[pair]['successful_trades'] += 1
            
            self.pair_metrics[pair]['success_rate'] = (
                self.pair_metrics[pair]['successful_trades'] / 
                self.pair_metrics[pair]['total_trades']
            ) if self.pair_metrics[pair]['total_trades'] > 0 else 0
            
            self.pair_metrics[pair]['avg_profit'] = (
                self.pair_metrics[pair]['total_profit'] / 
                self.pair_metrics[pair]['total_trades']
            ) if self.pair_metrics[pair]['total_trades'] > 0 else 0
            
            # Atualizar balanços
            self.exchange_balances[opportunity['buy_exchange']] -= buy_cost
            self.exchange_balances[opportunity['sell_exchange']] += sell_revenue
            
            # Calcular variação nos balanços
            new_buy_balance = self.exchange_balances.get(opportunity['buy_exchange'], 0)
            new_sell_balance = self.exchange_balances.get(opportunity['sell_exchange'], 0)
            buy_balance_change = new_buy_balance - old_buy_balance
            sell_balance_change = new_sell_balance - old_sell_balance
            
            # Informar resultado da operação e mudanças no balanço
            trans_logger.info(f"💹 ARBITRAGEM CONCLUÍDA: Lucro {profit:.6f} USDT ({profit_pct:.2f}%)")
            trans_logger.info(f"💰 BALANÇO APÓS OPERAÇÃO:")
            trans_logger.info(f"   * {opportunity['buy_exchange']}: {new_buy_balance:.4f} USDT ({buy_balance_change:+.4f})")
            trans_logger.info(f"   * {opportunity['sell_exchange']}: {new_sell_balance:.4f} USDT ({sell_balance_change:+.4f})")
            trans_logger.info(f"   * Total: {new_buy_balance + new_sell_balance:.4f} USDT (Variação: {buy_balance_change + sell_balance_change:+.4f})")
            trans_logger.info(f"📊 RESUMO DA OPERAÇÃO {arb_id}:")
            trans_logger.info(f"   * Par: {pair}")
            trans_logger.info(f"   * Compra: {actual_quantity} {pair.split('/')[0]} a {actual_price} em {opportunity['buy_exchange']}")
            trans_logger.info(f"   * Venda: {sell_quantity} {pair.split('/')[0]} a {sell_price_actual} em {opportunity['sell_exchange']}")
            trans_logger.info(f"   * Resultado: {profit:.6f} USDT ({profit_pct:.2f}%)")
            
            # Registrar sucesso
            logger.info(f"Arbitragem concluída: {pair} lucro={profit:.6f} USDT ({profit_pct:.2f}%)")
            
            return {
                'success': True,
                'profit': profit,
                'profit_pct': profit_pct,
                'buy_exchange': opportunity['buy_exchange'],
                'sell_exchange': opportunity['sell_exchange'],
                'quantity': actual_quantity,
                'buy_price': actual_price,
                'sell_price': sell_price_actual
            }
            
        except Exception as e:
            logger.error(f"Erro na execução da arbitragem: {e}")
            self.active_arbitrages[arb_id]['status'] = 'error'
            self.active_arbitrages[arb_id]['error'] = str(e)
            trans_logger.error(f"❌ ERRO NA ARBITRAGEM: {str(e)}")
            return {'success': False, 'reason': 'execution_error', 'error': str(e)}
    
    def _update_metrics(
        self,
        pair: str,
        buy_exchange: str,
        sell_exchange: str,
        profit: float,
        success: bool,
        buy_time: float,
        sell_time: float,
        buy_slippage: float,
        sell_slippage: float
    ) -> None:
        """Atualiza métricas de performance"""
        # Métricas do par
        self.pair_metrics[pair]['total_trades'] += 1
        
        if success:
            self.pair_metrics[pair]['successful_trades'] += 1
            self.pair_metrics[pair]['total_profit'] += profit
            
            if profit > self.pair_metrics[pair]['best_trade']:
                self.pair_metrics[pair]['best_trade'] = profit
                
            if profit < self.pair_metrics[pair]['worst_trade']:
                self.pair_metrics[pair]['worst_trade'] = profit
        else:
            self.pair_metrics[pair]['failed_trades'] += 1
        
        # Atualizar taxa de sucesso
        if self.pair_metrics[pair]['total_trades'] > 0:
            self.pair_metrics[pair]['success_rate'] = (
                self.pair_metrics[pair]['successful_trades'] / 
                self.pair_metrics[pair]['total_trades']
            )
        
        # Atualizar lucro médio
        if self.pair_metrics[pair]['successful_trades'] > 0:
            self.pair_metrics[pair]['avg_profit'] = (
                self.pair_metrics[pair]['total_profit'] / 
                self.pair_metrics[pair]['successful_trades']
            )
        
        # Métricas por exchange
        for ex_id in [buy_exchange, sell_exchange]:
            if ex_id not in self.pair_metrics[pair]['exchange_stats']:
                self.pair_metrics[pair]['exchange_stats'][ex_id] = {
                    'trades': 0,
                    'profit': 0.0,
                    'success_rate': 0.0,
                    'avg_execution_time': 0.0,
                    'slippage': 0.0,
                    'profit_ratio': 1.0
                }
            
            stats = self.pair_metrics[pair]['exchange_stats'][ex_id]
            stats['trades'] += 1
            
            if success:
                stats['profit'] += profit / 2  # Dividir lucro entre as exchanges
            
            # Atualizar tempo médio de execução
            execution_time = buy_time if ex_id == buy_exchange else sell_time
            stats['avg_execution_time'] = (
                (stats['avg_execution_time'] * (stats['trades'] - 1) + execution_time) / 
                stats['trades']
            )
            
            # Atualizar slippage médio
            slippage = buy_slippage if ex_id == buy_exchange else sell_slippage
            stats['slippage'] = (
                (stats['slippage'] * (stats['trades'] - 1) + slippage) / 
                stats['trades']
            )
            
            # Calcular profit ratio para alocação
            if stats['trades'] > 0:
                success_factor = (stats['success_rate'] if 'success_rate' in stats else 0.5)
                slippage_factor = 1.0 - stats['slippage']
                execution_factor = 1.0 / (1.0 + stats['avg_execution_time'] / 10.0)  # Normalizar para 0-1
                
                stats['profit_ratio'] = (
                    0.4 * success_factor + 
                    0.4 * slippage_factor + 
                    0.2 * execution_factor
                )
    
    async def rebalance_exchanges(self, target_allocation: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """Rebalanceia fundos entre exchanges"""
        if target_allocation is None:
            target_allocation = self.calculate_allocation()
        
        # Calcular transferências necessárias
        transfers = []
        min_transfer = self.config["min_transfer"]
        
        # Calcular desvios
        deviations = {}
        for ex_id, balance in self.exchange_balances.items():
            target = target_allocation.get(ex_id, 0)
            deviations[ex_id] = target - balance
        
        # Identificar exchanges com excesso e déficit
        surplus = {k: -v for k, v in deviations.items() if v < 0}
        deficit = {k: v for k, v in deviations.items() if v > 0}
        
        # Ordenar para otimizar transferências
        surplus_exchanges = sorted(surplus.keys(), key=lambda x: surplus[x], reverse=True)
        deficit_exchanges = sorted(deficit.keys(), key=lambda x: deficit[x], reverse=True)
        
        # Calcular transferências ótimas
        for from_ex in surplus_exchanges:
            available = surplus[from_ex]
            
            if available < min_transfer:
                continue
                
            for to_ex in deficit_exchanges:
                needed = deficit[to_ex]
                
                if needed < min_transfer:
                    continue
                    
                transfer_amount = min(available, needed)
                
                if transfer_amount >= min_transfer:
                    transfers.append({
                        'from_exchange': from_ex,
                        'to_exchange': to_ex,
                        'amount': transfer_amount
                    })
                    
                    # Atualizar valores
                    available -= transfer_amount
                    deficit[to_ex] -= transfer_amount
                    
                    if available < min_transfer:
                        break
        
        # Executar transferências
        executed_transfers = []
        
        for transfer in transfers:
            try:
                from_ex_id = transfer['from_exchange']
                to_ex_id = transfer['to_exchange']
                amount = transfer['amount']
                
                # Obter objetos de exchange
                from_exchange = next(ex for ex in self.exchanges if ex.exchange_id == from_ex_id)
                to_exchange = next(ex for ex in self.exchanges if ex.exchange_id == to_ex_id)
                
                # Executar transferência
                # Nota: Esta é uma implementação simplificada
                # Na prática, seria necessário implementar a lógica específica de transferência
                
                logger.info(f"Transferindo {amount} USDT de {from_ex_id} para {to_ex_id}")
                
                # Simular transferência bem-sucedida
                executed_transfers.append({
                    'from_exchange': from_ex_id,
                    'to_exchange': to_ex_id,
                    'amount': amount,
                    'status': 'success',
                    'timestamp': datetime.now().isoformat()
                })
                
                # Atualizar balanços
                self.exchange_balances[from_ex_id] -= amount
                self.exchange_balances[to_ex_id] += amount
                
            except Exception as e:
                logger.error(f"Erro na transferência: {e}")
                executed_transfers.append({
                    'from_exchange': transfer['from_exchange'],
                    'to_exchange': transfer['to_exchange'],
                    'amount': transfer['amount'],
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Atualizar estado
        self.state["last_rebalance"] = datetime.now().isoformat()
        
        return executed_transfers
    
    async def run_strategy_cycle(self) -> Dict[str, Any]:
        """Executa um ciclo completo da estratégia"""
        results = {
            'opportunities_detected': 0,
            'arbitrages_executed': 0,
            'successful_arbitrages': 0,
            'total_profit': 0.0,
            'rebalance_executed': False,
            'rebalance_transfers': 0,
            'errors': []
        }
        
        try:
            # Atualizar balanços
            await self.update_balances()
            
            # Detectar oportunidades
            opportunities = await self._detect_opportunities()
            results['opportunities_detected'] = len(opportunities)
            
            # Executar arbitragens
            for opportunity in opportunities:
                if opportunity['expected_profit'] <= 0:
                    continue
                    
                result = await self.execute_arbitrage(opportunity)
                results['arbitrages_executed'] += 1
                
                if result['success']:
                    results['successful_arbitrages'] += 1
                    results['total_profit'] += result.get('profit', 0)
                    
                    # Atualizar histórico retrocausal
                    self.retrocausal_optimizer.trade_history.append({
                        "state": {
                            "spread": opportunity["spread"],
                            "threshold": opportunity["threshold"],
                            "entropy": opportunity["entropy"],
                            "fractal_dimension": opportunity["fractal_dimension"]
                        },
                        "success": True,
                        "profit": result.get('profit', 0)
                    })
                else:
                    results['errors'].append({
                        'type': 'arbitrage_execution',
                        'reason': result.get('reason', 'unknown'),
                        'details': result
                    })
            
            # Adaptar parâmetros do sistema
            performance_metrics = {
                "success_rate": results['successful_arbitrages'] / results['arbitrages_executed'] if results['arbitrages_executed'] > 0 else 0.5,
                "avg_profit": results['total_profit'] / results['successful_arbitrages'] if results['successful_arbitrages'] > 0 else 0.0,
                "volatility": float(np.std([float(o["spread"]) for o in opportunities])) if opportunities else 0.5
            }
            
            adapted_config = self.adaptive_system.adapt_parameters(self.config, performance_metrics)
            self.config.update(adapted_config)
            
            # Verificar necessidade de rebalanceamento
            current_time = datetime.now()
            last_rebalance = datetime.fromisoformat(self.state["last_rebalance"]) if self.state["last_rebalance"] else None
            
            if last_rebalance is None or (current_time - last_rebalance) > timedelta(hours=1):
                target_allocation = self.calculate_allocation()
                
                needs_rebalance = False
                for ex_id, balance in self.exchange_balances.items():
                    target = target_allocation.get(ex_id, 0)
                    if balance > 0 and abs(target - balance) / balance > self.config["rebalance_threshold"]:
                        needs_rebalance = True
                        break
                
                if needs_rebalance:
                    transfers = await self.rebalance_exchanges(target_allocation)
                    results['rebalance_executed'] = True
                    results['rebalance_transfers'] = len(transfers)
            
            # Atualizar estado
            self.state["last_update"] = datetime.now().isoformat()
            self.state["performance"] = {
                'total_trades': sum(m['total_trades'] for m in self.pair_metrics.values()),
                'successful_trades': sum(m['successful_trades'] for m in self.pair_metrics.values()),
                'total_profit': sum(m['total_profit'] for m in self.pair_metrics.values()),
                'pairs': {pair: {
                    'success_rate': metrics['success_rate'],
                    'avg_profit': metrics['avg_profit'],
                    'total_profit': metrics['total_profit']
                } for pair, metrics in self.pair_metrics.items()}
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Erro no ciclo da estratégia: {e}")
            results['errors'].append({
                'type': 'strategy_cycle',
                'reason': 'execution_error',
                'details': str(e)
            })
            return results
    
    def save_state(self, filepath: str) -> None:
        """Salva o estado da estratégia em arquivo"""
        state_data = {
            'config': self.config,
            'volatility_matrix': self.volatility_matrix,
            'pair_metrics': self.pair_metrics,
            'exchange_balances': self.exchange_balances,
            'active_arbitrages': self.active_arbitrages,
            'state': self.state
        }
        
        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2)
        
        logger.info(f"Estado da estratégia salvo em {filepath}")
    
    def load_state(self, filepath: str) -> None:
        """Carrega o estado da estratégia de arquivo"""
        with open(filepath, 'r') as f:
            state_data = json.load(f)
        
        self.config = state_data.get('config', self.config)
        self.volatility_matrix = state_data.get('volatility_matrix', {})
        self.pair_metrics = state_data.get('pair_metrics', self._initialize_metrics())
        self.exchange_balances = state_data.get('exchange_balances', {})
        self.active_arbitrages = state_data.get('active_arbitrages', {})
        self.state = state_data.get('state', {
            "last_rebalance": None,
            "performance": {},
            "last_update": datetime.now().isoformat()
        })
        
        logger.info(f"Estado da estratégia carregado de {filepath}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Gera um resumo da performance da estratégia para exibir na interface
        
        Returns:
            Dict contendo métricas de performance
        """
        # Obter valores atuais
        total_balance = sum(self.exchange_balances.values())
        initial_capital = sum(self.config.get("capital_per_exchange", {}).values())
        total_profit = sum(m['total_profit'] for m in self.pair_metrics.values())
        
        # Calcular ganho/perda total
        total_gain_pct = 0
        if initial_capital > 0:
            total_gain_pct = ((total_balance - initial_capital) / initial_capital) * 100
            
        # Resumo por exchange
        exchange_summary = {}
        for ex in self.exchanges:
            ex_id = ex.exchange_id
            initial_ex_capital = self.config.get("capital_per_exchange", {}).get(ex_id, 0)
            current_ex_balance = self.exchange_balances.get(ex_id, 0)
            
            ex_gain_loss = current_ex_balance - initial_ex_capital
            ex_gain_pct = 0
            if initial_ex_capital > 0:
                ex_gain_pct = (ex_gain_loss / initial_ex_capital) * 100
                
            exchange_summary[ex_id] = {
                "initial_capital": initial_ex_capital,
                "current_balance": current_ex_balance,
                "gain_loss": ex_gain_loss,
                "gain_loss_pct": ex_gain_pct
            }
            
        # Resumo por par
        pair_summary = {}
        for pair, metrics in self.pair_metrics.items():
            pair_summary[pair] = {
                "total_trades": metrics['total_trades'],
                "successful_trades": metrics['successful_trades'],
                "success_rate": metrics['success_rate'] * 100 if 'success_rate' in metrics else 0,
                "total_profit": metrics['total_profit'],
                "avg_profit": metrics['avg_profit'] if 'avg_profit' in metrics else 0
            }
            
        # Resumo geral de arbitragens
        total_trades = sum(m['total_trades'] for m in self.pair_metrics.values())
        successful_trades = sum(m['successful_trades'] for m in self.pair_metrics.values())
        success_rate = (successful_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Histórico de operações recentes (últimas 10)
        recent_arbitrages = []
        sorted_arbitrages = sorted(
            self.active_arbitrages.items(), 
            key=lambda x: x[1].get('start_time', ''), 
            reverse=True
        )
        
        for arb_id, arb_data in sorted_arbitrages[:10]:
            # Verificar se tem os dados necessários
            if all(k in arb_data for k in ['pair', 'status', 'buy_exchange', 'sell_exchange']):
                # Calcular profit, se disponível
                profit = arb_data.get('profit', 0)
                profit_pct = arb_data.get('profit_pct', 0)
                
                recent_arbitrages.append({
                    "id": arb_id,
                    "pair": arb_data['pair'],
                    "status": arb_data['status'],
                    "buy_exchange": arb_data['buy_exchange'],
                    "sell_exchange": arb_data['sell_exchange'],
                    "profit": profit,
                    "profit_pct": profit_pct,
                    "timestamp": arb_data.get('start_time', '')
                })
        
        return {
            "summary": {
                "initial_capital": initial_capital,
                "current_balance": total_balance,
                "total_profit": total_profit,
                "gain_loss_pct": total_gain_pct,
                "total_trades": total_trades,
                "successful_trades": successful_trades,
                "success_rate": success_rate
            },
            "exchanges": exchange_summary,
            "pairs": pair_summary,
            "recent_arbitrages": recent_arbitrages,
            "updated_at": datetime.now().isoformat()
        } 