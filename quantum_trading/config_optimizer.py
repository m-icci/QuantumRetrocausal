"""
Otimizador de Configuração baseado em Dados Históricos
Ajusta parâmetros de configuração da estratégia WAVE baseado em dados de mercado
"""

import numpy as np
import pandas as pd
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from .market_api import MarketAPI

logger = logging.getLogger("config_optimizer")

class HistoricalDataAnalyzer:
    """Analisa dados históricos para determinar parâmetros otimizados"""
    
    def __init__(self, exchanges: List[MarketAPI], pairs: List[str], window_days: int = 30, capital_per_exchange: Dict[str, float] = None):
        """
        Inicializa o analisador
        
        Args:
            exchanges: Lista de APIs de exchanges
            pairs: Lista de pares de trading
            window_days: Janela de dados históricos (em dias)
            capital_per_exchange: Dicionário com capital disponível por exchange
        """
        self.exchanges = exchanges
        self.pairs = pairs
        self.window_days = window_days
        self.capital_per_exchange = capital_per_exchange or {ex.exchange_id: 1000.0 for ex in exchanges}
        
        # Armazenamento de dados
        self.historical_data = {}
        self.spread_data = {}
        self.volatility_data = {}
        self.volume_data = {}
        self.execution_data = {}
        self.fee_data = {}
        
        self.logger = logging.getLogger("historical_analyzer")
        self.logger.info(f"Analisador histórico inicializado: {len(exchanges)} exchanges, {len(pairs)} pares, {window_days} dias")
        self.logger.info(f"Capital disponível: {self.capital_per_exchange}")
    
    async def collect_historical_data(self) -> Dict[str, Any]:
        """Coleta dados históricos de todas as exchanges e pares"""
        self.logger.info("Iniciando coleta de dados históricos...")
        
        for pair in self.pairs:
            self.historical_data[pair] = {}
            
            for exchange in self.exchanges:
                ex_id = exchange.exchange_id
                self.logger.info(f"Coletando dados históricos para {pair} em {ex_id}...")
                
                # Coletar dados diários
                try:
                    # Pegar dados com granularidade de 1 hora
                    ohlcv_data = await exchange.get_ohlcv(
                        pair, 
                        timeframe='1h', 
                        limit=self.window_days * 24  # horas totais
                    )
                    
                    if ohlcv_data:
                        df = pd.DataFrame(
                            ohlcv_data, 
                            columns=["timestamp", "open", "high", "low", "close", "volume"]
                        )
                        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                        
                        self.historical_data[pair][ex_id] = df
                        self.logger.info(f"Coletados {len(df)} registros para {pair} em {ex_id}")
                    else:
                        self.logger.warning(f"Sem dados históricos para {pair} em {ex_id}")
                        
                except Exception as e:
                    self.logger.error(f"Erro ao coletar dados históricos para {pair} em {ex_id}: {e}")
        
        return self.historical_data
    
    def analyze_spreads(self) -> Dict[str, Dict[str, float]]:
        """Analisa spreads históricos entre exchanges"""
        self.logger.info("Analisando spreads históricos...")
        
        spread_stats = {}
        
        for pair in self.pairs:
            spread_stats[pair] = {}
            spread_series = []
            
            # Obter todas as combinações de exchanges
            exchange_ids = [ex.exchange_id for ex in self.exchanges]
            
            for i, ex1 in enumerate(exchange_ids):
                for j, ex2 in enumerate(exchange_ids[i+1:], i+1):
                    key = f"{ex1}_vs_{ex2}"
                    
                    # Verificar se temos dados para ambas exchanges
                    if ex1 in self.historical_data[pair] and ex2 in self.historical_data[pair]:
                        df1 = self.historical_data[pair][ex1]
                        df2 = self.historical_data[pair][ex2]
                        
                        # Alinhar dados pelo timestamp
                        df = pd.merge(
                            df1[['timestamp', 'close']].rename(columns={'close': f'close_{ex1}'}),
                            df2[['timestamp', 'close']].rename(columns={'close': f'close_{ex2}'}),
                            on='timestamp', how='inner'
                        )
                        
                        if not df.empty:
                            # Calcular spread como diferença percentual absoluta
                            df['spread'] = abs(df[f'close_{ex1}'] - df[f'close_{ex2}']) / df[[f'close_{ex1}', f'close_{ex2}']].min(axis=1)
                            
                            # Calcular estatísticas
                            spread_mean = df['spread'].mean()
                            spread_median = df['spread'].median()
                            spread_p75 = df['spread'].quantile(0.75)  # 75º percentil
                            spread_p90 = df['spread'].quantile(0.90)  # 90º percentil
                            spread_p95 = df['spread'].quantile(0.95)  # 95º percentil
                            
                            spread_stats[pair][key] = {
                                'mean': float(spread_mean),
                                'median': float(spread_median),
                                'p75': float(spread_p75),
                                'p90': float(spread_p90),
                                'p95': float(spread_p95),
                                'count': len(df)
                            }
                            
                            # Armazenar série de spreads
                            df['exchange_pair'] = key
                            spread_series.append(df[['timestamp', 'exchange_pair', 'spread']])
                            
                            self.logger.info(f"Spread para {pair} ({key}): média={spread_mean:.6f}, mediana={spread_median:.6f}, p90={spread_p90:.6f}")
                        else:
                            self.logger.warning(f"Sem dados alinhados para {pair} ({key})")
            
            # Combinar todas as séries de spread
            if spread_series:
                self.spread_data[pair] = pd.concat(spread_series)
            
        return spread_stats
    
    def analyze_volatility(self) -> Dict[str, Dict[str, float]]:
        """Analisa volatilidade dos pares por hora do dia"""
        self.logger.info("Analisando volatilidade por hora do dia...")
        
        volatility_stats = {}
        
        for pair in self.pairs:
            volatility_stats[pair] = {'hourly': {}, 'daily': {}}
            hourly_vol = []
            
            for ex_id, df in self.historical_data[pair].items():
                if not df.empty:
                    # Adicionar retornos percentuais
                    df['returns'] = df['close'].pct_change()
                    
                    # Agrupar por hora do dia
                    df['hour'] = df['datetime'].dt.hour
                    hourly_returns = df.groupby('hour')['returns'].agg(['std']).reset_index()
                    
                    for _, row in hourly_returns.iterrows():
                        hour = int(row['hour'])
                        std = float(row['std'])
                        
                        if hour not in volatility_stats[pair]['hourly']:
                            volatility_stats[pair]['hourly'][hour] = []
                            
                        if not np.isnan(std):
                            volatility_stats[pair]['hourly'][hour].append(std)
                    
                    # Calcular volatilidade diária
                    daily_std = float(df['returns'].std())
                    if not np.isnan(daily_std):
                        volatility_stats[pair]['daily'][ex_id] = daily_std
                        self.logger.info(f"Volatilidade diária para {pair} em {ex_id}: {daily_std:.6f}")
            
            # Calcular médias por hora
            for hour, values in volatility_stats[pair]['hourly'].items():
                if values:
                    volatility_stats[pair]['hourly'][hour] = float(np.mean(values))
                    
        return volatility_stats
    
    def analyze_volume(self) -> Dict[str, Dict[str, float]]:
        """Analisa volumes de trading diários"""
        self.logger.info("Analisando volumes de trading...")
        
        volume_stats = {}
        
        for pair in self.pairs:
            volume_stats[pair] = {}
            
            for ex_id, df in self.historical_data[pair].items():
                if not df.empty:
                    # Calcular estatísticas de volume
                    volume_mean = float(df['volume'].mean())
                    volume_median = float(df['volume'].median())
                    volume_min = float(df['volume'].min())
                    volume_max = float(df['volume'].max())
                    
                    volume_stats[pair][ex_id] = {
                        'mean': volume_mean,
                        'median': volume_median,
                        'min': volume_min,
                        'max': volume_max
                    }
                    
                    self.logger.info(f"Volume para {pair} em {ex_id}: média={volume_mean:.2f}, mediana={volume_median:.2f}")
        
        return volume_stats
    
    def analyze_fees(self) -> Dict[str, Dict[str, float]]:
        """Analisa taxas por exchange e par"""
        self.logger.info("Analisando taxas de trading...")
        
        fee_stats = {}
        
        for exchange in self.exchanges:
            ex_id = exchange.exchange_id
            fee_stats[ex_id] = {}
            
            try:
                # Obter estrutura de taxas da exchange
                exchange_fees = exchange.get_fees() if hasattr(exchange, 'get_fees') else {'taker': 0.001, 'maker': 0.001}
                
                fee_stats[ex_id]['taker'] = exchange_fees.get('taker', 0.001)
                fee_stats[ex_id]['maker'] = exchange_fees.get('maker', 0.001)
                
                # Calcular taxa média
                fee_stats[ex_id]['average'] = (fee_stats[ex_id]['taker'] + fee_stats[ex_id]['maker']) / 2
                
                self.logger.info(f"Taxas para {ex_id}: maker={fee_stats[ex_id]['maker']:.6f}, taker={fee_stats[ex_id]['taker']:.6f}")
                
            except Exception as e:
                self.logger.error(f"Erro ao obter taxas para {ex_id}: {e}")
                fee_stats[ex_id] = {'taker': 0.001, 'maker': 0.001, 'average': 0.001}
        
        return fee_stats

    def analyze_execution_times(self) -> Dict[str, Dict[str, float]]:
        """Analisa tempos de execução por exchange"""
        self.logger.info("Analisando tempos de execução...")
        
        # Valores de fallback se não houver dados reais
        execution_stats = {
            'kucoin': {'avg_time': 1.2, 'p90_time': 2.5, 'p95_time': 3.0},
            'kraken': {'avg_time': 0.8, 'p90_time': 1.5, 'p95_time': 2.0}
        }
        
        # Implementar medição de tempos reais de resposta da API aqui
        # Por enquanto, usamos valores baseados em análises anteriores
        
        for exchange in self.exchanges:
            ex_id = exchange.exchange_id
            if ex_id not in execution_stats:
                execution_stats[ex_id] = {'avg_time': 1.0, 'p90_time': 2.0, 'p95_time': 2.5}
            
            self.logger.info(f"Tempos de execução para {ex_id}: média={execution_stats[ex_id]['avg_time']:.2f}s, P90={execution_stats[ex_id]['p90_time']:.2f}s")
        
        return execution_stats
        
    def analyze_pair_performance(self) -> Dict[str, Dict[str, float]]:
        """Analisa desempenho histórico dos pares para priorização"""
        self.logger.info("Analisando desempenho histórico dos pares...")
        
        pair_stats = {}
        
        for pair in self.pairs:
            pair_stats[pair] = {
                'spread_frequency': 0,  # Frequência de spreads acima do threshold
                'avg_spread': 0,        # Spread médio quando acima do threshold
                'avg_opportunity': 0,   # Oportunidade média em USD
                'volume_rank': 0,       # Ranking de volume
                'volatility_rank': 0,   # Ranking de volatilidade
                'combined_score': 0     # Pontuação combinada
            }
            
            # Calcular frequência de spreads significativos
            if pair in self.spread_data:
                df = self.spread_data[pair]
                threshold = 0.0005  # Threshold mínimo para considerar
                
                spreads_above = df[df['spread'] > threshold]
                if not df.empty:
                    pair_stats[pair]['spread_frequency'] = len(spreads_above) / len(df)
                    
                    if not spreads_above.empty:
                        pair_stats[pair]['avg_spread'] = float(spreads_above['spread'].mean())
            
            # Calcular volatilidade média
            volatility_sum = 0
            volatility_count = 0
            
            if pair in self.volatility_data:
                for hour, vol in self.volatility_data[pair]['hourly'].items():
                    volatility_sum += vol
                    volatility_count += 1
                
                if volatility_count > 0:
                    pair_stats[pair]['avg_volatility'] = volatility_sum / volatility_count
        
        # Calcular rankings
        if self.pairs:
            # Ranking por frequência de spread
            sorted_by_freq = sorted(self.pairs, key=lambda p: pair_stats[p]['spread_frequency'], reverse=True)
            for i, pair in enumerate(sorted_by_freq):
                pair_stats[pair]['spread_freq_rank'] = i + 1
            
            # Ranking por spread médio
            sorted_by_spread = sorted(self.pairs, key=lambda p: pair_stats[p]['avg_spread'], reverse=True)
            for i, pair in enumerate(sorted_by_spread):
                pair_stats[pair]['spread_size_rank'] = i + 1
            
            # Ranking por volatilidade
            pairs_with_vol = [p for p in self.pairs if 'avg_volatility' in pair_stats[p]]
            if pairs_with_vol:
                sorted_by_vol = sorted(pairs_with_vol, key=lambda p: pair_stats[p]['avg_volatility'], reverse=True)
                for i, pair in enumerate(sorted_by_vol):
                    pair_stats[pair]['volatility_rank'] = i + 1
            
            # Calcular pontuação combinada (menor é melhor)
            for pair in self.pairs:
                freq_rank = pair_stats[pair].get('spread_freq_rank', len(self.pairs))
                size_rank = pair_stats[pair].get('spread_size_rank', len(self.pairs))
                vol_rank = pair_stats[pair].get('volatility_rank', len(self.pairs))
                
                # Pesos: frequência (50%), tamanho (30%), volatilidade (20%)
                weighted_rank = 0.5 * freq_rank + 0.3 * size_rank + 0.2 * vol_rank
                pair_stats[pair]['combined_score'] = weighted_rank
        
        # Ordenar pares por pontuação combinada (menor é melhor)
        ranked_pairs = sorted(self.pairs, key=lambda p: pair_stats[p]['combined_score'])
        
        self.logger.info("Ranking de pares por potencial de arbitragem:")
        for i, pair in enumerate(ranked_pairs):
            self.logger.info(f"{i+1}. {pair} - Score: {pair_stats[pair]['combined_score']:.2f}")
        
        return pair_stats

    def calculate_optimal_min_transfer(self, fee_stats: Dict[str, Dict[str, float]]) -> float:
        """Calcula valor mínimo de transferência que compensa as taxas"""
        self.logger.info("Calculando valor mínimo de transferência ótimo...")
        
        # Obter taxa média entre todas as exchanges
        avg_fee = sum(ex_data['average'] for ex_id, ex_data in fee_stats.items()) / len(fee_stats)
        
        # Calcular valor mínimo que compensa (considerando duas operações: compra e venda)
        # Assumimos que queremos pelo menos 3x o valor das taxas como lucro mínimo
        min_profit_factor = 3.0
        
        # Calculamos: valor = (2 * taxa média * min_profit_factor) / (spread médio esperado)
        # Adicionamos um buffer de segurança
        expected_spread = 0.001  # Spread médio esperado (0.1%)
        safety_factor = 1.5
        
        min_transfer = (2 * avg_fee * min_profit_factor) / expected_spread * safety_factor
        
        # Limitar com base no capital disponível
        min_capital = min(self.capital_per_exchange.values())
        max_percentage = 0.1  # Não usar mais de 10% do capital mínimo
        capital_limit = min_capital * max_percentage
        
        # Escolher o menor valor entre o calculado e o limite baseado no capital
        optimal_min_transfer = min(min_transfer, capital_limit)
        
        self.logger.info(f"Valor mínimo de transferência calculado: {min_transfer:.2f} USDT")
        self.logger.info(f"Limite baseado no capital ({max_percentage*100:.0f}% de {min_capital:.2f}): {capital_limit:.2f} USDT")
        self.logger.info(f"Valor mínimo de transferência ótimo: {optimal_min_transfer:.2f} USDT")
        
        return optimal_min_transfer

    def calculate_adaptive_position_sizing(self) -> Dict[str, float]:
        """Calcula posicionamento adaptativo baseado no capital e volatilidade"""
        self.logger.info("Calculando tamanhos de posição adaptativos...")
        
        position_sizing = {}
        
        # Calcular volatilidade média por par
        pair_volatility = {}
        for pair in self.pairs:
            if pair in self.volatility_data and 'daily' in self.volatility_data[pair]:
                vol_values = list(self.volatility_data[pair]['daily'].values())
                if vol_values:
                    pair_volatility[pair] = np.mean(vol_values)
                else:
                    pair_volatility[pair] = 0.02  # Valor padrão se não houver dados
            else:
                pair_volatility[pair] = 0.02  # Valor padrão
        
        # Normalizar volatilidades
        max_vol = max(pair_volatility.values()) if pair_volatility else 0.02
        if max_vol > 0:
            normalized_vol = {p: v/max_vol for p, v in pair_volatility.items()}
        else:
            normalized_vol = {p: 1.0 for p in pair_volatility}
        
        # Calcular posições inversamente proporcionais à volatilidade
        # (menor volatilidade = maior posição)
        for pair in self.pairs:
            vol = normalized_vol.get(pair, 1.0)
            
            # Invertemos a relação: 1.0 / vol, normalizado para somar 1.0
            position_sizing[pair] = 1.0 / vol if vol > 0 else 1.0
        
        # Normalizar para que a soma seja 1.0
        total = sum(position_sizing.values())
        if total > 0:
            position_sizing = {p: v/total for p, v in position_sizing.items()}
        
        # Ajustar com base no capital disponível
        min_capital = min(self.capital_per_exchange.values())
        for pair in position_sizing:
            # Garantir que nenhuma posição seja maior que 25% do capital por exchange
            position_sizing[pair] = min(position_sizing[pair], 0.25)
            
            # Calcular valor em USD
            position_sizing[f"{pair}_usd"] = position_sizing[pair] * min_capital
            
            self.logger.info(f"Posição para {pair}: {position_sizing[pair]*100:.1f}% ({position_sizing[f'{pair}_usd']:.2f} USDT)")
        
        return position_sizing
    
    def calculate_optimal_timeout(self, execution_stats: Dict[str, Dict[str, float]]) -> int:
        """Calcula timeout ótimo baseado nos tempos de execução"""
        # Usar P95 do tempo de execução mais lento como base
        max_p95 = max(ex_data['p95_time'] for ex_id, ex_data in execution_stats.items())
        
        # Adicionar margem de segurança
        safety_factor = 3.0
        optimal_timeout = int(max_p95 * safety_factor)
        
        # Garantir valor mínimo e máximo razoáveis
        optimal_timeout = max(5, min(optimal_timeout, 60))
        
        self.logger.info(f"Timeout ótimo calculado: {optimal_timeout} segundos")
        return optimal_timeout
    
    def calculate_optimal_retry_params(self, execution_stats: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calcula parâmetros ótimos para tentativas de retry"""
        # Usar tempo médio de execução para o retry delay
        avg_times = [ex_data['avg_time'] for ex_id, ex_data in execution_stats.items()]
        avg_execution_time = np.mean(avg_times) if avg_times else 1.0
        
        # Retry delay baseado no tempo médio de execução
        optimal_retry_delay = max(1.0, avg_execution_time * 0.5)
        
        # Backoff entre 1.5 e 3.0 dependendo da variabilidade
        p90_times = [ex_data['p90_time'] for ex_id, ex_data in execution_stats.items()]
        avg_p90 = np.mean(p90_times) if p90_times else 2.0
        
        variability = avg_p90 / avg_execution_time if avg_execution_time > 0 else 2.0
        optimal_backoff = 1.5 + (variability - 1.0)
        optimal_backoff = max(1.5, min(optimal_backoff, 3.0))
        
        # Número de tentativas
        optimal_max_retries = 3
        
        retry_params = {
            'retry_delay': round(optimal_retry_delay, 1),
            'retry_backoff': round(optimal_backoff, 1),
            'max_retries': optimal_max_retries
        }
        
        self.logger.info(f"Parâmetros de retry ótimos: {retry_params}")
        return retry_params

    def get_optimal_parameters(self) -> Dict[str, Any]:
        """Determina parâmetros ótimos baseados na análise histórica"""
        self.logger.info("Calculando parâmetros ótimos...")
        
        # Analisar spreads
        spread_stats = self.analyze_spreads()
        
        # Analisar volatilidade
        volatility_stats = self.analyze_volatility()
        
        # Analisar volumes
        volume_stats = self.analyze_volume()
        
        # Analisar taxas
        fee_stats = self.analyze_fees()
        
        # Analisar tempos de execução
        execution_stats = self.analyze_execution_times()
        
        # Analisar desempenho dos pares
        pair_performance = self.analyze_pair_performance()
        
        # Calcular posicionamento adaptativo
        position_sizing = self.calculate_adaptive_position_sizing()
        
        # Calcular parâmetros ótimos
        # 1. min_spread - usamos como base o percentil 75 dos spreads observados
        all_spreads_p75 = []
        for pair in spread_stats:
            for key in spread_stats[pair]:
                all_spreads_p75.append(spread_stats[pair][key]['p75'])
        
        optimal_min_spread = np.median(all_spreads_p75) * 0.8 if all_spreads_p75 else 0.001
        
        # 2. base_threshold - similar ao min_spread mas um pouco mais conservador
        optimal_base_threshold = optimal_min_spread * 0.9
        
        # 3. min_volume_24h - usamos 20% da mediana dos volumes diários
        all_volumes_median = []
        for pair in volume_stats:
            for ex_id in volume_stats[pair]:
                all_volumes_median.append(volume_stats[pair][ex_id]['median'])
        
        optimal_min_volume = np.median(all_volumes_median) * 0.2 if all_volumes_median else 5000
        
        # 4. Matriz de volatilidade por hora
        volatility_matrix = {}
        for pair in volatility_stats:
            volatility_matrix[pair] = []
            for hour in range(24):
                if hour in volatility_stats[pair]['hourly']:
                    volatility_matrix[pair].append(volatility_stats[pair]['hourly'][hour])
                else:
                    # Valor padrão se não houver dados
                    volatility_matrix[pair].append(0.02)
        
        # 5. Valor mínimo de transferência
        optimal_min_transfer = self.calculate_optimal_min_transfer(fee_stats)
        
        # 6. Timeout ótimo
        optimal_timeout = self.calculate_optimal_timeout(execution_stats)
        
        # 7. Parâmetros de retry
        retry_params = self.calculate_optimal_retry_params(execution_stats)
        
        # 8. Rebalance threshold baseado no capital disponível
        min_capital = min(self.capital_per_exchange.values())
        if min_capital < 100:
            # Capital pequeno requer rebalanceamento mais frequente
            optimal_rebalance_threshold = 0.05  # 5%
        elif min_capital < 1000:
            optimal_rebalance_threshold = 0.1   # 10%
        else:
            optimal_rebalance_threshold = 0.15  # 15%
        
        # 9. Calcular pesos ótimos para métricas
        # Com capital baixo, priorizar slippage e volatilidade
        if min_capital < 100:
            optimal_weights = {
                'volatility_weight': 0.3,
                'slippage_weight': 0.5,
                'execution_time_weight': 0.2
            }
        else:
            optimal_weights = {
                'volatility_weight': 0.4,
                'slippage_weight': 0.4,
                'execution_time_weight': 0.2
            }
        
        # 10. Calcular prioridade dos pares
        ranked_pairs = sorted(self.pairs, key=lambda p: pair_performance[p]['combined_score'])
        
        # Parâmetros otimizados completos
        optimal_params = {
            # Parâmetros principais
            "min_spread": round(float(optimal_min_spread), 6),
            "base_threshold": round(float(optimal_base_threshold), 6),
            "min_volume_24h": round(float(optimal_min_volume), 2),
            "min_transfer": round(float(optimal_min_transfer), 2),
            "rebalance_threshold": optimal_rebalance_threshold,
            "max_position_pct": 0.25 if min_capital > 200 else 0.5,  # Maior % para capital pequeno
            "history_window": 24,
            "time_segments": 24,
            "execution_timeout": optimal_timeout,
            "cycle_interval": 30 if min_capital > 200 else 45,  # Mais lento para capital menor
            "state_save_interval": 300,
            
            # Configurações avançadas
            "advanced_config": {
                "volatility_weight": optimal_weights['volatility_weight'],
                "slippage_weight": optimal_weights['slippage_weight'],
                "execution_time_weight": optimal_weights['execution_time_weight'],
                "max_retries": retry_params['max_retries'],
                "retry_delay": retry_params['retry_delay'],
                "retry_backoff": retry_params['retry_backoff'],
                "confidence_threshold": 0.65 if min_capital > 200 else 0.75  # Mais conservador para capital menor
            },
            
            # Dados adicionais
            "volatility_matrix": volatility_matrix,
            "pair_ranking": {pair: i+1 for i, pair in enumerate(ranked_pairs)},
            "position_sizing": {p: float(v) for p, v in position_sizing.items() if not p.endswith('_usd')},
            "capital_aware": {
                "min_capital": float(min_capital),
                "optimal_allocation": {p: float(position_sizing[f"{p}_usd"]) for p in self.pairs if f"{p}_usd" in position_sizing}
            }
        }
        
        self.logger.info(f"Parâmetros ótimos calculados")
        return optimal_params

class ConfigOptimizer:
    """Otimiza configurações do sistema baseado em dados históricos"""
    
    def __init__(self, config_path: str, exchanges: List[MarketAPI], pairs: List[str], capital_per_exchange: Dict[str, float] = None):
        """
        Inicializa o otimizador
        
        Args:
            config_path: Caminho para o arquivo de configuração
            exchanges: Lista de APIs de exchanges
            pairs: Lista de pares de trading
            capital_per_exchange: Dicionário com capital disponível por exchange
        """
        self.config_path = config_path
        self.exchanges = exchanges
        self.pairs = pairs
        self.capital_per_exchange = capital_per_exchange or {ex.exchange_id: 1000.0 for ex in exchanges}
        self.data_analyzer = HistoricalDataAnalyzer(exchanges, pairs, capital_per_exchange=self.capital_per_exchange)
        
        self.logger = logging.getLogger("config_optimizer")
        self.logger.info(f"Otimizador inicializado: {len(exchanges)} exchanges, {len(pairs)} pares")
        self.logger.info(f"Capital disponível: {self.capital_per_exchange}")
    
    async def load_current_config(self) -> Dict[str, Any]:
        """Carrega configuração atual"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return config
        else:
            return {}
    
    async def update_config_with_optimal_values(self) -> Dict[str, Any]:
        """Atualiza configuração com valores otimizados"""
        self.logger.info(f"Atualizando configuração em {self.config_path}...")
        
        # Carregar configuração atual
        current_config = await self.load_current_config()
        
        # Coletar dados históricos
        await self.data_analyzer.collect_historical_data()
        
        # Obter parâmetros ótimos
        optimal_params = self.data_analyzer.get_optimal_parameters()
        
        # Atualizar configuração
        new_config = current_config.copy()
        
        # Atualizar parâmetros principais
        for key in ["min_spread", "base_threshold", "min_volume_24h", "max_position_pct", 
                   "cycle_interval", "min_transfer", "rebalance_threshold", "execution_timeout"]:
            if key in optimal_params:
                new_config[key] = optimal_params[key]
        
        # Atualizar configuração avançada
        if "advanced_config" in new_config and "advanced_config" in optimal_params:
            for key in ["volatility_weight", "slippage_weight", "execution_time_weight", 
                       "max_retries", "retry_delay", "retry_backoff", "confidence_threshold"]:
                if key in optimal_params["advanced_config"]:
                    new_config["advanced_config"][key] = optimal_params["advanced_config"][key]
        
        # Salvar atualização
        with open(self.config_path, 'w') as f:
            json.dump(new_config, f, indent=2)
        
        # Salvar dados adicionais em arquivos separados
        config_dir = os.path.dirname(self.config_path)
        
        # Salvar matriz de volatilidade
        volatility_path = os.path.join(config_dir, "volatility_matrix.json")
        if "volatility_matrix" in optimal_params:
            with open(volatility_path, 'w') as f:
                json.dump(optimal_params["volatility_matrix"], f, indent=2)
            self.logger.info(f"Matriz de volatilidade salva em {volatility_path}")
        
        # Salvar ranking de pares
        pair_ranking_path = os.path.join(config_dir, "pair_ranking.json")
        if "pair_ranking" in optimal_params:
            with open(pair_ranking_path, 'w') as f:
                json.dump(optimal_params["pair_ranking"], f, indent=2)
            self.logger.info(f"Ranking de pares salvo em {pair_ranking_path}")
        
        # Salvar alocação de capital
        allocation_path = os.path.join(config_dir, "capital_allocation.json")
        if "capital_aware" in optimal_params:
            with open(allocation_path, 'w') as f:
                json.dump(optimal_params["capital_aware"], f, indent=2)
            self.logger.info(f"Alocação de capital salva em {allocation_path}")
        
        # Salvar posicionamento
        position_path = os.path.join(config_dir, "position_sizing.json")
        if "position_sizing" in optimal_params:
            with open(position_path, 'w') as f:
                json.dump(optimal_params["position_sizing"], f, indent=2)
            self.logger.info(f"Tamanhos de posição salvos em {position_path}")
        
        self.logger.info(f"Configuração atualizada com valores otimizados")
        return new_config

async def optimize_config(
    config_path: str, 
    exchanges: List[MarketAPI], 
    pairs: List[str], 
    capital_per_exchange: Dict[str, float],
    days: int = 7,
    iterations: int = 50
) -> Dict[str, Any]:
    """
    Otimiza os parâmetros de configuração baseado em dados históricos
    
    Args:
        config_path: Caminho do arquivo de configuração
        exchanges: Lista de exchanges para teste
        pairs: Lista de pares para teste
        capital_per_exchange: Capital por exchange
        days: Número de dias para análise de dados históricos
        iterations: Número de iterações para otimização
        
    Returns:
        Configuração otimizada
    """
    logger.info(f"Iniciando otimização de configuração...")
    
    # Definir capital padrão se não fornecido
    if capital_per_exchange is None:
        capital_per_exchange = {ex.exchange_id: 1000.0 for ex in exchanges}
    
    logger.info(f"Capital disponível: {capital_per_exchange}")
    
    optimizer = ConfigOptimizer(config_path, exchanges, pairs, capital_per_exchange)
    optimized_config = await optimizer.update_config_with_optimal_values()
    
    logger.info(f"Otimização de configuração concluída")
    return optimized_config 