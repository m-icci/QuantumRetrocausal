#!/usr/bin/env python3
"""
Spectra Strategy (Módulo de Análise Avançada)
==============================================
Este módulo implementa a estratégia Spectra, que combina análise de sentimento
com aprendizado por reforço profundo para gerar sinais complementares para o sistema
de trading QUALIA.

A estratégia inclui:
- Análise de sentimento do mercado usando processamento de linguagem natural
- Deep Reinforcement Learning para otimização de decisões de trading
- Integração com sinais externos e fatores de mercado
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Any, Optional
import time
from datetime import datetime

logger = logging.getLogger("spectra_strategy")

class SentimentAnalyzer:
    """
    Analisador de sentimento para criptomoedas.
    
    Utiliza processamento de linguagem natural para analisar notícias,
    tweets e fóruns, gerando uma pontuação de sentimento do mercado.
    """
    
    def __init__(self, config=None):
        """
        Inicializa o analisador de sentimento
        
        Args:
            config: Configuração opcional
        """
        self.config = config or {}
        self.sources = self.config.get("sources", ["twitter", "reddit", "news"])
        self.weight_social = self.config.get("weight_social", 0.6)
        self.weight_news = self.config.get("weight_news", 0.4)
        self.sentiment_history = []
        logger.info(f"SentimentAnalyzer inicializado com {len(self.sources)} fontes")
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analisa o sentimento de um texto específico
        
        Args:
            text: Texto para análise
            
        Returns:
            Resultado da análise de sentimento
        """
        # Simulação simplificada de análise de sentimento
        # Em uma implementação real, usaria NLP com modelos pré-treinados
        
        # Palavras indicativas de sentimento positivo/negativo (simplificadas)
        positive_words = ["bull", "up", "grow", "profit", "gain", "good", "positive", "rise"]
        negative_words = ["bear", "down", "crash", "loss", "bad", "negative", "fall", "risk"]
        
        # Contagem simplificada
        text_lower = text.lower()
        positive_count = sum(word in text_lower for word in positive_words)
        negative_count = sum(word in text_lower for word in negative_words)
        
        total_words = len(text_lower.split())
        if total_words == 0:
            return {"score": 0, "magnitude": 0, "confidence": 0}
        
        # Cálculo de score (-1 a 1)
        if positive_count + negative_count > 0:
            score = (positive_count - negative_count) / (positive_count + negative_count)
        else:
            score = 0
            
        # Magnitude e confiança
        magnitude = (positive_count + negative_count) / max(1, total_words)
        confidence = min(1.0, (positive_count + negative_count) / 10)
        
        return {
            "score": score,  # Varia de -1 (negativo) a 1 (positivo)
            "magnitude": magnitude,  # Força do sentimento
            "confidence": confidence  # Confiança na análise
        }
    
    async def analyze_market_sentiment(self, pair: str) -> Dict[str, Any]:
        """
        Analisa o sentimento do mercado para um par específico
        
        Args:
            pair: Par de trading (ex: BTC/USDT)
            
        Returns:
            Análise completa de sentimento
        """
        # Simula latência de busca de dados
        await asyncio.sleep(0.1)
        
        # Em uma implementação real, faria scraping de múltiplas fontes
        # e analisaria com modelos de NLP
        
        # Simulação de sentimento variável com componente aleatório
        base_symbol = pair.split('/')[0]
        timestamp = datetime.now().timestamp()
        
        # Sentimento cíclico + ruído
        cycle_component = np.sin(timestamp / 86400)  # Ciclo diário
        random_component = np.random.normal(0, 0.3)
        
        sentiment_score = 0.5 * cycle_component + 0.5 * random_component
        sentiment_score = max(-1.0, min(1.0, sentiment_score))  # Clamp entre -1 e 1
        
        # Armazena no histórico
        self.sentiment_history.append(sentiment_score)
        if len(self.sentiment_history) > 100:
            self.sentiment_history.pop(0)
        
        # Calcula métricas adicionais
        avg_sentiment = np.mean(self.sentiment_history) if self.sentiment_history else sentiment_score
        sentiment_momentum = sentiment_score - avg_sentiment
        
        # Cria resultado
        result = {
            "pair": pair,
            "sentiment_score": sentiment_score,
            "sentiment_magnitude": abs(sentiment_score),
            "bullish_ratio": (sentiment_score + 1) / 2,  # Converte para 0-1
            "source_count": len(self.sources),
            "sentiment_momentum": sentiment_momentum,
            "confidence": 0.7 + 0.3 * np.random.random(),  # Simulação de confiança
            "timestamp": timestamp
        }
        
        return result


class DeepReinforcementLearner:
    """
    Agente de Deep Reinforcement Learning para trading.
    
    Utiliza técnicas de Q-Learning e redes neurais para aprender
    estratégias ótimas de trading a partir da experiência.
    """
    
    def __init__(self, config=None):
        """
        Inicializa o agente de RL
        
        Args:
            config: Configuração opcional
        """
        self.config = config or {}
        self.learning_rate = self.config.get("learning_rate", 0.001)
        self.discount_factor = self.config.get("discount_factor", 0.95)
        self.exploration_rate = self.config.get("exploration_rate", 0.2)
        self.memory_size = self.config.get("memory_size", 1000)
        
        # Estado interno
        self.memory = []
        self.model_version = "0.1"
        self.last_state = None
        self.last_action = None
        
        logger.info(f"DeepReinforcementLearner inicializado (v{self.model_version})")
    
    def get_action(self, state: Dict[str, Any]) -> int:
        """
        Determina a melhor ação para o estado atual
        
        Args:
            state: Estado atual do mercado
            
        Returns:
            Ação: 0 (não fazer nada), 1 (comprar), 2 (vender)
        """
        # Simulação simplificada de política de RL
        # Em implementação real, usaria uma rede neural treinada
        
        # Extrai features relevantes
        price_momentum = state.get("price_momentum", 0)
        volume_change = state.get("volume_change", 0)
        sentiment = state.get("sentiment", 0)
        
        # Exploração (random) vs. exploitation (baseado em regras)
        if np.random.random() < self.exploration_rate:
            action = np.random.choice([0, 1, 2])  # Ação aleatória
        else:
            # Lógica simplificada baseada em regras
            combined_signal = 0.5 * price_momentum + 0.3 * volume_change + 0.2 * sentiment
            
            if combined_signal > 0.2:
                action = 1  # Comprar
            elif combined_signal < -0.2:
                action = 2  # Vender
            else:
                action = 0  # Não fazer nada
        
        # Armazena estado e ação para aprendizado futuro
        self.last_state = state
        self.last_action = action
        
        return action
    
    def update(self, new_state: Dict[str, Any], reward: float):
        """
        Atualiza o modelo com feedback da última ação
        
        Args:
            new_state: Novo estado após a ação
            reward: Recompensa recebida
        """
        if self.last_state is None or self.last_action is None:
            return
            
        # Armazena experiência (state, action, reward, next_state)
        experience = (self.last_state, self.last_action, reward, new_state)
        self.memory.append(experience)
        
        # Limita tamanho da memória
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        
        # Simulação de aprendizado
        # Em implementação real, atualizaria pesos da rede neural
        self.exploration_rate *= 0.999  # Reduz exploração gradualmente
        
    def predict_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prediz a ação ótima para o estado atual com métricas de confiança
        
        Args:
            state: Estado atual do mercado
            
        Returns:
            Predição com ação e métricas
        """
        action = self.get_action(state)
        
        # Mapeia código de ação para nome
        action_map = {0: "hold", 1: "buy", 2: "sell"}
        action_name = action_map[action]
        
        # Gera métricas associadas
        confidence = 0.5 + np.random.random() * 0.5  # Simulação de confiança
        
        return {
            "action": action_name,
            "action_code": action,
            "confidence": confidence,
            "expected_reward": np.random.random() * 0.3,  # Simulação de expectativa
            "exploration_rate": self.exploration_rate,
            "model_version": self.model_version,
            "state_value": 0.5 + 0.5 * np.random.random()  # Simulação de valor de estado
        }


class SpectraStrategy:
    """
    Estratégia Spectra de análise e trading.
    
    Combina análise de sentimento com deep reinforcement learning
    para gerar sinais complementares ao sistema QUALIA.
    """
    
    def __init__(self, exchanges: List[Any], pairs: List[str], config: Optional[Dict[str, Any]] = None):
        """
        Inicializa a estratégia Spectra
        
        Args:
            exchanges: Lista de APIs de exchanges
            pairs: Lista de pares de trading
            config: Configuração personalizada
        """
        self.exchanges = exchanges
        self.pairs = pairs
        self.config = config or {}
        
        # Configurações de trading
        self.min_spread = self.config.get("min_spread", 0.001)
        self.base_threshold = self.config.get("base_threshold", 0.001)
        self.sentiment_weight = self.config.get("sentiment_weight", 0.3)
        self.rl_weight = self.config.get("rl_weight", 0.7)
        
        # Inicializar componentes
        self.sentiment_analyzer = SentimentAnalyzer(self.config)
        self.rl_agent = DeepReinforcementLearner(self.config)
        
        # Estado interno
        self.last_run = None
        self.opportunities = []
        self.sentiment_scores = {}
        self.rl_predictions = {}
        
        logger.info(f"SpectraStrategy inicializada com {len(exchanges)} exchanges e {len(pairs)} pares")
    
    async def collect_market_data(self, pair: str) -> Dict[str, Any]:
        """
        Coleta dados de mercado para um par específico
        
        Args:
            pair: Par de trading
            
        Returns:
            Dados de mercado consolidados
        """
        # Coleta preços de todas as exchanges
        prices = {}
        volumes = {}
        
        for ex in self.exchanges:
            try:
                # Coleta preço atual
                price = await asyncio.to_thread(ex.get_price, pair)
                prices[ex.exchange_id] = price
                
                # Coleta volume
                volume = await asyncio.to_thread(ex.get_24h_volume, pair)
                volumes[ex.exchange_id] = volume
            except Exception as e:
                logger.warning(f"Erro ao coletar dados de {ex.exchange_id} para {pair}: {e}")
        
        # Calcula métricas consolidadas
        if not prices:
            return {}
            
        avg_price = sum(prices.values()) / len(prices)
        total_volume = sum(volumes.values())
        
        # Calcula spread entre exchanges
        if len(prices) >= 2:
            max_price = max(prices.values())
            min_price = min(prices.values())
            spread = (max_price - min_price) / min_price
        else:
            spread = 0
            
        return {
            "pair": pair,
            "prices": prices,
            "volumes": volumes,
            "avg_price": avg_price,
            "total_volume": total_volume,
            "spread": spread,
            "timestamp": time.time()
        }
    
    async def analyze_pair(self, pair: str) -> Dict[str, Any]:
        """
        Análise completa para um par de trading
        
        Args:
            pair: Par de trading
            
        Returns:
            Análise completa com sentimento e RL
        """
        # Coleta dados de mercado
        market_data = await self.collect_market_data(pair)
        if not market_data:
            return {"pair": pair, "valid": False}
            
        # Análise de sentimento
        sentiment_result = await self.sentiment_analyzer.analyze_market_sentiment(pair)
        
        # Preparação de estado para RL
        state = {
            "price": market_data["avg_price"],
            "volume": market_data["total_volume"],
            "spread": market_data["spread"],
            "sentiment": sentiment_result["sentiment_score"],
            "price_momentum": 0,  # Em implementação real, calcularia do histórico
            "volume_change": 0,   # Em implementação real, calcularia do histórico
        }
        
        # Obter predição de RL
        rl_prediction = self.rl_agent.predict_action(state)
        
        # Armazena resultados
        self.sentiment_scores[pair] = sentiment_result
        self.rl_predictions[pair] = rl_prediction
        
        # Combina sinais
        combined_signal = (
            self.sentiment_weight * sentiment_result["sentiment_score"] + 
            self.rl_weight * (1 if rl_prediction["action"] == "buy" else -1 if rl_prediction["action"] == "sell" else 0)
        )
        
        # Determina oportunidade
        opportunity = {
            "pair": pair,
            "valid": True,
            "sentiment_score": sentiment_result["sentiment_score"],
            "sentiment_confidence": sentiment_result["confidence"],
            "rl_action": rl_prediction["action"],
            "rl_confidence": rl_prediction["confidence"],
            "combined_signal": combined_signal,
            "is_opportunity": abs(combined_signal) > self.base_threshold and market_data["spread"] > self.min_spread,
            "timestamp": time.time()
        }
        
        return opportunity
    
    async def run_strategy_cycle(self) -> Dict[str, Any]:
        """
        Executa um ciclo completo da estratégia Spectra
        
        Returns:
            Resultados do ciclo
        """
        start_time = time.time()
        logger.info("Iniciando ciclo do Spectra...")
        
        # Limpa oportunidades anteriores
        self.opportunities = []
        
        # Analisa todos os pares em paralelo
        analysis_tasks = [self.analyze_pair(pair) for pair in self.pairs]
        results = await asyncio.gather(*analysis_tasks)
        
        # Filtra oportunidades válidas
        opportunities = [r for r in results if r.get("valid") and r.get("is_opportunity")]
        self.opportunities = opportunities
        
        # Calcula métricas agregadas
        sentiment_scores = [r.get("sentiment_score", 0) for r in results if r.get("valid")]
        rl_confidences = [r.get("rl_confidence", 0) for r in results if r.get("valid")]
        
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
        avg_rl_confidence = np.mean(rl_confidences) if rl_confidences else 0
        
        # Simula lucro total (em implementação real seria calculado a partir de trades)
        total_profit = 0
        if opportunities:
            # Simula alguns ganhos proporcionais à força do sinal
            for opp in opportunities:
                profit_factor = abs(opp["combined_signal"]) * 0.01
                total_profit += profit_factor
        
        # Atualiza timestamp de última execução
        self.last_run = time.time()
        
        execution_time = time.time() - start_time
        logger.info(f"Ciclo do Spectra concluído em {execution_time:.2f}s com {len(opportunities)} oportunidades")
        
        return {
            "opportunities": len(opportunities),
            "opportunities_details": opportunities,
            "signals_generated": len(results),
            "total_profit": total_profit,
            "sentiment_score_avg": avg_sentiment,
            "rl_confidence_avg": avg_rl_confidence,
            "execution_time": execution_time,
            "timestamp": self.last_run
        }


if __name__ == "__main__":
    # Exemplo de uso independente
    logging.basicConfig(level=logging.INFO)
    
    # Dummy exchange para teste
    class DummyExchange:
        def __init__(self, exchange_id):
            self.exchange_id = exchange_id
        def get_price(self, pair):
            return 100 + np.random.normal(0, 2)
        def get_24h_volume(self, pair):
            return 10000 + np.random.normal(0, 1000)
    
    async def run_test():
        # Cria exchanges e estratégia
        exchanges = [DummyExchange("Binance"), DummyExchange("KuCoin")]
        pairs = ["BTC/USDT", "ETH/USDT", "XMR/USDT"]
        
        strategy = SpectraStrategy(exchanges, pairs)
        
        # Executa 3 ciclos
        for i in range(3):
            logger.info(f"Executando ciclo {i+1}...")
            result = await strategy.run_strategy_cycle()
            logger.info(f"Resultado: {result}")
            await asyncio.sleep(1)
    
    # Executa o teste
    if __name__ == "__main__":
        asyncio.run(run_test()) 