#!/usr/bin/env python3
"""
QUALIA Core Engine: Orquestrador central que integra LSTM, QCNN e Performance Analyzer.
Implementa um sistema adaptativo de tomada de decisão para trading quântico.
"""

import pandas as pd
import numpy as np
import uuid
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

# Importar componentes do sistema QUALIA
from quantum_trading.neural.lstm_predictor import LSTMPredictor
from quantum_trading.metrics.performance_analyzer import PerformanceMetrics
from qcnn_wrapper import QCNNWrapper
from quantum_trading.integration.helix_controller import HelixController

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("qualia_engine")

class QUALIAEngine:
    """
    Motor central do sistema QUALIA que integra os componentes principais:
    - LSTM Predictor para timing ótimo
    - QCNN para análise quântica
    - Performance Analyzer para monitoramento e adaptação
    """
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 enable_quantum: bool = True,
                 enable_helix: bool = True,  # Nova opção para ativar o Helix
                 lstm_threshold: float = 0.7,
                 adaptation_rate: float = 0.01,
                 lstm_predictor: Optional[LSTMPredictor] = None,
                 qcnn_wrapper: Optional[QCNNWrapper] = None,
                 performance_analyzer: Optional[PerformanceMetrics] = None):
        """
        Inicializa o motor QUALIA.
        
        Args:
            config: Configurações opcionais.
            enable_quantum: Se deve ativar o componente quântico.
            enable_helix: Se deve ativar o componente Helix.
            lstm_threshold: Limiar para ativação do componente quântico.
            adaptation_rate: Taxa de adaptação para ajustes baseados em feedback.
            lstm_predictor: Instância opcional do LSTMPredictor.
            qcnn_wrapper: Instância opcional do QCNNWrapper.
            performance_analyzer: Instância opcional do PerformanceMetrics.
        """
        self.config = config or {}
        self.enable_quantum = enable_quantum
        self.enable_helix = enable_helix
        self.lstm_threshold = lstm_threshold
        self.adaptation_rate = adaptation_rate
        
        # Inicializar componentes
        self.lstm = lstm_predictor or LSTMPredictor(
            model_path=self.config.get('lstm_model_path'),
            lookback_periods=self.config.get('lookback_periods', 10)
        )
        self.performance = performance_analyzer or PerformanceMetrics(
            max_history=self.config.get('max_history', 1000),
            save_path=self.config.get('metrics_save_path', 'metrics/qualia_performance.json')
        )
        if self.enable_quantum:
            self.qcnn = qcnn_wrapper or QCNNWrapper(config=self.config.get('qcnn_config'))
        
        # Inicializar Helix se habilitado
        if self.enable_helix:
            self.helix = HelixController(config=self.config.get('helix_config'))
            # Evolução inicial do Helix
            self.helix.evolve_and_analyze(steps=5)
            helix_params = self.helix.derive_trading_parameters()
            self.adaptive_params = helix_params.copy()
        
        # Iniciar ciclo atual
        self.current_cycle_id = self.performance.start_cycle()
        
        # Métricas adaptativas iniciais
        self.adaptive_params = {
            'lstm_threshold': self.lstm_threshold,
            'quantum_coherence': 0.5,
            'quantum_complexity': 0.3,
            'success_rate': 0.0,
            'avg_profit': 0.0
        }
        
        # Definir parâmetros padrão para referência
        self.default_adaptive_params = self.adaptive_params.copy()
        
        logger.info("QUALIA Engine inicializado")
    
    def update_helix_state(self) -> Dict[str, Any]:
        """
        Atualiza o estado do Helix e ajusta os parâmetros adaptativos.
        
        Returns:
            Métricas atualizadas do Helix.
        """
        if not self.enable_helix:
            return {}
        helix_metrics = self.helix.evolve_and_analyze(steps=1)
        helix_params = self.helix.derive_trading_parameters()
        self.adaptive_params.update(helix_params)
        logger.debug(f"Parâmetros adaptativos atualizados com Helix: {helix_params}")
        return helix_metrics

    def process_opportunity(self, 
                          market_data: Dict[str, Any], 
                          candlesticks: List[Dict[str, Any]],
                          quantum_state: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Processa uma oportunidade de mercado, aplicando análise adaptativa.
        
        Args:
            market_data: Dados de mercado (preços, volumes, etc.)
            candlesticks: Lista de dados de candlesticks
            quantum_state: Estado quântico opcional para o QCNN
            
        Returns:
            Decisão de trading e insights
        """
        # Atualiza o estado do Helix, se ativado
        if self.enable_helix:
            helix_metrics = self.update_helix_state()
        
        # Converter dados para DataFrame para análise LSTM
        market_df = pd.DataFrame([market_data])
        candle_np = np.array(candlesticks)
        
        # Registrar a oportunidade para análise
        opportunity_id = str(uuid.uuid4())
        self.performance.record_opportunity(opportunity_id, {
            'timestamp': market_data.get('timestamp', datetime.now().isoformat()),
            'market_data': market_data,
            'adaptive_params': self.adaptive_params.copy()
        })
        
        # Obter previsão do LSTM
        lstm_result = self.lstm.predict(candle_np)
        confidence = lstm_result.get('confidence', 0.0)
        direction = lstm_result.get('direction', 'none')
        should_enter = lstm_result.get('should_enter', False)
        
        # Preparar resultado
        result = {
            'opportunity_id': opportunity_id,
            'timestamp': market_data.get('timestamp', datetime.now().isoformat()),
            'lstm_result': lstm_result,
            'should_enter': should_enter,
            'confidence': confidence,
            'direction': direction,
            'adaptive_params': self.adaptive_params.copy()
        }
        
        # Aplicar análise quântica se confiança for alta o suficiente
        if should_enter and confidence >= self.adaptive_params.get('lstm_threshold', self.lstm_threshold) and self.enable_quantum:
            consciousness_state = {
                'coherence': self.adaptive_params.get('quantum_coherence', 0.5),
                'complexity': self.adaptive_params.get('quantum_complexity', 0.3),
                'batch_size': 1
            }
            
            # Usar o estado quântico do Helix se disponível
            if self.enable_helix and quantum_state is None:
                quantum_state = self.helix.get_quantum_state()
            elif quantum_state is None:
                quantum_state = self._generate_quantum_state(pd.DataFrame([market_data]))
            
            qcnn_result = self.qcnn.process_candlestick_data(
                candle_np,
                quantum_state=quantum_state,
                consciousness_state=consciousness_state
            )
            
            result['quantum_insights'] = qcnn_result
            result['should_enter'] = self._integrate_lstm_qcnn_decisions(lstm_result, qcnn_result)
            
            if self.enable_helix:
                result['helix_insights'] = {
                    'quantum_metrics': self.helix.quantum_metrics,
                    'fractal_metrics': self.helix.fractal_metrics
                }
        
        # Decisão final de trading
        result['execute_trade'] = result['should_enter']
        
        # Registrar resultado
        self.performance.update_opportunity(opportunity_id, result)
        
        return result
    
    def execute_trade(self, 
                      opportunity: Dict[str, Any], 
                      amount: float, 
                      expected_profit: float) -> str:
        """
        Executa uma operação de trading.
        
        Args:
            opportunity: Oportunidade detectada.
            amount: Montante a ser negociado.
            expected_profit: Lucro esperado.
            
        Returns:
            ID da operação.
        """
        trade_id = str(uuid.uuid4())
        self.performance.record_trade_start(
            trade_id=trade_id,
            opportunity_id=opportunity['opportunity_id'],
            pair=opportunity.get('pair', "ETH-USDT"),
            exchange_a=opportunity.get('exchange_a', "kucoin"),
            exchange_b=opportunity.get('exchange_b', "kraken"),
            amount=amount,
            expected_profit=expected_profit,
            cycle_id=self.current_cycle_id
        )
        logger.info(f"Iniciada operação {trade_id} para oportunidade {opportunity['opportunity_id']}")
        return trade_id
    
    def complete_trade(self, trade_id: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Completa uma operação e atualiza métricas de desempenho.
        
        Args:
            trade_id: ID da operação
            results: Resultados da operação
            
        Returns:
            Métricas atualizadas
        """
        # Registrar resultados
        self.performance.complete_trade(trade_id, results)
        
        # Atualizar métricas adaptativas
        self._update_adaptive_metrics(results)
        
        return self.adaptive_params
    
    def end_current_cycle(self) -> Dict[str, Any]:
        """
        Finaliza o ciclo atual e inicia um novo.
        
        Returns:
            Métricas do ciclo finalizado.
        """
        cycle_metrics = self.performance.end_cycle(self.current_cycle_id)
        self.current_cycle_id = self.performance.start_cycle()
        logger.info(f"Finalizado ciclo #{cycle_metrics['cycle_id']}, iniciado novo ciclo #{self.current_cycle_id}")
        return cycle_metrics
    
    def generate_report(self) -> str:
        """
        Gera um relatório de desempenho.
        
        Returns:
            Caminho para o relatório gerado.
        """
        return self.performance.generate_performance_report()
    
    def _update_adaptive_metrics(self, results: Dict[str, Any]) -> None:
        """
        Atualiza métricas adaptativas com base nos resultados da operação.
        
        Args:
            results: Resultados da operação
        """
        success = results.get('success', False)
        profit = results.get('profit_percentage', 0.0)
        
        # Atualizar taxa de sucesso e lucro médio
        success_rate = self.adaptive_params.get('success_rate', 0.0)
        avg_profit = self.adaptive_params.get('avg_profit', 0.0)
        
        # Aplicar adaptação gradual
        self.adaptive_params['success_rate'] = success_rate * (1 - self.adaptation_rate) + (1.0 if success else 0.0) * self.adaptation_rate
        self.adaptive_params['avg_profit'] = avg_profit * (1 - self.adaptation_rate) + profit * self.adaptation_rate
        
        # Ajustar threshold do LSTM com base no desempenho
        lstm_threshold = self.adaptive_params.get('lstm_threshold', self.lstm_threshold)
        if success and profit > 0:
            # Se for bem-sucedido, reduzir levemente o threshold (mais oportunidades)
            lstm_threshold = max(0.6, lstm_threshold - 0.01 * self.adaptation_rate)
        else:
            # Se falhar, aumentar o threshold (mais seletivo)
            lstm_threshold = min(0.9, lstm_threshold + 0.02 * self.adaptation_rate)
        
        self.adaptive_params['lstm_threshold'] = lstm_threshold
        
        logger.debug(f"Métricas adaptativas atualizadas: {self.adaptive_params}")
    
    def _integrate_lstm_qcnn_decisions(self, lstm_result: Dict[str, Any], qcnn_result: Dict[str, Any]) -> bool:
        """
        Integra decisões do LSTM e QCNN para uma decisão final.
        
        Args:
            lstm_result: Resultado da análise LSTM
            qcnn_result: Resultado da análise QCNN
            
        Returns:
            Decisão integrada (True/False)
        """
        lstm_confidence = lstm_result.get('confidence', 0.0)
        quantum_confidence = qcnn_result.get('quantum_influence', {}).get('confidence', 0.0)
        
        # Verificar se o QCNN confirma a direção do LSTM
        lstm_direction = lstm_result.get('direction', 'none')
        qcnn_direction = 'up' if np.mean(qcnn_result.get('predictions', [0])[:5]) > 0 else 'down'
        
        # Se direções são opostas, usar menor confiança
        if lstm_direction != 'none' and lstm_direction != qcnn_direction:
            quantum_confidence *= 0.5
        
        # Integrar confiança
        integrated_confidence = lstm_confidence * 0.7 + quantum_confidence * 0.3
        
        # Gerar decisão final
        return integrated_confidence >= self.adaptive_params.get('lstm_threshold', self.lstm_threshold)
    
    def _generate_quantum_state(self, market_data: pd.DataFrame) -> np.ndarray:
        """
        Gera um estado quântico a partir dos dados de mercado.
        
        Args:
            market_data: DataFrame com dados de mercado.
            
        Returns:
            Array representando o estado quântico.
        """
        features = []
        if 'spread' in market_data.columns:
            features.append(market_data['spread'].mean())
            features.append(market_data['spread'].std())
        if 'volatility_a' in market_data.columns:
            features.append(market_data['volatility_a'].mean())
            features.append(market_data['volatility_a'].std())
        if 'volatility_b' in market_data.columns:
            features.append(market_data['volatility_b'].mean())
            features.append(market_data['volatility_b'].std())
        if 'entropy' in market_data.columns:
            features.append(market_data['entropy'].mean())
        if 'fractal_dimension' in market_data.columns:
            features.append(market_data['fractal_dimension'].mean())
        while len(features) < 16:
            features.append(0.0)
        features = np.array(features[:16])
        features = (features - np.mean(features)) / (np.std(features) + 1e-8)
        return features 