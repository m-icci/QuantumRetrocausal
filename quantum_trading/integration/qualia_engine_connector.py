#!/usr/bin/env python3
"""
Conector entre o HelixController e o QUALIAEngine.
Responsável pela integração completa dos sistemas Helix e QUALIA.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from quantum_trading.integration.qualia_core_engine import QUALIAEngine
from quantum_trading.integration.helix_controller import HelixController
from quantum_trading.neural.lstm_predictor import LSTMPredictor
from quantum_trading.metrics.performance_analyzer import PerformanceMetrics
from qcnn_wrapper import QCNNWrapper

logger = logging.getLogger("qualia_engine_connector")

class QUALIAEngineConnector:
    """
    Conector para integrar o HelixController com o QUALIAEngine,
    permitindo uma adaptação dinâmica e evolução consciente.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa o conector.
        
        Args:
            config: Configurações opcionais para todos os componentes
        """
        self.config = config or {}
        logger.info("Inicializando QUALIAEngineConnector...")
        
        # Inicializar Helix Controller
        helix_config = self.config.get('helix', {})
        self.helix_controller = HelixController(helix_config)
        logger.info("HelixController inicializado")
        
        # Inicializar dependências para o QUALIA Engine
        self.lstm_predictor = LSTMPredictor()
        self.qcnn_wrapper = QCNNWrapper()
        self.performance_analyzer = PerformanceMetrics()
        
        # Inicializar o motor QUALIA principal
        qualia_config = self.config.get('qualia', {})
        self.qualia_engine = QUALIAEngine(
            lstm_predictor=self.lstm_predictor,
            qcnn_wrapper=self.qcnn_wrapper,
            performance_analyzer=self.performance_analyzer,
            config=qualia_config
        )
        logger.info("QUALIAEngine inicializado")
        
        # Estado interno
        self.evolution_cycle = 0
        self.helix_results_history = []
        self.trade_results_history = []
        
        logger.info("QUALIAEngineConnector pronto")
    
    def process_market_data(self, market_data: Dict[str, Any], candlesticks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Processa os dados de mercado através do pipeline completo Helix-QUALIA
        
        Args:
            market_data: Dados de mercado (preços atuais, volumes, etc.)
            candlesticks: Lista de dados de candlestick
            
        Returns:
            Resultados do processamento e decisões de trading
        """
        # Passo 1: Evolução e análise do Helix
        logger.info(f"Iniciando ciclo de evolução {self.evolution_cycle}")
        helix_results = self.helix_controller.evolve_and_analyze(steps=1)
        self.helix_results_history.append(helix_results)
        
        # Passo 2: Derivar parâmetros adaptativos
        trading_params = self.helix_controller.derive_trading_parameters()
        
        # Passo 3: Atualizar parâmetros no QUALIA Engine
        self._update_qualia_parameters(trading_params)
        
        # Passo 4: Obter o estado quântico atual do Helix para o QCNN
        quantum_state = self.helix_controller.get_quantum_state()
        
        # Passo 5: Processar oportunidade de mercado com o QUALIA Engine
        trade_decision = self.qualia_engine.process_opportunity(
            market_data=market_data,
            candlesticks=candlesticks,
            quantum_state=quantum_state
        )
        
        # Registrar resultados
        self.trade_results_history.append({
            'cycle': self.evolution_cycle,
            'timestamp': market_data.get('timestamp'),
            'decision': trade_decision,
            'trading_params': trading_params,
        })
        
        # Incrementar o ciclo de evolução
        self.evolution_cycle += 1
        
        return {
            'cycle': self.evolution_cycle - 1,  # O ciclo atual (já incrementado)
            'helix_metrics': helix_results,
            'trading_params': trading_params,
            'trade_decision': trade_decision
        }
    
    def complete_trade(self, trade_id: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Completa uma operação e atualiza métricas de desempenho
        
        Args:
            trade_id: ID da operação
            results: Resultados da operação
            
        Returns:
            Métricas atualizadas
        """
        # Completar a operação no QUALIA Engine
        updated_metrics = self.qualia_engine.complete_trade(trade_id, results)
        
        # Aplicar feedback retrocausal no Helix
        self._apply_retrocausal_feedback(trade_id, results)
        
        return updated_metrics
    
    def run_backtesting(self, historical_data: List[Dict[str, Any]], candlesticks_history: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Executa backtesting com dados históricos
        
        Args:
            historical_data: Lista de dados históricos de mercado
            candlesticks_history: Lista de listas de candlesticks históricos
            
        Returns:
            Resultados do backtesting
        """
        logger.info(f"Iniciando backtesting com {len(historical_data)} pontos de dados")
        
        backtest_results = []
        trades_executed = 0
        trades_successful = 0
        total_profit = 0.0
        
        # Resetar estado para o backtesting
        self._reset_state()
        
        # Processar cada ponto de dados históricos
        for i, (market_data, candlesticks) in enumerate(zip(historical_data, candlesticks_history)):
            # Processar dados
            result = self.process_market_data(market_data, candlesticks)
            
            # Simular resultado da operação (em backtesting)
            trade_decision = result.get('trade_decision', {})
            if trade_decision.get('execute_trade', False):
                trades_executed += 1
                
                # Simular resultado baseado no próximo ponto de dados (se disponível)
                if i + 1 < len(historical_data):
                    next_price = historical_data[i + 1].get('price', market_data.get('price', 0))
                    entry_price = market_data.get('price', 0)
                    
                    # Calcular lucro/prejuízo simulado
                    if trade_decision.get('direction') == 'buy':
                        profit = (next_price - entry_price) / entry_price * 100
                    else:  # 'sell'
                        profit = (entry_price - next_price) / entry_price * 100
                    
                    # Ajustar para taxas simuladas
                    profit -= 0.1  # 0.1% de taxa
                    
                    trade_result = {
                        'trade_id': f"backtest_{i}_{self.evolution_cycle}",
                        'profit_percentage': profit,
                        'success': profit > 0,
                        'entry_price': entry_price,
                        'exit_price': next_price,
                        'direction': trade_decision.get('direction'),
                    }
                    
                    # Registrar resultado
                    trades_successful += 1 if profit > 0 else 0
                    total_profit += profit
                    
                    # Completar operação para feedback
                    self.complete_trade(trade_result['trade_id'], trade_result)
                    
                    backtest_results.append({
                        'market_data': market_data,
                        'processing_result': result,
                        'trade_result': trade_result
                    })
        
        # Compilar estatísticas
        win_rate = trades_successful / trades_executed if trades_executed > 0 else 0
        
        return {
            'cycles_processed': self.evolution_cycle,
            'trades_executed': trades_executed,
            'trades_successful': trades_successful,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'average_profit': total_profit / trades_executed if trades_executed > 0 else 0,
            'detailed_results': backtest_results
        }
    
    def visualize_helix_field(self) -> np.ndarray:
        """
        Obtém uma representação visual do campo da hélice atual
        
        Returns:
            Array 2D representando o campo da hélice
        """
        return self.helix_controller.helix_analyzer.visualize_field()
    
    def _update_qualia_parameters(self, trading_params: Dict[str, Any]) -> None:
        """
        Atualiza parâmetros no QUALIA Engine baseado nas saídas do Helix
        
        Args:
            trading_params: Parâmetros derivados do Helix
        """
        # Atualizar parâmetros no QUALIA Engine
        for param, value in trading_params.items():
            if hasattr(self.qualia_engine, param):
                setattr(self.qualia_engine, param, value)
            elif param in self.qualia_engine.adaptive_params:
                self.qualia_engine.adaptive_params[param] = value
        
        # Atualizar parâmetros específicos
        self.qualia_engine.lstm_threshold = trading_params.get('lstm_threshold', self.qualia_engine.lstm_threshold)
        self.qualia_engine.quantum_coherence = trading_params.get('quantum_coherence', self.qualia_engine.quantum_coherence)
        self.qualia_engine.quantum_complexity = trading_params.get('quantum_complexity', self.qualia_engine.quantum_complexity)
    
    def _apply_retrocausal_feedback(self, trade_id: str, results: Dict[str, Any]) -> None:
        """
        Aplica feedback retrocausal ao Helix baseado nos resultados de trading
        
        Args:
            trade_id: ID da operação
            results: Resultados da operação
        """
        # Extrair métricas relevantes
        profit = results.get('profit_percentage', 0)
        success = results.get('success', False)
        
        # Ajustar parâmetros do feedback retrocausal
        feedback_strength = min(1.0, abs(profit) / 10.0)  # Normalizar para 0-1
        feedback_direction = 1.0 if success else -0.5  # Positivo para sucesso, negativo para falha
        
        # Aplicar feedback ao integrador retrocausal
        self.helix_controller.retrocausal_integrator.add_external_feedback(
            step=self.helix_controller.current_step,
            feedback_value=feedback_strength * feedback_direction,
            metadata={
                'trade_id': trade_id,
                'profit': profit,
                'success': success
            }
        )
        
        logger.debug(f"Feedback retrocausal aplicado para trade {trade_id}, strength={feedback_strength}, direction={feedback_direction}")
    
    def _reset_state(self) -> None:
        """Reseta o estado interno para backtesting ou novos ciclos"""
        self.evolution_cycle = 0
        self.helix_results_history = []
        self.trade_results_history = []
        
        # Reinicializar o campo da hélice
        self.helix_controller.helix_analyzer.initialize_helix()
        self.helix_controller.current_step = 0
        
        # Resetar métricas de desempenho
        self.performance_analyzer.reset_metrics()
        
        # Resetar estado adaptativo no QUALIA Engine
        self.qualia_engine.adaptive_params = self.qualia_engine.default_adaptive_params.copy()
        
        logger.info("Estado interno resetado") 