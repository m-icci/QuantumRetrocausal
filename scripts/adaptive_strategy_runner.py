#!/usr/bin/env python
"""
Executor de Estratégia Adaptativa QUALIA
Integra o analisador de performance e o preditor LSTM com a estratégia de arbitragem
"""

import os
import sys
import logging
import argparse
import time
import uuid
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import signal
import threading

# Adicionar diretório principal ao path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

# Importar módulos QUALIA
from quantum_trading.metrics.performance_analyzer import PerformanceMetrics
from quantum_trading.neural.lstm_predictor import LSTMPredictor
from quantum_trading.utils.logger import setup_logger
from quantum_trading.config.constants import (
    MARKET_CHECK_INTERVAL, OPPORTUNITY_THRESHOLD
)

# Tente importar a estratégia principal (WAVE)
try:
    from quantum_trading.strategies.wave_strategy import WAVEStrategy, WAVEState
    default_strategy = "wave"
except ImportError:
    logging.warning("Estratégia WAVE não encontrada. Tentando importar estratégia alternativa.")
    try:
        # Tente importar outra estratégia disponível como fallback
        from quantum_trading.strategies.arbitrage_strategy import ArbitrageStrategy
        default_strategy = "arbitrage"
    except ImportError:
        logging.error("Nenhuma estratégia de arbitragem encontrada.")
        sys.exit(1)

# Status global do sistema
running = True

# Configuração de logs
log_dir = os.path.join(root_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'adaptive_strategy_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logger = setup_logger('adaptive_strategy', log_file)

class AdaptiveStrategyRunner:
    """
    Executor de estratégia adaptativa que integra LSTM e análise de performance 
    com a estratégia base
    """
    
    def __init__(self, 
                strategy_name=default_strategy,
                performance_metrics_path=None,
                lstm_model_path=None,
                test_mode=False,
                min_confidence=0.6,
                history_window=100):
        """
        Inicializa o executor adaptativo
        
        Args:
            strategy_name: Nome da estratégia base a utilizar
            performance_metrics_path: Caminho para salvar métricas de performance
            lstm_model_path: Caminho para o modelo LSTM
            test_mode: Se True, executa em modo de teste (sem operações reais)
            min_confidence: Confiança mínima para aceitar previsões do LSTM
            history_window: Tamanho da janela de histórico para análise
        """
        self.strategy_name = strategy_name
        self.test_mode = test_mode
        self.min_confidence = min_confidence
        self.history_window = history_window
        
        # Definir caminhos padrão se não especificados
        if performance_metrics_path is None:
            performance_metrics_path = os.path.join(root_dir, 'metrics', 'qualia_performance.json')
        
        if lstm_model_path is None:
            lstm_model_path = os.path.join(root_dir, 'models', 'lstm_predictor.h5')
        
        # Inicializar componentes
        self.performance_metrics = PerformanceMetrics(save_path=performance_metrics_path)
        
        # Verificar se o modelo LSTM existe e inicializar
        self.lstm_enabled = os.path.exists(lstm_model_path)
        if self.lstm_enabled:
            self.lstm_predictor = LSTMPredictor(model_path=lstm_model_path)
            logger.info(f"Preditor LSTM inicializado com modelo: {lstm_model_path}")
        else:
            self.lstm_predictor = None
            logger.warning(f"Modelo LSTM não encontrado em {lstm_model_path}. "
                          "Continuando sem predições neurais.")
        
        # Carregar estratégia base apropriada
        self._load_strategy()
        
        # Buffer para armazenar dados de mercado para o LSTM
        self.market_data_buffer = pd.DataFrame()
        
        # Buffer para oportunidades detectadas
        self.opportunities_buffer = []
        
        # Estatísticas da sessão atual
        self.session_stats = {
            'start_time': datetime.now(),
            'cycles_completed': 0,
            'opportunities_detected': 0,
            'trades_executed': 0,
            'lstm_predictions': 0,
            'lstm_positive_predictions': 0,
            'total_profit': 0.0
        }
        
        # ID do ciclo atual
        self.current_cycle_id = None
        
        logger.info(f"Executor de Estratégia Adaptativa inicializado com estratégia base: {strategy_name}")
        logger.info(f"Modo de teste: {test_mode}")
        
    def _load_strategy(self):
        """
        Carrega a estratégia base apropriada
        """
        try:
            if self.strategy_name.lower() == "wave":
                from quantum_trading.strategies.wave_strategy import WAVEStrategy, WAVEState
                self.strategy = WAVEStrategy(test_mode=self.test_mode)
                logger.info("Estratégia WAVE carregada com sucesso")
            elif self.strategy_name.lower() == "arbitrage":
                from quantum_trading.strategies.arbitrage_strategy import ArbitrageStrategy
                self.strategy = ArbitrageStrategy(test_mode=self.test_mode)
                logger.info("Estratégia de Arbitragem carregada com sucesso")
            else:
                logger.error(f"Estratégia desconhecida: {self.strategy_name}")
                raise ValueError(f"Estratégia desconhecida: {self.strategy_name}")
        except Exception as e:
            logger.error(f"Erro ao carregar estratégia: {e}")
            raise
    
    def start(self):
        """
        Inicia a execução da estratégia adaptativa
        """
        logger.info("Iniciando execução da estratégia adaptativa...")
        
        # Inicializar estratégia base
        self.strategy.initialize()
        
        # Loop principal
        try:
            while running:
                # Iniciar novo ciclo
                self.current_cycle_id = self.performance_metrics.start_cycle()
                cycle_start_time = time.time()
                
                # Executar ciclo de detecção e execução
                self._run_cycle()
                
                # Finalizar ciclo
                cycle_metrics = self.performance_metrics.end_cycle(self.current_cycle_id)
                cycle_duration = time.time() - cycle_start_time
                
                # Atualizar estatísticas da sessão
                self.session_stats['cycles_completed'] += 1
                
                # Salvar métricas periodicamente
                if self.session_stats['cycles_completed'] % 10 == 0:
                    self.performance_metrics.save_metrics()
                    
                    # Gerar relatório a cada 100 ciclos
                    if self.session_stats['cycles_completed'] % 100 == 0:
                        self.performance_metrics.generate_performance_report()
                
                # Log de progresso
                logger.info(f"Ciclo #{self.current_cycle_id} completado em {cycle_duration:.2f}s "
                           f"({cycle_metrics.get('opportunities_detected', 0)} oportunidades, "
                           f"{cycle_metrics.get('trades_executed', 0)} operações)")
                
                # Aguardar intervalo entre ciclos
                time.sleep(MARKET_CHECK_INTERVAL)
                
        except KeyboardInterrupt:
            logger.info("Execução interrompida pelo usuário.")
        except Exception as e:
            logger.error(f"Erro durante execução: {e}")
        finally:
            # Salvar métricas finais
            self.performance_metrics.save_metrics()
            self.performance_metrics.generate_performance_report()
            
            # Exibir resumo
            self._print_session_summary()
    
    def _run_cycle(self):
        """
        Executa um ciclo completo de detecção e execução de oportunidades
        """
        try:
            # 1. Verificar mercados e detectar oportunidades
            opportunities = self.strategy.check_markets()
            
            # Registrar oportunidades detectadas
            num_opportunities = len(opportunities)
            self.session_stats['opportunities_detected'] += num_opportunities
            
            if num_opportunities > 0:
                logger.info(f"Detectadas {num_opportunities} oportunidades de arbitragem")
                
                # Armazenar dados para LSTM
                self._update_market_data(opportunities)
                
                # Registrar no analisador de performance
                for opp in opportunities:
                    opportunity_id = str(uuid.uuid4())
                    self.performance_metrics.record_opportunity(
                        opportunity_id=opportunity_id,
                        pair=opp.get('pair', 'unknown'),
                        exchange_a=opp.get('exchange_a', 'unknown'),
                        exchange_b=opp.get('exchange_b', 'unknown'),
                        spread=opp.get('spread', 0.0),
                        cycle_id=self.current_cycle_id
                    )
                    
                    # Adicionar ID para referência futura
                    opp['opportunity_id'] = opportunity_id
                    
                # Filtrar oportunidades com LSTM se disponível
                if self.lstm_predictor and self.lstm_enabled and len(self.market_data_buffer) > 0:
                    filtered_opportunities = self._filter_opportunities_with_lstm(opportunities)
                else:
                    filtered_opportunities = opportunities
                
                # 2. Executar operações para oportunidades filtradas
                if len(filtered_opportunities) > 0:
                    for opp in filtered_opportunities:
                        # Executar operação
                        trade_id = str(uuid.uuid4())
                        opportunity_id = opp.get('opportunity_id')
                        pair = opp.get('pair', 'unknown')
                        exchange_a = opp.get('exchange_a', 'unknown')
                        exchange_b = opp.get('exchange_b', 'unknown')
                        amount = opp.get('amount', 0.0)
                        expected_profit = opp.get('expected_profit', 0.0)
                        
                        # Registrar início da operação
                        self.performance_metrics.record_trade_start(
                            trade_id=trade_id,
                            opportunity_id=opportunity_id,
                            pair=pair,
                            exchange_a=exchange_a,
                            exchange_b=exchange_b,
                            amount=amount,
                            expected_profit=expected_profit,
                            cycle_id=self.current_cycle_id
                        )
                        
                        # Executar a operação através da estratégia base
                        try:
                            result = self.strategy.execute_arbitrage(opp)
                            
                            # Registrar conclusão da operação
                            success = result.get('success', False)
                            actual_profit = result.get('profit', 0.0)
                            error = result.get('error') if not success else None
                            
                            self.performance_metrics.record_trade_completion(
                                trade_id=trade_id,
                                success=success,
                                actual_profit=actual_profit,
                                error=error
                            )
                            
                            # Atualizar estatísticas
                            self.session_stats['trades_executed'] += 1
                            self.session_stats['total_profit'] += actual_profit
                            
                            logger.info(f"Operação {trade_id} concluída: sucesso={success}, "
                                       f"lucro={actual_profit:.6f} USDT")
                            
                        except Exception as e:
                            # Registrar erro na operação
                            self.performance_metrics.record_trade_completion(
                                trade_id=trade_id,
                                success=False,
                                actual_profit=0.0,
                                error=str(e)
                            )
                            logger.error(f"Erro ao executar operação: {e}")
        
        except Exception as e:
            logger.error(f"Erro durante ciclo de execução: {e}")
    
    def _update_market_data(self, opportunities):
        """
        Atualiza o buffer de dados de mercado para o LSTM
        
        Args:
            opportunities: Lista de oportunidades detectadas
        """
        if not self.lstm_enabled or not opportunities:
            return
        
        try:
            # Extrair dados relevantes para o modelo
            timestamp = datetime.now()
            
            # Calcular valores agregados
            avg_spread = np.mean([opp.get('spread', 0.0) for opp in opportunities])
            max_spread = np.max([opp.get('spread', 0.0) for opp in opportunities])
            
            # Obter dados da estratégia
            strategy_data = self.strategy.get_market_state()
            
            # Construir registro
            record = {
                'timestamp': timestamp,
                'spread': avg_spread,
                'max_spread': max_spread,
                'num_opportunities': len(opportunities),
                'volume_a': strategy_data.get('volume_a', 0.0),
                'volume_b': strategy_data.get('volume_b', 0.0),
                'entropy': strategy_data.get('entropy', 0.0),
                'fractal_dimension': strategy_data.get('fractal_dimension', 0.0),
                'volatility_a': strategy_data.get('volatility_a', 0.0),
                'volatility_b': strategy_data.get('volatility_b', 0.0),
            }
            
            # Adicionar ao buffer
            self.market_data_buffer = pd.concat([
                self.market_data_buffer, 
                pd.DataFrame([record])
            ], ignore_index=True)
            
            # Manter apenas os últimos history_window registros
            self.market_data_buffer = self.market_data_buffer.tail(self.history_window)
            
        except Exception as e:
            logger.error(f"Erro ao atualizar dados de mercado: {e}")
    
    def _filter_opportunities_with_lstm(self, opportunities):
        """
        Filtra oportunidades com base nas previsões do LSTM
        
        Args:
            opportunities: Lista de oportunidades detectadas
            
        Returns:
            Lista de oportunidades filtradas
        """
        if not self.lstm_enabled or not self.lstm_predictor or len(opportunities) == 0:
            return opportunities
        
        try:
            # Incrementar contador de previsões
            self.session_stats['lstm_predictions'] += 1
            
            # Obter previsão do modelo
            prediction = self.lstm_predictor.predict_optimal_entry(
                self.market_data_buffer,
                min_confidence=self.min_confidence
            )
            
            should_enter = prediction.get('should_enter', False)
            confidence = prediction.get('confidence', 0.0)
            
            if should_enter:
                self.session_stats['lstm_positive_predictions'] += 1
                logger.info(f"LSTM recomenda entrada com confiança {confidence:.2%}: {prediction.get('reason', '')}")
                return opportunities
            else:
                logger.info(f"LSTM não recomenda entrada: {prediction.get('reason', '')}")
                return []
                
        except Exception as e:
            logger.error(f"Erro ao filtrar oportunidades com LSTM: {e}")
            return opportunities
    
    def _print_session_summary(self):
        """
        Exibe um resumo da sessão atual
        """
        duration = datetime.now() - self.session_stats['start_time']
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        summary = f"""
=== RESUMO DA SESSÃO QUALIA ===
Duração: {int(hours)}h {int(minutes)}m {int(seconds)}s
Ciclos completados: {self.session_stats['cycles_completed']}
Oportunidades detectadas: {self.session_stats['opportunities_detected']}
Operações executadas: {self.session_stats['trades_executed']}
Lucro total: {self.session_stats['total_profit']:.6f} USDT
"""
        
        if self.lstm_enabled:
            predictions = self.session_stats['lstm_predictions']
            positive = self.session_stats['lstm_positive_predictions']
            approval_rate = (positive / predictions) if predictions > 0 else 0
            
            summary += f"""
Predições LSTM: {predictions}
Entradas aprovadas pelo LSTM: {positive} ({approval_rate:.1%})
"""

        summary += "=============================="
        
        logger.info(summary)
        print(summary)

def signal_handler(sig, frame):
    """
    Manipulador de sinais para terminar graciosamente
    """
    global running
    print("\nTerminando execução...")
    running = False

def main():
    parser = argparse.ArgumentParser(description='Executor de Estratégia Adaptativa QUALIA')
    parser.add_argument('--strategy', type=str, default=default_strategy,
                        help=f'Estratégia base a utilizar (padrão: {default_strategy})')
    parser.add_argument('--metrics-path', type=str, 
                        help='Caminho para salvar métricas de performance')
    parser.add_argument('--lstm-model', type=str, 
                        help='Caminho para o modelo LSTM')
    parser.add_argument('--test-mode', action='store_true', default=False,
                        help='Executar em modo de teste sem operações reais')
    parser.add_argument('--min-confidence', type=float, default=0.6,
                        help='Confiança mínima para aceitar previsões do LSTM')
    parser.add_argument('--history-window', type=int, default=100,
                        help='Tamanho da janela de histórico para análise')
    
    args = parser.parse_args()
    
    # Registrar handler de sinais
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Criar e iniciar executor
        runner = AdaptiveStrategyRunner(
            strategy_name=args.strategy,
            performance_metrics_path=args.metrics_path,
            lstm_model_path=args.lstm_model,
            test_mode=args.test_mode,
            min_confidence=args.min_confidence,
            history_window=args.history_window
        )
        
        runner.start()
        
    except Exception as e:
        logger.error(f"Erro fatal: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 