"""
Performance Analyzer para o Sistema QUALIA
Monitora e analisa métricas de desempenho do sistema de trading quântico adaptativo
"""

import numpy as np
import pandas as pd
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
import os
import matplotlib.pyplot as plt
from collections import deque

logger = logging.getLogger("performance_analyzer")

class PerformanceMetrics:
    """
    Classe responsável por capturar, analisar e reportar métricas de desempenho
    do sistema QUALIA
    """
    
    def __init__(self, max_history: int = 1000, save_path: str = "metrics/qualia_performance.json"):
        """
        Inicializa o analisador de performance
        
        Args:
            max_history: Número máximo de operações a manter no histórico
            save_path: Caminho para salvar as métricas
        """
        # Configurações
        self.max_history = max_history
        self.save_path = save_path
        
        # Métricas de oportunidades
        self.opportunities_history = deque(maxlen=max_history)
        self.opportunities_count = 0
        self.opportunities_per_cycle = []
        
        # Métricas de operações
        self.trades_history = deque(maxlen=max_history)
        self.trade_count = 0
        self.successful_trades = 0
        self.failed_trades = 0
        
        # Métricas de tempo
        self.detection_timestamps = {}
        self.execution_timestamps = {}
        self.completion_timestamps = {}
        
        # Métricas financeiras
        self.profit_history = []
        self.cumulative_profit = 0.0
        self.best_trade = 0.0
        self.worst_trade = 0.0
        
        # Métricas por ciclo
        self.cycle_metrics = []
        self.current_cycle = 0
        
        # Métricas de eficiência
        self.detection_to_execution_times = []
        self.execution_to_completion_times = []
        self.total_operation_times = []
        
        logger.info("Analisador de Performance QUALIA inicializado")
    
    def start_cycle(self) -> int:
        """
        Inicia um novo ciclo de análise
        
        Returns:
            ID do ciclo
        """
        self.current_cycle += 1
        timestamp = datetime.now()
        
        cycle_data = {
            "cycle_id": self.current_cycle,
            "start_time": timestamp,
            "opportunities_detected": 0,
            "trades_executed": 0,
            "successful_trades": 0,
            "profit": 0.0,
            "end_time": None,
            "duration": None
        }
        
        self.cycle_metrics.append(cycle_data)
        logger.debug(f"Iniciado ciclo #{self.current_cycle}")
        
        return self.current_cycle
    
    def end_cycle(self, cycle_id: int) -> Dict[str, Any]:
        """
        Finaliza um ciclo de análise
        
        Args:
            cycle_id: ID do ciclo a finalizar
            
        Returns:
            Métricas do ciclo
        """
        timestamp = datetime.now()
        
        # Encontrar o ciclo correspondente
        cycle_index = None
        for i, cycle in enumerate(self.cycle_metrics):
            if cycle["cycle_id"] == cycle_id:
                cycle_index = i
                break
        
        if cycle_index is None:
            logger.warning(f"Ciclo #{cycle_id} não encontrado")
            return {}
        
        # Atualizar métricas do ciclo
        cycle = self.cycle_metrics[cycle_index]
        cycle["end_time"] = timestamp
        start_time = cycle["start_time"]
        if start_time:
            cycle["duration"] = (timestamp - start_time).total_seconds()
        
        # Calcular métricas por segundo
        duration = cycle["duration"] or 0.001  # Evitar divisão por zero
        cycle["opportunities_per_second"] = cycle["opportunities_detected"] / duration
        cycle["trades_per_second"] = cycle["trades_executed"] / duration
        
        logger.debug(f"Finalizado ciclo #{cycle_id} com duração de {cycle['duration']:.2f}s")
        
        # Armazenar métricas de oportunidades por ciclo
        self.opportunities_per_cycle.append(cycle["opportunities_detected"])
        
        # Manter apenas os últimos max_history ciclos
        if len(self.opportunities_per_cycle) > self.max_history:
            self.opportunities_per_cycle = self.opportunities_per_cycle[-self.max_history:]
        
        return cycle
    
    def record_opportunity(self, 
                         opportunity_id: str,
                         pair: str,
                         exchange_a: str,
                         exchange_b: str,
                         spread: float,
                         cycle_id: int) -> None:
        """
        Registra uma oportunidade detectada
        
        Args:
            opportunity_id: Identificador único da oportunidade
            pair: Par de trading
            exchange_a: Primeira exchange
            exchange_b: Segunda exchange
            spread: Spread detectado
            cycle_id: ID do ciclo atual
        """
        timestamp = datetime.now()
        
        opportunity = {
            "id": opportunity_id,
            "timestamp": timestamp,
            "pair": pair,
            "exchange_a": exchange_a,
            "exchange_b": exchange_b,
            "spread": spread,
            "cycle_id": cycle_id
        }
        
        self.opportunities_history.append(opportunity)
        self.opportunities_count += 1
        self.detection_timestamps[opportunity_id] = timestamp
        
        # Atualizar métricas do ciclo
        for cycle in self.cycle_metrics:
            if cycle["cycle_id"] == cycle_id:
                cycle["opportunities_detected"] += 1
                break
        
        logger.debug(f"Registrada oportunidade {opportunity_id} com spread {spread:.6f}")
    
    def record_trade_start(self,
                         trade_id: str,
                         opportunity_id: str,
                         pair: str,
                         exchange_a: str,
                         exchange_b: str, 
                         amount: float,
                         expected_profit: float,
                         cycle_id: int) -> None:
        """
        Registra o início de uma operação
        
        Args:
            trade_id: Identificador único da operação
            opportunity_id: ID da oportunidade relacionada
            pair: Par de trading
            exchange_a: Primeira exchange (compra)
            exchange_b: Segunda exchange (venda)
            amount: Montante da operação
            expected_profit: Lucro esperado
            cycle_id: ID do ciclo atual
        """
        timestamp = datetime.now()
        
        trade = {
            "id": trade_id,
            "opportunity_id": opportunity_id,
            "start_timestamp": timestamp,
            "pair": pair,
            "exchange_a": exchange_a,
            "exchange_b": exchange_b,
            "amount": amount,
            "expected_profit": expected_profit,
            "cycle_id": cycle_id,
            "status": "executing",
            "actual_profit": None,
            "completion_timestamp": None,
            "execution_time": None
        }
        
        self.trades_history.append(trade)
        self.trade_count += 1
        self.execution_timestamps[trade_id] = timestamp
        
        # Calcular tempo entre detecção e execução
        if opportunity_id in self.detection_timestamps:
            detection_time = self.detection_timestamps[opportunity_id]
            execution_delay = (timestamp - detection_time).total_seconds()
            self.detection_to_execution_times.append(execution_delay)
            logger.debug(f"Tempo entre detecção e execução: {execution_delay:.3f}s")
        
        # Atualizar métricas do ciclo
        for cycle in self.cycle_metrics:
            if cycle["cycle_id"] == cycle_id:
                cycle["trades_executed"] += 1
                break
        
        logger.debug(f"Iniciada operação {trade_id} para oportunidade {opportunity_id}")
    
    def record_trade_completion(self,
                              trade_id: str,
                              success: bool,
                              actual_profit: float,
                              error: Optional[str] = None) -> Dict[str, Any]:
        """
        Registra a conclusão de uma operação
        
        Args:
            trade_id: Identificador único da operação
            success: Se a operação foi bem-sucedida
            actual_profit: Lucro real obtido
            error: Mensagem de erro, se houver
            
        Returns:
            Métricas da operação
        """
        timestamp = datetime.now()
        
        # Encontrar a operação correspondente
        trade_index = None
        for i, trade in enumerate(self.trades_history):
            if trade["id"] == trade_id:
                trade_index = i
                break
        
        if trade_index is None:
            logger.warning(f"Operação {trade_id} não encontrada")
            return {}
        
        # Atualizar dados da operação
        trade = self.trades_history[trade_index]
        trade["status"] = "completed" if success else "failed"
        trade["actual_profit"] = actual_profit
        trade["completion_timestamp"] = timestamp
        trade["error"] = error
        
        start_time = trade["start_timestamp"]
        if start_time:
            trade["execution_time"] = (timestamp - start_time).total_seconds()
        
        self.completion_timestamps[trade_id] = timestamp
        
        # Atualizar contadores
        if success:
            self.successful_trades += 1
            # Atualizar métricas do ciclo
            cycle_id = trade.get("cycle_id")
            if cycle_id:
                for cycle in self.cycle_metrics:
                    if cycle["cycle_id"] == cycle_id:
                        cycle["successful_trades"] += 1
                        cycle["profit"] += actual_profit
                        break
        else:
            self.failed_trades += 1
        
        # Atualizar histórico de lucros
        self.profit_history.append(actual_profit)
        self.cumulative_profit += actual_profit
        
        # Atualizar recordes
        if actual_profit > self.best_trade:
            self.best_trade = actual_profit
        if actual_profit < self.worst_trade:
            self.worst_trade = actual_profit
        
        # Calcular tempo entre execução e conclusão
        if trade_id in self.execution_timestamps:
            execution_time = self.execution_timestamps[trade_id]
            completion_time = (timestamp - execution_time).total_seconds()
            self.execution_to_completion_times.append(completion_time)
            
            # Calcular tempo total da operação
            opportunity_id = trade.get("opportunity_id")
            if opportunity_id and opportunity_id in self.detection_timestamps:
                detection_time = self.detection_timestamps[opportunity_id]
                total_time = (timestamp - detection_time).total_seconds()
                self.total_operation_times.append(total_time)
                logger.debug(f"Tempo total da operação: {total_time:.3f}s")
        
        logger.debug(f"Operação {trade_id} concluída: sucesso={success}, lucro={actual_profit:.6f} USDT")
        
        return trade
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Obtém um resumo das métricas de desempenho
        
        Returns:
            Resumo das métricas
        """
        # Calcular métricas de oportunidades
        avg_opportunities_per_cycle = np.mean(self.opportunities_per_cycle) if self.opportunities_per_cycle else 0
        
        # Calcular métricas de operações
        success_rate = (self.successful_trades / self.trade_count) if self.trade_count > 0 else 0
        
        # Calcular métricas de tempo
        avg_detection_to_execution = np.mean(self.detection_to_execution_times) if self.detection_to_execution_times else 0
        avg_execution_to_completion = np.mean(self.execution_to_completion_times) if self.execution_to_completion_times else 0
        avg_total_operation_time = np.mean(self.total_operation_times) if self.total_operation_times else 0
        
        # Calcular métricas financeiras
        avg_profit_per_trade = np.mean(self.profit_history) if self.profit_history else 0
        
        # Construir resumo
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_cycles": self.current_cycle,
            "total_opportunities": self.opportunities_count,
            "total_trades": self.trade_count,
            "successful_trades": self.successful_trades,
            "failed_trades": self.failed_trades,
            "success_rate": success_rate,
            "avg_opportunities_per_cycle": avg_opportunities_per_cycle,
            "avg_detection_to_execution_time": avg_detection_to_execution,
            "avg_execution_to_completion_time": avg_execution_to_completion,
            "avg_total_operation_time": avg_total_operation_time,
            "cumulative_profit": self.cumulative_profit,
            "avg_profit_per_trade": avg_profit_per_trade,
            "best_trade": self.best_trade,
            "worst_trade": self.worst_trade
        }
        
        return summary
    
    def save_metrics(self) -> bool:
        """
        Salva as métricas em arquivo
        
        Returns:
            True se salvou com sucesso, False caso contrário
        """
        try:
            # Criar diretório se não existir
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            
            # Obter resumo das métricas
            summary = self.get_summary()
            
            # Salvar em arquivo
            with open(self.save_path, 'w') as f:
                json.dump(summary, f, indent=4)
            
            logger.info(f"Métricas salvas em {self.save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao salvar métricas: {e}")
            return False
    
    def generate_performance_report(self, output_path: str = "metrics/qualia_performance_report.html") -> str:
        """
        Gera um relatório visual de desempenho
        
        Args:
            output_path: Caminho para salvar o relatório
            
        Returns:
            Caminho para o relatório gerado
        """
        try:
            # Criar diretório se não existir
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Obter resumo das métricas
            summary = self.get_summary()
            
            # Criar gráficos e visualizações
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            
            # Gráfico 1: Oportunidades por ciclo
            if self.opportunities_per_cycle:
                axs[0, 0].plot(self.opportunities_per_cycle)
                axs[0, 0].set_title('Oportunidades por Ciclo')
                axs[0, 0].set_xlabel('Ciclo')
                axs[0, 0].set_ylabel('Número de Oportunidades')
                axs[0, 0].grid(True)
            
            # Gráfico 2: Lucro por operação
            if self.profit_history:
                axs[0, 1].plot(self.profit_history)
                axs[0, 1].set_title('Lucro por Operação')
                axs[0, 1].set_xlabel('Operação')
                axs[0, 1].set_ylabel('Lucro (USDT)')
                axs[0, 1].grid(True)
            
            # Gráfico 3: Lucro cumulativo
            if self.profit_history:
                cumulative = np.cumsum(self.profit_history)
                axs[1, 0].plot(cumulative)
                axs[1, 0].set_title('Lucro Cumulativo')
                axs[1, 0].set_xlabel('Operação')
                axs[1, 0].set_ylabel('Lucro Cumulativo (USDT)')
                axs[1, 0].grid(True)
            
            # Gráfico 4: Tempos de operação
            if self.total_operation_times:
                axs[1, 1].hist(self.total_operation_times, bins=20)
                axs[1, 1].set_title('Distribuição de Tempos de Operação')
                axs[1, 1].set_xlabel('Tempo (s)')
                axs[1, 1].set_ylabel('Frequência')
                axs[1, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig('metrics/performance_graphs.png')
            
            # Criar relatório HTML
            html_content = f"""<!DOCTYPE html>
            <html>
            <head>
                <title>QUALIA Performance Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #2c3e50; }}
                    h2 {{ color: #3498db; }}
                    .metrics {{ display: flex; flex-wrap: wrap; }}
                    .metric-card {{ 
                        background-color: #f9f9f9; 
                        border-radius: 5px; 
                        padding: 15px; 
                        margin: 10px; 
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                        width: 300px;
                    }}
                    .metric-value {{ 
                        font-size: 24px; 
                        font-weight: bold; 
                        color: #2980b9; 
                    }}
                    .graphs {{ margin-top: 30px; }}
                </style>
            </head>
            <body>
                <h1>QUALIA Performance Report</h1>
                <p>Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>Métricas Gerais</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <h3>Total de Ciclos</h3>
                        <div class="metric-value">{summary['total_cycles']}</div>
                    </div>
                    <div class="metric-card">
                        <h3>Total de Oportunidades</h3>
                        <div class="metric-value">{summary['total_opportunities']}</div>
                    </div>
                    <div class="metric-card">
                        <h3>Total de Operações</h3>
                        <div class="metric-value">{summary['total_trades']}</div>
                    </div>
                    <div class="metric-card">
                        <h3>Taxa de Sucesso</h3>
                        <div class="metric-value">{summary['success_rate']:.2%}</div>
                    </div>
                </div>
                
                <h2>Métricas de Tempo</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <h3>Tempo Médio Detecção-Execução</h3>
                        <div class="metric-value">{summary['avg_detection_to_execution_time']:.3f}s</div>
                    </div>
                    <div class="metric-card">
                        <h3>Tempo Médio Execução-Conclusão</h3>
                        <div class="metric-value">{summary['avg_execution_to_completion_time']:.3f}s</div>
                    </div>
                    <div class="metric-card">
                        <h3>Tempo Médio Total</h3>
                        <div class="metric-value">{summary['avg_total_operation_time']:.3f}s</div>
                    </div>
                </div>
                
                <h2>Métricas Financeiras</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <h3>Lucro Cumulativo</h3>
                        <div class="metric-value">{summary['cumulative_profit']:.6f} USDT</div>
                    </div>
                    <div class="metric-card">
                        <h3>Lucro Médio por Operação</h3>
                        <div class="metric-value">{summary['avg_profit_per_trade']:.6f} USDT</div>
                    </div>
                    <div class="metric-card">
                        <h3>Melhor Operação</h3>
                        <div class="metric-value">{summary['best_trade']:.6f} USDT</div>
                    </div>
                    <div class="metric-card">
                        <h3>Pior Operação</h3>
                        <div class="metric-value">{summary['worst_trade']:.6f} USDT</div>
                    </div>
                </div>
                
                <div class="graphs">
                    <h2>Visualizações</h2>
                    <img src="performance_graphs.png" alt="Performance Graphs" width="100%">
                </div>
            </body>
            </html>"""
            
            with open(output_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Relatório de desempenho gerado em {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Erro ao gerar relatório de desempenho: {e}")
            return "" 