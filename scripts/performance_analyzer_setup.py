#!/usr/bin/env python
"""
Script para configurar o analisador de performance do QUALIA
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import matplotlib
import numpy as np

# Configurar matplotlib para ambiente sem GUI
matplotlib.use('Agg')

# Adicionar diretório principal ao path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

# Importar módulos QUALIA
from quantum_trading.metrics.performance_analyzer import PerformanceMetrics

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(root_dir, 'logs', 'performance_setup.log'))
    ]
)

logger = logging.getLogger("performance_setup")

def setup_performance_analyzer(metrics_dir=None, create_sample=False):
    """
    Configura o analisador de performance

    Args:
        metrics_dir: Diretório onde serão salvos os relatórios
        create_sample: Se True, cria um relatório de amostra
    """
    # Definir diretório de métricas
    if metrics_dir is None:
        metrics_dir = os.path.join(root_dir, 'metrics')
    
    # Criar diretório se não existir
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Criar diretório de logs se não existir
    os.makedirs(os.path.join(root_dir, 'logs'), exist_ok=True)
    
    logger.info(f"Configurando analisador de performance no diretório {metrics_dir}")
    
    # Inicializar analisador
    performance_analyzer = PerformanceMetrics(
        save_path=os.path.join(metrics_dir, 'qualia_performance.json')
    )
    
    # Criar relatório de amostra se solicitado
    if create_sample:
        logger.info("Gerando dados de amostra para o relatório de desempenho")
        
        # Simulação de ciclos
        for i in range(10):
            cycle_id = performance_analyzer.start_cycle()
            
            # Simular oportunidades
            num_opportunities = np.random.randint(5, 20)
            for j in range(num_opportunities):
                performance_analyzer.record_opportunity(
                    opportunity_id=f"opp_{i}_{j}",
                    pair="ETH-USDT",
                    exchange_a="kucoin",
                    exchange_b="kraken",
                    spread=np.random.uniform(0.001, 0.02),
                    cycle_id=cycle_id
                )
            
            # Simular trades
            num_trades = np.random.randint(0, min(5, num_opportunities))
            for j in range(num_trades):
                trade_id = f"trade_{i}_{j}"
                opportunity_id = f"opp_{i}_{j}"
                
                # Iniciar trade
                performance_analyzer.record_trade_start(
                    trade_id=trade_id,
                    opportunity_id=opportunity_id,
                    pair="ETH-USDT",
                    exchange_a="kucoin",
                    exchange_b="kraken",
                    amount=100.0,
                    expected_profit=1.5,
                    cycle_id=cycle_id
                )
                
                # Concluir trade
                success = np.random.random() > 0.3  # 70% de chance de sucesso
                actual_profit = np.random.uniform(0.8, 2.0) if success else np.random.uniform(-1.0, 0.0)
                
                performance_analyzer.record_trade_completion(
                    trade_id=trade_id,
                    success=success,
                    actual_profit=actual_profit,
                    error=None if success else "Timeout na execução"
                )
            
            # Finalizar ciclo
            performance_analyzer.end_cycle(cycle_id)
        
        # Salvar métricas
        performance_analyzer.save_metrics()
        
        # Gerar relatório
        report_path = performance_analyzer.generate_performance_report(
            output_path=os.path.join(metrics_dir, 'qualia_performance_report.html')
        )
        
        logger.info(f"Relatório de amostra gerado em {report_path}")
    
    return performance_analyzer

def main():
    parser = argparse.ArgumentParser(description='Configuração do Analisador de Performance QUALIA')
    parser.add_argument('--metrics-dir', type=str, 
                        help='Diretório para armazenar as métricas')
    parser.add_argument('--create-sample', action='store_true', default=False,
                        help='Criar um relatório de amostra para testes')
    
    args = parser.parse_args()
    
    try:
        setup_performance_analyzer(
            metrics_dir=args.metrics_dir,
            create_sample=args.create_sample
        )
        logger.info("Configuração do analisador de performance concluída com sucesso!")
    except Exception as e:
        logger.error(f"Erro durante a configuração: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 