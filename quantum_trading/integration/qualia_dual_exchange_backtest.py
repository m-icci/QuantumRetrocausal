#!/usr/bin/env python3
"""
QUALIA Dual-Exchange Backtest
=============================
Wrapper para o sistema unificado de trading quântico.

Este script é mantido para compatibilidade com fluxos de trabalho existentes,
mas redireciona toda a funcionalidade para o sistema unificado em 
quantum_trading/run_integrated_system.py.

Para executar backtests de arbitragem entre duas exchanges, use:
python qualia_dual_exchange_backtest.py --mode dual_backtest [outras opções]
"""

import os
import sys
import asyncio
import argparse
import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("backtest.log")
    ]
)
logger = logging.getLogger("dual_exchange_backtest")

# Adiciona diretório raiz ao PYTHONPATH se necessário
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Importa o sistema unificado
try:
    from quantum_trading.run_integrated_system import UnifiedTradingSystem
except ImportError as e:
    logger.error(f"Erro ao importar sistema unificado: {e}")
    logger.info("Certifique-se de que o diretório quantum_trading está no PYTHONPATH")
    sys.exit(1)

async def main():
    """Função principal que configura e executa o sistema unificado no modo dual_backtest"""
    parser = argparse.ArgumentParser(description="QUALIA Dual-Exchange Backtest")
    parser.add_argument("--config", type=str, help="Caminho para arquivo de configuração")
    parser.add_argument("--data-dir", type=str, help="Diretório com dados históricos")
    parser.add_argument("--output-dir", type=str, help="Diretório para resultados")
    parser.add_argument("--cycles", type=int, help="Número de ciclos a executar")
    parser.add_argument("--pair", type=str, help="Par de trading (ex: BTC/USDT)")
    parser.add_argument("--window", type=int, help="Tamanho da janela de análise")
    parser.add_argument("--mode", type=str, default="dual_backtest", 
                      choices=["dual_backtest", "backtest", "simulated", "real"],
                      help="Modo de operação")
    
    args = parser.parse_args()
    
    # Se o usuário não especificou explicitamente 'dual_backtest' como modo,
    # verificamos se está usando este script sem --mode, o que implica dual_backtest
    if args.mode != "dual_backtest" and "--mode" not in sys.argv:
        logger.info("Usando este script implica no modo dual_backtest, mas foi especificado "
                   f"modo={args.mode}. Isso pode causar comportamento inesperado.")
    
    # Cria configuração base
    config = {
        "system": {
            "mode": args.mode,
            "data_dir": args.data_dir or "./data",
            "output_dir": args.output_dir or "./output"
        },
        "backtest": {}
    }
    
    # Adiciona opções específicas
    if args.cycles:
        config["backtest"]["cycles"] = args.cycles
    if args.pair:
        config["backtest"]["pair"] = args.pair
    if args.window:
        config["backtest"]["window_size"] = args.window
    
    try:
        # Inicializa sistema unificado
        if args.config:
            logger.info(f"Inicializando com arquivo de configuração: {args.config}")
            system = UnifiedTradingSystem(args.config)
            
            # Sobrescreve configurações com argumentos da linha de comando
            if "system" not in system.config:
                system.config["system"] = {}
            system.config["system"]["mode"] = args.mode
            
            if args.data_dir:
                system.config["system"]["data_dir"] = args.data_dir
            if args.output_dir:
                system.config["system"]["output_dir"] = args.output_dir
            
            if "backtest" not in system.config:
                system.config["backtest"] = {}
            
            if args.cycles:
                system.config["backtest"]["cycles"] = args.cycles
            if args.pair:
                system.config["backtest"]["pair"] = args.pair
            if args.window:
                system.config["backtest"]["window_size"] = args.window
        else:
            logger.info("Inicializando com configuração padrão")
            system = UnifiedTradingSystem(None)
            system.config.update(config)
        
        # Define explicitamente o modo
        system.mode = args.mode
        
        # Inicializa e executa o sistema
        logger.info(f"Iniciando sistema unificado em modo: {system.mode}")
        if await system.initialize():
            results = await system.run()
            
            if system.mode in ["backtest", "dual_backtest"] and isinstance(results, dict) and "summary" in results:
                logger.info("\n" + "="*50)
                logger.info("RESUMO DO BACKTEST:")
                for key, value in results["summary"].items():
                    if isinstance(value, float):
                        logger.info(f"  {key}: {value:.6f}")
                    else:
                        logger.info(f"  {key}: {value}")
                logger.info("="*50)
        else:
            logger.error("Falha ao inicializar o sistema")
            
    except KeyboardInterrupt:
        logger.info("Sistema interrompido pelo usuário")
    except Exception as e:
        logger.error(f"Erro na execução: {str(e)}", exc_info=True)
    finally:
        if 'system' in locals():
            await system.stop()

if __name__ == "__main__":
    # Configura política de loop de eventos para Windows
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Executa o sistema
    asyncio.run(main()) 