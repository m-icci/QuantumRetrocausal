#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script para executar o sistema integrado de scalping quântico.

Este script executa o Sistema Integrado de Scalping Quântico (QUALIA),
permitindo diferentes modos de operação.
"""

# Importações padrão da biblioteca
import sys
import os
import argparse
import time
from datetime import datetime
from pathlib import Path

# Configurar logging diretamente usando o módulo Python padrão
# Evite importar o módulo logging personalizado do projeto
import importlib
if 'logging' in sys.modules:
    del sys.modules['logging']  # Remover qualquer importação anterior
import logging as python_logging  # Renomear para evitar conflitos

# Configurar logging básico
python_logging.basicConfig(
    level=python_logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[python_logging.StreamHandler()]
)
logger = python_logging.getLogger(__name__)

# Criar diretórios necessários antes de importar outros módulos
logger.info("Criando diretórios necessários...")
os.makedirs('logs', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('output', exist_ok=True)

# Importações do sistema QUALIA
try:
    # Importar sistema unificado diretamente, não via pacote
    sys.path.insert(0, os.path.abspath('.'))
    from quantum_trading.run_integrated_system import UnifiedTradingSystem
    logger.info("Módulos importados com sucesso")
except ImportError as e:
    logger.error(f"Erro ao importar dependências: {e}")
    sys.exit(1)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='QUALIA Integrated Quantum Scalping System')
    parser.add_argument('--mode', type=str, default='simulated',
                      choices=['simulated', 'real', 'backtest', 'dual_backtest'],
                      help='Modo de operação (padrão: simulated)')
    parser.add_argument('--config', type=str, default='config/scalping_config.json',
                      help='Caminho para arquivo de configuração (padrão: config/scalping_config.json)')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', 
                      help='Par de trading (padrão: BTCUSDT)')
    parser.add_argument('--exchange', type=str, default='kucoin',
                      help='Exchange para trading (padrão: kucoin)')
    parser.add_argument('--debug', action='store_true',
                      help='Ativar modo de debug')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_arguments()
    
    # Configurar nível de log baseado em argumentos
    log_level = python_logging.DEBUG if args.debug else python_logging.INFO
    python_logging.getLogger().setLevel(log_level)
    
    logger.info(f"🚀 Iniciando QUALIA em modo {args.mode}")
    logger.info(f"⚙️ Usando configuração: {args.config}")
    logger.info(f"💱 Trading par: {args.symbol} em {args.exchange}")
    
    try:
        # Criar sistema unificado
        logger.info("Inicializando sistema unificado...")
        unified_system = UnifiedTradingSystem(
            config_path=args.config,
            mode=args.mode,
            symbol=args.symbol,
            exchange_id=args.exchange
        )
        
        # Executar sistema com base no modo selecionado
        logger.info(f"Executando sistema em modo {args.mode}...")
        if args.mode == 'simulated':
            result = unified_system.run_simulation()
        elif args.mode == 'real':
            result = unified_system.run_real_trading()
        elif args.mode == 'backtest':
            result = unified_system.run_backtest()
        elif args.mode == 'dual_backtest':
            result = unified_system.run_dual_backtest()
        else:
            logger.error(f"Modo não reconhecido: {args.mode}")
            return 1
            
        logger.info("✅ Sistema executado com sucesso!")
        if isinstance(result, dict) and not result.get('success', True):
            logger.error(f"❌ Falha na execução: {result.get('error', 'Erro desconhecido')}")
            return 1
        return 0
        
    except KeyboardInterrupt:
        logger.info("👋 Sistema interrompido pelo usuário")
        return 0
    except Exception as e:
        logger.error(f"❌ Erro ao executar sistema: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 