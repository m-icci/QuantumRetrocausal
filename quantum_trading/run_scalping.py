"""
Script para executar o sistema de scalping
"""

import asyncio
import os
import logging
import argparse
from datetime import datetime, timedelta
from .scalping import ScalpingSystem

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('run_scalping.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def main():
    """Função principal"""
    try:
        # Configura argumentos
        parser = argparse.ArgumentParser(description='Sistema de Scalping QUALIA')
        parser.add_argument('--mode', choices=['real', 'simulated'], default='simulated',
                          help='Modo de operação (real ou simulado)')
        parser.add_argument('--start-date', type=str, 
                          default=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                          help='Data inicial para simulação (YYYY-MM-DD)')
        parser.add_argument('--end-date', type=str,
                          default=datetime.now().strftime('%Y-%m-%d'),
                          help='Data final para simulação (YYYY-MM-DD)')
        parser.add_argument('--initial-balance', type=float, default=10000,
                          help='Saldo inicial para simulação')
        args = parser.parse_args()
        
        # Carrega configuração
        config = {
            'exchange': {
                'name': 'binance',
                'api_key': os.getenv('EXCHANGE_API_KEY'),
                'api_secret': os.getenv('EXCHANGE_API_SECRET')
            },
            'trading': {
                'symbol': 'BTC/USDT',
                'timeframe': '1m',
                'initial_balance': args.initial_balance,
                'mode': args.mode
            },
            'risk': {
                'max_position_size': 0.1,
                'max_daily_loss': 0.02,
                'max_drawdown': 0.05
            },
            'scalping': {
                'min_profit_threshold': 0.0005,  # 0.05%
                'max_loss_threshold': 0.0003,    # 0.03%
                'max_position_time': 300,        # 5 minutos
                'min_volume_threshold': 1000,    # Volume mínimo
                'max_spread_threshold': 0.0002,  # 0.02%
                'exchange_fee': 0.0004,          # 0.04%
                'slippage': 0.0001,             # 0.01%
                'min_trade_size': 0.001         # Tamanho mínimo do trade
            },
            'simulation': {
                'start_date': args.start_date,
                'end_date': args.end_date,
                'initial_balance': args.initial_balance
            }
        }
        
        # Verifica variáveis de ambiente apenas para modo real
        if args.mode == 'real':
            if not os.getenv('EXCHANGE_API_KEY') or not os.getenv('EXCHANGE_API_SECRET'):
                logger.error("Variáveis de ambiente não configuradas para trading real")
                return
        else:
            logger.info(f"Iniciando simulação de {args.start_date} até {args.end_date}")
            logger.info(f"Saldo inicial: ${args.initial_balance:,.2f}")
            
        # Inicia sistema
        logger.info(f"Iniciando sistema de scalping em modo {args.mode}")
        scalping_system = ScalpingSystem(config)
        await scalping_system.run()
        
    except Exception as e:
        logger.error(f"Erro na execução principal: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 