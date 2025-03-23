#!/usr/bin/env python3
"""
Script principal para iniciar a estratégia WAVE do sistema QUALIA
"""
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
import argparse
import asyncio
import platform

# Configurar asyncio para Windows
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Adicionar diretório raiz ao PYTHONPATH
root_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(root_dir)
os.chdir(root_dir)

# Carregar variáveis de ambiente
load_dotenv(override=True)

# Configurar logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wave_strategy.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Logar informações de ambiente
logger.info(f"Diretório de trabalho: {os.getcwd()}")
logger.info(f"Arquivo .env presente: {os.path.exists('.env')}")
logger.info(f"Variáveis de ambiente carregadas:")
for key in ["KUCOIN_API_KEY", "KUCOIN_API_SECRET", "KUCOIN_API_PASSPHRASE", 
           "KRAKEN_API_KEY", "KRAKEN_API_SECRET"]:
    value = os.getenv(key)
    logger.info(f"  {key}: {'*' * len(value) if value else 'None'}")

# Import a função main do run_wave_strategy
from quantum_trading.run_wave_strategy import main as run_wave_strategy

async def initialize_exchanges(exchange_ids):
    """Inicializa as exchanges assincronamente"""
    from quantum_trading.market_api import MarketAPI
    
    exchanges = []
    for exchange_id in exchange_ids:
        try:
            exchange = MarketAPI(exchange_id=exchange_id)
            await exchange.initialize()
            exchanges.append(exchange)
            logger.info(f"Exchange {exchange_id} inicializada com sucesso")
        except Exception as e:
            logger.error(f"Erro ao inicializar exchange {exchange_id}: {e}")
    
    return exchanges

def main():
    """Função principal para iniciar a estratégia WAVE"""
    parser = argparse.ArgumentParser(description='Iniciar estratégia WAVE')
    parser.add_argument('--optimize', action='store_true', help='Otimizar parâmetros')
    parser.add_argument('--fetch-balances', action='store_true', help='Buscar saldos reais')
    parser.add_argument('--kucoin-capital', type=float, default=50.0, help='Capital inicial KuCoin')
    parser.add_argument('--kraken-capital', type=float, default=50.0, help='Capital inicial Kraken')
    parser.add_argument('--config', type=str, help='Arquivo de configuração')
    parser.add_argument('--days', type=int, default=7, help='Número de dias para análise histórica')
    
    args = parser.parse_args()
    
    logger.info("Iniciando QUALIA - Sistema de Trading Quântico")
    logger.info("Componentes disponíveis:")
    logger.info("- Estratégia WAVE")
    logger.info("- Otimização de Parâmetros")
    logger.info("- Análise de Mercado")
    
    capital_per_exchange = {
        'kucoin': args.kucoin_capital,
        'kraken': args.kraken_capital
    }
    
    try:
        # Executar loop assíncrono
        asyncio.run(run_wave_strategy(
            config_path=args.config,
            optimize=args.optimize,
            days=args.days,
            fetch_balances=args.fetch_balances,
            capital_per_exchange=capital_per_exchange
        ))
    except KeyboardInterrupt:
        logger.info("Interrompendo estratégia WAVE...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Erro ao executar estratégia WAVE: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 