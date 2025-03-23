#!/usr/bin/env python3
"""
Test Exchange Connection
=======================
Script para testar a conexão com as exchanges Kraken e KuCoin.
"""

import asyncio
import logging
import sys

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("test_exchange")

# Importar adaptadores de exchange
try:
    from quantum_trading.exchanges.kucoin_adapter import KuCoinAdapter
    from quantum_trading.exchanges.kraken_adapter import KrakenAdapter
    from quantum_trading.exchanges.factory import ExchangeFactory
except ImportError as e:
    logger.error(f"Erro ao importar módulos de exchange: {e}")
    logger.info("Verifique se os módulos estão no PYTHONPATH.")
    sys.exit(1)

async def test_exchange(exchange_id):
    """Testa uma conexão de exchange específica."""
    logger.info(f"Testando conexão com {exchange_id}...")
    
    try:
        # Cria adaptador de exchange usando a factory
        exchange = ExchangeFactory.create_exchange(exchange_id)
        
        if not exchange:
            logger.error(f"Não foi possível criar adaptador para {exchange_id}")
            return False
            
        # Testa obtenção de ticker
        logger.info(f"Obtendo ticker BTC/USDT de {exchange_id}...")
        ticker_price = await exchange.get_price_async("BTC/USDT")
        logger.info(f"Preço atual de BTC/USDT em {exchange_id}: ${ticker_price:.2f}")
        
        # Testa obtenção de saldo (não requer autenticação)
        logger.info(f"Tentando obter saldo de {exchange_id}...")
        balance = await exchange.get_balance_async("USDT")
        logger.info(f"Saldo de USDT em {exchange_id}: {balance}")
        
        # Testa obtenção de livro de ordens
        if hasattr(exchange, 'get_orderbook'):
            logger.info(f"Obtendo livro de ordens de BTC/USDT em {exchange_id}...")
            orderbook = exchange.get_orderbook("BTC/USDT", 1)
            if orderbook and orderbook.get("bids") and orderbook.get("asks"):
                logger.info(f"Melhor oferta de compra: ${orderbook['bids'][0][0]}")
                logger.info(f"Melhor oferta de venda: ${orderbook['asks'][0][0]}")
            
        # Fechando conexão assíncrona
        await exchange.close_async()
        logger.info(f"Teste de {exchange_id} concluído com sucesso")
        return True
        
    except Exception as e:
        logger.error(f"Erro durante teste de {exchange_id}: {e}")
        return False

async def main():
    """Função principal para testar as exchanges."""
    logger.info("Iniciando testes de conexão com exchanges...")
    
    # Lista de exchanges a testar
    exchanges = ["kucoin", "kraken"]
    
    # Executa testes para cada exchange
    results = {}
    for exchange_id in exchanges:
        success = await test_exchange(exchange_id)
        results[exchange_id] = "SUCCESS" if success else "FAILED"
    
    # Relatório de resultados
    logger.info("\n=== RELATÓRIO DE TESTES ===")
    for exchange_id, result in results.items():
        logger.info(f"{exchange_id.upper()}: {result}")
    
    # Testa a factory para criar todas as exchanges habilitadas
    logger.info("\nTestando ExchangeFactory.create_all_enabled_exchanges()...")
    try:
        enabled_exchanges = ExchangeFactory.create_all_enabled_exchanges()
        logger.info(f"Exchanges habilitadas: {[ex.exchange_id for ex in enabled_exchanges]}")
    except Exception as e:
        logger.error(f"Erro ao criar todas as exchanges habilitadas: {e}")

if __name__ == "__main__":
    # Configurar loop de eventos para Windows se necessário
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Executa os testes
    asyncio.run(main())
