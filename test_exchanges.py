#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import asyncio
import logging
from dotenv import load_dotenv
from quantum_trading.market_api import MarketAPI

# Configuração do logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_exchanges')

async def test_exchange(exchange_id: str):
    """Testa a obtenção de saldos em uma exchange"""
    logger.info(f"\n===== TESTANDO EXCHANGE {exchange_id.upper()} =====")
    
    # Usar context manager para garantir fechamento correto da sessão
    async with MarketAPI(exchange_id) as api:
        # Testar obtenção de saldos
        currencies = ['USDT', 'BTC', 'ETH', 'DOGE', 'SOL']
        total_value = 0.0
        
        for currency in currencies:
            try:
                balance = await api.get_balance(currency)
                logger.info(f"Saldo de {currency}: {balance}")
                
                if balance > 0:
                    if currency == 'USDT':
                        value_usdt = balance
                    else:
                        ticker = await api.get_ticker(f"{currency}/USDT")
                        if ticker and 'price' in ticker:
                            price = ticker['price']
                            value_usdt = balance * price
                            logger.info(f"Valor em USDT: {value_usdt:.2f} (preço: {price})")
                        else:
                            logger.warning(f"Não foi possível obter preço para {currency}")
                            value_usdt = 0
                            
                    total_value += value_usdt
                    
            except Exception as e:
                logger.error(f"Erro ao obter saldo de {currency}: {str(e)}")
        
        logger.info(f"Valor total em {exchange_id}: {total_value:.2f} USDT")
        return total_value

async def main():
    """Função principal"""
    # Carregar variáveis de ambiente
    load_dotenv()
    
    # Verificar credenciais
    exchanges = ['kucoin', 'kraken']
    total_portfolio = 0.0
    
    for exchange in exchanges:
        try:
            value = await test_exchange(exchange)
            total_portfolio += value
        except Exception as e:
            logger.error(f"Erro ao testar {exchange}: {str(e)}")
    
    logger.info(f"\n===== RESUMO =====")
    logger.info(f"Valor total do portfólio: {total_portfolio:.2f} USDT")

if __name__ == '__main__':
    # Usar asyncio.run para executar o código assíncrono
    asyncio.run(main()) 