#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para testar a conexão com as exchanges Kucoin e Kraken usando as credenciais do arquivo .env
"""

import os
import logging
import asyncio
import json
import time
from dotenv import load_dotenv
from quantum_trading.integration.exchange import Exchange

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# Função para imprimir dicionários de forma mais legível
def pretty_print_dict(data, prefix=""):
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                print(f"{prefix}{key}:")
                pretty_print_dict(value, prefix + "  ")
            else:
                print(f"{prefix}{key}: {value}")
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, (dict, list)):
                print(f"{prefix}[{i}]:")
                pretty_print_dict(item, prefix + "  ")
            else:
                print(f"{prefix}[{i}]: {item}")
    else:
        print(f"{prefix}{data}")

async def test_exchange_connection():
    """Testa a conexão com as exchanges configuradas"""
    
    logger.info("Iniciando teste de conexão com exchanges em modo sandbox...")
    
    # Testa Kucoin em modo sandbox
    logger.info("=== Testando Kucoin SANDBOX ===")
    kucoin = Exchange("Kucoin", maker_fee=0.0016, taker_fee=0.0024, use_sandbox=True)
    
    # Verificar se a conexão foi bem-sucedida
    if kucoin.exchange is None:
        logger.error("Falha ao inicializar conexão com Kucoin Sandbox")
    else:
        logger.info("Kucoin Sandbox inicializado com sucesso")
        
        # Tenta carregar mercados explicitamente
        try:
            logger.info("Carregando mercados da Kucoin...")
            markets = kucoin.exchange.load_markets()
            logger.info(f"Mercados carregados: {len(markets)}")
            # Lista alguns pares de trading como exemplo
            sample_pairs = list(markets.keys())[:5] if markets else []
            logger.info(f"Exemplos de pares: {sample_pairs}")
        except Exception as e:
            logger.error(f"Erro ao carregar mercados da Kucoin: {str(e)}")
        
        # Testa obtenção de saldo
        logger.info("Obtendo saldo em Kucoin Sandbox...")
        try:
            balance = kucoin.exchange.fetch_balance()
            logger.info("Saldo obtido com sucesso")
            
            # Verifica se há saldos
            if 'total' in balance and balance['total']:
                # Filtra saldos não-zero
                non_zero = {k: v for k, v in balance['total'].items() if v > 0}
                if non_zero:
                    logger.info(f"Saldos não-zero em Kucoin Sandbox:")
                    for currency, amount in non_zero.items():
                        logger.info(f"  {currency}: {amount}")
                else:
                    logger.info("Nenhum saldo não-zero encontrado em Kucoin Sandbox")
            else:
                logger.info("Formato de saldo não esperado ou saldo vazio")
        except Exception as e:
            logger.error(f"Erro ao obter saldo em Kucoin Sandbox: {str(e)}")
    
    # Testa obtenção de dados OHLCV
    logger.info("\nObtendo candles BTC/USDT em Kucoin Sandbox...")
    try:
        btc_candles = kucoin.get_ohlcv('BTC/USDT', '1h', 5)  # Reduzimos para 5 candles para clareza
        logger.info(f"Obtidos {len(btc_candles)} candles de BTC/USDT em Kucoin Sandbox")
        if btc_candles and len(btc_candles) > 0:
            logger.info("Últimos candles (timestamp, open, high, low, close, volume):")
            for candle in btc_candles[-2:]:  # Mostra os 2 últimos candles
                logger.info(f"  {candle}")
    except Exception as e:
        logger.error(f"Erro ao obter candles: {str(e)}")
    
    # Tenta criar uma ordem simulada no sandbox
    logger.info("\nCriando ordem de teste em Kucoin Sandbox...")
    try:
        order = kucoin.create_order(
            symbol='BTC/USDT',
            type='limit',
            side='buy',
            amount=0.001,
            price=20000.0  # Preço baixo para garantir que não será executado
        )
        logger.info("Detalhes da ordem:")
        if isinstance(order, dict):
            for key, value in order.items():
                if key != 'info':  # 'info' geralmente contém dados muito detalhados
                    logger.info(f"  {key}: {value}")
    except Exception as e:
        logger.error(f"Erro ao criar ordem: {str(e)}")
    
    # Testa Kraken (não tem sandbox, então usamos o mesmo ambiente)
    logger.info("\n=== Testando Kraken ===")
    kraken = Exchange("Kraken", maker_fee=0.0016, taker_fee=0.0026, use_sandbox=False)
    
    # Verificar se a conexão foi bem-sucedida
    if kraken.exchange is None:
        logger.error("Falha ao inicializar conexão com Kraken")
    else:
        logger.info("Kraken inicializado com sucesso")
        
        # Tenta carregar mercados explicitamente
        try:
            logger.info("Carregando mercados da Kraken...")
            markets = kraken.exchange.load_markets()
            logger.info(f"Mercados carregados: {len(markets)}")
            # Lista alguns pares de trading como exemplo
            sample_pairs = list(markets.keys())[:5] if markets else []
            logger.info(f"Exemplos de pares: {sample_pairs}")
        except Exception as e:
            logger.error(f"Erro ao carregar mercados da Kraken: {str(e)}")
        
        # Testa obtenção de saldo
        logger.info("Obtendo saldo em Kraken...")
        try:
            balance = kraken.exchange.fetch_balance()
            logger.info("Saldo obtido com sucesso")
            
            # Verifica se há saldos
            if 'total' in balance and balance['total']:
                # Filtra saldos não-zero
                non_zero = {k: v for k, v in balance['total'].items() if v > 0}
                if non_zero:
                    logger.info(f"Saldos não-zero em Kraken:")
                    for currency, amount in non_zero.items():
                        logger.info(f"  {currency}: {amount}")
                else:
                    logger.info("Nenhum saldo não-zero encontrado em Kraken")
            else:
                logger.info("Formato de saldo não esperado ou saldo vazio")
        except Exception as e:
            logger.error(f"Erro ao obter saldo em Kraken: {str(e)}")
    
    logger.info("\nTestes concluídos!")

if __name__ == "__main__":
    asyncio.run(test_exchange_connection()) 