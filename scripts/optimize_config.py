#!/usr/bin/env python3
"""
Script para otimização automática de configurações baseada em dados históricos
Analisa dados de mercado para determinar parâmetros ótimos para a estratégia WAVE
"""

import os
import sys
import json
import logging
import asyncio
import argparse
from datetime import datetime
from pathlib import Path

# Adicionar diretório pai ao path para importar módulos
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantum_trading.market_api import MarketAPI
from quantum_trading.config_optimizer import optimize_config

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/config_optimizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)

logger = logging.getLogger("optimizer_script")

async def fetch_exchange_balances(simulation_mode=False, exchange_ids=None):
    """Obtém os saldos USDT disponíveis em cada exchange via API"""
    if exchange_ids is None:
        exchange_ids = ["kucoin", "kraken"]
    
    logger.info("Obtendo saldos disponíveis via API...")
    capital_per_exchange = {}
    
    for ex_id in exchange_ids:
        try:
            exchange = MarketAPI(simulation_mode=simulation_mode, exchange_id=ex_id)
            if simulation_mode:
                # Em modo de simulação, usar valores de simulação
                balance = await exchange.get_balance("USDT")
                capital_per_exchange[ex_id] = float(balance) if balance is not None else 100.0
            else:
                # Em modo real, obter saldo real da API
                balance = await exchange.get_balance("USDT")
                capital_per_exchange[ex_id] = float(balance) if balance is not None else 0.0
            
            logger.info(f"Saldo em {ex_id}: {capital_per_exchange[ex_id]:.2f} USDT")
        except Exception as e:
            logger.error(f"Erro ao obter saldo em {ex_id}: {e}")
            capital_per_exchange[ex_id] = 0.0
    
    return capital_per_exchange

async def main():
    """Função principal para execução do script"""
    parser = argparse.ArgumentParser(description="Otimizador de configuração baseado em dados históricos")
    parser.add_argument("--config", type=str, default="config/wave_config.json", help="Caminho para o arquivo de configuração")
    parser.add_argument("--simulation", action="store_true", help="Executar em modo de simulação")
    parser.add_argument("--days", type=int, default=30, help="Número de dias para análise histórica")
    parser.add_argument("--kucoin-capital", type=float, help="Capital disponível na KuCoin (USDT) - opcional, saldo será obtido via API se não fornecido")
    parser.add_argument("--kraken-capital", type=float, help="Capital disponível na Kraken (USDT) - opcional, saldo será obtido via API se não fornecido")
    parser.add_argument("--binance-capital", type=float, help="Capital disponível na Binance (USDT) - opcional, saldo será obtido via API se não fornecido")
    parser.add_argument("--other-capital", type=str, help="Capital em outras exchanges no formato 'exchange:valor,exchange2:valor2'")
    parser.add_argument("--fetch-balances", action="store_true", help="Obter saldos automaticamente via API")
    args = parser.parse_args()
    
    logger.info(f"Iniciando otimização de configuração: {args.config}")
    logger.info(f"Modo: {'simulação' if args.simulation else 'real'}")
    logger.info(f"Período de análise: {args.days} dias")
    
    # Carregar configuração atual para obter informações básicas
    config_path = args.config
    
    try:
        with open(config_path, 'r') as f:
            current_config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Erro ao carregar configuração: {e}")
        sys.exit(1)
    
    # Obter exchanges e pares da configuração
    exchange_ids = current_config.get("exchanges", ["kucoin", "kraken"])
    pairs = current_config.get("pairs", ["BTC/USDT", "ETH/USDT"])
    
    logger.info(f"Exchanges: {exchange_ids}")
    logger.info(f"Pares: {pairs}")
    
    # Inicializar APIs de mercado
    exchanges = []
    for ex_id in exchange_ids:
        try:
            exchange = MarketAPI(simulation_mode=args.simulation, exchange_id=ex_id)
            exchanges.append(exchange)
            logger.info(f"Exchange {ex_id} inicializada")
        except Exception as e:
            logger.error(f"Erro ao inicializar exchange {ex_id}: {e}")
    
    if not exchanges:
        logger.error("Nenhuma exchange inicializada. Abortando.")
        sys.exit(1)
    
    # Preparar dicionário de capital por exchange
    capital_per_exchange = {}
    
    # Verifica se é para obter saldos automaticamente via API 
    # ou se nenhum saldo foi especificado manualmente
    if args.fetch_balances or (args.kucoin_capital is None and args.kraken_capital is None and args.binance_capital is None and args.other_capital is None):
        logger.info("Obtendo saldos automaticamente via API...")
        capital_per_exchange = await fetch_exchange_balances(args.simulation, exchange_ids)
    else:
        # Usar valores fornecidos na linha de comando
        if args.kucoin_capital is not None and "kucoin" in exchange_ids:
            capital_per_exchange["kucoin"] = args.kucoin_capital
            
        if args.kraken_capital is not None and "kraken" in exchange_ids:
            capital_per_exchange["kraken"] = args.kraken_capital
            
        if args.binance_capital is not None and "binance" in exchange_ids:
            capital_per_exchange["binance"] = args.binance_capital
        
        # Adicionar outras exchanges se especificadas
        if args.other_capital:
            try:
                other_exchanges = args.other_capital.split(',')
                for ex_entry in other_exchanges:
                    ex_id, capital = ex_entry.split(':')
                    ex_id = ex_id.strip()
                    if ex_id in exchange_ids:
                        capital_per_exchange[ex_id] = float(capital.strip())
            except Exception as e:
                logger.error(f"Erro ao processar capital de outras exchanges: {e}")
                logger.error("Formato esperado: 'exchange:valor,exchange2:valor2'")
        
        # Verificar se existem exchanges configuradas que não têm capital definido
        missing_exchanges = [ex_id for ex_id in exchange_ids if ex_id not in capital_per_exchange]
        if missing_exchanges:
            logger.info(f"Obtendo saldos para exchanges sem capital definido: {', '.join(missing_exchanges)}")
            missing_balances = await fetch_exchange_balances(args.simulation, missing_exchanges)
            capital_per_exchange.update(missing_balances)
    
    logger.info(f"Capital por exchange: {capital_per_exchange}")
    
    # Executar otimização com capital obtido
    try:
        # Executar otimização
        optimized_config = await optimize_config(config_path, exchanges, pairs, capital_per_exchange)
        
        # Salvar cópia de backup da configuração original
        backup_path = f"{config_path}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with open(backup_path, 'w') as f:
            json.dump(current_config, f, indent=2)
        
        logger.info(f"Backup da configuração original salvo em {backup_path}")
        logger.info(f"Configuração otimizada salva em {config_path}")
        
        # Exibir parâmetros otimizados
        for key in ["min_spread", "base_threshold", "min_volume_24h", "max_position_pct", 
                    "cycle_interval", "min_transfer", "rebalance_threshold", "execution_timeout"]:
            if key in optimized_config:
                logger.info(f"Parâmetro otimizado: {key} = {optimized_config[key]}")
        
        if "advanced_config" in optimized_config:
            logger.info("Parâmetros avançados otimizados:")
            for key, value in optimized_config["advanced_config"].items():
                logger.info(f"  {key} = {value}")
        
        logger.info("Otimização concluída com sucesso!")
        
        # Exibir sumário de alocação
        config_dir = os.path.dirname(config_path)
        allocation_path = os.path.join(config_dir, "capital_allocation.json")
        
        if os.path.exists(allocation_path):
            try:
                with open(allocation_path, 'r') as f:
                    allocation = json.load(f)
                
                if "optimal_allocation" in allocation:
                    logger.info("Alocação ótima de capital por par:")
                    for pair, amount in allocation["optimal_allocation"].items():
                        logger.info(f"  {pair}: {amount:.2f} USDT")
            except Exception as e:
                logger.error(f"Erro ao carregar alocação de capital: {e}")
        
    except Exception as e:
        logger.error(f"Erro durante otimização: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    # Criar diretório de logs se não existir
    os.makedirs("logs", exist_ok=True)
    
    # Executar loop assíncrono
    asyncio.run(main()) 