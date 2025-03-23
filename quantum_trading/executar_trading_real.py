"""
Script de Execução do Trading em Tempo Real
"""

import argparse
import logging
import time
from datetime import datetime
from typing import Dict, List, Any
from .market_api import MarketAPI
from .real_time_trader import RealTimeTrader

# Configure logging
logger = logging.getLogger("executar_trading")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def parse_args():
    """
    Processa argumentos da linha de comando
    
    Returns:
        Argumentos processados
    """
    parser = argparse.ArgumentParser(description="Sistema de Trading Quântico QUALIA")
    
    # Parâmetros obrigatórios
    parser.add_argument(
        "--pares",
        type=str,
        required=True,
        help="Pares de trading separados por vírgula (ex: BTC/USDT,ETH/USDT)"
    )
    
    # Parâmetros opcionais
    parser.add_argument(
        "--duracao",
        type=int,
        default=60,
        help="Duração da sessão em minutos"
    )
    parser.add_argument(
        "--drawdown",
        type=float,
        default=2.0,
        help="Drawdown máximo permitido em porcentagem"
    )
    parser.add_argument(
        "--confianca",
        type=float,
        default=0.7,
        help="Confiança mínima para operar"
    )
    parser.add_argument(
        "--tamanho-max",
        type=float,
        default=0.1,
        help="Tamanho máximo da posição como fração do saldo"
    )
    parser.add_argument(
        "--teste",
        action="store_true",
        help="Executar em modo de teste"
    )
    parser.add_argument(
        "--monitoramento-avancado",
        action="store_true",
        help="Ativar monitoramento avançado"
    )
    
    return parser.parse_args()

def main():
    """
    Função principal de execução
    """
    try:
        # Processar argumentos
        args = parse_args()
        
        # Configurar logging
        if args.monitoramento_avancado:
            logger.setLevel(logging.DEBUG)
        
        logger.info("Iniciando Sistema de Trading Quântico QUALIA")
        logger.info(f"Modo: {'Teste' if args.teste else 'Real'}")
        
        # Inicializar API do mercado
        market_api = MarketAPI(
            simulation_mode=args.teste
        )
        
        # Inicializar trader
        trader = RealTimeTrader(
            market_api=market_api,
            max_memory_capacity=2048,
            min_confidence=args.confianca,
            max_position_size=args.tamanho_max
        )
        
        # Loop principal
        start_time = datetime.now()
        end_time = start_time.timestamp() + (args.duracao * 60)
        
        while datetime.now().timestamp() < end_time:
            try:
                # Processar cada par
                for pair in args.pares.split(','):
                    # Obter dados do mercado
                    ticker = market_api.get_ticker(pair)
                    if not ticker:
                        continue
                    
                    # Analisar oportunidade
                    opportunity = trader.analyze_trading_opportunity(
                        symbol=pair,
                        price=ticker['price'],
                        volume=ticker['volume']
                    )
                    
                    # Executar trade se houver sinal
                    if opportunity['action'] != 'hold':
                        result = trader.execute_trade(opportunity)
                        
                        if result['success']:
                            logger.info(
                                f"Trade executado: {result['action']} "
                                f"{result['symbol']} @ {result['price']:.8f}"
                            )
                    
                    # Monitoramento avançado
                    if args.monitoramento_avancado:
                        state = trader.get_current_state()
                        logger.debug(f"Estado atual: {state}")
                    
                # Aguardar próximo ciclo
                time.sleep(1)
                
            except KeyboardInterrupt:
                logger.info("Interrupção manual detectada")
                break
                
            except Exception as e:
                logger.error(f"Erro no ciclo de trading: {str(e)}")
                if not args.teste:
                    break
        
        # Exibir resumo
        state = trader.get_current_state()
        logger.info("=== Resumo da Sessão ===")
        logger.info(f"Total de trades: {state['total_trades']}")
        logger.info(f"Saldo final: {state['balance']:.8f}")
        
        if state['position']:
            logger.info(
                f"Posição atual: {state['position']['side']} "
                f"{state['position']['symbol']} @ {state['position']['price']:.8f}"
            )
        
    except Exception as e:
        logger.error(f"Erro fatal: {str(e)}")
        raise

if __name__ == "__main__":
    main() 