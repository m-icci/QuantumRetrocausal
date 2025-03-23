"""
Script de execução da estratégia WAVE (Weighted Adaptive Volatility Exploitation)
"""

import asyncio
import logging
import os
import sys
import time
import signal
import json
from datetime import datetime
from typing import Dict, List, Any
import argparse
from pathlib import Path

# Configurar logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("wave_strategy.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("run_wave_strategy")

# Importar componentes do sistema
from .market_api import MarketAPI
from .strategies.wave_strategy import WAVEStrategy

# Configurações padrão
DEFAULT_CONFIG = {
    "min_spread": 0.0005,  # 0.05% spread mínimo (reduzido de 0.1%)
    "base_threshold": 0.0004,  # Threshold base para spreads (reduzido)
    "min_volume_24h": 5000,  # Volume mínimo em 24h (reduzido de 10000)
    "min_transfer": 10,  # Transferência mínima para rebalanceamento
    "rebalance_threshold": 0.1,  # 10% de desvio para rebalanceamento
    "max_position_pct": 0.25,  # Máximo 25% do balanço por operação
    "history_window": 24,  # Janela de histórico em horas
    "time_segments": 24,  # Segmentos de tempo para análise
    "execution_timeout": 30,  # Timeout para execução em segundos
    "cycle_interval": 30,  # Intervalo entre ciclos em segundos (reduzido de 60)
    "state_save_interval": 300,  # Intervalo para salvar estado em segundos
    "exchanges": ["kucoin", "kraken"],  # Exchanges padrão
    "pairs": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XMR/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT"]  # Pares padrão expandidos
}

# Sinalizador para controle de execução
running = True

def load_config(config_path: str = None) -> Dict[str, Any]:
    """Carrega configuração a partir de arquivo ou usa padrão"""
    config = DEFAULT_CONFIG.copy()
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                config.update(loaded_config)
            logger.info(f"Configuração carregada de {config_path}")
        except Exception as e:
            logger.error(f"Erro ao carregar configuração: {e}")
    
    return config

async def init_exchanges(config: Dict[str, Any]) -> List[MarketAPI]:
    """Inicializa APIs de exchanges"""
    exchanges = []
    
    for exchange_id in config['exchanges']:
        try:
            exchange = MarketAPI(exchange_id=exchange_id)
            await exchange.initialize()
            exchanges.append(exchange)
            logger.info(f"Exchange {exchange_id} inicializada")
        except Exception as e:
            logger.error(f"Erro ao inicializar exchange {exchange_id}: {e}")
    
    return exchanges

def signal_handler(sig, frame):
    """Manipulador de sinais para encerramento gracioso"""
    global running
    logger.info("Sinal de encerramento recebido, finalizando...")
    running = False

async def run_strategy(config: Dict[str, Any], state_path: str = None):
    """Executa a estratégia WAVE"""
    global running
    
    # Inicializar exchanges
    exchanges = await init_exchanges(config)
    
    if not exchanges:
        logger.error("Nenhuma exchange inicializada, abortando")
        return
    
    # Inicializar estratégia
    strategy = WAVEStrategy(
        exchanges=exchanges,
        pairs=config['pairs'],
        config=config
    )
    
    # Carregar estado anterior se existir
    if state_path and os.path.exists(state_path):
        try:
            strategy.load_state(state_path)
            logger.info(f"Estado carregado de {state_path}")
        except Exception as e:
            logger.error(f"Erro ao carregar estado: {e}")
    
    # Inicializar estratégia
    try:
        await strategy.initialize()
    except Exception as e:
        logger.error(f"Erro na inicialização da estratégia: {e}")
        return
    
    # Loop principal
    cycle_count = 0
    last_save_time = time.time()
    
    logger.info("Iniciando loop principal da estratégia WAVE")
    
    while running:
        cycle_start = time.time()
        cycle_count += 1
        
        logger.info(f"Iniciando ciclo #{cycle_count}")
        
        try:
            # Executar ciclo da estratégia
            results = await strategy.run_strategy_cycle()
            
            # Logar resultados
            if results["opportunities_detected"] > 0:
                logger.info(f"Ciclo #{cycle_count} concluído: {results['opportunities_detected']} oportunidades detectadas, " + 
                            f"{results['arbitrages_executed']} arbitragens executadas, " +
                            f"{results['successful_arbitrages']} com sucesso, " +
                            f"lucro total: {results['total_profit']:.6f} USDT")
            else:
                logger.debug(f"Ciclo #{cycle_count} concluído: Nenhuma oportunidade detectada")
                
            # Verificar se é hora de salvar o estado
            current_time = time.time()
            if current_time - last_save_time > config['state_save_interval']:
                if state_path:
                    strategy.save_state(state_path)
                    logger.info(f"Estado salvo em {state_path}")
                last_save_time = current_time
            
            # Calcular tempo para próximo ciclo
            cycle_duration = time.time() - cycle_start
            sleep_time = max(1, config['cycle_interval'] - cycle_duration)
            
            logger.info(f"Ciclo #{cycle_count} durou {cycle_duration:.2f}s, aguardando {sleep_time:.2f}s até o próximo ciclo")
            
            # Aguardar até o próximo ciclo
            await asyncio.sleep(sleep_time)
            
        except Exception as e:
            logger.error(f"Erro no ciclo #{cycle_count}: {e}")
            await asyncio.sleep(5)  # Pequena pausa em caso de erro

async def main(
    config_path: str = None,
    simulation_mode: bool = False,
    optimize: bool = False,
    days: int = 7,
    fetch_balances: bool = False,
    capital_per_exchange: Dict[str, float] = None
):
    """
    Função principal para execução da estratégia WAVE
    
    Args:
        config_path: Caminho para o arquivo de configuração
        simulation_mode: Se True, executa em modo de simulação
        optimize: Se True, otimiza parâmetros antes de executar
        days: Número de dias para análise histórica
        fetch_balances: Se True, busca saldos reais das exchanges
        capital_per_exchange: Dicionário com capital disponível por exchange
    """
    # Configuração de logging
    setup_logging()
    
    # Variável para contagem de ciclos
    cycle_count = 0
    
    # Definir caminho padrão de configuração
    if config_path is None:
        config_path = os.path.join("config", "wave_config.json")
    
    # Carregar configuração
    try:
        config = load_config(config_path)
        pairs = config.get("pairs", ["BTC/USDT", "ETH/USDT"])
        exchange_ids = config.get("exchanges", ["kucoin", "kraken"])
        cycle_interval = config.get("cycle_interval", 60)
        state_save_interval = config.get("state_save_interval", 300)
    except Exception as e:
        logger.error(f"Erro ao carregar configuração: {e}")
        return
    
    # Inicializar capital por exchange se não fornecido
    if capital_per_exchange is None:
        capital_per_exchange = {ex_id: 1000.0 for ex_id in exchange_ids}
    else:
        # Garantir que todas as exchanges configuradas tenham um valor de capital
        for ex_id in exchange_ids:
            if ex_id not in capital_per_exchange:
                capital_per_exchange[ex_id] = 0.0
                logger.warning(f"Capital não especificado para {ex_id}, definindo como 0 USDT")
    
    logger.info(f"Capital por exchange: {capital_per_exchange}")
    
    # Configurar diretório para estados
    states_dir = os.path.join("states")
    os.makedirs(states_dir, exist_ok=True)
    
    # Inicializar exchanges
    exchanges = []
    for exchange_id in exchange_ids:
        try:
            exchange = MarketAPI(exchange_id=exchange_id)
            # Inicializar de forma assíncrona
            await exchange.initialize()
            exchanges.append(exchange)
            logger.info(f"Exchange {exchange_id} inicializada")
        except Exception as e:
            logger.error(f"Erro ao inicializar exchange {exchange_id}: {e}")
    
    if not exchanges:
        logger.error("Nenhuma exchange inicializada. Abortando.")
        return
    
    # Otimizar configuração se solicitado
    if optimize:
        try:
            logger.info("Iniciando otimização de parâmetros...")
            from .config_optimizer import optimize_config
            
            # Fazer backup da configuração original
            backup_path = f"{config_path}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with open(backup_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Executar otimização com informações de capital
            optimized_config = await optimize_config(
                config_path,
                exchanges,
                pairs,
                capital_per_exchange,
                days=days
            )
            
            # Recarregar configuração otimizada
            config = load_config(config_path)
            cycle_interval = config.get("cycle_interval", 60)
            state_save_interval = config.get("state_save_interval", 300)
            
            logger.info("Configuração otimizada carregada")
            
            # Verificar se existem arquivos de alocação de capital
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
            logger.warning("Continuando com a configuração original")
    
    # Inicializar estratégia com informações de capital
    try:
        # Adicionar informações de capital à configuração para uso pela estratégia
        config["capital_per_exchange"] = capital_per_exchange
        
        strategy = WAVEStrategy(exchanges, pairs, config)
        
        # Verificar se existe estado salvo
        state_file = os.path.join(states_dir, "wave_strategy_state.json")
        if os.path.exists(state_file):
            try:
                strategy.load_state(state_file)
                logger.info(f"Estado carregado de {state_file}")
            except Exception as e:
                logger.error(f"Erro ao carregar estado: {e}")
    except Exception as e:
        logger.error(f"Erro ao inicializar estratégia: {e}")
        return
    
    # Configurar manipuladores de sinais para shutdown gracioso
    should_exit = False
    
    def signal_handler(sig, frame):
        nonlocal should_exit
        logger.info(f"Sinal recebido ({sig}), finalizando...")
        should_exit = True
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Inicializar a estratégia
    try:
        await strategy.initialize()
        logger.info("Estratégia WAVE inicializada")
    except Exception as e:
        logger.error(f"Erro na inicialização da estratégia: {e}")
        return
    
    # Loop principal
    last_state_save = time.time()
    
    try:
        logger.info(f"Iniciando ciclos de estratégia a cada {cycle_interval} segundos")
        
        while not should_exit:
            # Executar ciclo da estratégia
            cycle_start = time.time()
            
            try:
                results = await strategy.run_strategy_cycle()
                
                # Logar resultados
                if results["opportunities_detected"] > 0:
                    logger.info(f"Ciclo concluído: {results['opportunities_detected']} oportunidades detectadas, " + 
                                f"{results['arbitrages_executed']} arbitragens executadas, " +
                                f"{results['successful_arbitrages']} com sucesso, " +
                                f"lucro total: {results['total_profit']:.6f} USDT")
                else:
                    logger.debug("Ciclo concluído: Nenhuma oportunidade detectada")
                
                # Gerar e logar resumo de performance a cada 5 ciclos
                cycle_count += 1
                
                if cycle_count % 5 == 0:
                    try:
                        performance = strategy.get_performance_summary()
                        
                        # Criar logger específico para performance
                        perf_logger = logging.getLogger("performance")
                        
                        # Logar resumo geral
                        summary = performance["summary"]
                        perf_logger.info("=" * 50)
                        perf_logger.info("RESUMO DE PERFORMANCE DA ESTRATÉGIA WAVE")
                        perf_logger.info("=" * 50)
                        perf_logger.info(f"Capital Inicial: {summary['initial_capital']:.2f} USDT")
                        perf_logger.info(f"Saldo Atual: {summary['current_balance']:.2f} USDT")
                        perf_logger.info(f"Lucro Total: {summary['total_profit']:.2f} USDT")
                        perf_logger.info(f"Ganho/Perda: {summary['gain_loss_pct']:.2f}%")
                        perf_logger.info(f"Operações: {summary['successful_trades']}/{summary['total_trades']} ({summary['success_rate']:.1f}%)")
                        
                        # Logar resumo por exchange
                        perf_logger.info("-" * 50)
                        perf_logger.info("RESUMO POR EXCHANGE:")
                        for ex_id, ex_data in performance["exchanges"].items():
                            perf_logger.info(f"{ex_id}: {ex_data['current_balance']:.2f} USDT ({ex_data['gain_loss_pct']:+.2f}%)")
                        
                        # Logar resumo por par
                        perf_logger.info("-" * 50)
                        perf_logger.info("RESUMO POR PAR:")
                        for pair, pair_data in performance["pairs"].items():
                            if pair_data["total_trades"] > 0:
                                perf_logger.info(f"{pair}: {pair_data['successful_trades']}/{pair_data['total_trades']} operações, "
                                                f"lucro: {pair_data['total_profit']:.2f} USDT, "
                                                f"taxa de sucesso: {pair_data['success_rate']:.1f}%")
                        
                        # Logar arbitragens recentes
                        if performance["recent_arbitrages"]:
                            perf_logger.info("-" * 50)
                            perf_logger.info("OPERAÇÕES RECENTES:")
                            for arb in performance["recent_arbitrages"][:5]:  # Mostrar apenas as 5 mais recentes
                                timestamp = datetime.fromisoformat(arb["timestamp"]).strftime("%H:%M:%S")
                                perf_logger.info(f"{timestamp} - {arb['pair']}: {arb['buy_exchange']} → {arb['sell_exchange']}, "
                                                f"status: {arb['status']}, lucro: {arb['profit']:.6f} USDT ({arb['profit_pct']:.2f}%)")
                        
                        perf_logger.info("=" * 50)
                        
                        # Salvar o resumo em arquivo JSON para a interface
                        perf_path = os.path.join("states", "performance_summary.json")
                        with open(perf_path, 'w') as f:
                            json.dump(performance, f, indent=2)
                            
                        logger.info(f"Resumo de performance atualizado em {perf_path}")
                    except Exception as e:
                        logger.error(f"Erro ao gerar resumo de performance: {e}")
                
                # Verificar se é necessário salvar o estado
                current_time = time.time()
                if current_time - last_state_save >= state_save_interval:
                    strategy.save_state(state_file)
                    last_state_save = current_time
                    logger.info(f"Estado salvo em {state_file}")
                
            except Exception as e:
                logger.error(f"Erro durante ciclo da estratégia: {e}")
                import traceback
                logger.error(traceback.format_exc())
            
            # Calcular tempo restante para próximo ciclo
            cycle_duration = time.time() - cycle_start
            sleep_time = max(0, cycle_interval - cycle_duration)
            
            if sleep_time > 0:
                # Utilizar asyncio.sleep para permitir cancelamento
                try:
                    await asyncio.sleep(sleep_time)
                except asyncio.CancelledError:
                    should_exit = True
            
    finally:
        # Salvar estado final
        try:
            strategy.save_state(state_file)
            logger.info(f"Estado final salvo em {state_file}")
        except Exception as e:
            logger.error(f"Erro ao salvar estado final: {e}")
        
        logger.info("Estratégia WAVE finalizada")

def setup_logging():
    """Configura o sistema de logging"""
    # Configurar diretório de logs
    logs_dir = os.path.join("logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Configurar arquivo de log principal
    qualia_log = os.path.join(logs_dir, "qualia.log")
    
    # Configurar arquivo de log específico para a estratégia
    wave_log = os.path.join(logs_dir, "wave_strategy.log")
    
    # Configurar arquivo de log para dados de mercado e spreads
    market_log = os.path.join(logs_dir, "market_data.log")
    
    # Configurar arquivo de log para performance
    performance_log = os.path.join(logs_dir, "performance.log")
    
    # Configurar arquivo de log para transações e mudanças de balanço
    transactions_log = os.path.join(logs_dir, "transactions.log")
    
    # Configuração de logging
    logging.basicConfig(
        level=logging.DEBUG,  # Aumentado para DEBUG para capturar mais informações
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(qualia_log),
            logging.FileHandler(wave_log)
        ]
    )
    
    # Configurar loggers específicos
    market_logger = logging.getLogger("market_data")
    market_logger.setLevel(logging.DEBUG)
    market_handler = logging.FileHandler(market_log)
    market_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    market_logger.addHandler(market_handler)
    
    # Configurar logger para spreads
    spread_logger = logging.getLogger("spread_analysis")
    spread_logger.setLevel(logging.DEBUG)
    spread_logger.addHandler(market_handler)  # Usar o mesmo arquivo para ambos
    
    # Configurar logger para performance
    perf_logger = logging.getLogger("performance")
    perf_logger.setLevel(logging.INFO)
    perf_handler = logging.FileHandler(performance_log)
    perf_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    perf_logger.addHandler(perf_handler)
    
    # Adicionar handler de console para performance (para mostrar na tela)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - PERFORMANCE - %(message)s'))
    perf_logger.addHandler(console_handler)
    
    # Configurar logger para transações
    trans_logger = logging.getLogger("transactions")
    trans_logger.setLevel(logging.INFO)
    transactions_handler = logging.FileHandler(transactions_log)
    transactions_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    trans_logger.addHandler(transactions_handler)
    
    # Adicionar handler de console para transações (para mostrar na tela)
    trans_console = logging.StreamHandler()
    trans_console.setFormatter(logging.Formatter('%(asctime)s - TRANSAÇÃO - %(message)s'))
    trans_logger.addHandler(trans_console)
    
    # Obter logger para este módulo
    logger = logging.getLogger("quantum_trading")
    logger.info(f"Configuração de logging inicializada")
    
    return logger

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Executa estratégia WAVE para arbitragem de criptomoedas")
    parser.add_argument("--config", type=str, help="Caminho para o arquivo de configuração")
    parser.add_argument("--simulation", action="store_true", help="Executar em modo de simulação")
    parser.add_argument("--optimize", action="store_true", help="Otimizar parâmetros antes de executar")
    parser.add_argument("--kucoin-capital", type=float, default=50.0, help="Capital disponível na KuCoin (USDT)")
    parser.add_argument("--kraken-capital", type=float, default=50.0, help="Capital disponível na Kraken (USDT)")
    parser.add_argument("--days", type=int, default=7, help="Número de dias para análise histórica")
    parser.add_argument("--fetch-balances", action="store_true", help="Buscar saldos reais das exchanges")
    args = parser.parse_args()
    
    try:
        # Preparar capital por exchange
        capital_per_exchange = {
            "kucoin": args.kucoin_capital,
            "kraken": args.kraken_capital
        }
        
        # Executar loop assíncrono
        asyncio.run(main(args.config, args.simulation, args.optimize, args.days, args.fetch_balances, capital_per_exchange))
    except KeyboardInterrupt:
        print("\nEncerrando estratégia...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Erro fatal: {e}")
        sys.exit(1) 