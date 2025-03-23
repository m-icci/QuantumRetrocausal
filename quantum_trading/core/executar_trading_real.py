#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QUALIA Trading System - Executor de Trading Real
------------------------------------------------
Script para executar o sistema de trading QUALIA em modo real
"""

import os
import sys
import logging
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Importar componentes do sistema
from quantum_trading.real_time_trader import RealTimeTrader, TestModeTrader
from quantum_trading.nexus_quantico import NexusQuanticoAvancado

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"trading_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger("qualia_trading")

# Carregar variáveis de ambiente
load_dotenv()

def verificar_credenciais():
    """Verifica se as credenciais das APIs estão configuradas"""
    # Verificar credenciais KuCoin
    kucoin_vars = {
        'KUCOIN_API_KEY': os.getenv('KUCOIN_API_KEY'),
        'KUCOIN_API_SECRET': os.getenv('KUCOIN_API_SECRET'),
        'KUCOIN_API_PASSPHRASE': os.getenv('KUCOIN_API_PASSPHRASE')
    }
    
    # Verificar credenciais Kraken
    kraken_vars = {
        'KRAKEN_API_KEY': os.getenv('KRAKEN_API_KEY'),
        'KRAKEN_API_SECRET': os.getenv('KRAKEN_API_SECRET')
    }
    
    # Verificar se há credenciais faltando
    missing_kucoin = [k for k, v in kucoin_vars.items() if not v]
    missing_kraken = [k for k, v in kraken_vars.items() if not v]
    
    if missing_kucoin and missing_kraken:
        logger.error("Credenciais de API não encontradas para KuCoin e Kraken")
        logger.error(f"KuCoin faltando: {', '.join(missing_kucoin)}")
        logger.error(f"Kraken faltando: {', '.join(missing_kraken)}")
        return False
    
    if missing_kucoin:
        logger.warning(f"Credenciais de API KuCoin incompletas: {', '.join(missing_kucoin)}")
        logger.warning("O sistema usará apenas a Kraken para trading real")
    
    if missing_kraken:
        logger.warning(f"Credenciais de API Kraken incompletas: {', '.join(missing_kraken)}")
        logger.warning("O sistema usará apenas a KuCoin para trading real")
    
    if not missing_kucoin or not missing_kraken:
        logger.info("Pelo menos uma exchange tem credenciais válidas")
        return True
    
    return False

def main():
    """Função principal para executar o trading real"""
    parser = argparse.ArgumentParser(description="QUALIA Trading System - Executor de Trading Real")
    parser.add_argument("--duracao", type=int, default=60,
                      help="Duração da sessão em minutos (padrão: 60)")
    parser.add_argument("--pares", type=str, default="BTC/USDT,ETH/USDT",
                      help="Pares de trading (padrão: BTC/USDT,ETH/USDT)")
    parser.add_argument("--drawdown", type=float, default=2.0,
                      help="Drawdown máximo permitido em % (padrão: 2.0)")
    parser.add_argument("--teste", action="store_true",
                      help="Executar em modo de teste")
    parser.add_argument("--exchanges", type=str, default="kucoin,kraken",
                      help="Exchanges para operar (padrão: kucoin,kraken)")
    parser.add_argument("--confirmar", action="store_true",
                      help="Pular confirmação de início")
    parser.add_argument("--monitoramento-avancado", action="store_true",
                      help="Habilitar monitoramento quântico avançado")
    parser.add_argument("--intervalo-visualizacao", type=int, default=300,
                      help="Intervalo para visualizações em segundos (padrão: 300)")
    parser.add_argument("--intervalo-analise", type=int, default=60,
                      help="Intervalo para análise de tendências em segundos (padrão: 60)")
    parser.add_argument("--dimensao-quantica", type=int, default=2048,
                      help="Dimensão do campo quântico (padrão: 2048)")
    parser.add_argument("--cache-dir", type=str, default="./quantum_cache",
                      help="Diretório para cache quântico (padrão: ./quantum_cache)")
    
    args = parser.parse_args()
    
    # Processar argumentos
    trading_pairs = [pair.strip() for pair in args.pares.split(',')]
    exchanges = [ex.strip().lower() for ex in args.exchanges.split(',')]
    
    # Verificar credenciais apenas se não estiver em modo teste
    if not args.teste and not verificar_credenciais():
        logger.error("Credenciais insuficientes para trading real")
        return 1
    
    try:
        # Inicializar componentes
        logger.info("Inicializando componentes do sistema...")
        
        # Criar instância do Nexus Quântico com configurações avançadas
        nexus = NexusQuanticoAvancado(
            dimensao=args.dimensao_quantica,
            cache_dir=args.cache_dir
        )
        
        # Selecionar classe apropriada
        trader_class = TestModeTrader if args.teste else RealTimeTrader
        
        # Criar instância do trader
        trader = trader_class(
            duration_minutes=args.duracao,
            max_drawdown_pct=args.drawdown,
            trading_pairs=trading_pairs,
            exchanges=exchanges
        )
        
        # Integrar sistemas
        if not nexus.integrar_com_trader(trader):
            logger.error("Falha na integração quântica")
            return 1
        
        # Configurar monitoramento avançado
        if args.monitoramento_avancado:
            trader.enable_advanced_monitoring(
                visualization_interval=args.intervalo_visualizacao,
                trend_analysis_interval=args.intervalo_analise
            )
            logger.info(f"Monitoramento avançado configurado:")
            logger.info(f"- Intervalo de visualização: {args.intervalo_visualizacao}s")
            logger.info(f"- Intervalo de análise: {args.intervalo_analise}s")
            logger.info(f"- Dimensão quântica: {args.dimensao_quantica}")
            logger.info(f"- Cache dir: {args.cache_dir}")
        
        # Confirmar início do trading real
        if not args.teste and not args.confirmar:
            print("\n" + "="*80)
            print("ATENÇÃO: Você está prestes a iniciar operações REAIS nas exchanges!")
            print(f"O sistema operará por {args.duracao} minutos nos pares: {', '.join(trading_pairs)}")
            print(f"Drawdown máximo: {args.drawdown}%")
            print(f"Dimensão quântica: {args.dimensao_quantica}")
            print("="*80)
            
            confirmacao = input("\nDigite 'confirmar' para iniciar as operações reais: ")
            if confirmacao.lower() != "confirmar":
                print("Operação cancelada pelo usuário.")
                return 0
        
        # Iniciar sessão de trading
        print("\nIniciando sistema QUALIA...")
        trader.start_trading_session()
        
        # Monitorar progresso
        while trader.is_trading_active:
            try:
                summary = trader.get_session_summary()
                print(f"\rProgresso: {summary['duration_minutes']:.1f} min, "
                      f"P&L: {summary['profit_loss_pct']:.2f}%, "
                      f"Trades: {summary['trades_executed']}", end='')
                
                # Atualizar análise quântica
                for pair in trading_pairs:
                    dados_mercado = trader.get_market_data(pair)
                    if dados_mercado is not None:
                        analise = nexus.analisar_padrao_trading(dados_mercado)
                        if analise:
                            logger.debug(f"Análise quântica para {pair}: {analise}")
                
                time.sleep(10)
            except KeyboardInterrupt:
                print("\nInterrompendo sessão...")
                trader.stop_trading_session()
                break
        
        # Exibir resumo final
        final_summary = trader.get_session_summary()
        print("\n\n=== Resumo da Sessão ===")
        print(f"Duração: {final_summary['duration_minutes']:.1f} minutos")
        print(f"Trades Executados: {final_summary['trades_executed']}")
        print(f"Trades Bem Sucedidos: {final_summary['successful_trades']}")
        print(f"Resultado: {final_summary['profit_loss_pct']:.2f}%")
        print(f"Drawdown Máximo: {final_summary['max_drawdown_pct']:.2f}%")
        
        # Exibir métricas quânticas finais
        metricas_quanticas = nexus.calcular_metricas()
        print("\n=== Métricas Quânticas Finais ===")
        for metrica, valor in metricas_quanticas.items():
            print(f"{metrica.capitalize()}: {valor:.4f}")
        print("="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Erro durante execução: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
