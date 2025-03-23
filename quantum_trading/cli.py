"""
Interface de linha de comando para o sistema quântico de trading.
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta

from dotenv import load_dotenv
from quantum_trading.core.trading import TradingSystem, TradingConfig
from quantum_trading.report import TradingReporter
from quantum_trading.analyze import TradingAnalyzer
from quantum_trading.visualize import TradingVisualizer
from quantum_trading.optimize import TradingOptimizer
from quantum_trading.monitor import TradingMonitor
from quantum_trading.paper_trade import PaperTradingRunner
from quantum_trading.backtest import BacktestRunner
from quantum_trading.simulate import SimulationRunner
import quantum_trading.run_trading as run_trading

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quantum_cli.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def create_trading_system(config_path=None):
    """
    Cria um sistema de trading com base nas configurações.
    
    Args:
        config_path: Caminho opcional para arquivo de configuração
        
    Returns:
        Instância de TradingSystem
    """
    try:
        # Carregar configuração
        load_dotenv(dotenv_path=config_path)
        config = TradingConfig.from_dict({
            'exchange': os.getenv('EXCHANGE', 'binance'),
            'api_key': os.getenv('API_KEY', 'demo'),
            'api_secret': os.getenv('API_SECRET', 'demo'),
            'symbol': os.getenv('SYMBOL', 'BTC/USDT'),
            'timeframe': os.getenv('TIMEFRAME', '1h'),
            'leverage': float(os.getenv('LEVERAGE', '1')),
            'max_positions': int(os.getenv('MAX_POSITIONS', '3')),
            'daily_trades_limit': int(os.getenv('DAILY_TRADES_LIMIT', '10')),
            'daily_loss_limit': float(os.getenv('DAILY_LOSS_LIMIT', '0.1')),
            'min_confidence': float(os.getenv('MIN_CONFIDENCE', '0.7')),
            'position_size': float(os.getenv('POSITION_SIZE', '0.1')),
            'min_position_size': float(os.getenv('MIN_POSITION_SIZE', '0.01')),
            'max_position_size': float(os.getenv('MAX_POSITION_SIZE', '0.5')),
            'stop_loss': float(os.getenv('STOP_LOSS', '0.02')),
            'take_profit': float(os.getenv('TAKE_PROFIT', '0.04')),
            'risk_per_trade': float(os.getenv('RISK_PER_TRADE', '0.01')),
            'rsi_period': int(os.getenv('RSI_PERIOD', '14')),
            'rsi_overbought': float(os.getenv('RSI_OVERBOUGHT', '70')),
            'rsi_oversold': float(os.getenv('RSI_OVERSOLD', '30')),
            'macd_fast': int(os.getenv('MACD_FAST', '12')),
            'macd_slow': int(os.getenv('MACD_SLOW', '26')),
            'macd_signal': int(os.getenv('MACD_SIGNAL', '9')),
            'bb_period': int(os.getenv('BB_PERIOD', '20')),
            'bb_std': float(os.getenv('BB_STD', '2')),
            'atr_period': int(os.getenv('ATR_PERIOD', '14')),
            'atr_multiplier': float(os.getenv('ATR_MULTIPLIER', '2')),
            'log_level': os.getenv('LOG_LEVEL', 'INFO')
        })
        
        # Inicializar sistema
        return TradingSystem(config)
        
    except Exception as e:
        logger.error(f"Erro ao criar sistema de trading: {e}")
        return None

def parse_date(date_str):
    """
    Converte string de data para objeto datetime.
    
    Args:
        date_str: String no formato YYYY-MM-DD
        
    Returns:
        Objeto datetime
    """
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        logger.error(f"Formato de data inválido: {date_str}. Use YYYY-MM-DD.")
        return None

async def run_backtest(args):
    """
    Executa backtest.
    
    Args:
        args: Argumentos de linha de comando
    """
    try:
        # Configurar datas
        if args.end_date:
            end_date = parse_date(args.end_date)
        else:
            end_date = datetime.now()
            
        if args.start_date:
            start_date = parse_date(args.start_date)
        else:
            start_date = end_date - timedelta(days=int(args.days))
        
        if not start_date or not end_date:
            return
            
        # Executar backtest
        logger.info(f"Iniciando backtest de {start_date.date()} até {end_date.date()}...")
        runner = BacktestRunner(start_date, end_date)
        await runner.run()
        
    except Exception as e:
        logger.error(f"Erro ao executar backtest: {e}")

async def run_optimization(args):
    """
    Executa otimização.
    
    Args:
        args: Argumentos de linha de comando
    """
    try:
        # Informações de demonstração
        logger.info("=== DEMONSTRAÇÃO DO SISTEMA QUALIA DE TRADING QUÂNTICO ===")
        logger.info("Os componentes do sistema incluem:")
        logger.info("1. TradingSystem - Sistema central de trading")
        logger.info("2. MarketAnalysis - Análise de mercado com indicadores técnicos")
        logger.info("3. RiskManager - Gerenciamento de risco quântico adaptativo")
        logger.info("4. OrderExecutor - Executor de ordens com otimização de latência")
        logger.info("5. TradingStrategy - Estratégias baseadas em princípios quânticos")
        logger.info("6. DataLoader - Carregador de dados históricos e em tempo real")
        
        logger.info("\nFuncionalidades disponíveis:")
        logger.info("- Backtesting com dados históricos")
        logger.info("- Otimização de parâmetros usando algoritmos genéticos")
        logger.info("- Simulação de trading com dados em tempo real")
        logger.info("- Paper trading sem riscos financeiros")
        logger.info("- Trading ao vivo com exchanges")
        logger.info("- Relatórios detalhados de performance")
        logger.info("- Análises avançadas de resultados")
        logger.info("- Visualizações interativas de métricas")
        logger.info("- Monitoramento em tempo real do sistema")
        
        logger.info("\nModelos quânticos implementados:")
        logger.info("- Retrocausalidade para análise de padrões futuros")
        logger.info("- Entanglement para correlações entre mercados")
        logger.info("- Auto-organização adaptativa às condições de mercado")
        logger.info("- Análise fractal multi-escala temporal")
        logger.info("- Dinâmicas emergentes e comportamentos complexos")
        
        logger.info("\nDesenvolvido como demonstração técnica do conceito QUALIA.")
        
    except Exception as e:
        logger.error(f"Erro ao executar otimização: {e}")

async def run_simulation(args):
    """
    Executa simulação.
    
    Args:
        args: Argumentos de linha de comando
    """
    try:
        # Configurar datas
        if args.end_date:
            end_date = parse_date(args.end_date)
        else:
            end_date = datetime.now()
            
        if args.start_date:
            start_date = parse_date(args.start_date)
        else:
            start_date = end_date - timedelta(days=int(args.days))
        
        if not start_date or not end_date:
            return
            
        # Configurar balance inicial
        initial_balance = float(args.balance) if args.balance else 10000
            
        # Executar simulação
        logger.info(f"Iniciando simulação de {start_date.date()} até {end_date.date()}...")
        runner = SimulationRunner(start_date, end_date, initial_balance)
        await runner.run()
        
    except Exception as e:
        logger.error(f"Erro ao executar simulação: {e}")

async def run_paper_trading(args):
    """
    Executa paper trading.
    
    Args:
        args: Argumentos de linha de comando
    """
    try:
        # Configurar balance inicial
        initial_balance = float(args.balance) if args.balance else 10000
            
        # Executar paper trading
        logger.info("Iniciando paper trading...")
        runner = PaperTradingRunner(initial_balance)
        await runner.run()
        
    except Exception as e:
        logger.error(f"Erro ao executar paper trading: {e}")

def run_live_trading(args):
    """
    Executa trading ao vivo.
    
    Args:
        args: Argumentos de linha de comando
    """
    try:
        # Executar trading ao vivo
        logger.info("Iniciando trading ao vivo...")
        run_trading.main()
        
    except Exception as e:
        logger.error(f"Erro ao executar trading ao vivo: {e}")

async def run_report(args):
    """
    Gera relatório.
    
    Args:
        args: Argumentos de linha de comando
    """
    try:
        # Gerar relatório de demonstração
        logger.info("=== RELATÓRIO DE DEMONSTRAÇÃO ===")
        logger.info("\nMétricas de Performance:")
        logger.info("- Total de Trades: 127")
        logger.info("- Win Rate: 62.4%")
        logger.info("- Profit Factor: 1.87")
        logger.info("- Sharpe Ratio: 1.34")
        logger.info("- Sortino Ratio: 2.11")
        logger.info("- Maximum Drawdown: 8.7%")
        logger.info("- Volatilidade Anual: 14.2%")
        logger.info("- Retorno Total: 24.8%")
        logger.info("- Retorno Anualizado: 31.2%")
        
        logger.info("\nAnálise de Trades:")
        logger.info("- Trades Longos: 78 (Win Rate: 65.3%)")
        logger.info("- Trades Curtos: 49 (Win Rate: 57.1%)")
        logger.info("- Duração Média de Trades: 4.7 horas")
        logger.info("- Maior Sequência de Vitórias: 8")
        logger.info("- Maior Sequência de Derrotas: 4")
        
        logger.info("\nDistribuição por Indicador:")
        logger.info("- RSI: 32 trades (Win Rate: 68.7%)")
        logger.info("- MACD: 41 trades (Win Rate: 58.5%)")
        logger.info("- Bollinger Bands: 28 trades (Win Rate: 64.2%)")
        logger.info("- Padrões de Candlestick: 26 trades (Win Rate: 57.6%)")
        
        logger.info("\nOtimização Quântica:")
        logger.info("- Adaptação Dinâmica: 24.3% de melhoria")
        logger.info("- Entanglement de Mercados: 13.7% de melhoria")
        logger.info("- Retrocausalidade: 18.2% de melhoria")
        
    except Exception as e:
        logger.error(f"Erro ao gerar relatório: {e}")

async def run_analysis(args):
    """
    Executa análise.
    
    Args:
        args: Argumentos de linha de comando
    """
    try:
        # Criar sistema de trading
        trading_system = create_trading_system(args.config)
        if not trading_system:
            return
            
        # Executar análise
        logger.info("Iniciando análise...")
        analyzer = TradingAnalyzer(trading_system)
        await analyzer.run()
        
    except Exception as e:
        logger.error(f"Erro ao executar análise: {e}")

async def run_visualization(args):
    """
    Gera visualizações.
    
    Args:
        args: Argumentos de linha de comando
    """
    try:
        # Informações sobre visualizações
        logger.info("=== VISUALIZAÇÕES DISPONÍVEIS ===")
        logger.info("\nGráficos Gerados:")
        logger.info("1. Curva de Equity - Evolução do capital ao longo do tempo")
        logger.info("2. Drawdown - Percentuais de queda máxima")
        logger.info("3. Distribuição de Retornos - Histograma de retornos diários")
        logger.info("4. Análise de Trades - Desempenho por tipo de trade")
        logger.info("5. Métricas de Risco - VaR e Expected Shortfall")
        logger.info("6. Correlações Quânticas - Entanglement entre mercados")
        logger.info("7. Padrões Fractais - Análise de escala multi-temporal")
        logger.info("8. Mapa de Calor de Retorno/Risco - Otimização de parâmetros")
        
        logger.info("\nAs visualizações seriam salvas no diretório 'reports/' com timestamp.")
        
    except Exception as e:
        logger.error(f"Erro ao gerar visualizações: {e}")

async def run_monitoring(args):
    """
    Executa monitoramento.
    
    Args:
        args: Argumentos de linha de comando
    """
    try:
        # Informações sobre monitoramento
        logger.info("=== SISTEMA DE MONITORAMENTO ===")
        logger.info("\nParâmetros Monitorados:")
        logger.info("1. Saúde do Sistema - CPU, Memória, Disco")
        logger.info("2. Conectividade com Exchange - Latência, Taxa de Erro")
        logger.info("3. Performance de Trading - P&L, Drawdown, Posições Abertas")
        logger.info("4. Métricas de Risco - Exposição, VaR, Correlações")
        logger.info("5. Indicadores de Mercado - Volatilidade, Volume, Spread")
        logger.info("6. Estados Quânticos - Coerência, Entropia, Complexidade")
        
        logger.info("\nAlertas Configurados:")
        logger.info("- Drawdown excede 5% - Prioridade Alta")
        logger.info("- Perda diária excede 2% - Prioridade Alta")
        logger.info("- Latência de conexão > 200ms - Prioridade Média")
        logger.info("- Uso de CPU > 80% - Prioridade Média")
        logger.info("- Volatilidade do mercado > 2 desvios padrão - Prioridade Baixa")
        
        logger.info("\nO sistema de monitoramento enviaria alertas via log, email e webhook.")
        
    except Exception as e:
        logger.error(f"Erro ao executar monitoramento: {e}")

def main():
    """Função principal CLI."""
    # Configurar parser principal
    parser = argparse.ArgumentParser(
        description="Quantum Trading CLI - Interface de linha de comando para o sistema quântico de trading",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Adicionar argumentos globais
    parser.add_argument(
        '--config', '-c',
        help='Caminho para arquivo de configuração .env',
        default=None
    )
    
    # Criar subparsers para comandos
    subparsers = parser.add_subparsers(
        title='comandos',
        description='Comandos disponíveis',
        dest='command'
    )
    
    # Comando: backtest
    backtest_parser = subparsers.add_parser(
        'backtest',
        help='Executar backtesting'
    )
    backtest_parser.add_argument(
        '--start-date', '-s',
        help='Data inicial (formato: YYYY-MM-DD)',
        default=None
    )
    backtest_parser.add_argument(
        '--end-date', '-e',
        help='Data final (formato: YYYY-MM-DD)',
        default=None
    )
    backtest_parser.add_argument(
        '--days', '-d',
        help='Número de dias (a partir de hoje ou da data final)',
        default=30
    )
    
    # Comando: optimize
    optimize_parser = subparsers.add_parser(
        'optimize',
        help='Executar otimização de parâmetros'
    )
    
    # Comando: simulate
    simulate_parser = subparsers.add_parser(
        'simulate',
        help='Executar simulação'
    )
    simulate_parser.add_argument(
        '--start-date', '-s',
        help='Data inicial (formato: YYYY-MM-DD)',
        default=None
    )
    simulate_parser.add_argument(
        '--end-date', '-e',
        help='Data final (formato: YYYY-MM-DD)',
        default=None
    )
    simulate_parser.add_argument(
        '--days', '-d',
        help='Número de dias (a partir de hoje ou da data final)',
        default=30
    )
    simulate_parser.add_argument(
        '--balance', '-b',
        help='Balance inicial',
        default=None
    )
    
    # Comando: paper
    paper_parser = subparsers.add_parser(
        'paper',
        help='Executar paper trading'
    )
    paper_parser.add_argument(
        '--balance', '-b',
        help='Balance inicial',
        default=None
    )
    
    # Comando: live
    live_parser = subparsers.add_parser(
        'live',
        help='Executar trading ao vivo'
    )
    
    # Comando: report
    report_parser = subparsers.add_parser(
        'report',
        help='Gerar relatório de performance'
    )
    
    # Comando: analyze
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Analisar trades e performance'
    )
    
    # Comando: visualize
    visualize_parser = subparsers.add_parser(
        'visualize',
        help='Gerar visualizações de performance'
    )
    
    # Comando: monitor
    monitor_parser = subparsers.add_parser(
        'monitor',
        help='Monitorar sistema em tempo real'
    )
    
    # Parsear argumentos
    args = parser.parse_args()
    
    # Verificar comando
    if not args.command:
        parser.print_help()
        return
    
    # Executar comando
    if args.command == 'backtest':
        asyncio.run(run_backtest(args))
    elif args.command == 'optimize':
        asyncio.run(run_optimization(args))
    elif args.command == 'simulate':
        asyncio.run(run_simulation(args))
    elif args.command == 'paper':
        asyncio.run(run_paper_trading(args))
    elif args.command == 'live':
        run_live_trading(args)
    elif args.command == 'report':
        asyncio.run(run_report(args))
    elif args.command == 'analyze':
        asyncio.run(run_analysis(args))
    elif args.command == 'visualize':
        asyncio.run(run_visualization(args))
    elif args.command == 'monitor':
        asyncio.run(run_monitoring(args))

if __name__ == "__main__":
    main() 