"""
Exemplo de uso do sistema de trading quântico
"""
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
from pathlib import Path

from ..scalping.quantum_scalper import (
    QuantumScalper,
    ScalpSignal,
    ScalpingMetrics
)
from ..visualization.quantum_trading_dashboard import QuantumTradingDashboard
from ...exchange.kucoin.quantum_bridge import KuCoinQuantumBridge

async def run_backtest(
    historical_data: Dict[str, List[float]],
    initial_capital: float = 10000.0,
    risk_per_trade: float = 0.02
) -> None:
    """
    Executa backtesting da estratégia
    
    Args:
        historical_data: Dados históricos OHLCV
        initial_capital: Capital inicial
        risk_per_trade: Risco por operação
    """
    print("Iniciando backtesting...")
    
    # Configura scalper
    scalper = QuantumScalper(
        field_dimensions=8,
        coherence_threshold=0.75,
        resonance_threshold=0.7
    )
    
    # Executa backtest
    results = scalper.backtest(
        market_data=historical_data,
        initial_capital=initial_capital,
        risk_per_trade=risk_per_trade
    )
    
    # Analisa resultados
    analysis = scalper.analyze_backtest_results(results)
    
    print("\nResultados do Backtest:")
    print(f"Capital Final: ${results.final_capital:.2f}")
    print(f"Total de Trades: {results.total_trades}")
    print(f"Taxa de Acerto: {(results.winning_trades/results.total_trades*100):.1f}%")
    print(f"PnL Total: ${results.total_pnl:.2f}")
    print(f"Drawdown Máximo: {(results.max_drawdown*100):.1f}%")
    print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    
    print("\nMétricas Quânticas:")
    print(f"Coerência Média: {results.field_coherence_mean:.2f}")
    print(f"Ressonância φ Média: {results.phi_resonance_mean:.2f}")
    print(f"Padrões Detectados: {len(results.patterns)}")
    
    # Salva resultados
    save_dir = Path("reports/backtesting")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(save_dir / f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        json.dump({
            "performance": analysis["performance"],
            "quantum_metrics": analysis["quantum_metrics"],
            "trade_analysis": analysis["trade_analysis"]
        }, f, indent=2)

async def run_live_trading(
    api_key: str,
    api_secret: str,
    api_passphrase: str,
    symbol: str = "BTC-USDT",
    capital_per_pair: float = 1000.0,
    risk_per_trade: float = 0.02
) -> None:
    """
    Executa trading em tempo real
    
    Args:
        api_key: API key da KuCoin
        api_secret: API secret da KuCoin
        api_passphrase: API passphrase da KuCoin
        symbol: Par de trading
        capital_per_pair: Capital por par
        risk_per_trade: Risco por operação
    """
    print(f"Iniciando trading em tempo real para {symbol}...")
    
    # Configura exchange
    exchange = KuCoinQuantumBridge(
        api_key=api_key,
        api_secret=api_secret,
        api_passphrase=api_passphrase
    )
    
    # Configura dashboard
    dashboard = QuantumTradingDashboard({
        'base_path': 'data',
        'alert_thresholds': {
            'drawdown': 0.1,
            'coherence': 0.7,
            'resonance': 0.6,
            'pattern_strength': 0.8
        }
    })
    
    # Configura scalper
    scalper = QuantumScalper(
        field_dimensions=8,
        coherence_threshold=0.75,
        resonance_threshold=0.7
    )
    
    try:
        while True:
            # Obtém dados do mercado
            market_data = await exchange.get_market_data(symbol)
            
            # Analisa mercado
            signal = scalper.analyze_market(market_data)
            
            # Atualiza dashboard
            await dashboard.update_metrics(
                capital=await exchange.get_balance(),
                positions=await exchange.get_positions(symbol),
                trades=scalper.trades,
                field_metrics=scalper.field.get_metrics()
            )
            
            # Verifica alertas
            alerts = dashboard.get_alerts()
            for alert in alerts:
                print(f"ALERTA: {alert}")
            
            # Se há sinal de trade
            if signal.direction != 0 and signal.confidence >= scalper.field.coherence_threshold:
                # Calcula tamanho da posição
                size = scalper.get_position_size(
                    capital=capital_per_pair,
                    risk_per_trade=risk_per_trade,
                    atr=signal.atr_volatility
                )
                
                # Executa ordem
                if signal.direction > 0:
                    order = await exchange.create_buy_order(
                        symbol=symbol,
                        size=size,
                        price=market_data['closes'][-1]
                    )
                else:
                    order = await exchange.create_sell_order(
                        symbol=symbol,
                        size=size,
                        price=market_data['closes'][-1]
                    )
                
                print(f"Ordem executada: {order}")
                
                # Salva trade
                await dashboard.save_trade({
                    'id': order['id'],
                    'entry_time': datetime.now(),
                    'exit_time': datetime.now(),
                    'entry_price': order['price'],
                    'size': order['size'],
                    'direction': signal.direction,
                    'field_coherence': signal.field_coherence,
                    'phi_resonance': signal.phi_resonance,
                    'pattern_id': signal.pattern_id
                })
            
            # Aguarda próximo ciclo
            await asyncio.sleep(60)  # 1 minuto
            
    except KeyboardInterrupt:
        print("\nEncerrando trading...")
        # Fecha posições abertas
        await exchange.close_all_positions(symbol)
        
    finally:
        # Salva relatório final
        performance = dashboard.get_performance_metrics()
        field_analysis = dashboard.get_field_analysis()
        
        print("\nRelatório Final:")
        print(f"Total de Trades: {performance['total_trades']}")
        print(f"Taxa de Acerto: {(performance['winning_trades']/performance['total_trades']*100):.1f}%")
        print(f"PnL Total: ${performance['total_pnl']:.2f}")
        print(f"Drawdown Máximo: {(performance['max_drawdown']*100):.1f}%")
        
        print("\nAnálise do Campo:")
        print(f"Coerência: {field_analysis['coherence']['current']:.2f}")
        print(f"Ressonância: {field_analysis['resonance']['current']:.2f}")
        print(f"Força: {field_analysis['strength']['current']:.2f}")

async def main():
    """Função principal"""
    # Exemplo de backtesting
    # Carrega dados históricos (substitua pelo seu método de obtenção de dados)
    historical_data = {
        'opens': [...],    # Lista de preços de abertura
        'highs': [...],    # Lista de preços máximos
        'lows': [...],     # Lista de preços mínimos
        'closes': [...],   # Lista de preços de fechamento
        'volumes': [...]   # Lista de volumes
    }
    
    await run_backtest(
        historical_data=historical_data,
        initial_capital=10000.0,
        risk_per_trade=0.02
    )
    
    # Exemplo de trading em tempo real
    # Substitua com suas credenciais da KuCoin
    await run_live_trading(
        api_key='YOUR_API_KEY',
        api_secret='YOUR_API_SECRET',
        api_passphrase='YOUR_API_PASSPHRASE',
        symbol='BTC-USDT',
        capital_per_pair=1000.0,
        risk_per_trade=0.02
    )

if __name__ == '__main__':
    asyncio.run(main())
