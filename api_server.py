#!/usr/bin/env python3
"""
API Server para o Quantum Trading System
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Optional
import logging
import os
from datetime import datetime
from qualia.quantum_trader import QuantumTrader
from qualia.utils.logging import setup_logger
from uvicorn import run

# Configuração de logging
logger = setup_logger('api_server')

# Inicializa a aplicação FastAPI
app = FastAPI(title="Quantum Trading System API")

# Configuração do CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variável global para o trader
trader: Optional[QuantumTrader] = None

@app.on_event("startup")
async def startup_event():
    """Inicializa o trader quando o servidor inicia"""
    global trader
    try:
        logger.info("Iniciando Quantum Trading System...")
        trader = QuantumTrader(
            trading_pairs=['BTC/USDT'],
            quantum_dimension=64,
            consciousness_threshold=0.7
        )
        logger.info("Sistema inicializado com sucesso!")
    except Exception as e:
        logger.error(f"Erro ao inicializar sistema: {e}")
        raise e

@app.get("/status")
async def get_status() -> Dict[str, Any]:
    """Retorna o status atual do sistema"""
    if not trader:
        raise HTTPException(status_code=503, detail="Sistema não inicializado")
    
    try:
        return {
            "status": "running",
            "live_trading": trader.trading_config.get('live_trading', False),
            "auto_trading_active": trader.auto_trading_active,
            "last_update": datetime.now().isoformat(),
            "total_trades": len(trader.trade_history.trades) if hasattr(trader.trade_history, 'trades') else 0,
            "winning_trades": trader.trade_history.winning_trades if hasattr(trader.trade_history, 'winning_trades') else 0,
            "losing_trades": trader.trade_history.losing_trades if hasattr(trader.trade_history, 'losing_trades') else 0
        }
    except Exception as e:
        logger.error(f"Erro ao obter status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/balance")
async def get_balance() -> Dict[str, Any]:
    """Retorna saldos atuais"""
    if not trader:
        logger.error("Trader não inicializado ao tentar obter saldo")
        raise HTTPException(status_code=503, detail="Sistema não inicializado")
    
    try:
        balance = trader.get_balance()
        return balance
    except Exception as e:
        logger.error(f"Erro ao obter saldo: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auto-trading/start")
async def start_auto_trading(symbol: str = "BTC/USDT") -> Dict[str, Any]:
    """Inicia o auto trading"""
    if not trader:
        raise HTTPException(status_code=503, detail="Sistema não inicializado")
    
    try:
        trader.start_auto_trading(symbol)
        return {"status": "success", "message": f"Auto trading iniciado para {symbol}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/auto-trading/stop")
async def stop_auto_trading() -> Dict[str, Any]:
    """Para o auto trading"""
    if not trader:
        raise HTTPException(status_code=503, detail="Sistema não inicializado")
    
    try:
        trader.auto_trading_active = False
        return {"status": "success", "message": "Auto trading parado"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/get_system_metrics")
async def get_system_metrics() -> Dict[str, Any]:
    """Retorna métricas do sistema"""
    if not trader:
        logger.error("Trader não inicializado ao tentar obter métricas do sistema")
        raise HTTPException(status_code=503, detail="Sistema não inicializado")
    
    try:
        return {
            "status": "ONLINE",
            "last_update": str(datetime.now()),
            "live_trading": trader.trading_config.get('live_trading', False),
            "auto_trading_active": trader.auto_trading_active,
            "cpu_usage": 0,  # TODO: Implementar métricas reais
            "memory_usage": 0,
            "total_trades": trader.trade_history.total_trades,
            "winning_trades": trader.trade_history.winning_trades,
            "losing_trades": trader.trade_history.losing_trades,
            "total_profit": trader.trade_history.total_profit,
            "max_drawdown": trader.trade_history.max_drawdown,
            "win_rate": (trader.trade_history.winning_trades / trader.trade_history.total_trades * 100) if trader.trade_history.total_trades > 0 else 0
        }
    except Exception as e:
        logger.error(f"Erro ao obter métricas do sistema: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_market_data")
async def get_market_data() -> Dict[str, Any]:
    """Retorna dados do mercado"""
    if not trader:
        logger.error("Trader não inicializado ao tentar obter dados de mercado")
        raise HTTPException(status_code=503, detail="Sistema não inicializado")
    
    try:
        symbol = "BTC/USDT"
        ticker = trader.exchange.fetch_ticker(symbol)
        ohlcv = trader.exchange.fetch_ohlcv(symbol, '1m', limit=100)
        
        return {
            "symbol": symbol,
            "last_price": ticker['last'],
            "bid": ticker['bid'],
            "ask": ticker['ask'],
            "volume": ticker['baseVolume'],
            "timestamps": [candle[0] for candle in ohlcv],
            "prices": [candle[4] for candle in ohlcv]  # Closing prices
        }
    except Exception as e:
        logger.error(f"Erro ao obter dados do mercado: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_open_positions")
async def get_open_positions() -> Dict[str, Any]:
    """Retorna posições abertas"""
    if not trader:
        logger.error("Trader não inicializado ao tentar obter posições abertas")
        raise HTTPException(status_code=503, detail="Sistema não inicializado")
    
    try:
        positions = []
        for pos in trader.open_positions:
            positions.append({
                "symbol": pos.get('symbol', ''),
                "side": pos.get('side', ''),
                "entry_price": pos.get('entry_price', 0.0),
                "pnl": pos.get('pnl', 0.0)
            })
        return {"positions": positions}
    except Exception as e:
        logger.error(f"Erro ao obter posições: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_active_alerts")
async def get_active_alerts() -> Dict[str, Any]:
    """Retorna alertas ativos"""
    if not trader:
        logger.error("Trader não inicializado ao tentar obter alertas")
        raise HTTPException(status_code=503, detail="Sistema não inicializado")
    
    try:
        alerts = []
        if hasattr(trader, 'analyzer') and hasattr(trader.analyzer, 'check_alerts'):
            market_state = trader.market_states.get("BTC/USDT")
            if market_state:
                metrics = trader.analyzer.get_quantum_metrics(market_state)
                alerts = trader.analyzer.check_alerts(metrics)
        
        return {"alerts": alerts}
    except Exception as e:
        logger.error(f"Erro ao obter alertas: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_trading_config")
async def update_trading_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Atualiza configuração de trading"""
    if not trader:
        raise HTTPException(status_code=503, detail="Sistema não inicializado")
    
    try:
        trader.update_trading_config(config)
        return {"status": "success", "message": "Configuração atualizada com sucesso"}
    except Exception as e:
        logger.error(f"Erro ao atualizar configuração: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_trading_config")
async def get_trading_config() -> Dict[str, Any]:
    """Retorna configuração atual do trading"""
    if not trader:
        raise HTTPException(status_code=503, detail="Sistema não inicializado")
    
    try:
        return {
            "risk_factor": trader.trading_config.get('risk_factor', 0.01),
            "min_trade_interval": trader.trading_config.get('min_trade_interval', 60),
            "max_spread_pct": trader.trading_config.get('max_spread_pct', 0.001),
            "signal_threshold": trader.trading_config.get('signal_threshold', 0.7)
        }
    except Exception as e:
        logger.error(f"Erro ao obter configuração: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """Função principal"""
    try:
        # Inicia o servidor na porta 8000
        run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logger.error(f"Erro fatal: {e}")
        raise e

if __name__ == "__main__":
    main()