"""
QUALIA API Server
Modern FastAPI implementation for quantum consciousness and trading
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Dict, Any, List
import uvicorn

from qualia.quantum_trader import QuantumTrader
from qualia.quantum_layer import QuantumLayer
from qualia.quantum.quantum_consciousness_integrator import QuantumConsciousnessIntegrator
from qualia.quantum.quantum_transformation_manifesto import QuantumTransformationManifesto

app = FastAPI(
    title="QUALIA API",
    description="Quantum Consciousness and Trading System API",
    version="2.0.0"
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
trader = QuantumTrader(simulation_mode=True)
consciousness = QuantumConsciousnessIntegrator()
manifesto = QuantumTransformationManifesto()

@app.get("/")
async def root(request: Request):
    """
    Serve the main dashboard
    """
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "auto_trading": True,
            "balances": {"USDT": 1000.0, "BTC": 0.1},
            "metrics": {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_profit_usdt": 0.0,
                "total_profit_percent": 0.0
            },
            "profit_data": {"x": [], "y": []}
        }
    )

@app.get("/api/status")
async def get_status():
    """
    Get current system status
    """
    return {
        "name": "QUALIA API",
        "version": "2.0.0",
        "status": "operational",
        "quantum_metrics": consciousness.get_current_metrics()
    }

@app.post("/consciousness/explore")
async def explore_consciousness(context: Dict[str, Any]):
    """
    Explore quantum consciousness landscape
    """
    try:
        trajectory = consciousness.explore_consciousness_landscape()
        narrative = consciousness.generate_philosophical_narrative()
        
        return {
            "trajectory": trajectory,
            "narrative": narrative,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transformation/manifest")
async def quantum_transformation(potential: Dict[str, Any]):
    """
    Execute quantum transformation protocol
    """
    try:
        trajectory = manifesto.explore_transformation_landscape()
        narrative = manifesto.generate_emergent_narrative()
        
        return {
            "trajectory": trajectory,
            "narrative": narrative,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market/analysis/{symbol}")
async def analyze_market(symbol: str):
    """
    Get quantum market analysis
    """
    try:
        analysis = trader.analyze_market(symbol)
        return {
            "analysis": analysis,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trade/execute")
async def execute_trade(trade_params: Dict[str, Any]):
    """
    Execute trade with quantum consciousness integration
    """
    try:
        result = trader.execute_trade(
            symbol=trade_params["symbol"],
            tipo=trade_params["type"],
            quantidade=trade_params["amount"],
            stop_loss=trade_params["stop_loss"],
            take_profit=trade_params["take_profit"]
        )
        return {
            "result": result,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
