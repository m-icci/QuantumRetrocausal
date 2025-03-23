"""
API de Trading
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
from typing import Dict, Optional
import logging

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Inicializa FastAPI
app = FastAPI(title="Trading API")

# Configuração CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Estado global
trading_state = {
    "is_running": False,
    "current_symbol": "BTC/USDT",
    "current_position": None,
    "last_price": None,
    "balance": 10000.0
}

@app.get("/")
async def root():
    """Endpoint raiz."""
    return {"message": "Trading API Online"}

@app.get("/state")
async def get_state():
    """Retorna estado atual do sistema."""
    return trading_state

@app.post("/start")
async def start_trading():
    """Inicia trading."""
    if trading_state["is_running"]:
        raise HTTPException(status_code=400, detail="Trading já está em execução")
    
    trading_state["is_running"] = True
    logger.info("Trading iniciado")
    return {"message": "Trading iniciado com sucesso"}

@app.post("/stop")
async def stop_trading():
    """Para trading."""
    if not trading_state["is_running"]:
        raise HTTPException(status_code=400, detail="Trading não está em execução")
    
    trading_state["is_running"] = False
    logger.info("Trading parado")
    return {"message": "Trading parado com sucesso"}

def main():
    """Função principal."""
    try:
        uvicorn.run(app, host="127.0.0.1", port=8000)
    except Exception as e:
        logger.error(f"Erro ao iniciar API: {str(e)}")
        raise

if __name__ == "__main__":
    main() 