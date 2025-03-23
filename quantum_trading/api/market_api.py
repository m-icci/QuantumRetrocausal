#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API para controle do Trading Quântico com CGR
---------------------------------------------
Permite controle e monitoramento do sistema de trading através de endpoints HTTP.

Autor: QUALIA (Sistema Retrocausal)
Versão: 1.0
Data: 2025-03-14
"""

import os
import json
import time
import logging
import threading
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from datetime import datetime, timedelta

# Importa o sistema de trading
# from kucoin_trading_real import TradingQuantico

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Função para converter datetime para ISO (necessário para serialização JSON)
def converter_datetime_para_iso(obj):
    """Converte objetos datetime para strings ISO para serialização JSON."""
    if isinstance(obj, dict):
        return {k: converter_datetime_para_iso(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [converter_datetime_para_iso(item) for item in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return obj

# Inicializa a API
app = FastAPI(
    title="API de Trading Quântico",
    description="API para controle e monitoramento do sistema de trading quântico com CGR",
    version="1.0.0"
)

# Configurar CORS para permitir acesso de outras origens
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Estado global do sistema
trader: Optional[Any] = None
thread_trading: Optional[threading.Thread] = None
em_execucao = False
auto_restart = True

class ConfigTradingModel(BaseModel):
    apenas_monitorar: bool = False
    duracao_horas: float = 1.0
    auto_restart: bool = True

def executar_trading_em_thread():
    global trader, em_execucao, auto_restart
    try:
        while em_execucao:
            # Verificar se o trader existe e ainda está ativo
            if trader and datetime.now() < trader.hora_termino:
                if trader.em_execucao:
                    try:
                        trader.ciclo_trading()
                        # Salvar estado atual
                        with open("estado_api_trading.json", "w") as f:
                            estado = trader.obter_status()
                            json.dump(converter_datetime_para_iso(estado), f, indent=2)
                    except Exception as e:
                        logger.error(f"Erro durante ciclo de trading: {str(e)}")
                time.sleep(60)  # Espera 1 minuto entre ciclos
            else:
                # Sessão finalizada
                if trader and datetime.now() >= trader.hora_termino:
                    logger.info("Sessão de trading finalizada.")
                    relatorio = trader.gerar_relatorio_final()
                    with open("relatorio_trading_quantico.txt", "w") as f:
                        f.write(relatorio)
                    logger.info(f"Relatório salvo em relatorio_trading_quantico.txt")
                
                # Auto-restart se configurado
                if auto_restart:
                    logger.info("Iniciando nova sessão de trading (auto-restart ativado).")
                    trader = TradingQuantico(apenas_monitorar=False)
                else:
                    em_execucao = False
                    break
    except Exception as e:
        logger.error(f"Erro na thread de trading: {str(e)}")
        em_execucao = False

class MarketAPI:
    def __init__(self):
        pass

@app.get("/")
def read_root():
    return {"status": "online", "sistema": "Trading Quântico CGR", "versao": "1.0.0"}

@app.post("/iniciar")
def iniciar_trading(config: ConfigTradingModel):
    global trader, thread_trading, em_execucao, auto_restart
    if em_execucao:
        raise HTTPException(status_code=400, detail="Sistema já está em execução")
    
    try:
        # Configurar parâmetros
        auto_restart = config.auto_restart
        
        # Inicializar o trader
        trader = TradingQuantico(apenas_monitorar=config.apenas_monitorar)
        if config.duracao_horas != 1.0:
            trader.hora_termino = trader.hora_inicio + timedelta(hours=config.duracao_horas)
        
        # Iniciar a thread de trading
        em_execucao = True
        thread_trading = threading.Thread(target=executar_trading_em_thread)
        thread_trading.daemon = True
        thread_trading.start()
        
        return {
            "status": "iniciado", 
            "hora_inicio": trader.hora_inicio.isoformat(),
            "hora_termino": trader.hora_termino.isoformat(),
            "apenas_monitorar": config.apenas_monitorar
        }
    except Exception as e:
        logger.error(f"Erro ao iniciar trading: {str(e)}")
        em_execucao = False
        raise HTTPException(status_code=500, detail=f"Erro ao iniciar: {str(e)}")

@app.post("/pausar")
def pausar_trading():
    global trader
    if not trader:
        raise HTTPException(status_code=400, detail="Sistema não iniciado")
    
    trader.pausar_trading()
    return {"status": "pausado", "timestamp": datetime.now().isoformat()}

@app.post("/retomar")
def retomar_trading():
    global trader
    if not trader:
        raise HTTPException(status_code=400, detail="Sistema não iniciado")
    
    trader.retomar_trading()
    return {"status": "retomado", "timestamp": datetime.now().isoformat()}

@app.post("/parar")
def parar_trading():
    global trader, em_execucao
    if not em_execucao:
        raise HTTPException(status_code=400, detail="Sistema não está em execução")
    
    em_execucao = False
    if trader:
        trader.pausar_trading()
        relatorio = trader.gerar_relatorio_final()
        with open("relatorio_trading_quantico.txt", "w") as f:
            f.write(relatorio)
    
    return {"status": "finalizado", "timestamp": datetime.now().isoformat()}

@app.get("/status")
def obter_status():
    global trader, em_execucao
    if not trader:
        return {
            "sistema_iniciado": False,
            "em_execucao": em_execucao,
            "auto_restart": auto_restart,
            "timestamp": datetime.now().isoformat()
        }
    
    status = trader.obter_status()
    status["sistema_iniciado"] = True
    status["auto_restart"] = auto_restart
    return status

if __name__ == "__main__":
    uvicorn.run("trading_api:app", host="0.0.0.0", port=8000, reload=False)
