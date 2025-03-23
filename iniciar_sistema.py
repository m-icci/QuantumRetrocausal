#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para iniciar o sistema de trading quântico com CGR
--------------------------------------------------------
Inicia tanto a API quanto a interface Streamlit em processos separados.

Autor: QUALIA (Sistema Retrocausal)
Versão: 1.0
Data: 2025-03-14
"""

import os
import sys
import subprocess
import time
import platform
import signal
import webbrowser
from datetime import datetime

# Processos ativos
processos = []

def imprimir_cabecalho():
    """Imprime o cabeçalho do sistema."""
    print("\n" + "=" * 60)
    print("   SISTEMA DE TRADING QUÂNTICO COM CGR - INICIADOR")
    print("=" * 60)
    print("Iniciando componentes do sistema...\n")

def iniciar_api():
    """Inicia o servidor da API."""
    print("Iniciando API de trading...")
    if platform.system() == "Windows":
        processo = subprocess.Popen([sys.executable, "trading_api.py"], 
                                   creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        processo = subprocess.Popen([sys.executable, "trading_api.py"])
    
    processos.append(processo)
    print("API iniciada. Aguardando 2 segundos para estabilização...")
    time.sleep(2)

def iniciar_interface():
    """Inicia a interface Streamlit."""
    print("Iniciando interface Streamlit...")
    comando = f"{sys.executable} -m streamlit run interface_trading.py"
    
    if platform.system() == "Windows":
        processo = subprocess.Popen(comando, 
                                   creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        processo = subprocess.Popen(comando, shell=True)
    
    processos.append(processo)
    print("Interface Streamlit iniciada.")
    time.sleep(1)
    
    # Abrir navegador
    try:
        webbrowser.open("http://localhost:8501")
        print("Interface aberta no navegador.")
    except:
        print("Não foi possível abrir o navegador automaticamente.")
        print("Acesse manualmente: http://localhost:8501")

def encerrar_processos():
    """Encerra todos os processos abertos."""
    print("\nEncerrando componentes do sistema...")
    for processo in processos:
        if processo.poll() is None:  # Verifica se o processo ainda está rodando
            if platform.system() == "Windows":
                processo.terminate()
            else:
                os.kill(processo.pid, signal.SIGTERM)
    print("Todos os processos encerrados.")

def main():
    """Função principal."""
    imprimir_cabecalho()
    
    try:
        # Iniciar componentes
        iniciar_api()
        iniciar_interface()
        
        print("\nSistema iniciado com sucesso!")
        print("API: http://localhost:8000")
        print("Interface: http://localhost:8501")
        print("\nPressione Ctrl+C para encerrar todos os componentes.")
        
        # Manter rodando até interrupção
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nInterrupção detectada.")
    except Exception as e:
        print(f"\nErro ao iniciar o sistema: {str(e)}")
    finally:
        encerrar_processos()
        print("\nSistema encerrado.")

if __name__ == "__main__":
    main()
