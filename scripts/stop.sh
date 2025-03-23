#!/bin/bash

# Configurações
PID_FILE="qualia.pid"

# Verifica se existe processo rodando
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    
    # Verifica se o processo ainda existe
    if ps -p "$PID" > /dev/null; then
        echo "Parando QUALIA (PID: $PID)..."
        kill "$PID"
        rm "$PID_FILE"
    else
        echo "Processo QUALIA não encontrado. Removendo PID file..."
        rm "$PID_FILE"
    fi
else
    echo "QUALIA não está em execução."
fi

# Para containers Docker
echo "Parando serviços Docker..."
docker-compose down

echo "Sistema QUALIA parado com sucesso!" 