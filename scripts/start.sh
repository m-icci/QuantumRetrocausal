#!/bin/bash

# Configurações
LOG_DIR="logs"
DATA_DIR="data"
ENV_FILE=".env"

# Verifica se diretórios necessários existem
for dir in "$LOG_DIR" "$DATA_DIR"; do
    if [ ! -d "$dir" ]; then
        echo "Criando diretório: $dir"
        mkdir -p "$dir"
    fi
done

# Verifica se arquivo .env existe
if [ ! -f "$ENV_FILE" ]; then
    echo "Erro: Arquivo $ENV_FILE não encontrado!"
    echo "Por favor, copie .env.example para .env e configure as variáveis de ambiente."
    exit 1
fi

# Carrega variáveis de ambiente
source "$ENV_FILE"

# Verifica dependências
echo "Verificando dependências..."
poetry install --no-dev

# Inicia serviços Docker
echo "Iniciando serviços Docker..."
docker-compose up -d prometheus grafana

# Aguarda serviços iniciarem
echo "Aguardando serviços iniciarem..."
sleep 10

# Inicia QUALIA
echo "Iniciando QUALIA..."
poetry run python -m quantum_trading.executar_trading_real "$@"

# Captura CTRL+C
trap 'echo -e "\nEncerrando QUALIA..." && docker-compose down' INT

# Aguarda finalização
wait 