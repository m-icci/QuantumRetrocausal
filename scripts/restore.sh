#!/bin/bash

# Configurações
BACKUP_DIR="backups"

# Verifica se foi fornecido um arquivo de backup
if [ $# -ne 1 ]; then
    echo "Uso: $0 <nome_do_backup>"
    echo -e "\nBackups disponíveis:"
    ls -lh "${BACKUP_DIR}"
    exit 1
fi

BACKUP_FILE="${BACKUP_DIR}/$1"

# Verifica se o arquivo existe
if [ ! -f "$BACKUP_FILE" ]; then
    echo "Erro: Arquivo de backup não encontrado: $BACKUP_FILE"
    exit 1
fi

# Confirma restauração
echo "ATENÇÃO: Esta operação irá sobrescrever os dados atuais!"
read -p "Deseja continuar? (s/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Ss]$ ]]; then
    echo "Operação cancelada."
    exit 1
fi

# Cria backup dos dados atuais
DATE=$(date +%Y%m%d_%H%M%S)
CURRENT_BACKUP="qualia_pre_restore_${DATE}.tar.gz"
echo "Criando backup dos dados atuais..."
tar -czf "${BACKUP_DIR}/${CURRENT_BACKUP}" \
    --exclude="*.pyc" \
    --exclude="__pycache__" \
    --exclude=".git" \
    --exclude=".env" \
    --exclude="venv" \
    --exclude="*.egg-info" \
    "data" "logs" "quantum_trading"

# Restaura backup
echo "Restaurando backup: $BACKUP_FILE"
tar -xzf "$BACKUP_FILE"

# Verifica se restauração foi bem sucedida
if [ $? -eq 0 ]; then
    echo "Backup restaurado com sucesso!"
    echo "Um backup dos dados anteriores foi criado: ${BACKUP_DIR}/${CURRENT_BACKUP}"
else
    echo "Erro ao restaurar backup!"
    exit 1
fi