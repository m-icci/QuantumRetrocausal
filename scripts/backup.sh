#!/bin/bash

# Configurações
BACKUP_DIR="backups"
DATA_DIR="data"
LOGS_DIR="logs"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="qualia_backup_${DATE}.tar.gz"

# Verifica se diretório de backup existe
if [ ! -d "$BACKUP_DIR" ]; then
    mkdir -p "$BACKUP_DIR"
fi

# Cria backup
echo "Iniciando backup do sistema QUALIA..."
tar -czf "${BACKUP_DIR}/${BACKUP_NAME}" \
    --exclude="*.pyc" \
    --exclude="__pycache__" \
    --exclude=".git" \
    --exclude=".env" \
    --exclude="venv" \
    --exclude="*.egg-info" \
    "${DATA_DIR}" "${LOGS_DIR}" "quantum_trading"

# Verifica se backup foi criado com sucesso
if [ $? -eq 0 ]; then
    echo "Backup criado com sucesso: ${BACKUP_DIR}/${BACKUP_NAME}"
    
    # Remove backups antigos (mantém últimos 7 dias)
    find "${BACKUP_DIR}" -name "qualia_backup_*.tar.gz" -mtime +7 -delete
    
    # Lista backups disponíveis
    echo -e "\nBackups disponíveis:"
    ls -lh "${BACKUP_DIR}"
else
    echo "Erro ao criar backup!"
    exit 1
fi 