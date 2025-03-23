"""
Configuração de Logging
=====================

Configuração centralizada do sistema de logging do QUALIA.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

# Configurações
LOG_DIR = "logs"
LOG_FILE = "qualia.log"
ERROR_FILE = "error.log"
METRICS_FILE = "metrics.log"
MAX_BYTES = 10 * 1024 * 1024  # 10MB
BACKUP_COUNT = 5

# Cria diretório de logs se não existir
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Formatos de log
DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
METRICS_FORMAT = "%(asctime)s %(message)s"

# Configuração do logger principal
def setup_logger(name=None):
    """Configura e retorna um logger."""
    logger = logging.getLogger(name or "qualia")
    logger.setLevel(logging.INFO)
    
    # Handler para arquivo principal
    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, LOG_FILE),
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT
    )
    file_handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))
    logger.addHandler(file_handler)
    
    # Handler para erros
    error_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, ERROR_FILE),
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))
    logger.addHandler(error_handler)
    
    # Handler para console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))
    logger.addHandler(console_handler)
    
    return logger

# Configuração do logger de métricas
def setup_metrics_logger():
    """Configura e retorna um logger para métricas."""
    logger = logging.getLogger("qualia.metrics")
    logger.setLevel(logging.INFO)
    
    handler = RotatingFileHandler(
        os.path.join(LOG_DIR, METRICS_FILE),
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT
    )
    handler.setFormatter(logging.Formatter(METRICS_FORMAT))
    logger.addHandler(handler)
    
    return logger

# Logger principal
logger = setup_logger()

# Logger de métricas
metrics_logger = setup_metrics_logger() 