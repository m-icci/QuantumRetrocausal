"""
Configuração de logging para o sistema QUALIA.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# Configurações
LOG_FILE = os.getenv('LOG_FILE', 'qualia.log')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
ENABLE_ADVANCED_MONITORING = os.getenv('ENABLE_ADVANCED_MONITORING', 'True').lower() == 'true'

def setup_logging():
    """
    Configura o sistema de logging com handlers para arquivo e console.
    """
    # Cria o logger
    logger = logging.getLogger('QUALIA')
    logger.setLevel(getattr(logging, LOG_LEVEL))

    # Formata as mensagens
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Handler para arquivo com rotação
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Handler para console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Configurações avançadas de monitoramento
    if ENABLE_ADVANCED_MONITORING:
        setup_advanced_monitoring(logger)

    return logger

def setup_advanced_monitoring(logger):
    """
    Configura monitoramento avançado com métricas detalhadas.
    """
    # Formata as mensagens com métricas adicionais
    advanced_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - '
        '%(quantum_metrics)s - %(morphic_field)s - '
        '%(consciousness_level)s - %(message)s'
    )

    # Handler para métricas avançadas
    metrics_handler = RotatingFileHandler(
        'metrics.log',
        maxBytes=20*1024*1024,  # 20MB
        backupCount=10
    )
    metrics_handler.setFormatter(advanced_formatter)
    logger.addHandler(metrics_handler)

def get_logger(name):
    """
    Retorna um logger configurado para o módulo especificado.
    
    Args:
        name (str): Nome do módulo requisitando o logger
        
    Returns:
        logging.Logger: Logger configurado
    """
    return logging.getLogger(f'QUALIA.{name}')

# Configura o logger principal
logger = setup_logging()

# Exemplo de uso:
if __name__ == '__main__':
    test_logger = get_logger('TEST')
    test_logger.info('Sistema de logging inicializado com sucesso!')
    test_logger.debug('Modo debug ativado')
    test_logger.warning('Exemplo de aviso')
    test_logger.error('Exemplo de erro')
    
    # Exemplo com métricas avançadas
    if ENABLE_ADVANCED_MONITORING:
        extra = {
            'quantum_metrics': 'coherence=0.85',
            'morphic_field': 'resonance=0.92',
            'consciousness_level': 'phi=0.78'
        }
        test_logger.info('Análise quântica completa', extra=extra) 