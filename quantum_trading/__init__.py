"""
QUALIA - Sistema Quântico de Trading

QUALIA é um sistema quântico-computacional avançado e auto-evolutivo projetado para
trading de alta frequência, com foco especial em scalping e arbitragem adaptativa.
"""

import logging
from logging.handlers import RotatingFileHandler
import os
import sys
from typing import Dict, Any

__version__ = "0.2.0"
__author__ = "Quantum Consciousness Team"

# Configurar logger raiz
logger = logging.getLogger("quantum_trading")
logger.setLevel(logging.INFO)

# Verificar se diretório de logs existe
if not os.path.exists("logs"):
    os.makedirs("logs", exist_ok=True)

# Configurar handler de arquivo
file_handler = RotatingFileHandler(
    "logs/qualia.log",
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
))
logger.addHandler(file_handler)

# Configurar handler de console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
))
logger.addHandler(console_handler)

# Componentes disponíveis
available_components = {
    "market_api": True,
    "scalping": True,
    "integrated_scalping": True,  # Novo sistema integrado
    "wave_strategy": True,
    "optimization": True,
    "backtesting": True,
    "monitoring": True,
    "visualization": True
}

def check_component(component_name: str) -> bool:
    """Verifica se um componente está disponível"""
    return available_components.get(component_name, False)

def get_version() -> str:
    """Retorna a versão atual do sistema"""
    return __version__

def get_system_info() -> Dict[str, Any]:
    """Retorna informações sobre o sistema"""
    return {
        "version": __version__,
        "author": __author__,
        "available_components": {k: v for k, v in available_components.items() if v},
        "python_version": sys.version
    }

# Inicialização
logger.info(f"QUALIA v{__version__} inicializado")
logger.info(f"Componentes disponíveis: {[k for k, v in available_components.items() if v]}")

# Imports principais
from .market_api import MarketAPI
from .strategies.wave_strategy import WAVEStrategy

# Importações do sistema integrado de scalping
try:
    from .integrated_quantum_scalping import IntegratedQuantumScalping
    logger.info("Integrated Quantum Scalping system loaded successfully")
except ImportError as e:
    logger.warning(f"Could not load Integrated Quantum Scalping system: {str(e)}")
    IntegratedQuantumScalping = None

# Detecção automática de estratégias
try:
    from .strategies import available_strategies
    logger.info(f"Available strategies: {available_strategies}")
except ImportError as e:
    logger.warning(f"Could not load strategies: {str(e)}")
    available_strategies = []

def init_system(config=None):
    """Initialize the quantum trading system with default settings"""
    if config is None:
        config = {}
    
    logger.info("Initializing QUALIA trading system...")
    
    try:
        if IntegratedQuantumScalping:
            scalping_system = IntegratedQuantumScalping(config)
            logger.info("Quantum scalping system initialized")
            return scalping_system
        else:
            logger.warning("Integrated Quantum Scalping system not available")
            return None
    except Exception as e:
        logger.error(f"Error initializing system: {str(e)}")
        return None
