"""
Constantes do Sistema
===================

Definições de constantes utilizadas em todo o sistema QUALIA.
"""

# Configurações gerais
MAX_MEMORY_STATES = 1000
SIMILARITY_THRESHOLD = 0.85
COHERENCE_THRESHOLD = 0.75
PROTECTION_THRESHOLD = 0.90

# Configurações de trading
MAX_POSITION_SIZE = 1.0  # 100% do capital disponível
MIN_CONFIDENCE = 0.80    # Confiança mínima para operar
STOP_LOSS = 0.02        # 2% de stop loss
TAKE_PROFIT = 0.05      # 5% de take profit

# Configurações de análise
PATTERN_SIZE = 144      # 24h * 6 (10 min candles)
MEMORY_DIMENSION = 256  # Dimensão da memória holográfica
CONSCIOUSNESS_LEVELS = 5  # Níveis de consciência

# Configurações de mercado
EXCHANGES = ["binance", "kucoin", "kraken"]
DEFAULT_TIMEFRAME = "10m"
DEFAULT_QUOTE = "USDT"
DEFAULT_PAIRS = [
    "BTC/USDT",
    "ETH/USDT",
    "XMR/USDT"
]

# Configurações de métricas
METRICS_INTERVAL = 5  # segundos
METRICS_PREFIX = "qualia_"
METRICS = [
    "quantum_coherence",
    "market_consciousness",
    "morphic_field_strength",
    "trading_performance"
]

# Configurações de backup
BACKUP_INTERVAL = 3600  # 1 hora
MAX_BACKUPS = 168      # 7 dias

# Configurações de API
API_HOST = "0.0.0.0"
API_PORT = 8000
METRICS_PORT = 8001
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}" 