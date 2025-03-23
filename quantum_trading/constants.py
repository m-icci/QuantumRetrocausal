"""
Constantes para o sistema QUALIA.
"""

from typing import Dict, List

# Configurações de Trading
SUPPORTED_EXCHANGES = ['kucoin', 'kraken']
DEFAULT_TIMEFRAME = '1m'
MAX_TRADES_PER_HOUR = 10
MIN_TRADE_INTERVAL = 60  # segundos
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY = 5  # segundos

# Limites de Operação
MIN_ORDER_SIZE = {
    'BTC/USDT': 0.0001,
    'ETH/USDT': 0.001,
    'XMR/USDT': 0.01
}

MAX_ORDER_SIZE = {
    'BTC/USDT': 1.0,
    'ETH/USDT': 10.0,
    'XMR/USDT': 100.0
}

# Configurações de Proteção
DEFAULT_STOP_LOSS = 0.02  # 2%
DEFAULT_TAKE_PROFIT = 0.03  # 3%
MAX_DRAWDOWN = 0.05  # 5%
MIN_CONFIDENCE = 0.7
MAX_POSITION_SIZE = 0.1  # 10% do saldo

# Métricas Quânticas
QUANTUM_THRESHOLDS = {
    'coherence': 0.7,
    'field_entropy': 0.5,
    'dark_ratio': 0.3
}

MORPHIC_THRESHOLDS = {
    'resonance': 0.6,
    'field_strength': 0.5,
    'consciousness': 0.7
}

# Configurações de Memória
MAX_MEMORY_STATES = 1000
MEMORY_PRUNING_THRESHOLD = 0.8  # 80% de uso
MIN_SIMILARITY_THRESHOLD = 0.6

# Configurações de Análise
ANALYSIS_TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']
MIN_DATA_POINTS = 100
MAX_PATTERN_LENGTH = 50
CORRELATION_THRESHOLD = 0.7

# Configurações de Rede
DEFAULT_API_PORT = 8000
DEFAULT_WS_PORT = 8001
REQUEST_TIMEOUT = 30  # segundos
MAX_CONNECTIONS = 100

# Configurações de Cache
CACHE_TTL = 60  # segundos
MAX_CACHE_SIZE = 1000
CACHE_PRUNING_THRESHOLD = 0.8  # 80% de uso

# Configurações de Backup
BACKUP_INTERVAL = 3600  # 1 hora
MAX_BACKUP_FILES = 24
BACKUP_RETENTION_DAYS = 7

# Mensagens de Erro
ERROR_MESSAGES = {
    'invalid_symbol': 'Símbolo de trading inválido: {}',
    'insufficient_balance': 'Saldo insuficiente para executar operação',
    'invalid_order_size': 'Tamanho de ordem inválido: {}',
    'api_error': 'Erro na API da exchange: {}',
    'quantum_protection': 'Proteção quântica ativada: {}',
    'memory_full': 'Memória holográfica cheia',
    'invalid_timeframe': 'Timeframe inválido: {}',
    'data_insufficient': 'Dados insuficientes para análise',
    'connection_error': 'Erro de conexão: {}'
}

# Mensagens de Log
LOG_MESSAGES = {
    'trade_executed': 'Trade executado: {} {} @ {}',
    'protection_activated': 'Proteção ativada: {}',
    'analysis_complete': 'Análise completa: {}',
    'memory_stored': 'Estado armazenado na memória: {}',
    'system_startup': 'Sistema QUALIA iniciado',
    'system_shutdown': 'Sistema QUALIA finalizado',
    'backup_complete': 'Backup completo: {}'
}

# Configurações de Visualização
CHART_COLORS = {
    'buy': '#00ff00',
    'sell': '#ff0000',
    'neutral': '#808080',
    'protection': '#ff8c00',
    'background': '#000000',
    'text': '#ffffff'
}

CHART_TIMEFRAMES = {
    '1m': {'candles': 60, 'label': '1 Minuto'},
    '5m': {'candles': 60, 'label': '5 Minutos'},
    '15m': {'candles': 60, 'label': '15 Minutos'},
    '1h': {'candles': 24, 'label': '1 Hora'},
    '4h': {'candles': 24, 'label': '4 Horas'},
    '1d': {'candles': 30, 'label': '1 Dia'}
}

# Configurações de Notificação
NOTIFICATION_LEVELS = {
    'info': 0,
    'warning': 1,
    'error': 2,
    'critical': 3
}

NOTIFICATION_CHANNELS = {
    'console': True,
    'log': True,
    'email': False,
    'telegram': False
}

# Configurações de Teste
TEST_CONFIG = {
    'simulation_mode': True,
    'initial_balance': 10000,
    'test_pairs': ['BTC/USDT', 'ETH/USDT'],
    'test_duration': 3600,  # 1 hora
    'random_seed': 42
}

# Exemplo de uso
if __name__ == '__main__':
    print("Exchanges Suportadas:", SUPPORTED_EXCHANGES)
    print("Limites de Ordem (BTC/USDT):", MIN_ORDER_SIZE['BTC/USDT'], "-", MAX_ORDER_SIZE['BTC/USDT'])
    print("Thresholds Quânticos:", QUANTUM_THRESHOLDS)
    print("Thresholds Mórficos:", MORPHIC_THRESHOLDS)
    print("Timeframes de Análise:", ANALYSIS_TIMEFRAMES)
    print("Cores do Gráfico:", CHART_COLORS) 