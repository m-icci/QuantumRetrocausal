"""
Exceções Customizadas
===================

Definição de exceções específicas do sistema QUALIA.
"""

import logging

logger = logging.getLogger(__name__)

class QualiaError(Exception):
    """Exceção base para erros do QUALIA."""
    pass

class MarketError(QualiaError):
    """Erro relacionado a operações de mercado."""
    pass

class APIError(QualiaError):
    """Erro relacionado a chamadas de API."""
    pass

class ConfigError(QualiaError):
    """Erro relacionado a configurações."""
    pass

class ValidationError(QualiaError):
    """Erro de validação de dados."""
    pass

class QuantumError(QualiaError):
    """Erro relacionado a operações quânticas."""
    pass

class MemoryError(QualiaError):
    """Erro relacionado à memória holográfica."""
    pass

class ConsciousnessError(QualiaError):
    """Erro relacionado à consciência de mercado."""
    pass

class MorphicError(QualiaError):
    """Erro relacionado ao campo mórfico."""
    pass

class TradingError(QualiaError):
    """Erro relacionado a operações de trading."""
    pass

class BackupError(QualiaError):
    """Erro relacionado a operações de backup."""
    pass

class StateError(QualiaError):
    """Erro relacionado ao estado do sistema."""
    pass

class ExchangeError(QualiaError):
    """Erro relacionado à comunicação com exchanges."""
    pass

class ExecutionError(QualiaError):
    """Erro relacionado à execução de operações."""
    def __init__(self, operation: str, message: str):
        self.operation = operation
        super().__init__(f"Erro na execução de {operation}: {message}")

class IntegrationError(QualiaError):
    """Erro relacionado à integração entre componentes."""
    def __init__(self, component: str, message: str):
        self.component = component
        super().__init__(f"Erro na integração com {component}: {message}")

class MetricsError(QualiaError):
    """Erro relacionado ao cálculo ou coleta de métricas."""
    def __init__(self, metric: str, message: str):
        self.metric = metric
        super().__init__(f"Erro na métrica {metric}: {message}")

class MonitoringError(QualiaError):
    """Erro relacionado ao monitoramento do sistema."""
    def __init__(self, component: str, message: str):
        self.component = component
        super().__init__(f"Erro no monitoramento de {component}: {message}")

class ApiError(ExchangeError):
    """Erro na API da exchange."""
    def __init__(self, exchange: str, message: str, status_code: int = None):
        self.exchange = exchange
        self.status_code = status_code
        super().__init__(f"Erro na API {exchange}: {message} (status: {status_code})")

class AuthenticationError(ExchangeError):
    """Erro de autenticação com a exchange."""
    def __init__(self, exchange: str, message: str):
        self.exchange = exchange
        super().__init__(f"Erro de autenticação na {exchange}: {message}")

class InsufficientBalanceError(ExchangeError):
    """Saldo insuficiente para executar operação."""
    def __init__(self, symbol: str, required: float, available: float):
        self.symbol = symbol
        self.required = required
        self.available = available
        super().__init__(
            f"Saldo insuficiente para {symbol}. "
            f"Necessário: {required}, Disponível: {available}"
        )

class InvalidOrderError(ExchangeError):
    """Ordem inválida."""
    def __init__(self, symbol: str, side: str, amount: float, reason: str):
        self.symbol = symbol
        self.side = side
        self.amount = amount
        self.reason = reason
        super().__init__(
            f"Ordem inválida para {symbol} {side} {amount}: {reason}"
        )

class QuantumProtectionError(QualiaError):
    """Proteção quântica ativada."""
    def __init__(self, metric: str, value: float, threshold: float):
        self.metric = metric
        self.value = value
        self.threshold = threshold
        super().__init__(
            f"Proteção quântica ativada: {metric} = {value:.2f} "
            f"(threshold: {threshold:.2f})"
        )

class PatternAnalysisError(QualiaError):
    """Erro na análise de padrões."""
    def __init__(self, pattern_type: str, message: str):
        self.pattern_type = pattern_type
        super().__init__(f"Erro na análise de padrão {pattern_type}: {message}")

class ConfigurationError(QualiaError):
    """Erro de configuração."""
    def __init__(self, parameter: str, value: str, message: str):
        self.parameter = parameter
        self.value = value
        super().__init__(f"Configuração inválida {parameter}={value}: {message}")

class DataError(QualiaError):
    """Erro relacionado a dados."""
    def __init__(self, data_type: str, message: str):
        self.data_type = data_type
        super().__init__(f"Erro nos dados {data_type}: {message}")

class NetworkError(QualiaError):
    """Erro de rede."""
    def __init__(self, host: str, port: int, message: str):
        self.host = host
        self.port = port
        super().__init__(f"Erro de rede {host}:{port}: {message}")

class TimeoutError(QualiaError):
    """Erro de timeout."""
    def __init__(self, operation: str, timeout: float):
        self.operation = operation
        self.timeout = timeout
        super().__init__(f"Timeout na operação {operation} após {timeout}s")

class AnalysisError(Exception):
    """Erro genérico para falhas na análise."""
    pass

class BacktestError(Exception):
    """Erro genérico para falhas em backtesting."""
    pass

class CommunicationError(Exception):
    """Erro genérico para falhas de comunicação."""
    pass

class OptimizationError(QualiaError):
    """Exceção para erros durante o processo de otimização."""
    pass

class LoggingError(QualiaError):
    """Exceção para erros relacionados ao sistema de logging."""
    pass

class SimulationError(QualiaError):
    """Exceção para erros durante a simulação."""
    pass

class StrategyError(QualiaError):
    """Exceção para erros relacionados a estratégias de negociação."""
    pass

class VisualizationError(QualiaError):
    """Exceção para erros durante a visualização de dados."""
    pass

__all__ = [
    'QualiaError',
    'MarketError',
    'APIError',
    'ConfigError',
    'ValidationError',
    'QuantumError',
    'MemoryError',
    'ConsciousnessError',
    'MorphicError',
    'TradingError',
    'BackupError',
    'StateError',
    'ExchangeError',
    'ExecutionError',
    'IntegrationError',
    'MetricsError',
    'MonitoringError',
    'ApiError',
    'AuthenticationError',
    'InsufficientBalanceError',
    'InvalidOrderError',
    'QuantumProtectionError',
    'PatternAnalysisError',
    'ConfigurationError',
    'DataError',
    'NetworkError',
    'TimeoutError',
    'AnalysisError',
    'BacktestError',
    'CommunicationError',
    'OptimizationError',
    'SimulationError',
    'StrategyError',
    'VisualizationError'
]

# Exemplo de uso
if __name__ == '__main__':
    try:
        # Simula erro de API
        raise ApiError(
            exchange="KuCoin",
            message="Rate limit exceeded",
            status_code=429
        )
    except ApiError as e:
        print(f"Capturado: {e}")
        print(f"Exchange: {e.exchange}")
        print(f"Status: {e.status_code}")
        
    try:
        # Simula proteção quântica
        raise QuantumProtectionError(
            metric="coherence",
            value=0.3,
            threshold=0.7
        )
    except QuantumProtectionError as e:
        print(f"\nCapturado: {e}")
        print(f"Métrica: {e.metric}")
        print(f"Valor: {e.value}")
        print(f"Threshold: {e.threshold}")
        
    try:
        # Simula erro de memória
        raise MemoryError(
            message="Memória cheia"
        )
    except MemoryError as e:
        print(f"\nCapturado: {e}")