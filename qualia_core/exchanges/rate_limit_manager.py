"""
Gerenciador de Rate Limit para APIs de Exchange
"""
import time
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, Optional
from core.logging.quantum_logger import quantum_logger

class RateLimitManager:
    def __init__(self, max_retries: int = 3, initial_delay: float = 2.0):
        """
        Inicializa gerenciador de rate limit com delays mais conservadores

        Args:
            max_retries: Número máximo de tentativas
            initial_delay: Delay inicial em segundos
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.last_request_time: Dict[str, datetime] = {}
        self.retry_counts: Dict[str, int] = {}

        # Métricas de performance
        self.wait_times: Dict[str, list] = {'private': [], 'public': []}
        self.error_history: Dict[str, list] = {'private': [], 'public': []}

        # Delays específicos por endpoint
        self.endpoint_delays = {
            'OHLC': 5.0,  # Delay maior para chamadas OHLC
            'default_public': 2.0,
            'default_private': 3.0
        }

    def _calculate_backoff(self, retry_count: int) -> float:
        """Calcula delay exponencial com jitter"""
        base_delay = self.initial_delay * (2 ** retry_count)
        jitter = base_delay * 0.1  # 10% de variação
        return base_delay + (jitter * (2 * time.time() % 1 - 0.5))

    def wait_if_needed(self, endpoint: str, method: str = None):
        """
        Aplica delay se necessário antes da próxima chamada

        Args:
            endpoint: Identificador do endpoint ('private' ou 'public')
            method: Método específico (ex: 'OHLC')
        """
        current_time = datetime.now()

        # Determina delay mínimo
        if method and method in self.endpoint_delays:
            min_delay = self.endpoint_delays[method]
        else:
            min_delay = (
                self.endpoint_delays['default_private']
                if endpoint == 'private'
                else self.endpoint_delays['default_public']
            )

        # Verifica último request
        if endpoint in self.last_request_time:
            time_since_last = (current_time - self.last_request_time[endpoint]).total_seconds()

            # Aplica backoff se houver retries
            if endpoint in self.retry_counts and self.retry_counts[endpoint] > 0:
                min_delay = self._calculate_backoff(self.retry_counts[endpoint])

            # Espera se necessário
            if time_since_last < min_delay:
                wait_time = min_delay - time_since_last

                # Registra tempo de espera para métricas
                self.wait_times[endpoint].append(wait_time)

                quantum_logger.debug(
                    f"Aguardando {wait_time:.2f}s antes da próxima chamada",
                    {
                        "endpoint": endpoint,
                        "method": method,
                        "backoff": min_delay,
                        "retry_count": self.retry_counts.get(endpoint, 0),
                        "avg_wait": sum(self.wait_times[endpoint][-10:]) / min(10, len(self.wait_times[endpoint]))
                    }
                )
                time.sleep(wait_time)

        # Atualiza timestamp
        self.last_request_time[endpoint] = current_time

    def handle_error(self, endpoint: str, error: Any) -> bool:
        """
        Gerencia erros e decide se deve tentar novamente

        Args:
            endpoint: Identificador do endpoint
            error: Erro retornado pela API

        Returns:
            bool: True se deve tentar novamente
        """
        # Registra erro no histórico
        error_entry = {
            'timestamp': datetime.now(),
            'error': str(error),
            'retry_count': self.retry_counts.get(endpoint, 0)
        }
        self.error_history[endpoint].append(error_entry)

        # Limita histórico a 100 entradas
        if len(self.error_history[endpoint]) > 100:
            self.error_history[endpoint].pop(0)

        # Inicializa contagem se necessário
        if endpoint not in self.retry_counts:
            self.retry_counts[endpoint] = 0

        # Incrementa contador
        self.retry_counts[endpoint] += 1

        # Verifica se é erro de rate limit
        is_rate_limit = False
        if isinstance(error, (list, str)):
            is_rate_limit = any(
                err in str(error)
                for err in ['EGeneral:Temporary lockout', 'EAPI:Rate limit exceeded']
            )

        if is_rate_limit:
            quantum_logger.warning(
                "Rate limit atingido",
                {
                    "endpoint": endpoint,
                    "retry_count": self.retry_counts[endpoint],
                    "backoff": self._calculate_backoff(self.retry_counts[endpoint]),
                    "error": str(error)
                }
            )

        # Decide se tenta novamente
        should_retry = (
            self.retry_counts[endpoint] < self.max_retries and 
            (is_rate_limit or isinstance(error, Exception))
        )

        if not should_retry:
            # Reseta contador se não vai tentar novamente
            self.retry_counts[endpoint] = 0

        return should_retry

    def reset(self, endpoint: str):
        """
        Reseta contadores para um endpoint

        Args:
            endpoint: Identificador do endpoint
        """
        if endpoint in self.retry_counts:
            self.retry_counts[endpoint] = 0

    def get_metrics(self) -> Dict[str, Any]:
        """
        Retorna métricas de performance

        Returns:
            Dict com métricas
        """
        metrics = {}
        for endpoint in ['private', 'public']:
            recent_waits = self.wait_times[endpoint][-100:]  # Últimos 100 waits
            recent_errors = self.error_history[endpoint][-100:]  # Últimos 100 erros

            metrics[endpoint] = {
                'avg_wait_time': sum(recent_waits) / max(1, len(recent_waits)),
                'max_wait_time': max(recent_waits) if recent_waits else 0,
                'error_rate': len([e for e in recent_errors if 'EAPI:Rate limit' in e['error']]) / max(1, len(recent_errors)),
                'total_errors': len(recent_errors),
                'current_retry_count': self.retry_counts.get(endpoint, 0)
            }

        return metrics