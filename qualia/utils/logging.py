"""
Enhanced logging configuration for QUALIA Trading System
Implements advanced logging with rotation, JSON formatting and structured logging
"""
import logging
import logging.handlers
import os
import time
import json
from datetime import datetime
from typing import Optional, Dict, Any, ContextManager
from pathlib import Path
from dataclasses import asdict, dataclass
from contextlib import contextmanager

@dataclass
class LogMetadata:
    """Structured metadata for logs"""
    component: str
    operation: str
    correlation_id: str = ""
    quantum_metrics: Optional[Dict[str, float]] = None
    execution_time: Optional[float] = None

class StructuredJsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'file': record.filename,
            'line': record.lineno
        }

        # Add extra fields if present
        if hasattr(record, 'metadata'):
            metadata = record.metadata
            if isinstance(metadata, LogMetadata):
                metadata = asdict(metadata)
            log_data['metadata'] = metadata

        # Add extra context if present
        if hasattr(record, 'event_data'):
            log_data['event_data'] = record.event_data

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_data)

def setup_logger(
    name: str,
    level: Optional[int] = None,
    log_dir: str = "logs",
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5,
    quantum_tracking: bool = True,
    structured: bool = True
) -> logging.Logger:
    """
    Configure a logger with advanced features and rotation

    Args:
        name: Logger name (usually __name__)
        level: Optional logging level override
        log_dir: Directory for log files
        max_bytes: Maximum size of each log file
        backup_count: Number of backup files to keep
        quantum_tracking: Enable quantum state tracking
        structured: Use JSON structured logging
    """
    logger = logging.getLogger(name)

    if level is not None:
        logger.setLevel(level)
    else:
        logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers.clear()

    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Console handler
    console_handler = logging.StreamHandler()
    if structured:
        console_formatter = StructuredJsonFormatter()
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Main log file handler
    main_log_file = log_path / f"{name.replace('.', '_')}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        main_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)

    # Quantum state tracking
    if quantum_tracking:
        quantum_log_file = log_path / f"{name.replace('.', '_')}_quantum.log"
        quantum_handler = logging.handlers.RotatingFileHandler(
            quantum_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        quantum_handler.setFormatter(console_formatter)
        logger.addHandler(quantum_handler)

    # Error tracking
    error_log_file = log_path / f"{name.replace('.', '_')}_error.log"
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    error_handler.setFormatter(console_formatter)
    error_handler.setLevel(logging.ERROR)
    logger.addHandler(error_handler)

    return logger

@contextmanager
def log_operation(
    logger: logging.Logger,
    operation: str,
    component: str,
    correlation_id: str = "",
    **extra_context
) -> ContextManager[Dict[str, Any]]:
    """
    Context manager for logging operations with timing

    Usage:
        with log_operation(logger, "trade_execution", "quantum_trader") as ctx:
            # Perform operation
            ctx['additional_info'] = some_value
    """
    start_time = time.time()
    metadata = LogMetadata(
        component=component,
        operation=operation,
        correlation_id=correlation_id or str(int(start_time * 1000))
    )

    context = {'metadata': metadata}
    try:
        logger.info(f"Starting {operation}", extra={'metadata': metadata})
        yield context

        execution_time = time.time() - start_time
        metadata.execution_time = execution_time
        context.update(extra_context)

        logger.info(
            f"Completed {operation}",
            extra={'metadata': metadata, **context}
        )

    except Exception as e:
        execution_time = time.time() - start_time
        metadata.execution_time = execution_time
        logger.exception(
            f"Failed {operation}",
            extra={'metadata': metadata, **context}
        )
        raise

def log_quantum_state(
    logger: logging.Logger,
    state_data: Dict[str, Any],
    component: str,
    correlation_id: str = ""
) -> None:
    """Log quantum state information in structured format"""
    try:
        metadata = LogMetadata(
            component=component,
            operation="quantum_state_update",
            correlation_id=correlation_id,
            quantum_metrics=state_data
        )

        logger.info(
            "Quantum state updated",
            extra={'metadata': metadata}
        )

    except Exception as e:
        logger.exception(
            "Failed to log quantum state",
            extra={
                'error': str(e),
                'state_data': state_data
            }
        )

def log_market_event(
    logger: logging.Logger,
    event_type: str,
    event_data: Dict[str, Any],
    correlation_id: str = ""
) -> None:
    """Log market events in structured format"""
    try:
        metadata = LogMetadata(
            component="market_events",
            operation=event_type,
            correlation_id=correlation_id
        )

        logger.info(
            f"Market event: {event_type}",
            extra={
                'metadata': metadata,
                'event_data': event_data
            }
        )

    except Exception as e:
        logger.exception(
            "Failed to log market event",
            extra={
                'error': str(e),
                'event_type': event_type,
                'event_data': event_data
            }
        )

def setup_performance_logging(logger: logging.Logger) -> None:
    """Setup performance monitoring with structured logging"""
    try:
        log_path = Path("logs")
        perf_log_file = log_path / f"{logger.name}_performance.log"

        perf_handler = logging.handlers.RotatingFileHandler(
            perf_log_file,
            maxBytes=10485760,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        perf_formatter = StructuredJsonFormatter()
        perf_handler.setFormatter(perf_formatter)
        logger.addHandler(perf_handler)

    except Exception as e:
        logger.exception("Failed to setup performance logging")

def log_performance_metric(
    logger: logging.Logger,
    metric_name: str,
    value: float,
    unit: str = "ms",
    correlation_id: str = "",
    **extra_context
) -> None:
    """Log performance metrics in structured format"""
    try:
        metadata = LogMetadata(
            component="performance",
            operation=metric_name,
            correlation_id=correlation_id
        )

        logger.info(
            f"Performance metric: {metric_name}",
            extra={
                'metadata': metadata,
                'metric_value': value,
                'unit': unit,
                **extra_context
            }
        )

    except Exception as e:
        logger.exception(
            "Failed to log performance metric",
            extra={
                'error': str(e),
                'metric_name': metric_name,
                'value': value
            }
        )