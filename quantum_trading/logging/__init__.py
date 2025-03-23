"""
Módulo de logging para o projeto QUALIA.
"""

# Importações padrão
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Renomeie a importação do módulo logging padrão
import logging as python_logging

@dataclass
class LogEntry:
    """Estrutura de dados para uma entrada de log."""
    timestamp: str
    level: str
    message: str
    quantum_data: Optional[Dict[str, Any]] = None
    error: Optional[Exception] = None

class LogFormatter(python_logging.Formatter):
    """Formatador personalizado para logs com dados quânticos."""
    
    def format(self, record):
        """Formata a mensagem de log com dados quânticos."""
        if hasattr(record, 'quantum_data'):
            quantum_info = json.dumps(record.quantum_data)
            record.msg = f"{record.msg} | Quantum Data: {quantum_info}"
        return super().format(record)

class LogHandler(python_logging.StreamHandler):
    """Handler personalizado para logs com armazenamento em memória."""
    
    def __init__(self):
        super().__init__()
        self.logs: List[LogEntry] = []
        self.setFormatter(LogFormatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    def emit(self, record):
        """Emite o log e o armazena em memória."""
        try:
            msg = self.format(record)
            quantum_data = getattr(record, 'quantum_data', None)
            error = getattr(record, 'error', None)
            
            log_entry = LogEntry(
                timestamp=datetime.fromtimestamp(record.created).isoformat(),
                level=record.levelname,
                message=msg,
                quantum_data=quantum_data,
                error=error
            )
            
            self.logs.append(log_entry)
            print(msg)
        except Exception:
            self.handleError(record)
    
    def get_logs(self) -> List[LogEntry]:
        """Retorna todos os logs armazenados."""
        return self.logs

class LogAnalyzer:
    """Analisador de logs com funcionalidades avançadas."""
    
    def __init__(self, logs: List[LogEntry]):
        self.logs = logs
    
    def filter_by_level(self, level: str) -> List[LogEntry]:
        """Filtra logs por nível."""
        return [log for log in self.logs if log.level == level]
    
    def filter_by_time_range(self, start_time: str, end_time: str) -> List[LogEntry]:
        """Filtra logs por intervalo de tempo."""
        return [
            log for log in self.logs 
            if start_time <= log.timestamp <= end_time
        ]
    
    def get_error_patterns(self) -> Dict[str, int]:
        """Analisa padrões de erros nos logs."""
        error_patterns = {}
        for log in self.logs:
            if log.error:
                error_type = type(log.error).__name__
                error_patterns[error_type] = error_patterns.get(error_type, 0) + 1
        return error_patterns
    
    def get_quantum_metrics(self) -> Dict[str, List[float]]:
        """Extrai métricas quânticas dos logs."""
        metrics = {}
        for log in self.logs:
            if log.quantum_data:
                for key, value in log.quantum_data.items():
                    if isinstance(value, (int, float)):
                        if key not in metrics:
                            metrics[key] = []
                        metrics[key].append(value)
        return metrics

class QuantumLogger:
    """Logger personalizado para o sistema QUALIA com suporte a dados quânticos."""
    
    def __init__(self, name: str = "QUALIA"):
        self.logger = python_logging.getLogger(name)
        self.logger.setLevel(python_logging.INFO)
        
        # Adiciona handler personalizado
        self.handler = LogHandler()
        self.logger.addHandler(self.handler)
    
    def log(self, level: int, message: str, quantum_data: Optional[Dict[str, Any]] = None, error: Optional[Exception] = None):
        """Registra uma mensagem de log com dados quânticos opcionais."""
        extra = {'quantum_data': quantum_data, 'error': error}
        self.logger.log(level, message, extra=extra)
    
    def info(self, message: str, quantum_data: Optional[Dict[str, Any]] = None):
        """Registra uma mensagem de informação."""
        self.log(python_logging.INFO, message, quantum_data)
    
    def warning(self, message: str, quantum_data: Optional[Dict[str, Any]] = None):
        """Registra uma mensagem de aviso."""
        self.log(python_logging.WARNING, message, quantum_data)
    
    def error(self, message: str, error: Optional[Exception] = None, quantum_data: Optional[Dict[str, Any]] = None):
        """Registra uma mensagem de erro."""
        self.log(python_logging.ERROR, message, quantum_data, error)
    
    def get_logs(self) -> List[LogEntry]:
        """Retorna todos os logs registrados."""
        return self.handler.get_logs()
    
    def get_analyzer(self) -> LogAnalyzer:
        """Retorna um analisador de logs."""
        return LogAnalyzer(self.get_logs())

class LoggingManager:
    """Gerenciador centralizado de logging para o sistema QUALIA."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.loggers: Dict[str, QuantumLogger] = {}
            self.initialized = True
    
    def get_logger(self, name: str) -> QuantumLogger:
        """Retorna ou cria um logger com o nome especificado."""
        if name not in self.loggers:
            self.loggers[name] = QuantumLogger(name)
        return self.loggers[name]
    
    def get_all_logs(self) -> Dict[str, List[LogEntry]]:
        """Retorna todos os logs de todos os loggers."""
        return {name: logger.get_logs() for name, logger in self.loggers.items()}
    
    def clear_logs(self):
        """Limpa todos os logs de todos os loggers."""
        for logger in self.loggers.values():
            logger.handler.logs.clear()

# Exporta símbolos
__all__ = [
    'LogEntry',
    'LogFormatter',
    'LogHandler',
    'LogAnalyzer',
    'QuantumLogger',
    'LoggingManager',
    'python_logging'  # Exporta o módulo renomeado
]