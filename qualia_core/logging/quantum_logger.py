"""
Sistema de logs detalhado para o trading quântico
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
import json

class QuantumLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Configurar logger principal
        self.logger = logging.getLogger("quantum_trading")
        self.logger.setLevel(logging.INFO)  # Mudado de DEBUG para INFO

        # Handler para arquivo mantém DEBUG para análise detalhada
        log_file = os.path.join(log_dir, f"quantum_trading_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Handler para console mostra apenas INFO e acima
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatador mais conciso para console
        console_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )

        # Formatador detalhado para arquivo
        file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s - %(context)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _format_context(self, context: Optional[Dict[str, Any]] = None) -> str:
        """Formata contexto adicional para o log de forma mais concisa"""
        if not context:
            return ""
        # Simplifica a saída JSON removendo espaços extras
        return json.dumps(context, separators=(',', ':'))

    def debug(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log nível DEBUG - Apenas para arquivo"""
        extra = {'context': self._format_context(context)}
        self.logger.debug(message, extra=extra)

    def info(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log nível INFO - Console e arquivo"""
        extra = {'context': self._format_context(context)}
        self.logger.info(message, extra=extra)

    def warning(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log nível WARNING"""
        extra = {'context': self._format_context(context)}
        self.logger.warning(message, extra=extra)

    def error(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log nível ERROR"""
        extra = {'context': self._format_context(context)}
        self.logger.error(message, extra=extra)

    def trade_executed(self, trade_info: Dict[str, Any]):
        """Log específico para execução de trades - Formato simplificado"""
        self.info("Trade executado", {
            'id': trade_info.get('order_id'),
            'status': 'sucesso' if trade_info.get('success', False) else 'falha',
            'tempo': f"{trade_info.get('execution_time', 0):.3f}s"
        })

# Instância global do logger
quantum_logger = QuantumLogger()