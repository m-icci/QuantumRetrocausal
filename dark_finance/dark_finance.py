"""
Dark Finance Core Implementation
Implements Singleton pattern to prevent multiple initializations
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime
import threading

logger = logging.getLogger(__name__)

class DarkFinance:
    _instance: Optional['DarkFinance'] = None
    _initialized: bool = False
    _lock = threading.Lock()
    _state: Dict[str, Any] = {}
    _last_update: datetime = datetime.now()
    _coupling_factor: float = 0.618  # Golden ratio conjugate

    def __new__(cls) -> 'DarkFinance':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not DarkFinance._initialized:
            with self._lock:
                if not DarkFinance._initialized:
                    self._init_core()
                    DarkFinance._initialized = True
                    logger.info("Dark Finance inicializado")

    def _init_core(self):
        """Initialize core components"""
        DarkFinance._last_update = datetime.now()
        DarkFinance._state = {}
        DarkFinance._coupling_factor = 0.618

    def initialize_with_coupling(self, coupling_factor: float = 0.618):
        """Initialize with specific coupling factor"""
        with self._lock:
            if coupling_factor != DarkFinance._coupling_factor:
                DarkFinance._coupling_factor = coupling_factor
                logger.info("Dark Finance inicializado com acoplamento aumentado")

    def get_state(self) -> Dict[str, Any]:
        """Get current state"""
        return {
            'coupling_factor': DarkFinance._coupling_factor,
            'last_update': DarkFinance._last_update,
            'state': DarkFinance._state
        }

    @classmethod
    def reset(cls):
        """Reset singleton state - only for testing"""
        with cls._lock:
            cls._instance = None
            cls._initialized = False
            cls._state = {}
            cls._last_update = datetime.now()
            cls._coupling_factor = 0.618