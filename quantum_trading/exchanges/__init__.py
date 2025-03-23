#!/usr/bin/env python3
"""
Quantum Trading Exchanges
========================
Módulo para interação com exchanges de criptomoedas.
"""

from quantum_trading.exchanges.exchange_base import ExchangeBase
from quantum_trading.exchanges.kucoin_adapter import KuCoinAdapter
from quantum_trading.exchanges.kraken_adapter import KrakenAdapter
from quantum_trading.exchanges.factory import ExchangeFactory

__all__ = [
    'ExchangeBase',
    'KuCoinAdapter',
    'KrakenAdapter',
    'ExchangeFactory'
] 