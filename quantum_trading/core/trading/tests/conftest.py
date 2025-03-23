"""
Configuração de testes.
"""

import pytest
import logging

@pytest.fixture(autouse=True)
def setup_logging():
    """Configura logging para testes."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ) 