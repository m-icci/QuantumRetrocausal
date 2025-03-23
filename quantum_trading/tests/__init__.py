"""
Testes do QUALIA.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch

# Configuração do pytest
def pytest_configure(config):
    """Configura o pytest."""
    config.addinivalue_line(
        "markers",
        "asyncio: mark test as async"
    )

# Fixtures globais
@pytest.fixture(scope="session")
def event_loop():
    """Cria um event loop para testes assíncronos."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_response():
    """Cria uma resposta mock para testes."""
    response = Mock()
    response.status = 200
    response.json = Mock(return_value={})
    return response

@pytest.fixture
def mock_session():
    """Cria uma sessão mock para testes."""
    session = Mock()
    session.get = Mock(return_value=mock_response())
    return session 