# Use uma imagem base Python
FROM python:3.10-slim

# Define variáveis de ambiente
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.7.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv"

# Adiciona Poetry ao PATH
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

# Define diretório de trabalho
WORKDIR $PYSETUP_PATH

# Instala dependências do sistema
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instala Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Copia arquivos de configuração
COPY poetry.lock pyproject.toml ./

# Instala dependências do projeto
RUN poetry install --no-dev --no-root

# Copia código fonte
COPY . .

# Instala o projeto
RUN poetry install --no-dev

# Expõe portas
EXPOSE 8000 8001

# Define comando de execução
CMD ["poetry", "run", "python", "-m", "quantum_trading.executar_trading_real"] 