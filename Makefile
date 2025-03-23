.PHONY: help install test lint format clean docker-build docker-run docker-stop

help:  ## Mostra esta ajuda
	@awk 'BEGIN {FS = ":.*##"; printf "\nUso:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

install:  ## Instala dependências do projeto
	poetry install

test:  ## Executa testes
	poetry run pytest tests/ -v --cov=quantum_trading --cov-report=term-missing

lint:  ## Executa linters
	poetry run black quantum_trading tests
	poetry run isort quantum_trading tests
	poetry run flake8 quantum_trading tests
	poetry run mypy quantum_trading tests
	poetry run bandit -r quantum_trading

format:  ## Formata código
	poetry run black quantum_trading tests
	poetry run isort quantum_trading tests

clean:  ## Limpa arquivos temporários
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type d -name "coverage_html" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type f -name ".DS_Store" -delete

docker-build:  ## Constrói imagem Docker
	docker-compose build

docker-run:  ## Executa containers Docker
	docker-compose up -d

docker-stop:  ## Para containers Docker
	docker-compose down

setup:  ## Configura ambiente de desenvolvimento
	pip install poetry
	poetry install
	poetry run pre-commit install
	cp .env.example .env
	mkdir -p logs backups
	chmod +x scripts/*.sh

run:  ## Executa o sistema
	poetry run python -m quantum_trading.executar_trading_real

run-test:  ## Executa o sistema em modo de teste
	poetry run python -m quantum_trading.executar_trading_real --teste

monitor:  ## Monitora logs do sistema
	tail -f logs/qualia.log

backup:  ## Executa backup manual
	scripts/backup.sh 