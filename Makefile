ENV_DB_HOST_PORT := $(shell grep -E '^DB_HOST_PORT=' .env 2>/dev/null | cut -d= -f2)
DB_HOST_PORT ?= $(or $(ENV_DB_HOST_PORT),5432)
TEST_DB_URL = postgresql+psycopg://app:local_password@localhost:$(DB_HOST_PORT)/rag_test
PYTHON ?= python3
VENV = .venv/bin
export DB_HOST_PORT

.DEFAULT_GOAL := help

.PHONY: help setup up down migrate ingest reset-docs eval test test-integration test-live lint format typecheck check

help: ## show this list
	@grep -E '^[a-z-]+:.*## ' Makefile | awk -F':.*## ' '{printf "  make %-18s %s\n", $$1, $$2}'

setup: ## local dev: create .venv, install requirements-dev.txt, copy .env.example
	$(PYTHON) -m venv .venv
	$(VENV)/pip install -r requirements-dev.txt
	test -f .env || cp .env.example .env

up: ## build images, start db/api/ui, run migrations
	docker compose up --build -d
	docker compose run --rm api alembic upgrade head

down: ## stop containers (data is kept)
	docker compose down

migrate: ## apply database migrations
	docker compose run --rm api alembic upgrade head

ingest: ## load data/documents.jsonl (safe to repeat)
	docker compose run --rm api python scripts/ingest.py

reset-docs: ## validate data/documents.jsonl, delete all documents, ingest again
	docker compose run --rm --no-deps api python scripts/ingest.py --check
	docker compose exec db psql -U app -d rag -c "DELETE FROM documents;"
	docker compose run --rm api python scripts/ingest.py

eval: ## run the 40 evaluation cases at 3 thresholds (calls Claude)
	docker compose run --rm -v ./reports:/app/reports api python scripts/run_eval.py --thresholds 0.25 0.28 0.35

test: ## unit tests (no network, no database)
	$(VENV)/pytest

test-integration: ## pgvector tests in a separate rag_test database
	docker compose up -d --wait db
	docker compose exec db psql -U app -d rag -tAc "SELECT 1 FROM pg_database WHERE datname='rag_test'" | grep -q 1 || docker compose exec db createdb -U app rag_test
	DATABASE_URL=$(TEST_DB_URL) LLM_PROVIDER=fake $(VENV)/alembic upgrade head
	DATABASE_URL=$(TEST_DB_URL) LLM_PROVIDER=fake $(VENV)/pytest -m integration

test-live: ## tests that call the real LLM provider (costs money)
	$(VENV)/pytest -m live

lint: ## ruff check + format check
	$(VENV)/ruff check .
	$(VENV)/ruff format --check .

format: ## auto-fix lint and formatting
	$(VENV)/ruff check --fix .
	$(VENV)/ruff format .

typecheck: ## mypy strict
	$(VENV)/mypy app

check: lint typecheck test ## lint + typecheck + unit tests

