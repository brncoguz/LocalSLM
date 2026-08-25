.DEFAULT_GOAL := help
SHELL := /bin/bash
export PYTHONPATH := src

PORT ?= 8420
VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

##@ Setup

$(VENV)/bin/activate:
	python3.13 -m venv $(VENV)
	$(PIP) install -q --upgrade pip

install: $(VENV)/bin/activate ## Install Python + UI dependencies
	$(PIP) install -q -r requirements-dev.txt
	cd ui && npm install --silent
	@[ -f .env ] || cp .env.example .env
	@echo "ready. run 'make serve'"

##@ Run

serve: ## Run API (:8420) and UI (:5273) together
	$(MAKE) -j2 api ui

api: ## Run the FastAPI backend only
	$(VENV)/bin/uvicorn localslm.api.main:app --reload --port $(PORT)

ui: ## Run the Vite dev server only
	cd ui && npm run dev

##@ Bench

bench: ## Run a benchmark preset headlessly: make bench PRESET=quick
	$(PY) -m localslm.cli bench --preset $(or $(PRESET),quick)

models: ## Show installed models with their live state
	$(PY) -m localslm.cli models

doctor: ## Check the host: ollama, disk, memory, power source
	$(PY) -m localslm.cli doctor

demo-data: ## Snapshot the API into the fixture the published demo serves
	$(PY) -m localslm.cli export-demo

demo: demo-data ## Build and preview the static demo locally
	cd ui && VITE_DEMO=1 npm run build && npm run preview

##@ Quality

test: ## Run the offline test suite
	$(VENV)/bin/pytest -q -m "not integration"

test-all: ## Run every test, including ones needing a live Ollama
	$(VENV)/bin/pytest -q

lint: ## Lint and type-check
	$(VENV)/bin/ruff check src tests
	$(VENV)/bin/ruff format --check src tests

fmt: ## Auto-format
	$(VENV)/bin/ruff format src tests
	$(VENV)/bin/ruff check --fix src tests

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache ui/dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

##@ Meta

help: ## Show this help
	@awk 'BEGIN {FS=":.*##"; printf "\nLocalSLM\n"} \
		/^##@/ {printf "\n\033[1m%s\033[0m\n", substr($$0,5)} \
		/^[a-zA-Z_-]+:.*?##/ {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo

.PHONY: install serve api ui bench models doctor demo-data demo test test-all lint fmt clean help
