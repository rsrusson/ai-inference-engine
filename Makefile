# Makefile — thin wrappers over the documented commands.
# The heavy stack lives in .venv-vllm (engine/tests) and .venv-torch (baseline).
# See docs/ARCHITECTURE.md for the system overview and loadtest/README.md for results.

VENV_VLLM := .venv-vllm
VENV_TORCH := .venv-torch
PY := $(VENV_VLLM)/bin/python

.PHONY: help setup setup-dev serve baseline bench load test lint clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Create both venvs and install dependencies
	python3 -m venv $(VENV_VLLM)
	$(VENV_VLLM)/bin/pip install --upgrade pip
	$(VENV_VLLM)/bin/pip install -r serve/requirements.txt
	python3 -m venv $(VENV_TORCH)
	$(VENV_TORCH)/bin/pip install -r v1-baseline/requirements-torch.txt

setup-dev: ## Install dev/test dependencies into .venv-vllm
	$(VENV_VLLM)/bin/pip install -r requirements-dev.txt

serve: ## Start the vLLM OpenAI server on :8000 (all WSL2 workarounds applied)
	./serve/vllm.serve.sh

baseline: ## Start the naive baseline server on :8001
	$(VENV_TORCH)/bin/python v1-baseline/main-torch.py

bench: ## Run a quick vLLM concurrency sweep (server must be up)
	$(PY) loadtest/bench.py --url http://127.0.0.1:8000/v1/chat/completions \
		--mode vllm --concurrency 8 --requests 16 --max-tokens 128

load: ## Run the full standard load set (server must be up)
	./loadtest/run_load.sh

test: ## Run the unit test suite (no GPU required)
	$(PY) -m pytest tests/ -q

lint: ## Shell + Python syntax checks (what CI runs)
	@for f in $$(git ls-files '*.sh'); do echo "bash -n $$f"; bash -n "$$f"; done
	$(PY) -m compileall -q loadtest v1-baseline tests
	@echo "syntax checks passed"

clean: ## Remove Python caches
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
