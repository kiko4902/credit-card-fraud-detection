# Works in Git Bash on Windows and on Linux (CI). Everything runs through uv.
.DEFAULT_GOAL := help

NOTEBOOKS := $(wildcard notebooks/*.ipynb)

.PHONY: help install lock data lint format test notebook train-baseline experiments api app

help:
	@echo "install         install dependencies exactly as pinned in uv.lock"
	@echo "lock            upgrade dependencies and rewrite uv.lock"
	@echo "data            download the Kaggle dataset into data/raw/ (skipped if present)"
	@echo "lint            ruff check + format check"
	@echo "format          ruff autofix + format"
	@echo "test            run pytest"
	@echo "notebook        execute the notebooks top to bottom"
	@echo "train-baseline, experiments, api, app: later phases"

# --locked fails if uv.lock is out of date with pyproject.toml instead of silently re-resolving.
install:
	uv sync --locked

lock:
	uv lock --upgrade

data:
	uv run --locked python -m fraud.data

lint:
	uv run --locked ruff check .
	uv run --locked ruff format --check .

format:
	uv run --locked ruff check --fix .
	uv run --locked ruff format .

test:
	uv run --locked pytest

notebook:
ifeq ($(strip $(NOTEBOOKS)),)
	@echo "notebook: not implemented yet (Phase 1)"
else
	uv run --locked jupyter nbconvert --to notebook --execute --inplace $(NOTEBOOKS)
endif

train-baseline:
	@echo "train-baseline: not implemented yet (Phase 2)"

experiments:
	@echo "experiments: not implemented yet (Phase 3)"

api:
	@echo "api: not implemented yet (Phase 5)"

app:
	@echo "app: not implemented yet (Phase 5)"
