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
	@echo "train-baseline  train the baseline and write its row to reports/results.csv"
	@echo "experiments     run the full grid, tuning and report (resumes; about an hour from scratch)"
	@echo "api, app: later phases"

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

# Logistic regression, no resampling: fit on train, evaluate on validation.
train-baseline:
	uv run --locked python -m fraud.train --model logreg --strategy none

# Full grid, tuning and report. Finished configurations are skipped, so it can be resumed.
# Subset: uv run python -m fraud.experiments --stage grid --model lightgbm --strategy smote
experiments:
	uv run --locked python -m fraud.experiments

api:
	@echo "api: not implemented yet (Phase 5)"

app:
	@echo "app: not implemented yet (Phase 5)"
