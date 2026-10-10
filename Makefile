# Works in Git Bash on Windows and on Linux (CI). Everything runs through uv.
.DEFAULT_GOAL := help

NOTEBOOKS := $(wildcard notebooks/*.ipynb)

IMAGE ?= fraud-api
PORT ?= 8000

.PHONY: help install lock data lint format test notebook train-baseline experiments evaluate \n	app-data api app docker-build docker-run

help:
	@echo "install         install dependencies exactly as pinned in uv.lock"
	@echo "lock            upgrade dependencies and rewrite uv.lock"
	@echo "data            download the Kaggle dataset into data/raw/ (skipped if present)"
	@echo "lint            ruff check + format check"
	@echo "format          ruff autofix + format"
	@echo "test            run pytest"
	@echo "notebook        execute the notebooks top to bottom"
	@echo "train-baseline  train the baseline and write its row to reports/results.csv"
	@echo "experiments     run experiment stages 1-3 and the report (finished runs are skipped)"
	@echo "evaluate        final evaluation: threshold, TEST SPLIT (logged), SHAP, report"
	@echo "app-data        export the validation scores used by the app (needs the dataset)"
	@echo "api             run the FastAPI service on http://127.0.0.1:$(PORT) (docs at /docs)"
	@echo "app             run the Streamlit app on http://localhost:8501"
	@echo "docker-build    build the API image ($(IMAGE))"
	@echo "docker-run      run the API image on http://127.0.0.1:$(PORT)"

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

# Stages 1-3 and the report. Finished configurations are skipped, so it can be resumed.
# Subset: uv run python -m fraud.experiments --stage 2 --model lightgbm --strategy smote_enn
# Runtime estimate for a stage: uv run python -m fraud.experiments --stage 1 --estimate
experiments:
	uv run --locked python -m fraud.experiments

# Phase 4. Loads the held-out test split and appends to reports/test_set_access.log.
# To check code changes without touching the test split:
#   uv run python -m fraud.final_eval --dry-run some/scratch/dir
# To rewrite the report from the saved metrics: uv run python -m fraud.final_eval --render-only
evaluate:
	uv run --locked python -m fraud.final_eval

# Validation scores for the threshold page of the app (score, label, amount only).
app-data:
	uv run --locked python -m fraud.app_data

api:
	uv run --locked uvicorn app.api:app --host 127.0.0.1 --port $(PORT)

# Needs only the committed model and app data, not the dataset or the API.
app:
	uv run --locked streamlit run app/streamlit_app.py --server.address localhost

docker-build:
	docker build --target api -t $(IMAGE) .

docker-run:
	docker run --rm -p $(PORT):8000 $(IMAGE)
