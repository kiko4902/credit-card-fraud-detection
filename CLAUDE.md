# Credit Card Fraud Detection

Fraud detection on a heavily imbalanced dataset (Kaggle `mlg-ulb/creditcardfraud`, ~0.17 %
fraud), with the focus on evaluating it correctly: leakage-free time-based validation, a
comparison of resampling vs cost-sensitive strategies, a cost-based threshold, SHAP, and
deployment as an API and a Streamlit app.

**`docs/PLAN.md` is the source of truth.** Read it before starting any phase. Work phase by
phase; do not start a later phase until the current one meets its "Done when" criteria.

## Tech stack

- Python 3.12, dependencies managed with uv (`pyproject.toml` + `uv.lock`)
- pandas, numpy, scikit-learn, imbalanced-learn, lightgbm, shap, matplotlib, seaborn
- FastAPI + uvicorn (API), Streamlit (UI), Docker, GitHub Actions
- pytest, ruff; optional extra `extras` = xgboost, mlflow (`uv sync --extra extras`)

## Repository structure

```
docs/PLAN.md          project plan            docs/images/     README images
data/raw|processed/   gitignored              notebooks/       01_eda.ipynb (Phase 1)
src/fraud/            config, data, features, plotting (+ models, evaluate, train, explain later)
models/               final model artifact    reports/         results.csv, figures/
app/                  api.py, streamlit_app.py
tests/                pytest, synthetic fixtures in conftest.py
.github/workflows/    ci.yml (lint + tests)
```

## Conventions

- Type hints on all functions; small, single-purpose functions.
- Fixed random seed everywhere: `fraud.config.RANDOM_SEED`. Paths and cost assumptions also
  come from `fraud.config`, never hard-coded.
- **No data leakage.** Resampling and scaling only inside an `imblearn.pipeline.Pipeline`,
  fit on training data only. Splits are time-based (no shuffling across time). The test set
  is used once, at the end.
- Primary metric is PR-AUC; never report accuracy as a headline number.
- English for code, comments, docs and commit messages. Plain commit messages, no
  attribution lines (`.claude/settings.json` disables them; keep it that way).
- Large or generated files are gitignored: `data/`, `models/*`, `mlruns/`. Un-ignore a
  specific small model artifact explicitly if it should be committed.
- LF line endings (`.gitattributes`). The Makefile must work in Git Bash on Windows and on
  the Linux CI runner: no PowerShell/cmd syntax, no OS-specific paths.
- Never run destructive commands (`rm -rf` on data, git history rewrites, deleting files
  outside the project) without asking first.
- Kaggle token lives in `~/.kaggle/` (or `KAGGLE_API_TOKEN`). Never print it, copy it into
  the project, or commit it.
- The machine's C: drive is nearly full; everything (caches, venv, data) stays on F:.

## Testing and CI

CI has no Kaggle access, so **tests must never need the real dataset**. Use the synthetic
`transactions` fixture / `make_synthetic_transactions()` in `tests/conftest.py` (same
columns as `creditcard.csv`, fixed seed). Import `kaggle` only lazily inside functions:
importing it triggers authentication.

## Makefile commands

| Command | What it does |
|---|---|
| `make install` | `uv sync --locked`; fails if `pyproject.toml` and `uv.lock` disagree |
| `make lock` | upgrade dependencies and rewrite `uv.lock` |
| `make data` | download the dataset into `data/raw/` (skipped if present) and validate it |
| `make lint` / `make format` | ruff check + format check / autofix |
| `make test` | pytest |
| `make notebook` | execute notebooks in place, outputs kept (needs data; never run in CI) |
| `make train-baseline`, `experiments`, `api`, `app` | not implemented yet |

## Current status

- [x] Phase 0 – Setup
- [ ] Phase 1 – EDA (in progress)
- [ ] Phase 2 – Training and evaluation pipeline
- [ ] Phase 3 – Experiments
- [ ] Phase 4 – Threshold and explainability
- [ ] Phase 5 – API, app, Docker
- [ ] Phase 6 – CI, deployment, documentation
