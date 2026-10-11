# Credit Card Fraud Detection

Fraud detection on a heavily imbalanced dataset (Kaggle `mlg-ulb/creditcardfraud`, ~0.17 %
fraud), with the focus on evaluating it correctly: leakage-free time-based validation, a
comparison of resampling vs cost-sensitive strategies, a cost-based threshold, SHAP, and
deployment as an API and a Streamlit app.

**`docs/PLAN.md` is the source of truth.** Read it before starting any phase. Work phase by
phase; do not start a later phase until the current one meets its "Done when" criteria.

## Tech stack

- Python 3.12, uv (`pyproject.toml` + `uv.lock`). Project dependencies are runtime only
  (pandas, scikit-learn, imbalanced-learn, xgboost); groups `train`, `api`, `app`, `dev` are
  all installed by default. Docker images install runtime + one group.
- Safe-Level / Polynom-Fit SMOTE are our own implementations in `fraud.samplers`.

## Repository structure

```
docs/PLAN.md          project plan            docs/images/     README images
data/raw|processed/   gitignored              notebooks/       01_eda.ipynb (Phase 1)
src/fraud/            config, data, features, samplers, models, cache, evaluate, train,
                      selection, experiments, report, threshold, explain, final_eval, plotting,
                      serving (runtime only: no shap/matplotlib imports), app_data
models/               final model artifact    reports/         results.csv, figures/
app/                  api.py, streamlit_app.py, data/ (committed sample + validation scores)
tests/                pytest, synthetic fixtures     .github/workflows/ci.yml  lint + tests
```

## Conventions

- Type hints on all functions; small, single-purpose functions.
- Fixed random seed everywhere: `fraud.config.RANDOM_SEED`. Paths and cost assumptions also
  come from `fraud.config`, never hard-coded.
- **No data leakage.** Resampling and scaling only inside an `imblearn.pipeline.Pipeline`,
  fit on training data only. Splits are time-based (no shuffling across time). The test set
  is used once, at the end: modelling code loads data with `load_train_val()`; the test split
  only via `load_test(allow_test=True)` in Phase 4. Raw `Time` is never a model feature.
- Primary metric is PR-AUC; never report accuracy as a headline number.
- English for code, comments, docs and commit messages. Plain commit messages, no
  attribution lines (`.claude/settings.json` disables them; keep it that way).
- Gitignored: `data/`, `mlruns/`, `models/*` except the final pipeline and its config.
- LF line endings (`.gitattributes`). The Makefile must work in Git Bash on Windows and on
  Linux CI: no PowerShell/cmd syntax, no OS-specific paths. Linux installs `xgboost-cpu`.
- Never run destructive commands (`rm -rf` on data, git history rewrites, deleting files
  outside the project) without asking first.
- Kaggle token lives in `~/.kaggle/` (or `KAGGLE_API_TOKEN`). Never print, copy or commit it.
- The machine's C: drive is nearly full; everything (caches, venv, data) stays on F:.

## Testing and CI

CI has no Kaggle access, so **tests must never need the real dataset**. Use the synthetic
`transactions` fixture in `tests/conftest.py` (same columns as `creditcard.csv`, fixed
seed). Import `kaggle` only lazily inside functions: importing it triggers authentication.

## Makefile commands

| Command | What it does |
|---|---|
| `make install` / `lock` | `uv sync --locked` (fails if `uv.lock` is out of date) / upgrade and rewrite `uv.lock` |
| `make data` | download the dataset into `data/raw/` (skipped if present) and validate it |
| `make lint` / `format` / `test` | ruff check + format check / ruff autofix / pytest |
| `make notebook` | execute notebooks in place, outputs kept (needs data; never run in CI) |
| `make train-baseline` | logistic regression, no resampling; upserts its row in `reports/results.csv` |
| `make experiments` | stages 1-3 + report, resumable (design: PLAN.md §6, rule: §14) |
| `make evaluate` | Phase 4: threshold, the one test-set evaluation (logged), SHAP, report |
| `make api` / `app` | FastAPI on :8000 / Streamlit on :8501; both use `fraud.serving` |
| `make docker-build` / `docker-run` | API image (`compose.yaml` adds the app) |
| `make app-data` / `app-requirements` | re-export app data / rewrite `app/requirements.txt` (Streamlit Cloud) from `uv.lock`, checked in CI |

## Current status

- [x] Phase 0 – Setup
- [x] Phase 1 – EDA
- [x] Phase 2 – Training and evaluation pipeline
- [x] Phase 3 – Experiments
- [x] Phase 4 – Threshold and explainability
- [x] Phase 5 – API, app, Docker
- [x] Phase 6 – CI, deployment, documentation
