# Project Plan: Credit Card Fraud Detection on Imbalanced Data

> This document is the source of truth for the project. Read it before starting any phase.
> Work phase by phase. Do not start a later phase until the current one meets its "Done when" criteria.

## 1. Goal

Build and deploy a fraud detection model on a heavily imbalanced dataset, and show **how to evaluate it correctly**.

The project is a portfolio piece for a data role and builds on the author's MSc thesis on SMOTE-based methods for class imbalance. It must demonstrate:
- leakage-free validation on imbalanced data
- a systematic comparison of resampling methods (SMOTE variants), undersampling and cost-sensitive learning
- appropriate metrics (PR-AUC, recall, precision), not accuracy
- choosing a decision threshold based on business cost
- model explainability (SHAP)
- deployment as an API and an interactive web app

## 2. Data

Kaggle **Credit Card Fraud Detection** (`mlg-ulb/creditcardfraud`):
- ~285,000 transactions over two days, ~0.17 % fraud (492 cases)
- Features: `Time` (seconds since first transaction), `V1`–`V28` (PCA components, anonymised), `Amount`, target `Class` (1 = fraud)
- Licence: ODbL. The raw CSV is **not committed** to the repo; a script downloads it via the Kaggle API (`make data`). This needs a Kaggle API token, either saved in `~/.kaggle/access_token` or set in the `KAGGLE_API_TOKEN` environment variable.

## 3. Tech stack

- Python 3.12: `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `lightgbm`, `xgboost` (optional), `shap`, `matplotlib`/`seaborn`, `pytest`, `ruff`
- Experiment tracking: `mlflow` (optional, local) or a results CSV
- Serving: `FastAPI` + `uvicorn`
- UI: `Streamlit`
- Docker, Makefile, GitHub Actions
- Code, comments, commit messages and README in English

## 4. Repository structure

```
credit-card-fraud-detection/
├── CLAUDE.md
├── README.md
├── Makefile
├── Dockerfile
├── pyproject.toml            # or requirements.txt
├── .gitignore
├── docs/
│   ├── PLAN.md               # this file
│   └── images/
├── data/                     # gitignored (raw/, processed/)
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   └── fraud/
│       ├── config.py         # paths, random seed, cost assumptions
│       ├── data.py           # download, load, time-based split
│       ├── features.py       # e.g. scaling Amount, hour-of-day from Time
│       ├── models.py         # model + resampler registry
│       ├── evaluate.py       # metrics, curves, cost function
│       ├── train.py          # run experiments: python -m fraud.train
│       └── explain.py        # SHAP
├── models/                   # final model artifact (small, can be committed or released)
├── reports/
│   ├── results.csv           # all experiment results
│   └── figures/
├── app/
│   ├── api.py                # FastAPI
│   └── streamlit_app.py
├── tests/
└── .github/workflows/ci.yml
```

## 5. Validation design (most important part)

- **Time-based split**: train on the earlier ~70 % of transactions (by `Time`), validate on the next ~15 %, test on the last ~15 %. No random shuffling across time.
- Hyperparameter tuning only on train/validation; the test set is used **once** at the end.
- **Resampling only on training data**, always inside an `imblearn.pipeline.Pipeline`, so it is applied within each CV fold and never touches validation/test data.
- Scaling is fit on training data only (also inside the pipeline).
- Fixed random seed everywhere for reproducibility.
- Report the number of fraud cases in each split.

## 6. Experiments

Models:
- Logistic Regression
- Random Forest
- LightGBM

Imbalance strategies (each combined with each model):
1. None (baseline)
2. Class weights / `scale_pos_weight` (cost-sensitive)
3. Random undersampling
4. SMOTE
5. Borderline-SMOTE
6. ADASYN
7. SMOTE-Tomek

Also try at least two sampling ratios for the SMOTE-family methods (e.g. minority:majority 0.1 and 0.5), not only full balancing.

All results go to `reports/results.csv` (model, strategy, params, metrics, train time).

## 7. Metrics

- Primary: **PR-AUC** (average precision)
- Also: ROC-AUC, precision, recall, F1 at the chosen threshold, confusion matrix
- Precision–recall curves comparing the best configurations
- README must briefly explain why accuracy is misleading here (predicting "never fraud" gives ~99.8 %)

## 8. Cost-based threshold

Define a simple cost model in `config.py`:
- missed fraud (FN) costs the transaction `Amount`
- false alarm (FP) costs a fixed review cost (e.g. €5)
- correctly caught fraud (TP) also incurs the review cost

Choose the threshold that minimises total cost on the validation set; report cost and savings vs "no model" on the test set. Plot cost vs threshold.

## 9. Explainability

- SHAP global summary plot for the final model
- SHAP explanation for individual predictions (reused in the Streamlit app)

## 10. Serving and deployment

- **FastAPI** `POST /predict`: input transaction features → fraud probability, decision at chosen threshold, top contributing features. `GET /health`.
- **Streamlit app**: pick a transaction from the test set (or enter values), show probability, decision, SHAP explanation; a second page with the experiment results table and PR curves.
- **Docker** image for the API.
- Deploy the Streamlit app to Streamlit Community Cloud or Hugging Face Spaces (free). The app must load a saved model and a small sample of test data, not the full dataset.

## 11. README must contain

1. Problem and key result (e.g. "LightGBM + X reaches PR-AUC Y; cost-optimal threshold saves Z % vs no model")
2. Link to live app + screenshot
3. Validation design and why it matters (leakage, time split)
4. Results table of the comparison
5. PR curves and cost-vs-threshold plot
6. How to run (download data, train, serve)
7. Link to the MSc thesis topic as motivation

## 12. Phases

### Phase 0 – Setup
Repo structure, `CLAUDE.md`, dependencies, Makefile, `.gitignore`, data download script via Kaggle API, README skeleton, git init.
**Done when:** `make install` and `make data` work; the CSV is in `data/raw/` and gitignored.

### Phase 1 – EDA
`notebooks/01_eda.ipynb`: class balance, Amount and Time distributions by class, fraud rate over time, correlations. Save key figures to `reports/figures/`.
**Done when:** notebook runs top to bottom, with short written conclusions.

### Phase 2 – Training and evaluation pipeline
`data.py` (load, time split), `features.py`, `models.py` (registry of models and strategies), `evaluate.py` (metrics), `train.py`. Run a baseline end to end.
**Done when:** `make train-baseline` produces metrics for one model; tests cover the split (no time overlap, resampling not applied to validation/test).

### Phase 3 – Experiments
Full grid from section 6, results to `reports/results.csv`, comparison plots.
**Done when:** `make experiments` runs the full grid reproducibly; best configuration identified.

### Phase 4 – Threshold and explainability
Cost model, threshold selection on validation, final evaluation on test, SHAP plots. Save final model to `models/`.
**Done when:** test-set results and cost savings reported; figures saved.

### Phase 5 – API, app, Docker
FastAPI, Streamlit app, Dockerfile, tests for the API.
**Done when:** `make api` and `make app` run locally; `docker build` + `docker run` serves predictions.

### Phase 6 – CI, deployment, documentation
GitHub Actions (lint + tests), deploy Streamlit app, final README.
**Done when:** CI is green, the app is reachable via a public link, README is complete.
