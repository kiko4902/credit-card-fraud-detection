# Credit Card Fraud Detection on Imbalanced Data

> Work in progress. See [docs/PLAN.md](docs/PLAN.md) for the full project plan.

## Problem and key result

_TODO (Phase 4): one-paragraph problem statement and headline result
(best model + strategy, PR-AUC, cost savings vs no model)._

## Live app

_TODO (Phase 6): link to the deployed Streamlit app and a screenshot._

## Validation design and why it matters

_TODO (Phase 2): time-based split, resampling and scaling only inside the training
pipeline, why leakage inflates results, why accuracy is misleading at ~0.17 % fraud._

## Results

_TODO (Phase 3): comparison table of models and imbalance strategies._

## PR curves and cost vs threshold

_TODO (Phases 3-4): precision-recall curves of the best configurations and the
cost-vs-threshold plot._

## How to run

Requirements: [uv](https://docs.astral.sh/uv/), GNU make, and a Kaggle API token
(`~/.kaggle/access_token` or the `KAGGLE_API_TOKEN` environment variable).

```bash
make install   # install dependencies from uv.lock
make data      # download the dataset into data/raw/ (not committed)
make notebook  # re-run the EDA notebook (notebooks/01_eda.ipynb)
make lint
make test
```

_TODO (Phases 2-5): training, experiments, API and app commands._

## Motivation: MSc thesis

_TODO (Phase 6): link to the MSc thesis on SMOTE-based methods for class imbalance._

## Data and licence

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(ULB Machine Learning Group), Open Database License (ODbL). The raw CSV is not part of
this repository.
