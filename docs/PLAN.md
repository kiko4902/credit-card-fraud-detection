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

- Python 3.12: `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `lightgbm`, `xgboost`, `shap`, `matplotlib`/`seaborn`, `pytest`, `ruff`
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

The project extends the author's MSc thesis on SMOTE-based methods (18 datasets, 8
classifiers, imbalance ratios up to 28:1) to a much more extreme case: about 540:1.

**Thesis findings to test here**

- Polynom-Fit SMOTE was the only method significantly better than SMOTE on F1, G-Mean and
  AUC-ROC. Safe-Level-SMOTE was best on F1 / MCC / AUC-PR and its advantage grew with the
  imbalance ratio. SMOTE-ENN was best on G-Mean. SMOTE-Tomek was indistinguishable from SMOTE.
- Borderline-SMOTE, ADASYN and random undersampling performed poorly. They are excluded.
- On average, oversampling barely changed AUC-PR (+0.001). Hypothesis: it mainly shifts the
  decision boundary instead of improving the ranking.
- The choice of oversampling method mattered most for logistic regression and k-NN, least
  for tree ensembles.

**Design: a staged search**, so that many classifiers are covered without excessive runtime.
Every configuration is scored with the same 3 time-ordered CV folds on the training split
and on the validation split. The test split is not used.

*Stage 1 - screening.* 9 classifiers x 3 strategies (none, class weights, SMOTE at ratio 0.1),
library default parameters:
logistic regression, decision tree, Gaussian naive Bayes, k-nearest neighbours, random
forest, extra trees, XGBoost, LightGBM and a multi-layer perceptron (with early stopping).
Class weights only where the classifier supports them (not Gaussian NB, k-NN, MLP), which
gives 24 configurations. XGBoost has no `class_weight`; its `scale_pos_weight` is computed
from the data of each fit. Classifiers are ranked by their best mean CV PR-AUC across the
strategies.

SVM with an RBF kernel is excluded: kernel SVM training scales between quadratically and
cubically with the number of rows, and probability estimates need an additional internal
5-fold fit, so a single configuration on about 200,000 rows would take hours.

*Stage 2 - SMOTE family.* The top 4 classifiers of stage 1 x SMOTE (ratios 0.5 and 1.0; 0.1
is already in stage 1), SMOTE-Tomek, SMOTE-ENN, Safe-Level-SMOTE and Polynom-Fit SMOTE (each
at ratios 0.1 and 0.5): 10 new configurations per classifier, 40 in total. Ratios are
minority:majority after resampling.

*Stage 2 extras (added after stage 1 was seen).* Stage 1 ranked four tree ensembles first;
logistic regression missed fourth place by 0.0001 mean CV PR-AUC and k-NN came seventh. The
thesis question about linear and neighbour-based classifiers cannot be answered from tree
ensembles alone, so logistic regression and k-NN get the same 10 stage 2 runs each, recorded
with `stage = stage2_extra`. These rows are used only for the comparison with the thesis:
they are excluded from the ranking, from the paired top-3 comparisons, from tuning and from
the selection of the final model.

*Stage 3 - tuning.* A small, fixed hyperparameter search for the two best classifiers, each
with its best strategy, scored with the same CV. Every tried setting is recorded.

**Sampler details**

- k = 5 neighbours for the neighbour-based oversamplers (SMOTE, the SMOTE step of
  SMOTE-Tomek and SMOTE-ENN, Safe-Level-SMOTE). The thesis reported the best result over
  k in {3, 5, 7}; here k is fixed, so the SMOTE-family results are not tuned over k.
  The cleaning step of SMOTE-ENN uses the standard 3-nearest-neighbour editing rule.
- SMOTE, SMOTE-Tomek and SMOTE-ENN come from imbalanced-learn. Safe-Level-SMOTE and
  Polynom-Fit SMOTE are implemented in `fraud.samplers` from the original papers (the thesis
  code is not available, so they are not guaranteed to match the thesis implementation
  detail for detail). They take the same minority:majority ratio as the other samplers and
  produce exactly that many minority rows.
  - Safe-Level-SMOTE (Bunkhumpornpat et al., 2009): the safe level of a row is the number
    of minority rows among its k nearest neighbours in the whole training data; the partner
    of a minority row is one of its k nearest *minority* rows; the gap follows the five
    cases of the paper and is drawn per attribute. The paper's single pass is repeated
    until the ratio is reached.
  - Polynom-Fit SMOTE (Gazzah & Amara, 2008): star topology only, i.e. evenly spaced points
    on the segments from each minority row to the minority mean. The bus, mesh and
    polynomial-curve topologies are not implemented. For a whole number of points per
    minority row the output is identical to the `smote-variants` package (checked once).
- Features are standardised before every sampler (fitted on the training part only), for
  every classifier, so all classifiers see the same resampled data.
- Resampled training sets are cached per training part, sampler, ratio and seed and reused
  across classifiers. The cache key includes the data itself, so a fold can never receive
  another fold's resampled set.

All results go to `reports/results.csv` (stage, model, strategy, ratio, parameters, metrics,
resampling / fit / predict time).

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

- **FastAPI** `POST /predict`: input transaction features → fraud score (not a probability, see section 15), decision at chosen threshold, top contributing features. Also `POST /predict/batch`, `GET /health`, `GET /model-info`.
- **Streamlit app**: pick a transaction from the test set (or enter values), show fraud score, decision, SHAP explanation; a second page with the experiment results table and PR curves.
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

## 13. Design decisions

Decisions taken after the EDA (`notebooks/01_eda.ipynb`). They override earlier sections
where they differ.

1. **Duplicates.** The 1,081 exact duplicates (identical in all 31 columns, 19 of them fraud)
   are dropped once, before splitting. Rows that repeat only at a different `Time` are kept:
   they are distinct transactions. In Phase 4 the test metrics are reported both with and
   without deduplication.
2. **Primary evaluation is the 70/15/15 time-based split**, because it simulates predicting
   the future. Known limitation: validation and test cover only the afternoon and evening of
   day 2 and contain no night hours, where the fraud rate is highest. This is documented in
   the README. In Phase 4 a clearly labelled **secondary** evaluation on a stratified random
   split of the deduplicated data is added, for comparison only.
3. **Raw `Time` is never a model feature.** It only encodes the position in the recording and
   is out of the training range in validation and test. The derived hour of day is used
   instead.
4. **Cross-validation inside the training split is time-ordered** (expanding window), never
   shuffled k-fold, and every validation fold must contain a minimum number of fraud cases.
5. **The test split is loaded only through `fraud.data.load_test(allow_test=True)`** and only
   in Phase 4.
6. **Scaling is applied before distance-based samplers** (SMOTE family) even for tree models;
   LightGBM uses `class_weight="balanced"`; cross-validation uses 3 time-ordered folds with at
   least 30 fraud cases in each validation fold.

## 14. Selection rule (Phase 3)

The rule below was committed before the first Phase 3 run and is unchanged. The experiment
design was replaced afterwards (section 6). At that point 11 logistic regression results of
the interrupted first run had been seen (no resampling, class weights, undersampling, SMOTE,
Borderline-SMOTE, ADASYN, one SMOTE-Tomek); no other result existed.

1. *Primary:* the mean PR-AUC over the 3 time-ordered CV folds of the training split (218
   fraud cases across the validation folds, against 55 in the validation split).
2. *Tie-break:* configurations within 0.005 of the best mean CV PR-AUC count as tied. Among
   them the simplest wins: no resampling, then class weights, then resampling; after that the
   higher CV mean.
3. *Confirmation:* validation PR-AUC with its stratified bootstrap 95 % CI is reported for
   every configuration. It confirms or questions the CV ranking but does not replace it. If
   the CV winner is clearly worse on validation than the configuration that is best on
   validation (paired bootstrap CI of the difference excludes zero), this is flagged in the
   summary and the choice is made by the author, not automatically.
4. *Paired comparisons:* for the top 3 configurations of stages 1 and 2 by the rule above, a
   paired stratified bootstrap on the validation split against (a) the same classifier with
   no resampling and (b) the overall baseline (logistic regression, no resampling), reporting
   the difference in PR-AUC with its 95 % CI.

**Stage transitions.** Stage 2 takes the 4 classifiers with the best mean CV PR-AUC over
their stage 1 strategies. Stage 3 takes the two best classifiers by the rule above, each
with its best strategy, and tries a fixed set of settings (5 to 12 per classifier, defined in
`fraud.experiments.tuning_space`), recorded with `stage = stage3`.

**Final model.** The configuration that wins under the same rule over all rows, fitted on
the training split. No decision threshold is chosen in Phase 3 and the test split is not
used.

## 15. Final evaluation protocol (Phase 4)

Written before the test split was loaded. The validation results of this section's analyses
had been seen at that point (a dry run that uses validation rows in place of the test split).

1. **Model.** The pipeline saved by Phase 3 (XGBoost, Safe-Level-SMOTE 0.5, tuned), fitted
   on the training split only. It is not refitted on train + validation: the threshold is
   chosen on validation scores of exactly this model, and a refitted model would have a
   different score distribution with no data left to choose a threshold on.
2. **Threshold.** A transaction is flagged when `score >= threshold`. The threshold is the
   one with the lowest total cost on validation (section 8): the sum of the amounts of
   missed fraud plus the review cost for every flagged transaction. Candidate thresholds
   are the midpoints between neighbouring distinct validation scores, plus "flag nothing"
   and "flag everything". A cost tie goes to the threshold that flags fewer transactions.
   The F1-maximising threshold is reported for comparison only.
3. **Thesis hypothesis** ("oversampling mainly shifts the decision boundary"), on
   validation: the selected model against XGBoost with class weights, XGBoost without
   resampling and the MLP without resampling (the stage 1 configurations, default
   parameters). Compared: PR-AUC, cost with every model at threshold 0.5, and cost with
   every model at its own cost-optimal threshold, as paired bootstrap differences. For the
   last one the threshold is chosen again inside every resample. Support for the
   hypothesis: the cost gap at 0.5 shrinks or disappears at the optimal thresholds.
4. **Test, once.** `python -m fraud.final_eval` loads the test split in one place and
   appends every load to `reports/test_set_access.log`. Reported: the selected model at its
   validation threshold, and the three reference models at their own validation
   thresholds (after selection, not used for it). Intervals are 95 % stratified bootstrap.
5. **With and without deduplication.** Primary: the test split of the deduplicated data.
   "Without deduplication": the same model and threshold on those rows plus the exact
   duplicates that were removed from the test period. Nothing is refitted.
6. **Secondary random split.** The selected configuration refitted on a stratified random
   70/15/15 split of the deduplicated data (seed 42), threshold from its validation part,
   evaluated on its test part. Labelled as secondary; it does not estimate performance on
   future data.
7. **Score, not probability.** The model is trained on oversampled data, so its output is
   reported as a fraud score unless the reliability curve on validation shows it is
   calibrated.
8. **SHAP.** TreeExplainer on the XGBoost step. Local examples are picked by rule: the
   caught and the missed fraud with the largest amount, and the false alarm with the
   highest score.
