# Experiment summary (Phase 3)

## Conclusions

- **Selected by the rule in PLAN.md §14: XGBoost with Safe-Level-SMOTE at ratio 0.5** (tuned:
  400 trees, learning rate 0.1, depth 6). Mean CV PR-AUC 0.767, validation PR-AUC 0.851
  [0.759, 0.933]. It is clearly better on validation than the same classifier without
  resampling (+0.181 [0.061, 0.289]) and than the logistic regression baseline
  (+0.100 [0.026, 0.169]).
- **The top of the ranking is a tie.** The best 14 configurations of stages 1 and 2 lie
  within 0.016 mean CV PR-AUC of each other, while the folds of a single configuration
  differ by about 0.07 (standard deviation). Tuning added 0.004 to the CV mean, which is
  noise. The selection is the outcome of a rule fixed in advance, not evidence that this
  configuration is better than its neighbours.
- **Validation does not confirm the CV winner as the best model.** The MLP without
  resampling has the highest validation PR-AUC (0.900) and the selected configuration is
  0.049 lower [-0.097, -0.006]. The MLP ranks only sixth by CV (0.713), so the two
  estimates disagree. Validation has 55 fraud cases and a CV fold 56 to 87, so neither is
  precise. The rule was followed and the disagreement is flagged for review, not resolved.
- **What matters most is doing something about the imbalance in the boosted models and in
  logistic regression.** With default parameters and no reweighting, XGBoost (CV 0.475) and
  especially LightGBM (0.134) rank poorly; class weights or any oversampling bring both to
  about 0.74 to 0.76. Random forest and extra trees are already at 0.741 without any
  handling and gain nothing.
- **Oversampling against class weights is a small difference.** For the four tree ensembles
  the best oversampling run is within -0.003 to +0.023 mean CV PR-AUC of class weights, and
  none of the validation differences excludes zero. Only logistic regression gains clearly
  (CV +0.089, validation +0.054 [0.023, 0.094]).
- **LightGBM is unstable on this data.** Three LightGBM runs collapse on validation while
  looking normal in CV or in neighbouring settings: no resampling (0.062), Safe-Level-SMOTE
  0.5 (CV 0.727, validation 0.396) and one tuning setting (0.359). This is a reason not to
  pick a LightGBM configuration from a single number.
- **Runtime.** Every stage ran well inside the budget: stage 2 took about 25 minutes for 60
  configurations and stage 3 about 5 minutes. SMOTE-Tomek and SMOTE-ENN cost 45 to 80 s of
  resampling on the full training split, the other samplers about a second.

## Comparison with the thesis

The thesis covered 18 datasets with imbalance up to 28:1; this is one dataset at about
540:1, one seed, k fixed at 5, and PR-AUC only (no threshold is chosen yet, so F1, MCC and
G-Mean are not compared). Safe-Level-SMOTE and Polynom-Fit SMOTE were re-implemented from
the papers because the thesis code was not available, so they may differ in detail from the
thesis implementation. The statements below are about mean CV PR-AUC unless stated; "clear"
means a 95 % paired bootstrap interval on validation that excludes zero.

**Does any SMOTE variant beat plain SMOTE?** Not consistently.

- For the tree ensembles the differences are small and of mixed sign (-0.056 to +0.020).
  The one consistent case is Safe-Level-SMOTE with XGBoost: better than SMOTE at both ratios
  in CV (+0.003, +0.020) and clearly better on validation (+0.033, +0.044).
- For k-NN both Safe-Level-SMOTE (+0.068, +0.124) and Polynom-Fit SMOTE (+0.112, +0.110) are
  far better than SMOTE, clearly so on validation. But SMOTE itself makes k-NN worse than no
  resampling (0.618 to 0.490), and even the best variant (0.603) stays below no resampling.
  The variants do less damage; they do not help.
- **SMOTE-Tomek is exactly SMOTE here**, which agrees with the thesis in the strongest
  possible way: the scaled training data has 16 Tomek links before oversampling and none
  after SMOTE (counted on the full training split at both ratios), so the cleaning step
  removes nothing.
- SMOTE-ENN is within noise of SMOTE for the ensembles and logistic regression and clearly
  worse for k-NN.

**Does oversampling improve PR-AUC over no resampling or class weights?** It depends on the
classifier, which the thesis average (+0.001) hides.

- Across the 9 classifiers of stage 1, SMOTE 0.1 minus no resampling has a median of -0.001,
  which matches the thesis, but the mean is +0.068 and the range is -0.283 (decision tree)
  to +0.617 (LightGBM).
- The large gains are for XGBoost, LightGBM and logistic regression, where class weights
  give most of the same gain. Against class weights, oversampling adds little (see
  Conclusions).
- The thesis hypothesis that oversampling moves the decision boundary and not the ranking
  holds for random forest and extra trees: PR-AUC is unchanged while recall at threshold 0.5
  rises (0.69 to 0.76, 0.71 to 0.75). It does not hold for the single decision tree, k-NN
  and the MLP, where SMOTE makes the ranking itself worse (CV -0.283, -0.128, -0.047), nor
  for the boosted models and logistic regression, where it makes the ranking better.

**Do Safe-Level-SMOTE and Polynom-Fit SMOTE keep their advantage at 540:1?** Only partly.

- Safe-Level-SMOTE: an advantage for XGBoost and (relative to SMOTE) for k-NN. No advantage
  for random forest or extra trees, worse than SMOTE for LightGBM in CV (-0.056, -0.026),
  and for logistic regression clearly worse on validation (-0.070, -0.126). The thesis
  result that its advantage grows with the imbalance ratio is not reproduced as a general
  effect. Why was not investigated.
- Polynom-Fit SMOTE: the best oversampler by CV for LightGBM, random forest, logistic
  regression and k-NN, and for logistic regression at ratio 0.5 the gain over SMOTE is large
  (+0.035). For the tree ensembles the margins over SMOTE are within noise. This is the
  closest match to the thesis, where it was the only method significantly better than
  SMOTE, but here the evidence is one dataset and mostly not significant.

**Is the choice of method most important for linear and neighbour-based classifiers?**
Mostly yes.

- Range of mean CV PR-AUC over the 11 oversampling runs: k-NN 0.204, logistic regression
  0.080, LightGBM 0.075, XGBoost 0.031, extra trees 0.030, random forest 0.015.
- k-NN and logistic regression are the most sensitive and random forest and extra trees the
  least, as in the thesis. LightGBM is the exception among the tree ensembles; its range
  comes from the unstable runs described above.
- Logistic regression and k-NN were not in the top 4 of stage 1. They were run in addition,
  after stage 1 had been seen, only for this comparison, and are not part of the ranking.

**Where the results contradict the thesis:** Safe-Level-SMOTE is not generally the best
method for PR-AUC at this imbalance ratio, and oversampling is not neutral for PR-AUC: it
helps some classifiers a lot and hurts others a lot. Where they agree: SMOTE-Tomek equals
SMOTE, tree ensembles are insensitive to the method, and Polynom-Fit SMOTE is the most
reliable variant.

<!-- generated:start -->
## Selected configuration

**XGBoost, Safe-Level-SMOTE 0.5** (parameters: learning_rate=0.1, max_depth=6, min_child_weight=1, n_estimators=400, subsample=1.0; from stage3). Mean CV PR-AUC 0.767; validation PR-AUC 0.851, 95 % CI [0.759, 0.933].

Validation check: Best on validation is MLP, No resampling (PR-AUC 0.900, CV 0.713). Selected minus best-on-validation: -0.049, 95 % CI [-0.097, -0.006]. **The interval excludes zero: the CV winner is clearly worse on validation. Flagged for review.**

Saved to `models/final_pipeline.joblib` (0.22 MB, fitted on the training split) with `models/final_config.json`. No decision threshold is chosen yet and the test split was not used.

## Stage 1: screening of 9 classifiers

Ranked by the best mean CV PR-AUC over the strategies. CV columns are means over the 3 time-ordered folds; validation has 55 fraud cases, so its intervals are wide. Fit time is resampling plus fitting on the full training split; predict time is scoring the validation split.

| Rank | Classifier | CV No resampling | CV Class weights | CV SMOTE 0.1 | Best strategy | CV folds of best | Val PR-AUC of best | Fit (s) | Predict (s) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | XGBoost | 0.475 | 0.740 | 0.756 | SMOTE 0.1 | 0.79 / 0.67 / 0.81 | 0.828 [0.729, 0.918] | 1.3 | 0.1 |
| 2 | LightGBM | 0.134 | 0.740 | 0.751 | SMOTE 0.1 | 0.79 / 0.65 / 0.81 | 0.828 [0.733, 0.916] | 1.8 | 0.1 |
| 3 | Extra trees | 0.741 | 0.723 | 0.740 | No resampling | 0.78 / 0.63 / 0.81 | 0.871 [0.784, 0.947] | 2.9 | 0.1 |
| 4 | Random forest | 0.741 | 0.739 | 0.732 | No resampling | 0.76 / 0.69 / 0.78 | 0.861 [0.775, 0.939] | 16.6 | 0.1 |
| 5 | Logistic regression | 0.561 | 0.663 | 0.741 | SMOTE 0.1 | 0.75 / 0.64 / 0.83 | 0.807 [0.719, 0.891] | 0.8 | 0.0 |
| 6 | MLP | 0.713 | n/a | 0.666 | No resampling | 0.73 / 0.62 / 0.79 | 0.900 [0.823, 0.965] | 10.2 | 0.0 |
| 7 | k-NN | 0.618 | n/a | 0.490 | No resampling | 0.71 / 0.49 / 0.66 | 0.800 [0.696, 0.901] | 0.2 | 7.6 |
| 8 | Decision tree | 0.464 | 0.381 | 0.181 | No resampling | 0.56 / 0.37 / 0.46 | 0.378 [0.269, 0.502] | 11.8 | 0.0 |
| 9 | Gaussian NB | 0.063 | n/a | 0.066 | SMOTE 0.1 | 0.05 / 0.05 / 0.10 | 0.050 [0.040, 0.062] | 0.4 | 0.1 |

### Ranking quality against decisions at threshold 0.5

No resampling against SMOTE 0.1 for every classifier. If oversampling mainly moves the decision boundary, recall and the number of flagged transactions change much more than PR-AUC.

| Classifier | CV PR-AUC none → SMOTE | Val PR-AUC none → SMOTE | Recall at 0.5 | Precision at 0.5 | Flagged at 0.5 |
|---|---|---|---|---|---|
| XGBoost | 0.475 → 0.756 | 0.670 → 0.828 | 0.67 → 0.78 | 0.88 → 0.81 | 42 → 53 |
| LightGBM | 0.134 → 0.751 | 0.062 → 0.828 | 0.29 → 0.78 | 0.15 → 0.67 | 106 → 64 |
| Extra trees | 0.741 → 0.740 | 0.871 → 0.864 | 0.71 → 0.75 | 1.00 → 1.00 | 39 → 41 |
| Random forest | 0.741 → 0.732 | 0.861 → 0.856 | 0.69 → 0.76 | 1.00 → 1.00 | 38 → 42 |
| Logistic regression | 0.561 → 0.741 | 0.750 → 0.807 | 0.53 → 0.89 | 0.71 → 0.25 | 41 → 194 |
| MLP | 0.713 → 0.666 | 0.900 → 0.798 | 0.85 → 0.78 | 0.90 → 0.53 | 52 → 81 |
| k-NN | 0.618 → 0.490 | 0.800 → 0.491 | 0.78 → 0.85 | 1.00 → 0.29 | 43 → 160 |
| Decision tree | 0.464 → 0.181 | 0.378 → 0.102 | 0.67 → 0.73 | 0.56 → 0.14 | 66 → 286 |
| Gaussian NB | 0.063 → 0.066 | 0.049 → 0.050 | 0.85 → 0.87 | 0.04 → 0.04 | 1150 → 1237 |

## Stage 2: SMOTE family for the top 4 classifiers

Mean CV PR-AUC. Logistic regression and k-NN did not reach the top 4 and were run in addition, for the comparison with the thesis only: they are not part of the ranking or of the model selection.

| Strategy | XGBoost | LightGBM | Extra trees | Random forest | Logistic regression | k-NN |
|---|---|---|---|---|---|---|
| No resampling | 0.475 | 0.134 | 0.741 | 0.741 | 0.561 | 0.618 |
| Class weights | 0.740 | 0.740 | 0.723 | 0.739 | 0.663 | n/a |
| SMOTE 0.1 | 0.756 | 0.751 | 0.740 | 0.732 | 0.741 | 0.490 |
| SMOTE 0.5 | 0.743 | 0.753 | 0.722 | 0.730 | 0.717 | 0.435 |
| SMOTE 1 | 0.747 | 0.684 | 0.730 | 0.723 | 0.673 | 0.433 |
| SMOTE-Tomek 0.1 | 0.756 | 0.751 | 0.740 | 0.732 | 0.741 | 0.490 |
| SMOTE-Tomek 0.5 | 0.743 | 0.753 | 0.722 | 0.730 | 0.717 | 0.435 |
| SMOTE-ENN 0.1 | 0.752 | 0.756 | 0.710 | 0.735 | 0.744 | 0.439 |
| SMOTE-ENN 0.5 | 0.732 | 0.758 | 0.732 | 0.727 | 0.721 | 0.399 |
| Safe-Level-SMOTE 0.1 | 0.759 | 0.695 | 0.733 | 0.729 | 0.716 | 0.558 |
| Safe-Level-SMOTE 0.5 | 0.763 | 0.727 | 0.713 | 0.732 | 0.721 | 0.559 |
| Polynom-Fit SMOTE 0.1 | 0.749 | 0.758 | 0.740 | 0.736 | 0.738 | 0.603 |
| Polynom-Fit SMOTE 0.5 | 0.741 | 0.744 | 0.738 | 0.721 | 0.752 | 0.545 |

### Each variant against plain SMOTE at the same ratio

CV difference is variant minus SMOTE; "folds won" is out of 3. The validation difference is a paired bootstrap. `*` marks a 95 % interval that excludes zero.

| Classifier | Variant | CV PR-AUC | CV of SMOTE | CV difference | Folds won | Val difference [95 % CI] |
|---|---|---|---|---|---|---|
| XGBoost | SMOTE-Tomek 0.1 | 0.756 | 0.756 | +0.000 | 0 | +0.000 [0.000, 0.000] |
| XGBoost | SMOTE-Tomek 0.5 | 0.743 | 0.743 | +0.000 | 0 | +0.000 [0.000, 0.000] |
| XGBoost | SMOTE-ENN 0.1 | 0.752 | 0.756 | -0.004 | 1 | +0.008 [-0.031, 0.047] |
| XGBoost | SMOTE-ENN 0.5 | 0.732 | 0.743 | -0.011 | 1 | -0.001 [-0.026, 0.021] |
| XGBoost | Safe-Level-SMOTE 0.1 | 0.759 | 0.756 | +0.003 | 2 | +0.033 [0.005, 0.066] * |
| XGBoost | Safe-Level-SMOTE 0.5 | 0.763 | 0.743 | +0.020 | 2 | +0.044 [0.003, 0.094] * |
| XGBoost | Polynom-Fit SMOTE 0.1 | 0.749 | 0.756 | -0.006 | 0 | +0.003 [-0.038, 0.045] |
| XGBoost | Polynom-Fit SMOTE 0.5 | 0.741 | 0.743 | -0.002 | 1 | +0.034 [0.000, 0.079] * |
| LightGBM | SMOTE-Tomek 0.1 | 0.751 | 0.751 | +0.000 | 0 | +0.000 [0.000, 0.000] |
| LightGBM | SMOTE-Tomek 0.5 | 0.753 | 0.753 | +0.000 | 0 | +0.000 [0.000, 0.000] |
| LightGBM | SMOTE-ENN 0.1 | 0.756 | 0.751 | +0.005 | 1 | -0.006 [-0.024, 0.015] |
| LightGBM | SMOTE-ENN 0.5 | 0.758 | 0.753 | +0.005 | 1 | +0.046 [0.000, 0.101] * |
| LightGBM | Safe-Level-SMOTE 0.1 | 0.695 | 0.751 | -0.056 | 1 | +0.025 [-0.008, 0.063] |
| LightGBM | Safe-Level-SMOTE 0.5 | 0.727 | 0.753 | -0.026 | 1 | -0.381 [-0.468, -0.279] * |
| LightGBM | Polynom-Fit SMOTE 0.1 | 0.758 | 0.751 | +0.008 | 2 | +0.016 [-0.014, 0.052] |
| LightGBM | Polynom-Fit SMOTE 0.5 | 0.744 | 0.753 | -0.009 | 1 | +0.075 [0.016, 0.143] * |
| Extra trees | SMOTE-Tomek 0.1 | 0.740 | 0.740 | +0.000 | 0 | +0.000 [0.000, 0.000] |
| Extra trees | SMOTE-Tomek 0.5 | 0.722 | 0.722 | +0.000 | 0 | +0.000 [0.000, 0.000] |
| Extra trees | SMOTE-ENN 0.1 | 0.710 | 0.740 | -0.030 | 0 | -0.003 [-0.017, 0.009] |
| Extra trees | SMOTE-ENN 0.5 | 0.732 | 0.722 | +0.010 | 2 | -0.001 [-0.013, 0.011] |
| Extra trees | Safe-Level-SMOTE 0.1 | 0.733 | 0.740 | -0.007 | 0 | +0.006 [-0.004, 0.018] |
| Extra trees | Safe-Level-SMOTE 0.5 | 0.713 | 0.722 | -0.008 | 0 | +0.010 [-0.003, 0.028] |
| Extra trees | Polynom-Fit SMOTE 0.1 | 0.740 | 0.740 | -0.000 | 1 | -0.002 [-0.022, 0.016] |
| Extra trees | Polynom-Fit SMOTE 0.5 | 0.738 | 0.722 | +0.016 | 3 | +0.011 [-0.001, 0.027] |
| Random forest | SMOTE-Tomek 0.1 | 0.732 | 0.732 | +0.000 | 0 | +0.000 [0.000, 0.000] |
| Random forest | SMOTE-Tomek 0.5 | 0.730 | 0.730 | +0.000 | 0 | +0.000 [0.000, 0.000] |
| Random forest | SMOTE-ENN 0.1 | 0.735 | 0.732 | +0.003 | 1 | +0.010 [-0.008, 0.034] |
| Random forest | SMOTE-ENN 0.5 | 0.727 | 0.730 | -0.003 | 1 | +0.004 [-0.020, 0.028] |
| Random forest | Safe-Level-SMOTE 0.1 | 0.729 | 0.732 | -0.002 | 2 | +0.004 [-0.032, 0.039] |
| Random forest | Safe-Level-SMOTE 0.5 | 0.732 | 0.730 | +0.002 | 2 | +0.010 [-0.027, 0.049] |
| Random forest | Polynom-Fit SMOTE 0.1 | 0.736 | 0.732 | +0.004 | 1 | +0.014 [-0.015, 0.047] |
| Random forest | Polynom-Fit SMOTE 0.5 | 0.721 | 0.730 | -0.009 | 1 | +0.016 [-0.014, 0.053] |
| Logistic regression | SMOTE-Tomek 0.1 | 0.741 | 0.741 | +0.000 | 0 | +0.000 [0.000, 0.000] |
| Logistic regression | SMOTE-Tomek 0.5 | 0.717 | 0.717 | +0.000 | 0 | +0.000 [0.000, 0.000] |
| Logistic regression | SMOTE-ENN 0.1 | 0.744 | 0.741 | +0.003 | 2 | +0.015 [0.004, 0.029] * |
| Logistic regression | SMOTE-ENN 0.5 | 0.721 | 0.717 | +0.004 | 3 | +0.000 [-0.004, 0.004] |
| Logistic regression | Safe-Level-SMOTE 0.1 | 0.716 | 0.741 | -0.025 | 0 | -0.070 [-0.113, -0.030] * |
| Logistic regression | Safe-Level-SMOTE 0.5 | 0.721 | 0.717 | +0.003 | 2 | -0.126 [-0.192, -0.063] * |
| Logistic regression | Polynom-Fit SMOTE 0.1 | 0.738 | 0.741 | -0.003 | 2 | +0.055 [0.022, 0.098] * |
| Logistic regression | Polynom-Fit SMOTE 0.5 | 0.752 | 0.717 | +0.035 | 2 | +0.010 [-0.001, 0.025] |
| k-NN | SMOTE-Tomek 0.1 | 0.490 | 0.490 | +0.000 | 0 | +0.000 [0.000, 0.000] |
| k-NN | SMOTE-Tomek 0.5 | 0.435 | 0.435 | +0.000 | 0 | +0.000 [0.000, 0.000] |
| k-NN | SMOTE-ENN 0.1 | 0.439 | 0.490 | -0.051 | 0 | -0.043 [-0.079, -0.017] * |
| k-NN | SMOTE-ENN 0.5 | 0.399 | 0.435 | -0.036 | 0 | -0.039 [-0.066, -0.018] * |
| k-NN | Safe-Level-SMOTE 0.1 | 0.558 | 0.490 | +0.068 | 2 | +0.302 [0.222, 0.373] * |
| k-NN | Safe-Level-SMOTE 0.5 | 0.559 | 0.435 | +0.124 | 3 | +0.395 [0.319, 0.467] * |
| k-NN | Polynom-Fit SMOTE 0.1 | 0.603 | 0.490 | +0.112 | 3 | +0.223 [0.140, 0.307] * |
| k-NN | Polynom-Fit SMOTE 0.5 | 0.545 | 0.435 | +0.110 | 3 | +0.236 [0.163, 0.315] * |

### Oversampling against no resampling and class weights

The best oversampling run of each classifier (chosen by CV). `*` marks a 95 % interval that excludes zero.

| Classifier | Best oversampling (by CV) | CV: none / weights / oversampling | Val: oversampling minus none [95 % CI] | Val: oversampling minus weights [95 % CI] |
|---|---|---|---|---|
| XGBoost | Safe-Level-SMOTE 0.5 | 0.475 / 0.740 / 0.763 | +0.180 [0.059, 0.289] * | +0.015 [-0.025, 0.056] |
| LightGBM | Polynom-Fit SMOTE 0.1 | 0.134 / 0.740 / 0.758 | +0.782 [0.680, 0.866] * | +0.013 [-0.027, 0.062] |
| Extra trees | SMOTE 0.1 | 0.741 / 0.723 / 0.740 | -0.008 [-0.019, 0.002] | +0.003 [-0.009, 0.015] |
| Random forest | Polynom-Fit SMOTE 0.1 | 0.741 / 0.739 / 0.736 | +0.009 [-0.005, 0.030] | +0.017 [-0.002, 0.043] |
| Logistic regression | Polynom-Fit SMOTE 0.5 | 0.561 / 0.663 / 0.752 | +0.113 [0.044, 0.178] * | +0.054 [0.023, 0.094] * |
| k-NN | Polynom-Fit SMOTE 0.1 | 0.618 / n/a / 0.603 | -0.087 [-0.167, -0.006] * | n/a |

### How much the choice of oversampling method matters per classifier

Spread of the mean CV PR-AUC over all oversampling runs of a classifier.

| Classifier | Oversampling runs | Worst | Best | Range | Std |
|---|---|---|---|---|---|
| XGBoost | 11 | 0.732 (SMOTE-ENN 0.5) | 0.763 (Safe-Level-SMOTE 0.5) | 0.031 | 0.009 |
| LightGBM | 11 | 0.684 (SMOTE 1) | 0.758 (Polynom-Fit SMOTE 0.1) | 0.075 | 0.026 |
| Extra trees | 11 | 0.710 (SMOTE-ENN 0.1) | 0.740 (SMOTE 0.1) | 0.030 | 0.011 |
| Random forest | 11 | 0.721 (Polynom-Fit SMOTE 0.5) | 0.736 (Polynom-Fit SMOTE 0.1) | 0.015 | 0.004 |
| Logistic regression | 11 | 0.673 (SMOTE 1) | 0.752 (Polynom-Fit SMOTE 0.5) | 0.080 | 0.022 |
| k-NN | 11 | 0.399 (SMOTE-ENN 0.5) | 0.603 (Polynom-Fit SMOTE 0.1) | 0.204 | 0.067 |

## Overall ranking (stages 1 and 2, top 15)

Ordered by the selection rule: mean CV PR-AUC, ties within 0.005 broken toward the simpler strategy.

| Rank | Classifier | Strategy | CV PR-AUC (mean ± std) | CV folds | Val PR-AUC [95 % CI] | Val ROC-AUC | Fit (s) | Predict (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | XGBoost | Safe-Level-SMOTE 0.5 | 0.763 ± 0.071 | 0.79 / 0.68 / 0.82 | 0.850 [0.759, 0.934] | 0.988 | 1.6 | 0.0 |
| 2 | XGBoost | Safe-Level-SMOTE 0.1 | 0.759 ± 0.070 | 0.79 / 0.68 / 0.81 | 0.860 [0.772, 0.939] | 0.989 | 1.5 | 0.0 |
| 3 | LightGBM | Polynom-Fit SMOTE 0.1 | 0.758 ± 0.074 | 0.79 / 0.67 / 0.81 | 0.844 [0.746, 0.928] | 0.980 | 1.5 | 0.1 |
| 4 | LightGBM | SMOTE-ENN 0.5 | 0.758 ± 0.048 | 0.79 / 0.70 / 0.78 | 0.823 [0.727, 0.912] | 0.991 | 80.3 | 0.1 |
| 5 | LightGBM | SMOTE-ENN 0.1 | 0.756 ± 0.058 | 0.79 / 0.69 / 0.79 | 0.822 [0.725, 0.910] | 0.991 | 44.3 | 0.1 |
| 6 | XGBoost | SMOTE 0.1 | 0.756 ± 0.072 | 0.79 / 0.67 / 0.81 | 0.828 [0.729, 0.918] | 0.990 | 1.3 | 0.1 |
| 7 | XGBoost | SMOTE-Tomek 0.1 | 0.756 ± 0.072 | 0.79 / 0.67 / 0.81 | 0.828 [0.729, 0.918] | 0.990 | 44.9 | 0.0 |
| 8 | LightGBM | SMOTE 0.5 | 0.753 ± 0.035 | 0.79 / 0.72 / 0.75 | 0.776 [0.673, 0.876] | 0.965 | 2.2 | 0.1 |
| 9 | LightGBM | SMOTE-Tomek 0.5 | 0.753 ± 0.035 | 0.79 / 0.72 / 0.75 | 0.776 [0.673, 0.876] | 0.965 | 81.9 | 0.1 |
| 10 | XGBoost | SMOTE-ENN 0.1 | 0.752 ± 0.058 | 0.79 / 0.68 / 0.78 | 0.835 [0.736, 0.920] | 0.981 | 43.8 | 0.0 |
| 11 | LightGBM | SMOTE 0.1 | 0.751 ± 0.085 | 0.79 / 0.65 / 0.81 | 0.828 [0.733, 0.916] | 0.993 | 1.8 | 0.1 |
| 12 | LightGBM | SMOTE-Tomek 0.1 | 0.751 ± 0.085 | 0.79 / 0.65 / 0.81 | 0.828 [0.733, 0.916] | 0.993 | 45.4 | 0.1 |
| 13 | XGBoost | Polynom-Fit SMOTE 0.1 | 0.749 ± 0.074 | 0.79 / 0.66 / 0.80 | 0.830 [0.740, 0.914] | 0.988 | 1.1 | 0.0 |
| 14 | XGBoost | SMOTE 1 | 0.747 ± 0.069 | 0.79 / 0.67 / 0.79 | 0.801 [0.697, 0.895] | 0.984 | 2.4 | 0.1 |
| 15 | Extra trees | No resampling | 0.741 ± 0.095 | 0.78 / 0.63 / 0.81 | 0.871 [0.784, 0.947] | 0.952 | 2.9 | 0.1 |

## Paired comparisons on validation

Paired stratified bootstrap of the PR-AUC difference (configuration minus reference) for the top 3 configurations of stages 1 and 2 and the selected one. An interval that includes zero means the difference cannot be told apart from noise on this validation set.

| Configuration | Versus | Val PR-AUC | Reference PR-AUC | Difference | 95 % CI of difference | Resamples where config wins |
|---|---|---|---|---|---|---|
| XGBoost, Safe-Level-SMOTE 0.5 | same classifier, no resampling | 0.850 | 0.670 | +0.180 | [0.059, 0.289] | 100% |
| XGBoost, Safe-Level-SMOTE 0.5 | baseline (logistic regression, no resampling) | 0.850 | 0.750 | +0.100 | [0.021, 0.169] | 99% |
| XGBoost, Safe-Level-SMOTE 0.1 | same classifier, no resampling | 0.860 | 0.670 | +0.191 | [0.073, 0.295] | 100% |
| XGBoost, Safe-Level-SMOTE 0.1 | baseline (logistic regression, no resampling) | 0.860 | 0.750 | +0.110 | [0.035, 0.180] | 100% |
| LightGBM, Polynom-Fit SMOTE 0.1 | same classifier, no resampling | 0.844 | 0.062 | +0.782 | [0.680, 0.866] | 100% |
| LightGBM, Polynom-Fit SMOTE 0.1 | baseline (logistic regression, no resampling) | 0.844 | 0.750 | +0.094 | [0.016, 0.172] | 99% |
| XGBoost, Safe-Level-SMOTE 0.5 (tuned) | same classifier, no resampling | 0.851 | 0.670 | +0.181 | [0.061, 0.289] | 100% |
| XGBoost, Safe-Level-SMOTE 0.5 (tuned) | baseline (logistic regression, no resampling) | 0.851 | 0.750 | +0.100 | [0.026, 0.169] | 100% |

## Stage 3: hyperparameter tuning

Every setting that was tried, with the untuned run of the same strategy for reference.

| Classifier | Strategy | Changed parameters | CV PR-AUC (mean ± std) | Val PR-AUC [95 % CI] | Fit (s) |
|---|---|---|---|---|---|
| XGBoost | Safe-Level-SMOTE 0.5 | learning_rate=0.1, max_depth=6, min_child_weight=1, n_estimators=400, subsample=1.0 | 0.767 ± 0.069 | 0.851 [0.759, 0.933] | 4.1 |
| XGBoost | Safe-Level-SMOTE 0.5 | defaults | 0.763 ± 0.071 | 0.850 [0.759, 0.934] | 1.6 |
| XGBoost | Safe-Level-SMOTE 0.5 | learning_rate=0.3, max_depth=6, min_child_weight=1, n_estimators=400, subsample=0.7 | 0.761 ± 0.062 | 0.857 [0.770, 0.939] | 4.7 |
| XGBoost | Safe-Level-SMOTE 0.5 | learning_rate=0.3, max_depth=8, min_child_weight=1, n_estimators=400, subsample=1.0 | 0.761 ± 0.072 | 0.859 [0.771, 0.937] | 3.6 |
| XGBoost | Safe-Level-SMOTE 0.5 | learning_rate=0.1, max_depth=3, min_child_weight=1, n_estimators=200, subsample=1.0 | 0.758 ± 0.070 | 0.868 [0.782, 0.944] | 1.9 |
| XGBoost | Safe-Level-SMOTE 0.5 | learning_rate=0.3, max_depth=8, min_child_weight=5, n_estimators=200, subsample=1.0 | 0.751 ± 0.066 | 0.856 [0.766, 0.936] | 2.3 |
| XGBoost | Safe-Level-SMOTE 0.5 | learning_rate=0.1, max_depth=8, min_child_weight=10, n_estimators=200, subsample=1.0 | 0.748 ± 0.061 | 0.862 [0.774, 0.938] | 2.8 |
| XGBoost | Safe-Level-SMOTE 0.5 | learning_rate=0.3, max_depth=8, min_child_weight=10, n_estimators=400, subsample=1.0 | 0.746 ± 0.057 | 0.849 [0.756, 0.929] | 3.8 |
| XGBoost | Safe-Level-SMOTE 0.5 | learning_rate=0.1, max_depth=8, min_child_weight=10, n_estimators=200, subsample=0.7 | 0.746 ± 0.068 | 0.862 [0.775, 0.940] | 2.9 |
| XGBoost | Safe-Level-SMOTE 0.5 | learning_rate=0.3, max_depth=4, min_child_weight=10, n_estimators=400, subsample=1.0 | 0.744 ± 0.071 | 0.861 [0.774, 0.940] | 3.2 |
| XGBoost | Safe-Level-SMOTE 0.5 | learning_rate=0.3, max_depth=3, min_child_weight=10, n_estimators=400, subsample=1.0 | 0.737 ± 0.064 | 0.871 [0.788, 0.944] | 3.5 |
| XGBoost | Safe-Level-SMOTE 0.5 | learning_rate=0.03, max_depth=3, min_child_weight=10, n_estimators=400, subsample=1.0 | 0.727 ± 0.075 | 0.863 [0.779, 0.938] | 3.3 |
| XGBoost | Safe-Level-SMOTE 0.5 | learning_rate=0.03, max_depth=4, min_child_weight=10, n_estimators=200, subsample=0.7 | 0.719 ± 0.053 | 0.857 [0.771, 0.936] | 2.6 |
| LightGBM | Polynom-Fit SMOTE 0.1 | learning_rate=0.05, min_child_samples=10, n_estimators=800, num_leaves=7 | 0.759 ± 0.067 | 0.832 [0.739, 0.919] | 5.7 |
| LightGBM | Polynom-Fit SMOTE 0.1 | defaults | 0.758 ± 0.074 | 0.844 [0.746, 0.928] | 1.5 |
| LightGBM | Polynom-Fit SMOTE 0.1 | n_estimators=800, num_leaves=15 | 0.758 ± 0.082 | 0.854 [0.766, 0.935] | 7.4 |
| LightGBM | Polynom-Fit SMOTE 0.1 | learning_rate=0.05, min_child_samples=50, n_estimators=800 | 0.758 ± 0.084 | 0.853 [0.765, 0.934] | 9.0 |
| LightGBM | Polynom-Fit SMOTE 0.1 | learning_rate=0.05, min_child_samples=50, n_estimators=400 | 0.753 ± 0.089 | 0.843 [0.750, 0.928] | 4.7 |
| LightGBM | Polynom-Fit SMOTE 0.1 | learning_rate=0.05, n_estimators=400 | 0.752 ± 0.090 | 0.847 [0.757, 0.932] | 4.7 |
| LightGBM | Polynom-Fit SMOTE 0.1 | learning_rate=0.02, n_estimators=400 | 0.748 ± 0.090 | 0.848 [0.757, 0.932] | 4.4 |
| LightGBM | Polynom-Fit SMOTE 0.1 | learning_rate=0.05, n_estimators=400, num_leaves=15 | 0.747 ± 0.082 | 0.844 [0.755, 0.929] | 3.8 |
| LightGBM | Polynom-Fit SMOTE 0.1 | min_child_samples=100, n_estimators=200, num_leaves=7 | 0.739 ± 0.087 | 0.827 [0.736, 0.920] | 1.8 |
| LightGBM | Polynom-Fit SMOTE 0.1 | learning_rate=0.05, min_child_samples=10, n_estimators=200, num_leaves=63 | 0.725 ± 0.102 | 0.854 [0.765, 0.936] | 2.9 |
| LightGBM | Polynom-Fit SMOTE 0.1 | learning_rate=0.02, n_estimators=400, num_leaves=63 | 0.716 ± 0.109 | 0.851 [0.759, 0.932] | 5.4 |
| LightGBM | Polynom-Fit SMOTE 0.1 | learning_rate=0.02, n_estimators=200, num_leaves=7 | 0.710 ± 0.071 | 0.773 [0.656, 0.896] | 1.7 |
| LightGBM | Polynom-Fit SMOTE 0.1 | min_child_samples=10, n_estimators=200, num_leaves=15 | 0.628 ± 0.165 | 0.359 [0.289, 0.467] | 2.0 |

## Figures

- `reports/figures/exp_stage1_ranking.png`
- `reports/figures/exp_stage2_cv_heatmap.png`
- `reports/figures/exp_pr_curves_best_per_model.png`
- `reports/figures/exp_pr_auc_vs_fit_time.png`
<!-- generated:end -->
