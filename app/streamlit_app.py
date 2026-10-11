"""Streamlit app: explore scored transactions, the threshold and how the model was chosen.

    streamlit run app/streamlit_app.py        (or ``make app``)

The app loads the committed model through :mod:`fraud.serving` (the same code the API
uses) and the small committed files in ``app/data/`` and ``reports/``. It needs neither the
API nor the Kaggle dataset.
"""

import json
import sys
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

try:
    import fraud  # noqa: F401
except ImportError:  # the package is not installed (e.g. a hosted app): use the sources
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud.config import (
    AMOUNT_COLUMN,
    APP_SAMPLE_CSV,
    APP_VALIDATION_SCORES,
    FIGURES_DIR,
    FINAL_METRICS_JSON,
    FRAUD_COLOR,
    LEGIT_COLOR,
    NEUTRAL_COLOR,
    PCA_COLUMNS,
    RESULTS_CSV,
    REVIEW_COST_EUR,
    TARGET_COLUMN,
    TIME_COLUMN,
)
from fraud.features import FEATURE_LABELS, MODEL_FEATURES, SECONDS_PER_DAY, SECONDS_PER_HOUR
from fraud.selection import MODEL_LABELS, strategy_label
from fraud.serving import FraudModel, Prediction
from fraud.threshold import CostCurve, Decisions

SHAP_PREFIX = "shap_"
OUTCOMES: dict[str, str] = {
    "caught_fraud": "Caught fraud",
    "missed_fraud": "Missed fraud",
    "false_alarm": "False alarm",
    "legitimate": "Legitimate",
}
ALL_OUTCOMES = "All"
SCORE_FLOOR = 1e-5  # lowest score shown on log-scaled axes
SELECTED_LABEL = "Selected (cost-optimal)"
F1_LABEL = "F1-optimal"
CURRENT_LABEL = "Your threshold"
MARK_COLORS = {SELECTED_LABEL: "#6250d6", F1_LABEL: "#1baf7a", CURRENT_LABEL: "#0b0b0b"}


# --- Cached loading ------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading the model...")
def load_model() -> FraudModel:
    return FraudModel.load()


@st.cache_data
def load_sample() -> pd.DataFrame:
    sample = pd.read_csv(APP_SAMPLE_CSV)
    sample.insert(0, "id", np.arange(1, len(sample) + 1))
    return sample


@st.cache_data
def load_validation() -> pd.DataFrame:
    return pd.read_csv(APP_VALIDATION_SCORES)


@st.cache_data
def load_metrics() -> dict[str, Any]:
    return json.loads(FINAL_METRICS_JSON.read_text(encoding="utf-8"))


@st.cache_data
def load_results() -> pd.DataFrame:
    return pd.read_csv(RESULTS_CSV)


@st.cache_data
def threshold_grid(selected: float, f1_optimal: float) -> list[float]:
    """Thresholds offered by the slider: log-spaced, plus the two marked ones."""
    grid = np.r_[np.geomspace(1e-4, 0.5, 140), np.linspace(0.5, 0.999, 60)]
    return sorted({float(f"{value:.4g}") for value in grid} | {selected, f1_optimal})


@st.cache_data
def cost_by_threshold(thresholds: list[float], review_cost: float) -> pd.DataFrame:
    """Validation cost at every threshold of the grid."""
    validation = load_validation()
    y_true = validation[TARGET_COLUMN].to_numpy()
    scores = validation["score"].to_numpy()
    amount = validation[AMOUNT_COLUMN].to_numpy(dtype=float)
    costs = [
        Decisions(y_true, scores, amount, threshold, review_cost).metrics()["cost"]
        for threshold in thresholds
    ]
    return pd.DataFrame({"threshold": thresholds, "cost": costs})


# --- Small building blocks -----------------------------------------------------------------


def eur(value: float) -> str:
    return f"€{value:,.0f}"


def decision_badge(flagged: bool) -> str:
    color, text = (FRAUD_COLOR, "FLAGGED FOR REVIEW") if flagged else (LEGIT_COLOR, "PASSED")
    return (
        f"<span style='background:{color};color:white;padding:0.25rem 0.7rem;"
        f"border-radius:0.4rem;font-weight:600'>{text}</span>"
    )


def score_strip(score: float, threshold: float) -> alt.LayerChart:
    """Where the score lies relative to the threshold, on a log scale."""
    shown = max(score, SCORE_FLOOR)
    flagged = score >= threshold
    scale = alt.Scale(type="log", domain=[SCORE_FLOOR, 1.0])
    axis = alt.Axis(title="Fraud score (log scale)", format="~g", grid=False)
    zones = pd.DataFrame(
        {
            "start": [SCORE_FLOOR, threshold],
            "end": [threshold, 1.0],
            "zone": ["passed", "flagged for review"],
        }
    )
    background = (
        alt.Chart(zones)
        .mark_rect(opacity=0.18)
        .encode(
            x=alt.X("start:Q", scale=scale, axis=axis),
            x2="end:Q",
            color=alt.Color(
                "zone:N",
                scale=alt.Scale(
                    domain=["passed", "flagged for review"], range=[LEGIT_COLOR, FRAUD_COLOR]
                ),
                legend=None,
            ),
        )
    )
    zones["middle"] = np.sqrt(zones["start"] * zones["end"])
    names = (
        alt.Chart(zones)
        .mark_text(baseline="middle", fontSize=12, color=NEUTRAL_COLOR)
        .encode(x=alt.X("middle:Q", scale=scale), y=alt.value(11), text="zone:N")
    )
    rule = (
        alt.Chart(pd.DataFrame({"x": [threshold], "label": [f"threshold {threshold:.3f}"]}))
        .mark_rule(color=NEUTRAL_COLOR, strokeDash=[4, 3], size=2)
        .encode(x=alt.X("x:Q", scale=scale))
    )
    point = (
        alt.Chart(pd.DataFrame({"x": [shown], "score": [score]}))
        .mark_point(
            filled=True,
            size=260,
            opacity=1,
            color=FRAUD_COLOR if flagged else LEGIT_COLOR,
            stroke="white",
            strokeWidth=2,
        )
        .encode(x=alt.X("x:Q", scale=scale), tooltip=[alt.Tooltip("score:Q", format=".5f")])
    )
    return (background + names + rule + point).properties(height=140)


def contribution_chart(contributions: pd.DataFrame) -> alt.Chart:
    """Horizontal bars: orange pushes the score towards fraud, blue away from it."""
    return (
        alt.Chart(contributions)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X(
                "contribution:Q", title="Contribution to the fraud score (SHAP value, log-odds)"
            ),
            y=alt.Y(
                "feature:N",
                sort=None,
                title=None,
                axis=alt.Axis(labelLimit=260, labelOverlap=False),
            ),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(
                    domain=["increases the score", "decreases the score"],
                    range=[FRAUD_COLOR, LEGIT_COLOR],
                ),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("feature:N"),
                alt.Tooltip("value:Q", format=".3f", title="feature value"),
                alt.Tooltip("contribution:Q", format="+.3f"),
            ],
        )
        .properties(height=120 + 30 * len(contributions))
    )


def contributions_frame(
    features: list[str], values: list[float], shap: list[float]
) -> pd.DataFrame:
    frame = pd.DataFrame({"name": features, "value": values, "contribution": shap})
    frame["feature"] = [
        f"{FEATURE_LABELS[name]} = {value:.2f}"
        for name, value in zip(features, values, strict=True)
    ]
    frame["direction"] = np.where(
        frame["contribution"] > 0, "increases the score", "decreases the score"
    )
    return frame


def stored_contributions(row: pd.Series, model: FraudModel, top_n: int = 10) -> pd.DataFrame:
    """The strongest SHAP values saved with a sample row, with readable feature values."""
    shap = row[[f"{SHAP_PREFIX}{name}" for name in MODEL_FEATURES]].astype(float).to_numpy()
    readable = model.pipeline.steps[0][1].transform(model.to_frame(row.to_frame().T))
    order = np.argsort(-np.abs(shap))[:top_n]
    return contributions_frame(
        [MODEL_FEATURES[i] for i in order],
        [float(readable[MODEL_FEATURES[i]].iloc[0]) for i in order],
        [float(shap[i]) for i in order],
    )


def prediction_contributions(prediction: Prediction) -> pd.DataFrame:
    top = prediction.top_features
    return contributions_frame(
        [item.feature for item in top],
        [item.value for item in top],
        [item.contribution for item in top],
    )


# --- Page 1: explore a transaction -----------------------------------------------------------


def page_explore() -> None:
    model, sample = load_model(), load_sample()
    st.title("Explore a transaction")
    st.write(
        "Pick a transaction from a sample of the held-out test period. The model gives it a "
        "**fraud score**; the transaction is flagged for manual review when the score "
        f"reaches the threshold of **{model.threshold:.3f}**."
    )

    choice = st.segmented_control(
        "Show", [ALL_OUTCOMES, *OUTCOMES.values()], default=ALL_OUTCOMES, key="outcome_filter"
    )
    wanted = [key for key, label in OUTCOMES.items() if label == choice]
    rows = sample[sample["outcome"].isin(wanted)] if wanted else sample
    counts = sample["outcome"].map(OUTCOMES).value_counts()
    st.caption(
        "In this sample: "
        + ", ".join(f"{counts.get(label, 0)} {label.lower()}" for label in OUTCOMES.values())
        + ". All fraud cases and false alarms of the test period are included, plus 100 "
        "randomly chosen legitimate transactions, so fraud is heavily over-represented here."
    )
    labels = {
        int(row.id): f"Transaction {row.id} · €{row.Amount:,.2f} · score {row.score:.3f}"
        for row in rows.sort_values("score", ascending=False).itertuples()
    }
    chosen = st.selectbox("Transaction", list(labels), format_func=labels.get, key="row")
    row = sample[sample["id"] == chosen].iloc[0]

    score, flagged, is_fraud = float(row["score"]), bool(row["flagged"]), bool(row[TARGET_COLUMN])
    left, middle, right = st.columns(3)
    left.metric(
        "Fraud score", f"{score:.4f}", help="A ranking score between 0 and 1. Not a probability."
    )
    middle.metric("Threshold", f"{model.threshold:.4f}")
    right.metric("Amount", f"€{row[AMOUNT_COLUMN]:,.2f}")
    decision, actual = st.columns(2)
    decision.markdown(f"**Decision**&nbsp;&nbsp;{decision_badge(flagged)}", unsafe_allow_html=True)
    actual_color = FRAUD_COLOR if is_fraud else LEGIT_COLOR
    actual.markdown(
        f"**Actual label** (for demo only)&nbsp;&nbsp;<span style='color:{actual_color};"
        f"font-weight:600'>{'Fraud' if is_fraud else 'Legitimate'}</span> · "
        f"{OUTCOMES[row['outcome']].lower()}",
        unsafe_allow_html=True,
    )
    st.caption(
        "The actual label is shown for the demo only. In real use it is not known when the "
        "transaction is scored."
    )
    st.altair_chart(score_strip(score, model.threshold), width="stretch")

    st.subheader("Why this score?")
    st.write(
        "Each bar shows how much one feature pushed the score up (orange) or down (blue) for "
        "this transaction. V1 to V28 are anonymised components of the original card data, so "
        "they have no business meaning."
    )
    st.altair_chart(contribution_chart(stored_contributions(row, model)), width="stretch")
    st.caption(
        f"SHAP values in log-odds. The model starts from a base value of {row['base_value']:.2f} "
        "and adds the contributions of all 31 features; the ten strongest are shown."
    )

    with st.expander("Edit this transaction and re-score"):
        edit_and_rescore(row, model)


def edit_and_rescore(row: pd.Series, model: FraudModel) -> None:
    """Let the user change the amount, the hour and the strongest components."""
    shap = row[[f"{SHAP_PREFIX}{name}" for name in PCA_COLUMNS]].astype(float).abs()
    strongest = [
        name.removeprefix(SHAP_PREFIX) for name in shap.sort_values(ascending=False).index[:5]
    ]
    st.write(
        "Change the amount, the time of day or the five components that mattered most for "
        "this transaction, then score it again with the live model."
    )
    with st.form(f"edit_{int(row['id'])}"):
        first, second = st.columns(2)
        amount = first.number_input(
            "Amount (€)",
            min_value=0.0,
            max_value=1_000_000.0,
            value=float(row[AMOUNT_COLUMN]),
            step=10.0,
        )
        hour = second.slider(
            "Hour of day", 0.0, 23.99, float(row[TIME_COLUMN]) / SECONDS_PER_HOUR, step=0.25
        )
        columns = st.columns(len(strongest))
        edited = {
            name: column.number_input(name, value=float(row[name]), step=0.5, format="%.3f")
            for name, column in zip(strongest, columns, strict=True)
        }
        submitted = st.form_submit_button("Re-score", type="primary")
    if not submitted:
        return
    transaction = {name: float(row[name]) for name in PCA_COLUMNS}
    transaction.update(edited)
    transaction[AMOUNT_COLUMN] = amount
    transaction[TIME_COLUMN] = (hour * SECONDS_PER_HOUR) % SECONDS_PER_DAY
    [prediction] = model.predict([transaction])
    before = float(row["score"])
    left, right = st.columns(2)
    left.metric(
        "New fraud score",
        f"{prediction.fraud_score:.4f}",
        delta=f"{prediction.fraud_score - before:+.4f}",
        delta_color="inverse",
    )
    right.markdown(
        f"**New decision**&nbsp;&nbsp;{decision_badge(prediction.flagged)}", unsafe_allow_html=True
    )
    st.altair_chart(score_strip(prediction.fraud_score, model.threshold), width="stretch")
    st.altair_chart(contribution_chart(prediction_contributions(prediction)), width="stretch")
    st.caption("The five strongest contributions for the edited transaction.")


# --- Page 2: choosing the threshold ----------------------------------------------------------


def page_threshold() -> None:
    model, validation = load_model(), load_validation()
    selected = model.threshold
    f1_optimal = float(model.config["decision"]["f1_optimal_threshold_for_comparison"])
    y_true = validation[TARGET_COLUMN].to_numpy()
    scores = validation["score"].to_numpy()
    amount = validation[AMOUNT_COLUMN].to_numpy(dtype=float)

    st.title("Choosing the threshold")
    st.write(
        f"The model scores every transaction; the threshold decides which ones a person "
        f"reviews. Here you can move it over the {len(validation):,} validation transactions "
        f"({int(y_true.sum())} of them fraud) and see what happens. The cost model: a **missed "
        "fraud costs its amount**, and **every flagged transaction costs one review**, "
        "whether it turns out to be fraud or not."
    )
    left, right = st.columns([3, 1])
    review_cost = right.number_input(
        "Cost of one review (€)", min_value=0.5, max_value=100.0, value=REVIEW_COST_EUR, step=0.5
    )
    grid = threshold_grid(selected, f1_optimal)
    threshold = left.select_slider(
        "Decision threshold on the fraud score",
        options=grid,
        value=selected,
        format_func=lambda value: f"{value:.4g}",
        key="threshold",
    )

    now = Decisions(y_true, scores, amount, threshold, review_cost).metrics()
    chosen = Decisions(y_true, scores, amount, selected, review_cost).metrics()
    extra_cost = now["cost"] - chosen["cost"]
    first, second, third, fourth = st.columns(4)
    first.metric(
        "Total cost",
        eur(now["cost"]),
        delta=f"{extra_cost:+,.0f} € vs selected" if round(extra_cost) else None,
        delta_color="inverse",
        help=f"Flagging nothing would cost {eur(now['cost_no_model'])}.",
    )
    second.metric(
        "Recall",
        f"{now['recall']:.0%}",
        help=f"{int(now['tp'])} of {int(y_true.sum())} fraud cases caught.",
    )
    third.metric(
        "Precision", f"{now['precision']:.0%}", help="Share of flagged transactions that are fraud."
    )
    fourth.metric(
        "False alarms",
        f"{int(now['fp']):,}",
        help=f"{int(now['tp'] + now['fp']):,} flagged in total.",
    )

    curve = cost_by_threshold(grid, review_cost)
    marks = pd.DataFrame(
        {
            "threshold": [selected, f1_optimal, threshold],
            "mark": [SELECTED_LABEL, F1_LABEL, CURRENT_LABEL],
        }
    )
    marks["cost"] = [
        Decisions(y_true, scores, amount, value, review_cost).metrics()["cost"]
        for value in marks["threshold"]
    ]
    scale = alt.Scale(type="log", domain=[1e-4, 1.0])
    ceiling = float(now["cost_no_model"]) * 1.3
    line = (
        alt.Chart(curve)
        .mark_line(color=NEUTRAL_COLOR, interpolate="step-after", clip=True)
        .encode(
            x=alt.X(
                "threshold:Q",
                scale=scale,
                title="Decision threshold (log scale)",
                axis=alt.Axis(format="~g"),
            ),
            y=alt.Y(
                "cost:Q", title="Total cost on validation (€)", scale=alt.Scale(domain=[0, ceiling])
            ),
        )
    )
    color = alt.Color(
        "mark:N",
        scale=alt.Scale(domain=list(MARK_COLORS), range=list(MARK_COLORS.values())),
        legend=alt.Legend(title=None, orient="bottom"),
    )
    rules = (
        alt.Chart(marks)
        .mark_rule(size=2, strokeDash=[5, 3])
        .encode(x=alt.X("threshold:Q", scale=scale), color=color)
    )
    points = (
        alt.Chart(marks)
        .mark_point(filled=True, size=130, opacity=1, clip=True)
        .encode(
            x=alt.X("threshold:Q", scale=scale),
            y="cost:Q",
            color=color,
            tooltip=[
                "mark:N",
                alt.Tooltip("threshold:Q", format=".4g"),
                alt.Tooltip("cost:Q", format=",.0f"),
            ],
        )
    )
    no_model = (
        alt.Chart(pd.DataFrame({"cost": [now["cost_no_model"]]}))
        .mark_rule(color=NEUTRAL_COLOR, strokeDash=[2, 2])
        .encode(y="cost:Q")
    )
    st.altair_chart((line + no_model + rules + points).properties(height=360), width="stretch")
    best_threshold, best_cost = CostCurve(y_true, scores, amount, review_cost).best()
    st.caption(
        f"Dotted horizontal line: the cost of flagging nothing ({eur(now['cost_no_model'])}). "
        f"With a review cost of €{review_cost:g} the cheapest threshold on these rows is "
        f"{best_threshold:.4g} (cost {eur(best_cost)}). The selected threshold "
        f"({selected:.4g}) was chosen this way for €{REVIEW_COST_EUR:g}; the F1-optimal one "
        f"({f1_optimal:.4g}) ignores amounts and review costs."
    )
    stability = load_metrics()["threshold_stability"]
    st.warning(
        "**The threshold is not well determined.** Validation has only "
        f"{int(y_true.sum())} fraud cases and the cost curve is flat around its minimum. Over "
        f"bootstrap resamples the cheapest threshold ranges from {stability['q025']:.3f} to "
        f"{stability['q975']:.2f}. It should be re-validated on fresh data before anyone "
        "relies on it."
    )


# --- Page 3: how the model was chosen --------------------------------------------------------


def stage1_table(results: pd.DataFrame) -> pd.DataFrame:
    """Best strategy of every classifier in the screening stage, best first."""
    stage1 = results[(results["stage"] == "stage1") & results["cv_pr_auc_mean"].notna()]
    best = stage1.loc[stage1.groupby("model")["cv_pr_auc_mean"].idxmax()]
    best = best.sort_values("cv_pr_auc_mean", ascending=False)
    return pd.DataFrame(
        {
            "Classifier": best["model"].map(MODEL_LABELS).to_numpy(),
            "Best strategy": [
                strategy_label(strategy, ratio)
                for strategy, ratio in zip(best["strategy"], best["sampling_ratio"], strict=True)
            ],
            "CV PR-AUC": best["cv_pr_auc_mean"].to_numpy(),
            "Validation PR-AUC": best["pr_auc"].to_numpy(),
        }
    )


def test_table(metrics: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for entry in metrics["test"]["models"].values():
        values = entry["metrics"]
        rows.append(
            {
                "Model": entry["label"],
                "PR-AUC": values["pr_auc"]["value"],
                "Recall": values["recall"]["value"],
                "Precision": values["precision"]["value"],
                "False alarms": int(values["fp"]["value"]),
                "Cost (€)": values["cost"]["value"],
                "Savings vs no model (€)": values["savings_vs_no_model"]["value"],
            }
        )
    return pd.DataFrame(rows)


def figure(name: str, caption: str) -> None:
    path = FIGURES_DIR / name
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")


def page_selection() -> None:
    results, metrics = load_results(), load_metrics()
    selected = metrics["test"]["models"]["selected"]["metrics"]
    st.title("How the model was chosen")
    st.write(
        "The data covers two days of card transactions with about one fraud in 540. Models "
        "were compared on **PR-AUC** with a **time-based split**: trained on the first 70 % of "
        "the period, tuned on the next 15 %, and tested once on the last 15 %. The rule for "
        "picking the final model was written down before any result was seen."
    )
    st.subheader("A staged search")
    st.markdown(
        "1. **Screening.** 9 classifiers, each with no resampling, class weights and SMOTE.\n"
        "2. **SMOTE family.** The best 4 classifiers with SMOTE, SMOTE-Tomek, SMOTE-ENN, "
        "Safe-Level-SMOTE and Polynom-Fit SMOTE at several ratios.\n"
        "3. **Tuning.** A small hyperparameter search for the best 2.\n\n"
        f"In total {results['config_id'].nunique()} configurations, each scored with 3 "
        "time-ordered cross-validation folds and on validation."
    )
    st.dataframe(
        stage1_table(results),
        hide_index=True,
        width="stretch",
        column_config={
            "CV PR-AUC": st.column_config.NumberColumn(format="%.3f"),
            "Validation PR-AUC": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    st.caption(
        "Screening stage: the best strategy of each classifier, ranked by cross-validated PR-AUC."
    )
    figure(
        "exp_stage2_cv_heatmap.png", "Mean cross-validated PR-AUC of every classifier and strategy."
    )

    st.subheader("Result on the test period")
    st.write(
        "The rule selected **XGBoost with Safe-Level-SMOTE**. On the test period it catches "
        f"{int(selected['tp']['value'])} of {int(selected['tp']['value'] + selected['fn']['value'])} "
        f"fraud cases with {int(selected['fp']['value'])} false alarms and removes "
        f"{selected['savings_share_of_no_model']['value']:.0%} of the fraud cost "
        f"(95 % interval {selected['savings_share_of_no_model']['ci_low']:.0%} to "
        f"{selected['savings_share_of_no_model']['ci_high']:.0%}). **XGBoost with class weights "
        "performed equally well**: the top configurations are statistically tied, and the "
        "selected model was kept because the rule was fixed in advance, not because it won."
    )
    st.dataframe(
        test_table(metrics),
        hide_index=True,
        width="stretch",
        column_config={
            "PR-AUC": st.column_config.NumberColumn(format="%.3f"),
            "Recall": st.column_config.NumberColumn(format="%.2f"),
            "Precision": st.column_config.NumberColumn(format="%.2f"),
            "Cost (€)": st.column_config.NumberColumn(format="%.0f"),
            "Savings vs no model (€)": st.column_config.NumberColumn(format="%.0f"),
        },
    )
    st.caption(
        "Each model at the threshold chosen for it on validation. The three reference models "
        "were evaluated after the selection and were not used for it."
    )
    figure("eval_test_pr_curves.png", "Precision-recall curves on the test period.")
    figure("eval_shap_summary.png", "Which features drive the fraud score (SHAP summary).")

    st.subheader("Comparison with the MSc thesis")
    st.write(
        "This project extends an MSc thesis that compared SMOTE-based oversampling methods on "
        "18 datasets with imbalance up to 28:1. Here the imbalance is about 540:1. "
        "As in the thesis, SMOTE-Tomek gave exactly the same results as SMOTE, and tree "
        "ensembles were the least sensitive to the choice of method while k-NN and logistic "
        "regression were the most sensitive. Unlike in the thesis, Safe-Level-SMOTE was not "
        "the best method in general: it helped XGBoost but not the other models. The thesis hypothesis "
        "that oversampling mainly shifts the decision threshold, and does not improve the "
        "ranking, is supported against class weights: both rank equally well, and the large "
        "cost difference at a threshold of 0.5 disappears once each model gets its own "
        "threshold. It is not supported against using no imbalance handling at all, where "
        "XGBoost ranks clearly worse."
    )


# --- Page 4: about and limitations -----------------------------------------------------------


def page_about() -> None:
    model, metrics = load_model(), load_metrics()
    splits, stability = metrics["splits"], metrics["threshold_stability"]
    st.title("About and limitations")
    st.subheader("Data")
    st.write(
        "The public Kaggle dataset *Credit Card Fraud Detection* (ULB Machine Learning Group): "
        "284,807 card transactions made by European cardholders over two days in September "
        "2013, 492 of them fraud (0.17 %). The features V1 to V28 are principal components of "
        "the original, confidential data; only the amount and the time are given as they were."
    )
    st.subheader("How it was evaluated")
    st.write(
        f"Exact duplicate rows were removed, then the data was split by time: "
        f"{splits['train']['rows']:,} transactions for training ({splits['train']['fraud']} "
        f"fraud), {splits['val']['rows']:,} for validation ({splits['val']['fraud']} fraud) and "
        f"{splits['evaluation']['rows']:,} for the final test ({splits['evaluation']['fraud']} "
        "fraud). The model never sees transactions from the future of what it is evaluated "
        "on. Oversampling and scaling are fitted inside the pipeline on training data only. "
        "The test period was used once."
    )
    st.subheader("Limitations")
    st.markdown(
        f"- **Unstable threshold.** The threshold ({model.threshold:.3f}) minimises cost on "
        f"{splits['val']['fraud']} validation fraud cases. Over bootstrap resamples the "
        f"cheapest threshold ranges from {stability['q025']:.3f} to {stability['q975']:.2f}, "
        "because the cost curve is flat around its minimum. Re-validate it on fresh data "
        "before relying on it.\n"
        f"- **Few fraud cases.** With about 50 fraud cases per evaluation period every number "
        "is uncertain: the savings on the test period are between "
        f"{metrics['test']['models']['selected']['metrics']['savings_share_of_no_model']['ci_low']:.0%} "
        f"and {metrics['test']['models']['selected']['metrics']['savings_share_of_no_model']['ci_high']:.0%} "
        "of the fraud cost.\n"
        "- **Two days of data.** Nothing here says how the model behaves on other weekdays or "
        "months later. Validation and test cover only the afternoon and evening of the second "
        "day, with no night hours, where fraud is most frequent.\n"
        "- **Anonymised features.** V1 to V28 have no business meaning, so the explanations "
        "show which components drive a score but not why in terms an analyst could act on.\n"
        "- **A fraud score, not a probability.** The model is trained on oversampled data, "
        "which distorts its outputs as probabilities, and there are too few cases to "
        "calibrate them. The score is only meaningful relative to the threshold and to other "
        "scores.\n"
        "- **A simple cost model.** One review costs a fixed amount and is assumed to always "
        "recognise fraud; blocking a legitimate customer costs nothing beyond the review."
    )
    st.subheader("Model")
    info = model.info()
    st.write(
        f"Version {info['model_version']}: XGBoost trained on data oversampled with "
        f"Safe-Level-SMOTE (ratio {info['sampling_ratio']}), threshold {info['threshold']:.4f}. "
        "The same model and the same code serve the API of this project."
    )


# --- Navigation ------------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Credit card fraud detection", page_icon="💳", layout="wide")
    pages = [
        st.Page(
            page_explore, title="Explore a transaction", icon="🔎", url_path="explore", default=True
        ),
        st.Page(page_threshold, title="Choosing the threshold", icon="🎚️", url_path="threshold"),
        st.Page(page_selection, title="How the model was chosen", icon="📊", url_path="selection"),
        st.Page(page_about, title="About and limitations", icon="ℹ️", url_path="about"),
    ]
    navigation = st.navigation(pages)
    model = load_model()
    with st.sidebar:
        st.markdown("### Credit card fraud detection")
        st.caption(
            "Fraud detection on heavily imbalanced data, evaluated with a time-based split "
            "and a cost-based threshold."
        )
        st.caption(f"Model version {model.version} · threshold {model.threshold:.3f}")
    navigation.run()


if __name__ == "__main__":
    main()
