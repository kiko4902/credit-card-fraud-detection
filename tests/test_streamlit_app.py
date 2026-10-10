"""The Streamlit pages run without errors on the committed model and app data."""

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from fraud.config import APP_SAMPLE_CSV, APP_VALIDATION_SCORES, PROJECT_ROOT
from fraud.serving import get_model
from fraud.threshold import cost_optimal_threshold

TIMEOUT = 120
APP_FILE = PROJECT_ROOT / "app" / "streamlit_app.py"


def explore() -> None:
    from app.streamlit_app import page_explore

    page_explore()


def threshold() -> None:
    from app.streamlit_app import page_threshold

    page_threshold()


def selection() -> None:
    from app.streamlit_app import page_selection

    page_selection()


def about() -> None:
    from app.streamlit_app import page_about

    page_about()


def run(page) -> AppTest:  # type: ignore[no-untyped-def]
    app = AppTest.from_function(page, default_timeout=TIMEOUT).run()
    assert not app.exception, app.exception
    return app


def test_whole_app_starts_on_the_first_page() -> None:
    app = AppTest.from_file(str(APP_FILE), default_timeout=TIMEOUT).run()
    assert not app.exception, app.exception
    assert app.title[0].value == "Explore a transaction"


def test_explore_page_shows_score_threshold_and_demo_label() -> None:
    app = run(explore)
    sample = pd.read_csv(APP_SAMPLE_CSV)
    assert app.metric[0].label == "Fraud score"
    assert app.metric[0].value == f"{sample['score'].max():.4f}"  # highest score comes first
    assert app.metric[1].value == f"{get_model().threshold:.4f}"
    assert any("for demo only" in block.value for block in app.markdown)


@pytest.mark.parametrize(
    ("label", "outcome"), [("Missed fraud", "missed_fraud"), ("False alarm", "false_alarm")]
)
def test_explore_page_filters_by_outcome(label: str, outcome: str) -> None:
    app = run(explore)
    app.button_group[0].set_value(label).run()
    assert not app.exception, app.exception
    sample = pd.read_csv(APP_SAMPLE_CSV)
    assert len(app.selectbox[0].options) == int((sample["outcome"] == outcome).sum())
    assert any(label.lower() in block.value for block in app.markdown)


def test_edited_transaction_is_rescored_by_the_live_model() -> None:
    app = run(explore)
    app.number_input[0].set_value(1.0)  # amount
    app.button[0].click().run()
    assert not app.exception, app.exception
    assert "New fraud score" in [metric.label for metric in app.metric]


def test_threshold_page_starts_at_the_selected_threshold() -> None:
    app = run(threshold)
    validation = pd.read_csv(APP_VALIDATION_SCORES)
    _, cost = cost_optimal_threshold(validation["Class"], validation["score"], validation["Amount"])
    assert app.select_slider[0].value == get_model().threshold
    assert app.metric[0].label == "Total cost" and app.metric[0].value == f"€{cost:,.0f}"
    assert len(app.warning) == 1 and "re-validated" in app.warning[0].value


def test_threshold_page_reacts_to_the_slider() -> None:
    app = run(threshold)
    before = [metric.value for metric in app.metric]
    app.select_slider[0].set_value(0.999).run()
    assert not app.exception, app.exception
    assert [metric.value for metric in app.metric] != before


def test_selection_and_about_pages_render() -> None:
    chosen = run(selection)
    assert len(chosen.dataframe) == 2
    assert any("performed equally well" in block.value for block in chosen.markdown)
    limits = run(about)
    text = " ".join(block.value for block in limits.markdown)
    phrases = ("Unstable threshold", "Few fraud cases", "Anonymised features", "not a probability")
    for phrase in phrases:
        assert phrase in text
