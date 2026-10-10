"""Shared plot style so every figure in the project looks the same."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from fraud.config import FIGURES_DIR

# Blue / orange stay distinguishable under the common colour-vision deficiencies.
# Legitimate is always blue and fraud always orange, in every figure.
LEGIT_COLOR: str = "#2a78d6"
FRAUD_COLOR: str = "#eb6834"
NEUTRAL_COLOR: str = "#52514e"
CLASS_COLORS: dict[int, str] = {0: LEGIT_COLOR, 1: FRAUD_COLOR}
CLASS_LABELS: dict[int, str] = {0: "Legitimate", 1: "Fraud"}
# Models get their own colours, distinct from the two class colours.
MODEL_COLORS: dict[str, str] = {
    "logreg": "#6250d6",
    "random_forest": "#1baf7a",
    "lightgbm": "#c98500",
}
TEXT_COLOR: str = "#0b0b0b"

SURFACE_COLOR: str = "#fcfcfb"
GRID_COLOR: str = "#e4e3df"
SAVE_DPI: int = 150


def set_style() -> None:
    """Apply the project-wide matplotlib style."""
    plt.rcParams.update(
        {
            "figure.dpi": 80,
            "figure.facecolor": SURFACE_COLOR,
            "axes.facecolor": SURFACE_COLOR,
            "savefig.facecolor": SURFACE_COLOR,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.labelcolor": NEUTRAL_COLOR,
            "axes.edgecolor": GRID_COLOR,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.8,
            "xtick.color": NEUTRAL_COLOR,
            "ytick.color": NEUTRAL_COLOR,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
        }
    )


def save_figure(fig: Figure, name: str, directory: Path = FIGURES_DIR) -> Path:
    """Save ``fig`` as ``<directory>/<name>.png`` and return the path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path, dpi=SAVE_DPI)
    return path
