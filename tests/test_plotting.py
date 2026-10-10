from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from fraud import plotting


def test_class_colours_are_distinct_and_cover_both_classes() -> None:
    assert set(plotting.CLASS_COLORS) == set(plotting.CLASS_LABELS) == {0, 1}
    assert plotting.CLASS_COLORS[0] != plotting.CLASS_COLORS[1]


def test_save_figure_writes_png(tmp_path: Path) -> None:
    plotting.set_style()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    path = plotting.save_figure(fig, "example", directory=tmp_path / "figures")
    plt.close(fig)
    assert path == tmp_path / "figures" / "example.png"
    assert path.stat().st_size > 0
