from fraud import config


def test_paths_are_inside_project_root() -> None:
    assert (config.PROJECT_ROOT / "pyproject.toml").exists()
    for path in (config.RAW_DIR, config.PROCESSED_DIR, config.MODELS_DIR, config.FIGURES_DIR):
        assert config.PROJECT_ROOT in path.parents


def test_expected_columns() -> None:
    assert len(config.EXPECTED_COLUMNS) == 31
    assert config.EXPECTED_COLUMNS[0] == "Time"
    assert config.EXPECTED_COLUMNS[-2:] == ["Amount", "Class"]


def test_cost_assumptions() -> None:
    assert config.REVIEW_COST_EUR > 0
    assert isinstance(config.RANDOM_SEED, int)
