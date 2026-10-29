from __future__ import annotations

import pandas as pd
import pytest

from football_predictor.core import FEATURE_COLUMNS, build_features, fixture_features, train_models


def sample_matches(repeats: int = 12) -> pd.DataFrame:
    teams = ["A", "B", "C", "D"]
    rows = []
    day = pd.Timestamp("2020-01-01")
    for index in range(repeats * 6):
        home = teams[index % 4]
        away = teams[(index + 1 + (index // 4)) % 4]
        if home == away:
            away = teams[(teams.index(away) + 1) % 4]
        home_goals, away_goals = index % 3, (index + 1) % 3
        result = "H" if home_goals > away_goals else "D" if home_goals == away_goals else "A"
        rows.append({"Date": day, "HomeTeam": home, "AwayTeam": away, "FTHG": home_goals, "FTAG": away_goals, "FTR": result})
        day += pd.offsets.Day(3)
    return pd.DataFrame(rows)


def test_current_result_cannot_leak_into_current_features() -> None:
    matches = sample_matches()
    changed = matches.copy()
    changed.loc[10, ["FTHG", "FTAG", "FTR"]] = [9, 0, "H"]
    original_features, _, _ = build_features(matches)
    changed_features, _, _ = build_features(changed)
    pd.testing.assert_series_equal(original_features.iloc[10], changed_features.iloc[10])
    assert not original_features.iloc[11:].equals(changed_features.iloc[11:])


def test_fixture_features_validate_teams() -> None:
    _, _, metadata = build_features(sample_matches())
    frame = fixture_features("A", "B", metadata)
    assert list(frame.columns) == FEATURE_COLUMNS
    with pytest.raises(ValueError, match="different"):
        fixture_features("A", "A", metadata)
    with pytest.raises(ValueError, match="training data"):
        fixture_features("A", "Unknown", metadata)


def test_training_is_chronological_and_probabilities_are_valid(tmp_path) -> None:
    matches = sample_matches(20)
    result = train_models(matches, tmp_path / "model.joblib")
    assert result["split"]["train_rows"] < len(matches)
    assert result["split"]["test_start"] > str(matches["Date"].min().date())
    features = fixture_features("A", "B", result["metadata"])
    probability = result["model"].predict_proba(features)[0]
    assert probability.sum() == pytest.approx(1.0)
    assert (tmp_path / "model.joblib").exists()
