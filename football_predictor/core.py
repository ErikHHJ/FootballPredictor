"""Data acquisition, leakage-safe feature engineering, and model evaluation."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from urllib.request import urlretrieve

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEASONS = ("1920", "2021", "2122", "2223", "2324", "2425")
SOURCE_TEMPLATE = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
CLASSES = np.array(["A", "D", "H"])
FEATURE_COLUMNS = [
    "home_points_5",
    "away_points_5",
    "home_goals_for_5",
    "away_goals_for_5",
    "home_goals_against_5",
    "away_goals_against_5",
    "home_win_rate_10",
    "away_win_rate_10",
    "home_rest_days",
    "away_rest_days",
    "home_advantage",
]


@dataclass
class TeamState:
    points: deque = field(default_factory=lambda: deque(maxlen=5))
    goals_for: deque = field(default_factory=lambda: deque(maxlen=5))
    goals_against: deque = field(default_factory=lambda: deque(maxlen=5))
    wins: deque = field(default_factory=lambda: deque(maxlen=10))
    last_date: pd.Timestamp | None = None


def download_data(data_dir: str | Path = "data/raw", seasons: Iterable[str] = SEASONS) -> list[Path]:
    """Download season CSVs without silently replacing existing inputs."""
    destination = Path(data_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for season in seasons:
        target = destination / f"E0_{season}.csv"
        if not target.exists():
            urlretrieve(SOURCE_TEMPLATE.format(season=season), target)
        paths.append(target)
    return paths


def load_matches(data_dir: str | Path = "data/raw", seasons: Iterable[str] = SEASONS) -> pd.DataFrame:
    frames = []
    for season, path in zip(seasons, download_data(data_dir, seasons), strict=True):
        frame = pd.read_csv(path, usecols=lambda c: c in {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"})
        frame["season"] = season
        frames.append(frame)
    matches = pd.concat(frames, ignore_index=True)
    matches["Date"] = pd.to_datetime(matches["Date"], dayfirst=True, errors="coerce")
    matches = matches.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
    matches = matches[matches["FTR"].isin(CLASSES)].sort_values("Date").reset_index(drop=True)
    return matches


def _mean(values: deque, default: float) -> float:
    return float(np.mean(values)) if values else default


def _state_features(state: TeamState, match_date: pd.Timestamp, prefix: str) -> dict[str, float]:
    rest = 7.0 if state.last_date is None else float(np.clip((match_date - state.last_date).days, 2, 30))
    return {
        f"{prefix}_points_5": _mean(state.points, 1.3),
        f"{prefix}_goals_for_5": _mean(state.goals_for, 1.35),
        f"{prefix}_goals_against_5": _mean(state.goals_against, 1.35),
        f"{prefix}_win_rate_10": _mean(state.wins, 0.34),
        f"{prefix}_rest_days": rest,
    }


def build_features(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Build each row before updating state with that match's result."""
    states: dict[str, TeamState] = defaultdict(TeamState)
    rows, labels, dates = [], [], []
    for match in matches.sort_values("Date").itertuples(index=False):
        home, away, match_date = match.HomeTeam, match.AwayTeam, pd.Timestamp(match.Date)
        row = {
            **_state_features(states[home], match_date, "home"),
            **_state_features(states[away], match_date, "away"),
            "home_advantage": 1.0,
        }
        rows.append(row)
        labels.append(match.FTR)
        dates.append(match_date)
        home_goals, away_goals = int(match.FTHG), int(match.FTAG)
        home_points = 3 if home_goals > away_goals else 1 if home_goals == away_goals else 0
        away_points = 3 if away_goals > home_goals else 1 if home_goals == away_goals else 0
        for team, goals_for, goals_against, points in (
            (home, home_goals, away_goals, home_points),
            (away, away_goals, home_goals, away_points),
        ):
            state = states[team]
            state.points.append(points)
            state.goals_for.append(goals_for)
            state.goals_against.append(goals_against)
            state.wins.append(float(points == 3))
            state.last_date = match_date
    metadata = {
        "dates": pd.Series(dates, name="Date"),
        "states": states,
        "teams": sorted(states),
        "last_match_date": max(dates),
    }
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS), pd.Series(labels, name="result"), metadata


def fixture_features(home: str, away: str, metadata: dict, match_date: date | pd.Timestamp | None = None) -> pd.DataFrame:
    if home == away:
        raise ValueError("Home and away teams must be different.")
    states = metadata["states"]
    if home not in states or away not in states:
        raise ValueError("Select teams present in the training data.")
    when = pd.Timestamp(
        match_date or (pd.Timestamp(metadata["last_match_date"]) + pd.offsets.Day(7))
    )
    row = {
        **_state_features(states[home], when, "home"),
        **_state_features(states[away], when, "away"),
        "home_advantage": 1.0,
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def _pipeline(estimator) -> Pipeline:
    transform = ColumnTransformer(
        [("numeric", Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())]), FEATURE_COLUMNS)]
    )
    return Pipeline([("features", transform), ("model", estimator)])


def _score(model, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    probability = model.predict_proba(x)
    return {
        "accuracy": float(accuracy_score(y, model.predict(x))),
        "log_loss": float(log_loss(y, probability, labels=model.classes_)),
        "brier": float(np.mean(np.sum((probability - (y.to_numpy()[:, None] == model.classes_)) ** 2, axis=1))),
    }


def train_models(matches: pd.DataFrame, artifact_path: str | Path | None = None) -> dict:
    x, y, metadata = build_features(matches)
    n = len(x)
    train_end, validation_end = int(n * 0.70), int(n * 0.85)
    slices = {"train": slice(0, train_end), "validation": slice(train_end, validation_end), "test": slice(validation_end, n)}
    candidates = {
        "Most frequent baseline": DummyClassifier(strategy="prior"),
        "Multinomial logistic regression": _pipeline(LogisticRegression(max_iter=2000, C=0.5)),
        "Histogram gradient boosting": _pipeline(
            HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=158)
        ),
    }
    validation_metrics = {}
    for name, model in candidates.items():
        model.fit(x.iloc[slices["train"]], y.iloc[slices["train"]])
        validation_metrics[name] = _score(model, x.iloc[slices["validation"]], y.iloc[slices["validation"]])
    model_names = [name for name in candidates if name != "Most frequent baseline"]
    selected_name = min(model_names, key=lambda name: validation_metrics[name]["log_loss"])
    selected = candidates[selected_name]
    selected.fit(x.iloc[:validation_end], y.iloc[:validation_end])
    test_metrics = _score(selected, x.iloc[slices["test"]], y.iloc[slices["test"]])
    baseline = DummyClassifier(strategy="prior").fit(x.iloc[:validation_end], y.iloc[:validation_end])
    result = {
        "model": selected,
        "selected_model": selected_name,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "baseline_test_metrics": _score(baseline, x.iloc[slices["test"]], y.iloc[slices["test"]]),
        "metadata": metadata,
        "split": {
            "train_rows": train_end,
            "validation_rows": validation_end - train_end,
            "test_rows": n - validation_end,
            "test_start": str(metadata["dates"].iloc[validation_end].date()),
            "test_end": str(metadata["dates"].iloc[-1].date()),
        },
        "source": SOURCE_TEMPLATE,
    }
    if artifact_path:
        target = Path(artifact_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(result, target)
    return result
