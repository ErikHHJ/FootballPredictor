"""Reproduce the prepared data, model comparison, and deployable artifact."""

from __future__ import annotations

import json
from pathlib import Path

from football_predictor import load_matches, train_models


def main() -> None:
    matches = load_matches()
    result = train_models(matches, "artifacts/premier_league_model.joblib")
    metrics = {
        "selected_model": result["selected_model"],
        "split": result["split"],
        "validation_metrics": result["validation_metrics"],
        "test_metrics": result["test_metrics"],
        "baseline_test_metrics": result["baseline_test_metrics"],
    }
    Path("artifacts/metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
