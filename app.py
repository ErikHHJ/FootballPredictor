"""Streamlit interface for pre-match Premier League predictions."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from football_predictor import fixture_features

ARTIFACT = Path("artifacts/premier_league_model.joblib")
LABELS = {"H": "Home win", "D": "Draw", "A": "Away win"}

st.set_page_config(page_title="Premier League Predictor", page_icon="⚽", layout="wide")


@st.cache_resource
def load_artifact() -> dict:
    if not ARTIFACT.exists():
        raise FileNotFoundError("Model artifact is missing. Run `python -m scripts.train` first.")
    return joblib.load(ARTIFACT)


st.title("Premier League Predictor")
st.caption("A leakage-safe estimate based only on each club's form before kick-off.")

try:
    artifact = load_artifact()
except Exception as exc:  # noqa: BLE001 - UI boundary must show corrupt-artifact failures
    st.error(f"Prediction service is unavailable: {exc}")
    st.stop()

teams = artifact["metadata"].get("teams", [])
if len(teams) < 2:
    st.warning("There are not enough teams in the trained model to create a fixture.")
    st.stop()

left, right = st.columns(2)
default_home = teams.index("Arsenal") if "Arsenal" in teams else 0
default_away = teams.index("Liverpool") if "Liverpool" in teams else 1
home = left.selectbox("Home team", teams, index=default_home)
away = right.selectbox("Away team", teams, index=default_away)

if home == away:
    st.info("Choose two different teams to calculate probabilities.")
    st.stop()

try:
    features = fixture_features(home, away, artifact["metadata"])
    probabilities = artifact["model"].predict_proba(features)[0]
    classes = artifact["model"].classes_
except Exception as exc:  # noqa: BLE001 - UI boundary must fail gracefully
    st.error(f"Could not calculate this fixture: {exc}")
    st.stop()

probability_map = {LABELS[code]: float(value) for code, value in zip(classes, probabilities, strict=True)}
winner = max(probability_map, key=probability_map.get)
st.subheader(f"{home} vs {away}")
st.markdown(f"### Most likely: {winner}")

columns = st.columns(3)
for column, label in zip(columns, ("Home win", "Draw", "Away win"), strict=True):
    column.metric(label, f"{probability_map[label]:.1%}")
st.bar_chart(pd.DataFrame({"Probability": probability_map}), horizontal=True)

with st.expander("Current form used by the model", expanded=True):
    form = pd.DataFrame(
        {
            home: [
                features.at[0, "home_points_5"],
                features.at[0, "home_goals_for_5"],
                features.at[0, "home_goals_against_5"],
                features.at[0, "home_win_rate_10"],
            ],
            away: [
                features.at[0, "away_points_5"],
                features.at[0, "away_goals_for_5"],
                features.at[0, "away_goals_against_5"],
                features.at[0, "away_win_rate_10"],
            ],
        },
        index=["Points / match (last 5)", "Goals scored / match (last 5)", "Goals allowed / match (last 5)", "Win rate (last 10)"],
    )
    st.dataframe(form.style.format("{:.2f}"), width="stretch")

metrics = artifact["test_metrics"]
baseline = artifact["baseline_test_metrics"]
with st.expander("Model card and held-out performance"):
    st.write(
        f"**{artifact['selected_model']}** · test period "
        f"{artifact['split']['test_start']} to {artifact['split']['test_end']}"
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Test accuracy", f"{metrics['accuracy']:.1%}", f"{metrics['accuracy'] - baseline['accuracy']:+.1%} vs baseline")
    c2.metric("Test log loss", f"{metrics['log_loss']:.3f}", f"{metrics['log_loss'] - baseline['log_loss']:+.3f} vs baseline", delta_color="inverse")
    c3.metric("Test Brier score", f"{metrics['brier']:.3f}", f"{metrics['brier'] - baseline['brier']:+.3f} vs baseline", delta_color="inverse")
    st.caption(
        "Train, validation, and test sets are chronological. Features are rolling summaries "
        "computed before each match; odds, final scores, and future matches are excluded."
    )

st.caption("Educational model, not betting advice. Data: football-data.co.uk.")
