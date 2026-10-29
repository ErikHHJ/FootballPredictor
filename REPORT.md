# DAT158 Project Report — Draft

## Objective

The project predicts Premier League match outcomes (home win, draw, away win) and
communicates calibrated class probabilities through a small web application.

## Data and preparation

Six completed seasons (2019/20–2024/25) are reproducibly downloaded from
[Football-Data.co.uk](https://www.football-data.co.uk/englandm.php). Records are
ordered by match date. For every fixture, trailing team form is calculated before
the result is used to update state. This strict sequence, plus the exclusion of odds
and post-match fields, prevents target leakage.

## Modelling and evaluation

The first 70% of matches train each candidate; the next 15% select between logistic
regression and histogram gradient boosting by log loss; a prior-only dummy classifier
provides context. The selected model is refit on the first 85% and evaluated once on
the final 15%. Logistic regression won validation selection. On the 342-match test
period (15 September 2024–25 May 2025), it reached 49.1% accuracy, 1.029 log loss,
and 0.618 Brier score, versus 41.5%, 1.077, and 0.653 for the prior-only baseline.
Exact generated results are stored in `artifacts/metrics.json`.

## Product

The Streamlit interface offers home/away selectors, three outcome probabilities,
the most likely result, both teams' recent form, and held-out model metrics. It
handles missing artifacts, missing team data, and duplicate team selections with
clear messages.

## Ethics, limitations, and reproducibility

The model omits injuries, line-ups, transfers, and managerial context and therefore
must not be presented as betting advice. The source is acknowledged without claiming
an open data license; raw files remain outside version control. A single training
command reconstructs the data, comparison, metrics, and serialized pipeline.
