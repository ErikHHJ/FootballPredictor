# Premier League Football Predictor

**Name:** Erik Haugervåg Hjulstad
**Course:** DAT158 ML
**Date:** 20/10/2026

# 1: DESCRIBE THE PROBLEM

The goal of this project is to predict the outcome of an English Premier League football match before it is played. For a selected home team and away team, the system estimates three probabilities: home win (`H`), draw (`D`), and away win (`A`). This is a multiclass classification problem because every match belongs to exactly one of three outcome classes.

Football results are difficult to predict. A strong team can lose because of injuries, tactics, a red card, or random variation. The project therefore returns probabilities rather than claiming that one result is certain. The website also shows recent form and held-out model metrics so that the result is easier to interpret.

The project covers the complete machine-learning lifecycle: data acquisition, cleaning, leakage-safe feature engineering, chronological evaluation, model comparison, serialization, testing, and a Streamlit interface. The main technical priority is to imitate a realistic prediction setting. A prediction for a match may only use facts that were available before that match started.

## Scope

The project is limited to the English Premier League and uses six completed seasons, from 2019/20 through 2024/25. It predicts the full-time outcome only. It does not predict the exact score, number of goals, or events during a match. It also does not use betting odds, post-match statistics, player-level data, line-ups, injuries, transfers, tactics, managers, or live information.

The intended output is an educational web application. It is not a betting product or professional advice. A user chooses two different teams and receives three probabilities, the most likely result, recent form values, and a short model summary. The project is compact enough for a school project while still demonstrating the complete ML lifecycle.

## Metrics

Three metrics are used:

- **Accuracy** is the percentage of matches where the outcome with the highest predicted probability is correct.
- **Multiclass log loss** evaluates all three predicted probabilities and strongly penalizes confident wrong predictions. It is the main model-selection metric because the website displays probabilities.
- **Multiclass Brier score** measures the squared difference between the predicted probabilities and the real outcome.

Higher accuracy is better, while lower log loss and Brier score are better. A `DummyClassifier(strategy="prior")` is included as a baseline. It only learns how common each outcome is, so a useful model based on team form should perform better.

# 2: DATA

The data comes from Football-Data.co.uk, which provides season CSV files for English football [1]. The code downloads the Premier League `E0.csv` file for each season from 2019/20 through 2024/25. It loads the match date, home team, away team, full-time home goals, full-time away goals, and full-time result. Rows with missing required values are removed, outcomes are restricted to `H`, `D`, and `A`, and matches are sorted by date. The final dataset contains **2,280 matches**.

The source files contain extra columns such as betting odds and match statistics. These are intentionally excluded. Statistics collected during or after a match would cause target leakage, while betting odds would make the model depend on bookmaker predictions instead of features created from team history.

The feature builder processes matches from oldest to newest and keeps a rolling state for every team. For each match, it first creates the feature row from both teams' previous results. It only updates the team states after this row has been created. This prevents the current result and future matches from affecting the prediction. An automated test changes a match score and verifies that the features for that same match stay unchanged, while later matches are allowed to change.

The model uses eleven features:

- Home and away mean points from the previous 5 matches
- Home and away mean goals scored from the previous 5 matches
- Home and away mean goals conceded from the previous 5 matches
- Home and away win rate from the previous 10 matches
- Home and away rest days, limited to 2–30 days
- A home-advantage indicator

When a team has no previous history, neutral default values are used. This allows the model to handle teams with little available data, although the defaults may not represent newly promoted teams accurately.

Football-Data.co.uk publishes field definitions and source information [2], but does not provide an explicit open-data licence and displays “All Rights Reserved.” Raw CSV files are therefore downloaded locally, excluded from Git, and not redistributed. The repository's MIT licence applies to the project code and documentation, not the downloaded data. The provider also publishes an accuracy and liability disclaimer [3].

# 3: MODELING

A random train/test split would mix older and newer matches. That would not represent how the model is used in practice, where predictions always move forward in time. The 2,280 matches are therefore sorted by date and split without shuffling:

| Partition  | Matches | Share | Purpose               |
| ---------- | ------: | ----: | --------------------- |
| Training   |   1,596 |   70% | Fit the models        |
| Validation |     342 |   15% | Select the best model |
| Test       |     342 |   15% | Final evaluation      |

The test period runs from 15 September 2024 to 25 May 2025. The test data is not used when choosing the model.

Three approaches were compared:

1. **Prior baseline:** `DummyClassifier(strategy="prior")`. This predicts using only the outcome frequencies in the training data.
2. **Multinomial logistic regression:** Missing values are handled with a `SimpleImputer`, features are standardized, and logistic regression is used with `C=0.5` and `max_iter=2000`.
3. **Histogram gradient boosting:** The same preprocessing is followed by `HistGradientBoostingClassifier` with 150 iterations, learning rate 0.05, a maximum of 15 leaf nodes, L2 regularization 1.0, and random seed 158.

The two main models were first trained on the training set. The model with the lowest validation log loss was selected.

## Validation results

| Model                       |  Accuracy |  Log loss | Brier score |
| --------------------------- | --------: | --------: | ----------: |
| Prior baseline              |     45.6% |     1.060 |       0.641 |
| Logistic regression         | **52.9%** | **0.984** |   **0.585** |
| Histogram gradient boosting |     50.6% |     1.051 |       0.621 |

Logistic regression had the lowest validation log loss and was selected. This also shows that the more complicated gradient-boosting model was not automatically better. Logistic regression was then trained again using both the training and validation data.

## Held-out test results

| Model                        |  Accuracy |  Log loss | Brier score |
| ---------------------------- | --------: | --------: | ----------: |
| Prior baseline               |     41.5% |     1.077 |       0.653 |
| Selected logistic regression | **49.1%** | **1.029** |   **0.618** |

The selected model improved accuracy by 7.6 percentage points compared with the baseline. It also achieved lower log loss and Brier score. The weaker test result compared with the validation result shows why a later held-out period is important. Exact values are saved in `artifacts/metrics.json`.

The project contains three automated tests covering leakage prevention, fixture validation, chronological splitting, probability normalization, invalid teams, and artifact creation. The submitted test run reports **3 passed**, and `ruff check .` reports no lint errors.

The model still has important limitations. It only uses six seasons from one league, and recent form may react slowly to sudden team changes. It does not include player or match context, and no separate probability-calibration study or rolling backtest was completed.

# 4: DEPLOYMENT

The selected model is saved with Joblib in `artifacts/premier_league_model.joblib`. The artifact contains the fitted preprocessing and model pipeline, model name, evaluation metrics, split information, team list, and final rolling team states. Exact metrics are also saved in `artifacts/metrics.json`.

The Streamlit application loads this artifact and provides home-team and away-team selectors. It displays all three outcome probabilities, highlights the most likely result, shows a probability chart and recent form, and presents held-out model metrics. It gives clear messages for missing or damaged artifacts, duplicate team selections, unknown teams, and prediction errors. The website also states that the output is educational and not betting advice.

The project can be run locally on Windows with Python 3.11 or newer:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m streamlit run app.py --server.port 8765
```

The model can be retrained and the project checked with:

```powershell
python -m scripts.train
python -m pytest -q
python -m ruff check .
```

The training command downloads missing source files and creates new model and metric artifacts. `pyproject.toml` records the dependencies, and gradient boosting uses random seed 158. A new download is not checksum-pinned, so changes made by the data provider could affect a future training run. The committed model and metrics preserve the submitted result.

The current project does not include automatic monitoring or retraining. A future deployed version could store non-sensitive prediction information and compare predictions with later results. Accuracy, log loss, Brier score, outcome frequencies, and feature distributions could be monitored over time. A clear performance decline could trigger investigation and retraining using only data available at that point.

**Website demonstration:**[Demo](image.png) URL

# 5: REFERENCES

[1] Football-Data.co.uk. (n.d.). _England football results and betting odds data files_. Retrieved October 26, 2026, from https://www.football-data.co.uk/englandm.php

[2] Football-Data.co.uk. (n.d.). _Notes for football data_. Retrieved October 26, 2026, from https://www.football-data.co.uk/notes.txt

[3] Football-Data.co.uk. (n.d.). _Liability disclaimer_. Retrieved October 26, 2026, from https://www.football-data.co.uk/disclaimer.php

[4] Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. _Journal of Machine Learning Research, 12_, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html

[5] Streamlit Inc. (n.d.). _Streamlit documentation_. https://docs.streamlit.io/

## AI-tool disclosure

OpenAI ChatGPT and Cursor were used for project planning.
