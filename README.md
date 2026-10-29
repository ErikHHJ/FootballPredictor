# DAT158 Premier League Football Predictor (REPORT)

**Group:** [GROUP NUMBER 20]  
**Members:** Erik Haugervåg Hjulstad  
**Repository:** [REPOSITORY URL]  
**Submission date:** [DATE]

## Overview

This project is an end-to-end machine-learning application that estimates the probabilities of a Premier League **home win**, **draw**, or **away win**. It uses only historical information available before kick-off, compares a baseline with two classification models, saves the selected model, and presents predictions through a Streamlit website. It is an educational DAT158 project, not betting advice.

## Data and licensing

The training script downloads six completed Premier League seasons (2019/20–2024/25) from Football-Data.co.uk `E0.csv` files [1]. The prepared dataset contains **2,280 matches**. We use the match date, teams, full-time goals, and full-time result. Betting odds and post-match statistics are excluded from the model.

Football-Data.co.uk documents its fields and sources [2], but does not publish an explicit open-data licence and marks its content “All Rights Reserved.” The raw CSV files are therefore downloaded locally, excluded from Git, and not redistributed. The repository's MIT licence covers our code and documentation, not the source data. The provider's disclaimer also applies [3].

## Machine-learning lifecycle

The project follows these steps:

1. Download or reuse the source CSV files.
2. Validate the required columns and sort matches by date.
3. Create rolling features from matches played before each fixture.
4. Split the data chronologically into training, validation, and test sets.
5. Train and compare a baseline and two classification models.
6. Select the model with the lowest validation log loss.
7. Refit the selected model and evaluate it once on the held-out test set.
8. Save the model, preprocessing pipeline, team states, and metrics for the website.

For every fixture, the feature row is created **before** that match updates either team's state. This prevents target leakage: the current result, score, and future matches cannot influence the current prediction.

The model uses eleven inputs:

- Home and away mean points over the last 5 matches
- Home and away mean goals scored over the last 5 matches
- Home and away mean goals conceded over the last 5 matches
- Home and away win rate over the last 10 matches
- Home and away rest days, limited to 2–30 days
- A home-advantage indicator

Teams without previous history receive neutral default values.

## Data split, models, and results

The 2,280 matches are sorted by date and divided without shuffling:

| Partition  | Matches | Share | Purpose               |
| ---------- | ------: | ----: | --------------------- |
| Training   |   1,596 |   70% | Fit models            |
| Validation |     342 |   15% | Select the best model |
| Test       |     342 |   15% | Final evaluation      |

The test period is **15 September 2024–25 May 2025**. We compared:

1. `DummyClassifier(strategy="prior")` as a class-prior baseline
2. Multinomial logistic regression with imputation and standardization
3. Histogram gradient boosting with the same preprocessing

### Validation results

| Model                       |  Accuracy |  Log loss | Brier score |
| --------------------------- | --------: | --------: | ----------: |
| Prior baseline              |     45.6% |     1.060 |       0.641 |
| Logistic regression         | **52.9%** | **0.984** |   **0.585** |
| Histogram gradient boosting |     50.6% |     1.051 |       0.621 |

Logistic regression had the lowest validation log loss and was selected. It was then refitted using the combined training and validation data.

### Held-out test results

| Model                        |  Accuracy |  Log loss | Brier score |
| ---------------------------- | --------: | --------: | ----------: |
| Prior baseline               |     41.5% |     1.077 |       0.653 |
| Selected logistic regression | **49.1%** | **1.029** |   **0.618** |

Higher accuracy is better, while lower log loss and Brier score are better. The selected model outperformed the baseline on all three metrics. Exact values are stored in `artifacts/metrics.json`.

## Website

The Streamlit website lets the user select a home team and an away team. It displays:

- Home-win, draw, and away-win probabilities
- The most likely outcome
- A probability chart
- Recent team form used by the model
- The selected model and held-out evaluation results

The application also handles duplicate team selections, missing or damaged model files, insufficient team data, and prediction errors with clear messages.

**Website demonstration:** [INSERT DEPLOYED WEBSITE OR VIDEO LINK]

## Project structure

```text
app.py                         Streamlit website
football_predictor/core.py     Data, features, training, and evaluation
scripts/train.py               Reproducible training command
artifacts/                     Saved model and exact metrics
tests/test_core.py             Leakage and pipeline tests
pyproject.toml                 Dependencies and tool configuration
README.md                      Project report and instructions
```

## Installation and use

Use PowerShell with Python 3.11 or newer from the project directory.

```powershell
# Create and activate a virtual environment
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

# Install the project
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

# Run the website with the included trained model
python -m streamlit run app.py --server.port 8765
```

Open `http://localhost:8765` if the browser does not open automatically.

To download the source data and retrain the model:

```powershell
python -m scripts.train
```

To run the automated checks:

```powershell
python -m pytest -q
python -m ruff check .
```

Our submitted checks report **3 passed tests** and no Ruff errors.

## Limitations and ethics

Football results are difficult to predict. The model does not include line-ups, injuries, suspensions, transfers, tactics, managers, match importance, or live information. Recent form may react slowly to sudden team changes, and neutral defaults may represent newly promoted teams poorly. Results from six Premier League seasons do not prove that the model will perform equally well in other leagues or future seasons.

The predictions are uncertain estimates and should not be treated as professional or gambling advice. A test accuracy of 49.1% does not guarantee future performance.

## Reproducibility

`pyproject.toml` records the required dependency ranges and Python version. One training command performs data acquisition, preprocessing, feature creation, model comparison, evaluation, and serialization. Gradient boosting uses random seed `158`, while the committed model and metrics preserve the submitted results.

Raw data is not included because no open redistribution licence was identified. A fresh training run also depends on Football-Data.co.uk remaining available and unchanged.

## Contributions

| Member | Contribution                                               |
| ------ | ---------------------------------------------------------- |
| [NAME] | [DATA / FEATURES / MODELLING / WEBSITE / TESTING / REPORT] |
| [NAME] | [CONTRIBUTION]                                             |
| [NAME] | [CONTRIBUTION]                                             |

## AI-tool disclosure

OpenAI ChatGPT and Cursor were used for project planning.

## References

[1] Football-Data.co.uk. (n.d.). _England football results and betting odds data files_. Retrieved October 26, 2026, from https://www.football-data.co.uk/englandm.php

[2] Football-Data.co.uk. (n.d.). _Notes for football data_. Retrieved October 26, 2026, from https://www.football-data.co.uk/notes.txt

[3] Football-Data.co.uk. (n.d.). _Liability disclaimer_. Retrieved October 26, 2026, from https://www.football-data.co.uk/disclaimer.php

[4] Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. _Journal of Machine Learning Research, 12_, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html

[5] Streamlit Inc. (n.d.). _Streamlit documentation_. https://docs.streamlit.io/
