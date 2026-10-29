# Premier League Football Predictor

This is our DAT158 school project. It is a small Streamlit website that predicts the chances of a **home win**, **draw**, or **away win** in a Premier League match.

Choose two teams and the website will show the three probabilities, the most likely result, recent team form, and a few numbers showing how the model performed. The model only uses information from matches played before the match being predicted.

This is just an educational project and should not be used as betting advice.

## What is included?

- Six Premier League seasons from 2019/20 to 2024/25
- Team form based on earlier matches
- A chronological train, validation, and test split
- A simple baseline, logistic regression, and gradient boosting
- A saved trained model
- A Streamlit website
- Automated tests

## Results

The dataset contains 2,280 matches. Logistic regression performed best on the validation data, so that is the model used by the website.

| Model               | Test accuracy |  Log loss | Brier score |
| ------------------- | ------------: | --------: | ----------: |
| Prior baseline      |         41.5% |     1.077 |       0.653 |
| Logistic regression |     **49.1%** | **1.029** |   **0.618** |

Higher accuracy is better, while lower log loss and Brier score are better.

## Project files

```text
app.py                         Streamlit website
football_predictor/core.py     Data, features, training, and evaluation
scripts/train.py               Trains the model
artifacts/                     Saved model and metrics
tests/test_core.py             Automated tests
pyproject.toml                 Dependencies and settings
REPORT.md                      Full project report
```

## Run it on Windows

Python 3.11 or newer is required. Open PowerShell in the project folder and run:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m streamlit run app.py --server.port 8765
```

Open http://localhost:8765 if the browser does not open automatically.

## Train the model again

```powershell
python -m scripts.train
```

This downloads the missing season files and rebuilds the model and metrics.

## Run the tests

```powershell
python -m pytest -q
python -m ruff check .
```

Our submitted run reports **3 passed tests** and no Ruff errors.

## Data

Match results come from [Football-Data.co.uk](https://www.football-data.co.uk/englandm.php). The raw CSV files are downloaded when training and are not included in the repository because the website does not provide a clear open-data licence.

The model does not know about injuries, line-ups, transfers, tactics, managers, or live events, so its predictions will never be perfect.

## Demo

[ADD A SCREENSHOT, VIDEO LINK, OR DEPLOYED WEBSITE LINK]
