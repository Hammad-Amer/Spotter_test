# Freight Rate Prediction

Predicts the posted rate (USD) for a freight load from its lane, distance, equipment, weight, date and market signals. Built for the Spotter Machine Learning Engineer assessment.

## Project structure

```
.
├── data/                  # Provided CSVs (not committed; see "Data" below)
├── notebooks/             # Exploratory data analysis
├── src/
│   ├── config.py          # Paths and constants
│   ├── data.py            # Loading and cleaning
│   ├── features.py        # Feature engineering
│   ├── split.py           # Time-based split and metrics
│   ├── train.py           # Train and save the model
│   └── predict.py         # Validation and December chart predictions
├── models/                # Saved model (generated)
├── outputs/               # Intermediate predictions (generated)
├── reports/               # Written report and figures
├── score.py               # Provided scorer (unchanged)
├── requirements.txt
└── validation_predictions.csv   # Final submission (generated)
```

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
python -m pip install -r requirements.txt
```

## Data

Place the four provided files in `data/`:

```
data/train_test.csv
data/validation.csv
data/validation_predictions_template.csv
data/december_chart_inputs.csv
```

## Run

```bash
python -m src.train
python -m src.predict
python score.py --predictions validation_predictions.csv --december-predictions outputs/december_chart_predictions.csv
```

The scorer validates both prediction files and writes `scorer_results/candidate_december.png`.
