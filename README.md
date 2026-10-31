# Freight Rate Prediction

Predicts the posted rate (USD) of a freight load from its lane, distance, equipment, weight and date. Built for the Spotter Machine Learning Engineer assessment.

- **Training data:** 48,000 labelled loads, Jan–Oct 2025 (`data/train_test.csv`)
- **Predictions:** 12,000 loads, Nov–Dec 2025 (`validation_predictions.csv`), plus a fixed-lane December chart
- **Report:** `reports/report.pdf` · **EDA:** `notebooks/01_eda.ipynb`

## Approach in brief

1. **Clean the inputs.** Fix negative weights (sign errors) and fill missing weights. Fill missing `market_index` with that day's median (it is a daily signal). Look up city coordinates for the December file. Remove about 1.4% broken prices (2–7× off their expected value) from training.
2. **Drop `quote_signal`.** It equals price per mile in some months, mirrors it in others, and is pure noise in Nov–Dec. A check that needs no labels (its correlation with distance) shows this. A model that used it would look great on a random split and fail on Nov–Dec.
3. **Validate by time.** Three rolling folds each train on the past and predict the next two months, the same shape as Jan–Oct → Nov–Dec. Models are compared on mean and worst-fold MAE.
4. **Model.**
   - LightGBM on log(rate) uses load attributes and coordinates. It has no city names, so new cities still get a price.
   - A **market price level** shared by all loads on a day is added on top: the recent level carried forward, plus the yearly upward trend, plus a ~4% ramp over the last four weeks of each quarter.
   - The validation `market_index` values give each future day its weekly cycle.

## Results

Time-based backtest: mean of 3 folds, each predicting the two months after its training data. Scores are on clean test loads.

| Model | MAE | Worst fold MAE | MAPE |
|---|---|---|---|
| **Blend + market level (selected)** | **$44.5** | **$51.4** | **1.89%** |
| LightGBM + market level | $45.8 | $54.7 | 1.96% |
| LightGBM + recent level only | $56.4 | $62.1 | 2.38% |
| LightGBM (no market level) | $77.5 | $117.6 | 3.24% |
| Naive $/mile baseline | $105.7 | $148.0 | 4.59% |

Full per-fold results, including RMSE, bias and scores on all loads with broken prices kept, are in `reports/backtest_results.csv`. The final model is retrained on all 47,323 clean Jan–Oct loads.

## Project structure

```
.
├── data/                  # Provided CSVs (not committed; see "Data")
├── notebooks/01_eda.ipynb # Exploratory analysis with charts
├── src/
│   ├── config.py          # Paths and constants
│   ├── data.py            # Loading, cleaning, outlier detection
│   ├── features.py        # Feature engineering (shared by train/validation/December)
│   ├── split.py           # Time-based folds and metrics
│   ├── models.py          # Baselines, LightGBM, market price level
│   ├── train.py           # Backtest candidates, train and save the final model
│   ├── diagnostics.py     # Evidence for the split, quote_signal and unseen-city decisions
│   ├── predict.py         # Validation and December chart predictions
│   └── report.py          # Builds reports/report.docx and .pdf
├── tests/                 # pytest suite (reuses score.py's validators)
├── reports/               # Report, figures, backtest and diagnostics results
├── score.py               # Provided scorer (unchanged)
└── validation_predictions.csv   # Final submission
```

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
python -m pip install -r requirements.txt        # pipeline only
python -m pip install -r requirements-dev.txt    # + tests, notebook, report
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
python -m src.train          # backtest all candidates, train the final model (~2 min)
python -m src.diagnostics    # split / quote_signal / unseen-city evidence (~2 min)
python -m src.predict        # writes validation_predictions.csv and the December predictions
python score.py --predictions validation_predictions.csv --december-predictions outputs/december_chart_predictions.csv
python -m pytest             # tests
```

`score.py` validates both prediction files and writes `scorer_results/candidate_december.png`.

Optional: re-run the notebook with `python -m nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb`, then rebuild the report with `python -m src.report` (PDF export needs Microsoft Word).
