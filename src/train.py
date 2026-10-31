"""Train the model on data/train_test.csv and save it to models/.

Run: python -m src.train
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from src import config, data, features, models, split

CANDIDATES = {
    "Naive $/mile": lambda: models.NaiveRatePerMile(),
    "Log-linear": lambda: models.LogLinear(),
    "LightGBM": lambda: models.LightGBM(),
    "LightGBM + recent level": lambda: models.LevelAdjusted(models.LightGBM(), use_ramp=False, use_trend=False),
    "LightGBM + market level": lambda: models.LevelAdjusted(models.LightGBM()),
    "Blend + market level": lambda: models.LevelAdjusted(
        models.Blend([models.LogLinear(), models.LightGBM()], weights=[0.3, 0.7])
    ),
}
SELECTED = "Blend + market level"
SELECTION_REASON = (
    "Blend + market level is selected: it has the lowest mean and worst-fold MAE. It averages a log-linear model "
    "(clean multiplicative structure and city effects) with LightGBM (non-linear effects; coordinates for unseen cities), "
    "and both share one market price level of recent level + trend + quarter-end ramp."
)


def prepare_training_data():
    """Cleaned training loads, their price-outlier flags, and the lookup tables used at prediction time."""
    train = data.load_train_test()
    tables = data.build_reference_tables(train, data.load_validation())
    train = data.clean_features(train, tables)
    return train, data.flag_price_outliers(train), tables


def backtest(train: pd.DataFrame, outlier: pd.Series, candidates=CANDIDATES) -> pd.DataFrame:
    """Score every candidate on every time fold.

    Models never see outliers during training. Test scores are reported on clean loads and on all
    loads ("raw_"), since the hidden validation labels may contain the same kind of errors.
    """
    X = features.build_features(train)
    y_log = np.log(train[config.TARGET])
    rows = []
    for name, make_model in candidates.items():
        for fold in split.FOLDS:
            fit_mask, test_mask = split.fold_masks(train["date"], fold)
            fit_mask &= ~outlier
            model = make_model().fit(X[fit_mask], y_log[fit_mask])
            predicted = np.exp(model.predict(X[test_mask]))
            actual = train.loc[test_mask, config.TARGET].to_numpy()
            clean = ~outlier[test_mask].to_numpy()
            raw_metrics = split.regression_metrics(actual, predicted)
            rows.append(
                {
                    "model": name,
                    "fold": fold.name,
                    **split.regression_metrics(actual[clean], predicted[clean]),
                    # Positive = predictions too low on average (median-style, in %).
                    "bias_pct": float(100 * np.mean(np.log(actual[clean] / predicted[clean]))),
                    **{f"raw_{metric}": value for metric, value in raw_metrics.items()},
                }
            )
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    summary = results.groupby("model").mean(numeric_only=True)
    summary.insert(1, "worst_fold_MAE", results.groupby("model")["MAE"].max())
    return summary.sort_values("MAE")


def main() -> None:
    train, outlier, tables = prepare_training_data()
    print(
        f"Loaded {len(train):,} training loads ({train['date'].min():%Y-%m-%d} to {train['date'].max():%Y-%m-%d}); "
        f"excluding {int(outlier.sum()):,} price outliers ({outlier.mean():.1%}) from training."
    )

    results = backtest(train, outlier)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.BACKTEST_RESULTS_PATH, index=False)
    summary = summarize(results)
    with pd.option_context("display.width", 140, "display.float_format", "{:,.2f}".format):
        print("\nBacktest (mean over time folds; MAE/RMSE in $, MAPE in %):")
        print(summary.to_string())
    print(f"\nSaved per-fold results to {config.BACKTEST_RESULTS_PATH}")

    clean = train[~outlier]
    model = CANDIDATES[SELECTED]().fit(features.build_features(clean), np.log(clean[config.TARGET]))
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "tables": tables}, config.MODEL_PATH)

    market_level = {
        # Relative to the average Jan-Oct level, in log units (~ fractions).
        "recent_level_log": model.level_ - float(model.daily_level_.mean()),
        "trend_per_month_log": model.trend_per_day_ * 30,
        "ramp_effect_log": model.ramp_effect_,
    }
    metadata = {
        "selected_model": SELECTED,
        "selection_reason": SELECTION_REASON,
        "training_rows": int(len(clean)),
        "excluded_outliers": int(outlier.sum()),
        "training_dates": [f"{train['date'].min():%Y-%m-%d}", f"{train['date'].max():%Y-%m-%d}"],
        "tree_features": models.TREE_FEATURES,
        "level_window_days": model.window_days,
        "market_level": market_level,
        "backtest_summary": summary.round(3).reset_index().to_dict(orient="records"),
    }
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(
        f"Trained '{SELECTED}' on {len(clean):,} loads. Market level: recent {np.expm1(market_level['recent_level_log']):+.2%} "
        f"vs Jan-Oct average, trend {np.expm1(market_level['trend_per_month_log']):+.2%}/month, "
        f"quarter-end ramp {np.expm1(market_level['ramp_effect_log']):+.2%}."
    )
    print(f"Saved model to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
