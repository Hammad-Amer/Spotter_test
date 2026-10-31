"""Evidence behind the validation design: split choice, the quote_signal trap and unseen cities.

Run: python -m src.diagnostics
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config, data, features, models, split
from src.train import prepare_training_data

DIAGNOSTICS_PATH = config.REPORTS_DIR / "diagnostics.json"
WITH_QUOTE_SIGNAL = models.TREE_FEATURES + ["quote_signal"]


def _mae(model, X, y_log, fit_mask, test_mask, actual) -> float:
    model.fit(X[fit_mask], y_log[fit_mask])
    return split.regression_metrics(actual[test_mask], np.exp(model.predict(X[test_mask])))["MAE"]


def split_comparison(train, outlier, X, y_log) -> list[dict]:
    """A random split looks far better than a time split, mostly because quote_signal leaks in some months."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    random_test = pd.Series(rng.random(len(train)) < 0.2, index=train.index)
    time_train, time_test = split.fold_masks(train["date"], split.FOLDS[-1])
    rows = []
    for split_name, fit_mask, test_mask in [
        ("Random 80/20", ~random_test, random_test),
        ("Time: Jan-Aug -> Sep-Oct", time_train, time_test),
    ]:
        for feature_name, feature_list in [("with quote_signal", WITH_QUOTE_SIGNAL), ("without quote_signal", models.TREE_FEATURES)]:
            mae = _mae(models.LightGBM(features=feature_list), X, y_log, fit_mask & ~outlier, test_mask & ~outlier, train[config.TARGET])
            rows.append({"split": split_name, "features": feature_name, "MAE": round(mae, 1)})
    return rows


def quote_signal_by_month(train, validation, outlier, X, y_log) -> list[dict]:
    """Leave-one-month-out error with and without quote_signal, next to label-free regime indicators.

    corr(quote_signal, distance) needs no labels, so it also shows which regime Nov-Dec are in.
    """
    clean = train[~outlier]
    rate_per_mile = clean[config.TARGET] / clean["distance"]
    rows = []
    for month in range(1, 11):
        in_month = train["date"].dt.month == month
        maes = {
            name: round(_mae(models.LightGBM(features=feature_list), X, y_log, ~in_month & ~outlier, in_month & ~outlier, train[config.TARGET]), 1)
            for name, feature_list in [("MAE_with_quote_signal", WITH_QUOTE_SIGNAL), ("MAE_without_quote_signal", models.TREE_FEATURES)]
        }
        month_rows = clean["date"].dt.month == month
        rows.append(
            {
                "month": f"2025-{month:02d}",
                "corr_quote_signal_rate_per_mile": round(clean.loc[month_rows, "quote_signal"].corr(rate_per_mile[month_rows]), 3),
                "corr_quote_signal_distance": round(clean.loc[month_rows, "quote_signal"].corr(clean.loc[month_rows, "distance"]), 3),
                **maes,
            }
        )
    for month in (11, 12):
        month_rows = validation["date"].dt.month == month
        rows.append(
            {
                "month": f"2025-{month:02d} (validation)",
                "corr_quote_signal_distance": round(validation.loc[month_rows, "quote_signal"].corr(validation.loc[month_rows, "distance"]), 3),
            }
        )
    return rows


def unseen_cities(train, outlier, X, y_log, n_cities: int = 8) -> list[dict]:
    """Hide n cities from training and score loads that touch them (validation has 8 new cities)."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    hidden = set(rng.choice(sorted(train["pickup"].unique()), n_cities, replace=False))
    touches_hidden = train["pickup"].isin(hidden) | train["delivery"].isin(hidden)
    rows = []
    for name, model in [
        ("Log-linear (city one-hot)", models.LogLinear()),
        ("LightGBM (coordinates)", models.LightGBM()),
        ("Blend", models.Blend([models.LogLinear(), models.LightGBM()], weights=[0.3, 0.7])),
    ]:
        mae = _mae(model, X, y_log, ~touches_hidden & ~outlier, touches_hidden & ~outlier, train[config.TARGET])
        rows.append({"model": name, "hidden_cities": sorted(hidden), "MAE_on_unseen_city_loads": round(mae, 1)})
    return rows


def main() -> None:
    train, outlier, tables = prepare_training_data()
    validation = data.clean_features(data.load_validation(), tables)
    X = features.build_features(train).assign(quote_signal=train["quote_signal"])
    y_log = np.log(train[config.TARGET])

    results = {
        "split_comparison": split_comparison(train, outlier, X, y_log),
        "quote_signal_by_month": quote_signal_by_month(train, validation, outlier, X, y_log),
        "unseen_cities": unseen_cities(train, outlier, X, y_log),
    }
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_PATH.write_text(json.dumps(results, indent=2))
    with pd.option_context("display.width", 160):
        for name, rows in results.items():
            print(f"\n{name}:")
            print(pd.DataFrame(rows).drop(columns=["hidden_cities"], errors="ignore").to_string(index=False))
    print(f"\nSaved to {DIAGNOSTICS_PATH}")


if __name__ == "__main__":
    main()
