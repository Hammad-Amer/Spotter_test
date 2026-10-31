"""Load the provided CSVs and clean data-quality issues (bad weights, missing values, rate outliers)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src import config

COORDINATE_COLUMNS = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]


def _read(path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def load_train_test() -> pd.DataFrame:
    return _read(config.TRAIN_TEST_PATH)


def load_validation() -> pd.DataFrame:
    return _read(config.VALIDATION_PATH)


def load_december_inputs() -> pd.DataFrame:
    return _read(config.DECEMBER_INPUTS_PATH)


@dataclass
class ReferenceTables:
    """Lookups built from input columns only (never from posted_rate), so they are safe at prediction time."""

    weight_median: dict[str, float]
    daily_market_index: pd.Series  # indexed by date
    city_coords: pd.DataFrame  # indexed by city, columns: lat, lon


def build_reference_tables(train: pd.DataFrame, validation: pd.DataFrame) -> ReferenceTables:
    # market_index is a daily signal (~98% of its variance is explained by date), so a daily median
    # is the natural fill value. Validation inputs extend the table into Nov-Dec for the December chart.
    inputs = pd.concat([train, validation], ignore_index=True)
    weight_median = train["weight"].abs().clip(upper=config.WEIGHT_CAP).groupby(train["equipment"]).median()
    pickups = inputs[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1)
    deliveries = inputs[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1)
    return ReferenceTables(
        weight_median=weight_median.to_dict(),
        daily_market_index=inputs.groupby("date")["market_index"].median(),
        city_coords=pd.concat([pickups, deliveries]).drop_duplicates("city").set_index("city"),
    )


def clean_features(frame: pd.DataFrame, tables: ReferenceTables) -> pd.DataFrame:
    """Fix input columns and add any that are absent (the December file has no coordinates or market_index)."""
    out = frame.copy()

    weight = out["weight"].abs().clip(upper=config.WEIGHT_CAP)
    out["weight_missing"] = weight.isna().astype(int)
    out["weight"] = weight.fillna(out["equipment"].map(tables.weight_median))

    if "market_index" not in out:
        out["market_index"] = np.nan
    out["market_index"] = out["market_index"].fillna(out["date"].map(tables.daily_market_index))

    for end in ("pickup", "delivery"):
        for axis in ("lat", "lon"):
            column = f"{end}_{axis}"
            if column not in out:
                out[column] = np.nan
            out[column] = out[column].fillna(out[end].map(tables.city_coords[axis]))

    required = ["distance", "weight", "market_index", *COORDINATE_COLUMNS]
    missing = [column for column in required if out[column].isna().any()]
    if missing:
        raise ValueError(f"could not fill missing values in: {missing}")
    return out


def price_residuals(frame: pd.DataFrame, threshold: float = config.OUTLIER_LOG_THRESHOLD) -> pd.Series:
    """Log distance between each load's rate and a robust log-linear estimate of it.

    The estimate uses lane, equipment, distance, weight, market_index and month, and is refit once
    without the flagged rows so the outliers do not pull it towards themselves.
    """
    X = pd.get_dummies(frame[["equipment", "pickup", "delivery"]], dtype=float)
    X = X.join(pd.get_dummies(frame["date"].dt.month, prefix="month", dtype=float))
    X["log_distance"] = np.log(frame["distance"])
    X["log_weight"] = np.log(frame["weight"])
    X["log_market_index"] = np.log(frame["market_index"])
    y = np.log(frame[config.TARGET] / frame["distance"])

    keep = pd.Series(True, index=frame.index)
    for _ in range(2):
        model = Ridge(alpha=1.0).fit(X[keep], y[keep])
        residual = y - model.predict(X)
        keep = residual.abs() <= threshold
    return residual


def flag_price_outliers(frame: pd.DataFrame, threshold: float = config.OUTLIER_LOG_THRESHOLD) -> pd.Series:
    """True for loads priced 1.8x+ above or below expectation: almost certainly data-entry errors."""
    return price_residuals(frame, threshold).abs() > threshold
