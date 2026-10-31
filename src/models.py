"""Candidate models. Every model is fit on log(rate) and predicts log(rate)."""

from __future__ import annotations

import copy

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

# Load-level inputs only. Calendar effects that move all loads together (trend, quarter-end ramp)
# belong to the market price level in LevelAdjusted, not to the per-load model.
TREE_FEATURES = [
    "distance",
    "log_distance",
    "equipment",
    "weight",
    "weight_missing",
    "market_index",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "straight_line_miles",
    "delta_lat",
    "delta_lon",
    "day_of_week",
]

DISTANCE_BANDS = [0, 150, 300, 500, 800, 1200, 2000, np.inf]


class NaiveRatePerMile:
    """Baseline: median $/mile for the equipment type and distance band, times distance."""

    def fit(self, X, y_log):
        rate_per_mile = np.exp(y_log) / X["distance"]
        self.table_ = rate_per_mile.groupby([X["equipment"], self._band(X)], observed=True).median()
        self.fallback_ = float(rate_per_mile.median())
        return self

    def predict(self, X):
        keys = pd.MultiIndex.from_arrays([X["equipment"], self._band(X)])
        rate_per_mile = self.table_.reindex(keys).fillna(self.fallback_).to_numpy()
        return np.log(rate_per_mile * X["distance"].to_numpy())

    @staticmethod
    def _band(X):
        return pd.cut(X["distance"], DISTANCE_BANDS)


class LogLinear:
    """Ridge regression on log-scaled inputs with one-hot equipment and lane endpoints.

    Unseen cities get no city effect (the encoder ignores unknown categories).
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y_log):
        encoder = ColumnTransformer(
            [
                ("onehot", OneHotEncoder(handle_unknown="ignore"), ["equipment", "pickup", "delivery", "day_of_week"]),
                ("numeric", StandardScaler(), ["log_distance", "log_distance_sq", "log_weight", "log_market_index", "weight_missing"]),
            ]
        )
        self.pipeline_ = make_pipeline(encoder, Ridge(alpha=self.alpha)).fit(self._inputs(X), y_log)
        return self

    def predict(self, X):
        return self.pipeline_.predict(self._inputs(X))

    @staticmethod
    def _inputs(X):
        return X.assign(
            equipment=X["equipment"].astype(str),
            log_distance_sq=X["log_distance"] ** 2,
            log_weight=np.log(X["weight"]),
            log_market_index=np.log(X["market_index"]),
        )


class LightGBM:
    """Gradient-boosted trees on coordinates and load attributes (no city names)."""

    def __init__(self, features=TREE_FEATURES, **params):
        self.features = list(features)
        self.params = {
            "n_estimators": 1500,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 50,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "random_state": config.RANDOM_SEED,
            "verbose": -1,
            **params,
        }

    def fit(self, X, y_log):
        self.model_ = lgb.LGBMRegressor(**self.params).fit(X[self.features], y_log)
        return self

    def predict(self, X):
        return self.model_.predict(X[self.features])


class Blend:
    """Weighted average of member predictions in log space."""

    def __init__(self, members, weights):
        self.members = members
        self.weights = weights

    def fit(self, X, y_log):
        for member in self.members:
            member.fit(X, y_log)
        return self

    def predict(self, X):
        return sum(weight * member.predict(X) for member, weight in zip(self.members, self.weights))


class LevelAdjusted:
    """Adds the market price level: the part of the rate shared by all loads on a day that the inputs don't explain.

    Fitting alternates between training the base model on level-adjusted targets and re-estimating a daily level
    from its residuals. The final daily level is measured on out-of-fold predictions (days held out in groups), because
    in-sample residuals are optimistic. It is then split by least squares into a linear trend and a quarter-end ramp
    (the `quarter_end_proximity` feature). The forecast for a future date is the recent level (mean of the last
    `window_days`, ramp removed), plus the trend continued from there, plus the ramp for that date.
    """

    def __init__(
        self,
        base,
        window_days: int = 14,
        n_iter: int = 5,
        oof_folds: int = 5,
        use_ramp: bool = True,
        use_trend: bool = True,
    ):
        self.base = base
        self.window_days = window_days
        self.n_iter = n_iter
        self.oof_folds = oof_folds
        self.use_ramp = use_ramp
        self.use_trend = use_trend

    def fit(self, X, y_log):
        dates = X["date"]
        y_log = pd.Series(np.asarray(y_log, dtype=float), index=X.index)
        level = pd.Series(0.0, index=X.index)
        for _ in range(self.n_iter):
            self.base.fit(X, y_log - level)
            daily_level = (y_log - self.base.predict(X)).groupby(dates).mean()
            level = dates.map(daily_level)
        daily_level = (y_log - self._out_of_fold(X, y_log - level)).groupby(dates).mean()

        days = self._days(daily_level.index)
        ramp = X["quarter_end_proximity"].groupby(dates).first().reindex(daily_level.index).to_numpy()
        design = np.column_stack([np.ones_like(days), days, ramp])
        _, trend, ramp_effect = np.linalg.lstsq(design, daily_level.to_numpy(), rcond=None)[0]
        self.trend_per_day_ = trend if self.use_trend else 0.0
        self.ramp_effect_ = ramp_effect if self.use_ramp else 0.0

        recent = daily_level.index > daily_level.index.max() - pd.Timedelta(days=self.window_days)
        self.anchor_day_ = float(days[recent].mean())
        self.level_ = float((daily_level - self.ramp_effect_ * ramp)[recent].mean())
        self.daily_level_ = daily_level
        return self

    def market_level(self, X):
        """Forecast log price level for each row's date."""
        days_ahead = self._days(X["date"]) - self.anchor_day_
        return self.level_ + self.trend_per_day_ * days_ahead + self.ramp_effect_ * X["quarter_end_proximity"].to_numpy()

    def predict(self, X):
        return self.base.predict(X) + self.market_level(X)

    def _out_of_fold(self, X, target):
        """Base-model predictions for each row from a copy trained without that row's day."""
        days = X["date"].unique()
        rng = np.random.default_rng(config.RANDOM_SEED)
        fold = X["date"].map(pd.Series(rng.integers(0, self.oof_folds, len(days)), index=days)).to_numpy()
        predicted = np.empty(len(X))
        for k in range(self.oof_folds):
            held_out = fold == k
            model = copy.deepcopy(self.base).fit(X[~held_out], target[~held_out])
            predicted[held_out] = model.predict(X[held_out])
        return predicted

    @staticmethod
    def _days(dates):
        return np.asarray((pd.DatetimeIndex(dates) - pd.Timestamp("2025-01-01")).days, dtype=float)
