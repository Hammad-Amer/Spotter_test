"""Candidate models. Every model is fit on log(rate) and predicts log(rate)."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

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
    "quarter_end_proximity",
]

DISTANCE_BANDS = [0, 150, 300, 500, 800, 1200, 2000, np.inf]


class NaiveRatePerMile:
    """Baseline: median $/mile for the equipment type and distance band, times distance."""

    def fit(self, X, y_log, dates=None, sample_weight=None):
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

    def fit(self, X, y_log, dates=None, sample_weight=None):
        encoder = ColumnTransformer(
            [
                ("onehot", OneHotEncoder(handle_unknown="ignore"), ["equipment", "pickup", "delivery", "day_of_week"]),
                (
                    "numeric",
                    StandardScaler(),
                    ["log_distance", "log_distance_sq", "log_weight", "log_market_index", "weight_missing", "quarter_end_proximity"],
                ),
            ]
        )
        self.pipeline_ = make_pipeline(encoder, Ridge(alpha=self.alpha))
        self.pipeline_.fit(self._inputs(X), y_log, ridge__sample_weight=sample_weight)
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

    def fit(self, X, y_log, dates=None, sample_weight=None):
        self.model_ = lgb.LGBMRegressor(**self.params)
        self.model_.fit(X[self.features], y_log, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.model_.predict(X[self.features])


class Blend:
    """Weighted average of member predictions in log space."""

    def __init__(self, members, weights):
        self.members = members
        self.weights = weights

    def fit(self, X, y_log, dates=None, sample_weight=None):
        for member in self.members:
            member.fit(X, y_log, dates=dates, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return sum(weight * member.predict(X) for member, weight in zip(self.members, self.weights))


class LevelAdjusted:
    """Adds the recent market price level that the inputs do not explain, carried forward.

    Training prices drift over the year independently of market_index. The model alternates between
    fitting the base model on level-adjusted targets and re-estimating a daily level from its residuals,
    then forecasts the level as the average over the last `window_days` days of training data.
    """

    def __init__(self, base, window_days: int = 14, n_iter: int = 5):
        self.base = base
        self.window_days = window_days
        self.n_iter = n_iter

    def fit(self, X, y_log, dates, sample_weight=None):
        dates = pd.Series(pd.to_datetime(np.asarray(dates)), index=X.index)
        y_log = pd.Series(np.asarray(y_log, dtype=float), index=X.index)
        level = pd.Series(0.0, index=X.index)
        for _ in range(self.n_iter):
            self.base.fit(X, y_log - level, dates=dates, sample_weight=sample_weight)
            residual = y_log - self.base.predict(X)
            daily_level = residual.groupby(dates).mean()
            level = dates.map(daily_level)
        recent = daily_level.index > daily_level.index.max() - pd.Timedelta(days=self.window_days)
        self.daily_level_ = daily_level
        self.level_ = float(daily_level[recent].mean())
        return self

    def predict(self, X):
        return self.base.predict(X) + self.level_
