import numpy as np
import pandas as pd
import pytest

from src import config, data, features, models


@pytest.fixture
def training_set(synthetic_loads):
    cleaned = data.clean_features(synthetic_loads, data.build_reference_tables(synthetic_loads, synthetic_loads.iloc[:0]))
    return features.build_features(cleaned), np.log(cleaned[config.TARGET]), cleaned["date"]


@pytest.mark.parametrize(
    "make_model",
    [models.NaiveRatePerMile, models.LogLinear, lambda: models.LightGBM(n_estimators=200)],
    ids=["naive", "log-linear", "lightgbm"],
)
def test_models_learn_the_pricing_formula(training_set, make_model):
    X, y_log, dates = training_set
    model = make_model().fit(X.iloc[:2500], y_log.iloc[:2500], dates=dates.iloc[:2500])
    error = np.abs(np.exp(model.predict(X.iloc[2500:])) / np.exp(y_log.iloc[2500:]) - 1)
    assert error.mean() < 0.06


def test_log_linear_handles_unseen_city(training_set):
    X, y_log, dates = training_set
    model = models.LogLinear().fit(X, y_log)
    unseen = X.head(1).assign(pickup="Nowhere")
    assert np.isfinite(model.predict(unseen)).all()


def test_level_adjusted_carries_forward_a_recent_price_shift(training_set):
    X, y_log, dates = training_set
    shifted = y_log + np.where(dates >= dates.max() - pd.Timedelta(days=30), 0.05, 0.0)

    model = models.LevelAdjusted(models.LogLinear(), window_days=14).fit(X, shifted, dates=dates)

    assert model.level_ - model.daily_level_[dates.min()] == pytest.approx(0.05, abs=0.01)
