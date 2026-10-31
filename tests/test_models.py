import numpy as np
import pytest

from src import config, data, features, models


@pytest.fixture
def training_set(synthetic_loads):
    cleaned = data.clean_features(synthetic_loads, data.build_reference_tables(synthetic_loads, synthetic_loads.iloc[:0]))
    return features.build_features(cleaned), np.log(cleaned[config.TARGET])


@pytest.mark.parametrize(
    "make_model",
    [models.NaiveRatePerMile, models.LogLinear, lambda: models.LightGBM(n_estimators=200)],
    ids=["naive", "log-linear", "lightgbm"],
)
def test_models_learn_the_pricing_formula(training_set, make_model):
    X, y_log = training_set
    model = make_model().fit(X.iloc[:2500], y_log.iloc[:2500])
    error = np.abs(np.exp(model.predict(X.iloc[2500:])) / np.exp(y_log.iloc[2500:]) - 1)
    assert error.mean() < 0.06


def test_log_linear_handles_unseen_city(training_set):
    X, y_log = training_set
    model = models.LogLinear().fit(X, y_log)
    unseen = X.head(1).assign(pickup="Nowhere")
    assert np.isfinite(model.predict(unseen)).all()


def test_level_adjusted_carries_forward_a_recent_price_shift(training_set):
    X, y_log = training_set
    shifted = y_log + np.where(X["date"] > X["date"].max() - np.timedelta64(30, "D"), 0.05, 0.0)

    model = models.LevelAdjusted(models.LogLinear(), use_ramp=False, use_trend=False).fit(X, shifted)

    assert model.level_ - model.daily_level_.iloc[0] == pytest.approx(0.05, abs=0.01)


def test_level_adjusted_recovers_quarter_end_ramp_and_trend(training_set):
    X, y_log = training_set
    days = (X["date"] - X["date"].min()).dt.days
    priced = y_log + 0.04 * X["quarter_end_proximity"] + 0.0002 * days

    model = models.LevelAdjusted(models.LogLinear()).fit(X, priced)

    assert model.ramp_effect_ == pytest.approx(0.04, abs=0.005)
    assert model.trend_per_day_ == pytest.approx(0.0002, abs=0.00005)
    # The forecast for the last day of the quarter includes the full ramp.
    last_day = X[X["quarter_end_proximity"] == 1].head(1)
    assert model.market_level(last_day)[0] - model.market_level(last_day.assign(quarter_end_proximity=0.0))[0] == pytest.approx(0.04, abs=0.005)
