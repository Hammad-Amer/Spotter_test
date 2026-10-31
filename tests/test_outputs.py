"""Checks on the generated submission files, reusing the validators from the provided score.py."""

import numpy as np
import pandas as pd
import pytest

import score
from src import config, predict

requires_outputs = pytest.mark.skipif(
    not (config.VALIDATION_PREDICTIONS_PATH.exists() and config.DECEMBER_PREDICTIONS_PATH.exists()),
    reason="run `python -m src.train` and `python -m src.predict` first",
)


@requires_outputs
def test_validation_predictions_pass_the_official_checks():
    predictions = pd.read_csv(config.VALIDATION_PREDICTIONS_PATH)
    score.validate_predictions(predictions)
    template = pd.read_csv(config.TEMPLATE_PATH)
    assert predictions["load_id"].tolist() == template["load_id"].tolist()


@requires_outputs
def test_validation_predictions_are_plausible_per_mile():
    predictions = pd.read_csv(config.VALIDATION_PREDICTIONS_PATH)
    validation = pd.read_csv(config.VALIDATION_PATH)
    merged = predictions.merge(validation[["load_id", "distance"]], on="load_id", validate="one_to_one")
    rate_per_mile = merged["predicted_rate"] / merged["distance"]
    low, high = predict.RATE_PER_MILE_BOUNDS
    assert rate_per_mile.between(low, high).all()


@requires_outputs
def test_december_predictions_pass_the_official_checks():
    december = pd.read_csv(config.DECEMBER_PREDICTIONS_PATH)
    checked = score.validate_december(december)
    # Only the date changes, so day-to-day moves should be modest and the line not flat.
    rates = checked["predicted_rate"].to_numpy()
    assert rates.std() > 0
    assert np.abs(np.diff(rates) / rates[:-1]).max() < 0.05


def test_check_rates_rejects_implausible_values():
    with pytest.raises(ValueError, match="plausible"):
        predict.check_rates(np.array([100.0, 50_000.0]), np.array([100.0, 100.0]), "test")
