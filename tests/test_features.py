import pandas as pd
import pytest

from src import data, features, models
from tests.conftest import requires_data


def test_haversine_matches_known_distance():
    # New York to Los Angeles is ~2,445 miles in a straight line.
    miles = features.haversine_miles(40.7128, -74.0060, 34.0522, -118.2437)
    assert miles == pytest.approx(2445, rel=0.01)


@pytest.mark.parametrize(
    ("day", "expected"),
    [("2025-12-31", 1.0), ("2025-12-17", 0.5), ("2025-12-03", 0.0), ("2025-11-30", 0.0), ("2025-09-30", 1.0)],
)
def test_quarter_end_proximity(synthetic_loads, day, expected):
    frame = synthetic_loads.head(1).assign(date=pd.Timestamp(day), weight_missing=0)
    assert features.build_features(frame)["quarter_end_proximity"].iloc[0] == pytest.approx(expected)


def test_quote_signal_is_never_a_feature(synthetic_loads):
    X = features.build_features(synthetic_loads.assign(weight_missing=0))
    assert "quote_signal" not in X.columns
    assert "quote_signal" not in models.TREE_FEATURES


@requires_data
def test_train_validation_and_december_get_identical_complete_features():
    train, validation, december = data.load_train_test(), data.load_validation(), data.load_december_inputs()
    tables = data.build_reference_tables(train, validation)
    built = [features.build_features(data.clean_features(frame, tables)) for frame in (train, validation, december)]

    assert built[0].columns.tolist() == built[1].columns.tolist() == built[2].columns.tolist()
    for X in built:
        assert not X[models.TREE_FEATURES].isna().any().any()
    # Every December day gets its own market_index, so the chart is not flat by construction.
    assert built[2]["market_index"].nunique() == 31
