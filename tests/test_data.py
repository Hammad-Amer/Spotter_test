import numpy as np
import pandas as pd
import pytest

from src import config, data


def test_clean_fixes_negative_capped_and_missing_weights(synthetic_loads):
    frame = synthetic_loads.copy()
    frame.loc[0, "weight"] = -30_000
    frame.loc[1, "weight"] = np.nan
    frame.loc[2, "weight"] = 60_000
    tables = data.build_reference_tables(frame, frame.iloc[:0])

    cleaned = data.clean_features(frame, tables)

    assert cleaned.loc[0, "weight"] == 30_000
    assert cleaned.loc[1, "weight"] == tables.weight_median[frame.loc[1, "equipment"]]
    assert cleaned.loc[2, "weight"] == config.WEIGHT_CAP
    assert cleaned["weight_missing"].tolist()[:3] == [0, 1, 0]
    assert (cleaned["weight"] > 0).all()


def test_clean_fills_market_index_with_daily_median(synthetic_loads):
    frame = synthetic_loads.copy()
    day = frame.loc[0, "date"]
    frame.loc[0, "market_index"] = np.nan
    tables = data.build_reference_tables(frame, frame.iloc[:0])

    cleaned = data.clean_features(frame, tables)

    same_day = frame.loc[(frame["date"] == day) & frame["market_index"].notna(), "market_index"]
    assert cleaned.loc[0, "market_index"] == pytest.approx(same_day.median())


def test_clean_adds_missing_columns_for_december_style_rows(synthetic_loads):
    tables = data.build_reference_tables(synthetic_loads, synthetic_loads.iloc[:0])
    day = synthetic_loads["date"].iloc[0]
    december_style = pd.DataFrame(
        {"pickup": ["Alpha"], "delivery": ["Bravo"], "distance": [400.0], "equipment": ["Dry Van"], "weight": [32_000.0], "date": [day]}
    )

    cleaned = data.clean_features(december_style, tables)

    assert cleaned.loc[0, ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]].tolist() == [35.0, -90.0, 40.0, -85.0]
    assert cleaned.loc[0, "market_index"] == pytest.approx(tables.daily_market_index[day])


def test_clean_fails_loudly_for_unknown_city_without_coordinates(synthetic_loads):
    tables = data.build_reference_tables(synthetic_loads, synthetic_loads.iloc[:0])
    unknown = pd.DataFrame(
        {"pickup": ["Nowhere"], "delivery": ["Bravo"], "distance": [400.0], "equipment": ["Dry Van"], "weight": [32_000.0], "date": [synthetic_loads["date"].iloc[0]]}
    )

    with pytest.raises(ValueError, match="pickup_lat"):
        data.clean_features(unknown, tables)


def test_flag_price_outliers_catches_planted_errors_only(synthetic_loads):
    frame = synthetic_loads.copy()
    planted_high, planted_low = [10, 20, 30], [40, 50, 60]
    frame.loc[planted_high, config.TARGET] *= 4
    frame.loc[planted_low, config.TARGET] *= 0.25
    cleaned = data.clean_features(frame, data.build_reference_tables(frame, frame.iloc[:0]))

    flagged = data.flag_price_outliers(cleaned)

    assert sorted(flagged[flagged].index) == planted_high + planted_low
