"""Feature engineering shared by training, validation and December chart inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.offsets import QuarterEnd

from src import config
from src.data import COORDINATE_COLUMNS

EARTH_RADIUS_MILES = 3958.8

# Rates rise ~2% over the last ~10 days of each quarter (end of Mar, Jun and Sep in training).
QUARTER_END_RAMP_DAYS = 10


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(a))


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Model inputs from a cleaned frame.

    quote_signal is deliberately left out: its relationship with the rate flips between months and is
    pure noise in Nov-Dec. City names are kept only for the linear model; the tree model uses
    coordinates so it can price cities it has never seen.
    """
    X = pd.DataFrame(index=frame.index)
    X["distance"] = frame["distance"]
    X["log_distance"] = np.log(frame["distance"])
    X["equipment"] = pd.Categorical(frame["equipment"], categories=config.EQUIPMENT_TYPES)
    X["weight"] = frame["weight"]
    X["weight_missing"] = frame["weight_missing"]
    X["market_index"] = frame["market_index"]
    for column in COORDINATE_COLUMNS:
        X[column] = frame[column]
    X["straight_line_miles"] = haversine_miles(
        frame["pickup_lat"], frame["pickup_lon"], frame["delivery_lat"], frame["delivery_lon"]
    )
    X["delta_lat"] = frame["delivery_lat"] - frame["pickup_lat"]
    X["delta_lon"] = frame["delivery_lon"] - frame["pickup_lon"]
    X["day_of_week"] = frame["date"].dt.dayofweek
    days_to_quarter_end = (frame["date"] + QuarterEnd(0) - frame["date"]).dt.days
    # 0 until 10 days before the quarter ends, rising linearly to 1 on its last day.
    X["quarter_end_proximity"] = (1 - days_to_quarter_end / QUARTER_END_RAMP_DAYS).clip(lower=0)
    X["pickup"] = frame["pickup"]
    X["delivery"] = frame["delivery"]
    return X
