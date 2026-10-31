import numpy as np
import pandas as pd
import pytest

from src import config

CITIES = {
    "Alpha": (35.0, -90.0),
    "Bravo": (40.0, -85.0),
    "Charlie": (33.0, -100.0),
    "Delta": (42.0, -75.0),
}

requires_data = pytest.mark.skipif(not config.TRAIN_TEST_PATH.exists(), reason="provided data files are not in data/")


@pytest.fixture
def synthetic_loads() -> pd.DataFrame:
    """Loads priced by a known multiplicative formula with 2% noise, in the provided file layout."""
    rng = np.random.default_rng(0)
    n = 3000
    names = list(CITIES)
    pickup = rng.choice(names, n)
    delivery = np.array([rng.choice([c for c in names if c != p]) for p in pickup])
    dates = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 120, n), unit="D")
    frame = pd.DataFrame(
        {
            "load_id": [f"TR-{i:06d}" for i in range(n)],
            "pickup": pickup,
            "delivery": delivery,
            "pickup_lat": [CITIES[c][0] for c in pickup],
            "pickup_lon": [CITIES[c][1] for c in pickup],
            "delivery_lat": [CITIES[c][0] for c in delivery],
            "delivery_lon": [CITIES[c][1] for c in delivery],
            "distance": rng.uniform(100, 2500, n).round(1),
            "equipment": rng.choice(config.EQUIPMENT_TYPES, n),
            "weight": rng.uniform(10_000, 45_000, n).round(0),
            "date": dates,
            "market_index": rng.uniform(0.8, 1.2, n),
            "quote_signal": rng.uniform(1.5, 2.5, n),
        }
    )
    equipment_factor = frame["equipment"].map({"Dry Van": 1.0, "Flatbed": 1.08, "Reefer": 1.13})
    rate_per_mile = 2.9 * (frame["distance"] / 100) ** -0.13 * equipment_factor * frame["market_index"] ** 0.14
    frame[config.TARGET] = (frame["distance"] * rate_per_mile * np.exp(rng.normal(0, 0.02, n))).round(2)
    return frame
