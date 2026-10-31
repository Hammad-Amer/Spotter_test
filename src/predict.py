"""Predict data/validation.csv and data/december_chart_inputs.csv with the saved model.

Run: python -m src.predict
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src import config, data, features

# Plausible $/mile range for a clean load; cleaned training rates sit between ~$1.6 and ~$3.6 per mile.
RATE_PER_MILE_BOUNDS = (0.8, 8.0)


def predict_rates(model, tables: data.ReferenceTables, frame: pd.DataFrame) -> np.ndarray:
    cleaned = data.clean_features(frame, tables)
    return np.exp(model.predict(features.build_features(cleaned)))


def check_rates(rates, distance, label: str) -> None:
    rate_per_mile = np.asarray(rates) / np.asarray(distance)
    low, high = RATE_PER_MILE_BOUNDS
    if not np.isfinite(rates).all() or (rate_per_mile < low).any() or (rate_per_mile > high).any():
        raise ValueError(f"{label}: predictions outside the plausible ${low}-${high} per mile range")


def main() -> None:
    bundle = joblib.load(config.MODEL_PATH)
    model, tables = bundle["model"], bundle["tables"]

    validation = data.load_validation()
    rates = predict_rates(model, tables, validation)
    check_rates(rates, validation["distance"], "validation")
    predicted = pd.DataFrame({config.ID_COLUMN: validation[config.ID_COLUMN], "predicted_rate": rates.round(2)})
    template = pd.read_csv(config.TEMPLATE_PATH, usecols=[config.ID_COLUMN])
    submission = template.merge(predicted, on=config.ID_COLUMN, how="left", validate="one_to_one")
    if submission["predicted_rate"].isna().any():
        raise ValueError("template contains load_ids that are missing from validation.csv")
    submission.to_csv(config.VALIDATION_PREDICTIONS_PATH, index=False)
    print(
        f"Wrote {len(submission):,} predictions to {config.VALIDATION_PREDICTIONS_PATH} "
        f"(median ${submission['predicted_rate'].median():,.0f}, "
        f"median ${np.median(rates / validation['distance']):.2f}/mile)"
    )

    # Fill the December file without re-parsing it, so its original columns and formatting are kept.
    december = pd.read_csv(config.DECEMBER_INPUTS_PATH)
    december_rates = predict_rates(model, tables, data.load_december_inputs())
    check_rates(december_rates, december["distance"], "December")
    december["predicted_rate"] = december_rates.round(2)
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    december.to_csv(config.DECEMBER_PREDICTIONS_PATH, index=False)
    print(
        f"Wrote {len(december)} December predictions to {config.DECEMBER_PREDICTIONS_PATH} "
        f"(${december['predicted_rate'].min():,.0f} to ${december['predicted_rate'].max():,.0f})"
    )


if __name__ == "__main__":
    main()
