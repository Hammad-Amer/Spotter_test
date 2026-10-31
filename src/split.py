"""Time-based train/test splitting and evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    """Train on everything up to train_end, test on the following days up to test_end."""

    name: str
    train_end: str
    test_end: str


# Each fold mirrors the real task: learn from the past, predict the next two months.
FOLDS = [
    Fold("Jan-Apr -> May-Jun", "2025-04-30", "2025-06-30"),
    Fold("Jan-Jun -> Jul-Aug", "2025-06-30", "2025-08-31"),
    Fold("Jan-Aug -> Sep-Oct", "2025-08-31", "2025-10-31"),
]


def fold_masks(dates: pd.Series, fold: Fold) -> tuple[pd.Series, pd.Series]:
    train_end = pd.Timestamp(fold.train_end)
    train = dates <= train_end
    test = (dates > train_end) & (dates <= pd.Timestamp(fold.test_end))
    return train, test


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAPE": float(100 * np.mean(np.abs(error) / y_true)),
    }
