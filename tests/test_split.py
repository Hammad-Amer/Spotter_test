import pandas as pd
import pytest

from src import split


@pytest.mark.parametrize("fold", split.FOLDS, ids=lambda fold: fold.name)
def test_folds_train_strictly_before_test(fold):
    dates = pd.Series(pd.date_range("2025-01-01", "2025-10-31", freq="D"))
    train, test = split.fold_masks(dates, fold)

    assert train.any() and test.any()
    assert not (train & test).any()
    assert dates[train].max() < dates[test].min()
    # Each fold forecasts roughly two months ahead, like Nov-Dec after Jan-Oct.
    assert 59 <= test.sum() <= 62


def test_regression_metrics_known_values():
    metrics = split.regression_metrics([100, 200], [110, 180])
    assert metrics["MAE"] == pytest.approx(15)
    assert metrics["RMSE"] == pytest.approx((250) ** 0.5)
    assert metrics["MAPE"] == pytest.approx(10)
