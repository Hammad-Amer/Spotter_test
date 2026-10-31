"""Project paths and constants shared across the pipeline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
TRAIN_TEST_PATH = DATA_DIR / "train_test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_INPUTS_PATH = DATA_DIR / "december_chart_inputs.csv"

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_PATH = MODELS_DIR / "model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
BACKTEST_RESULTS_PATH = REPORTS_DIR / "backtest_results.csv"
VALIDATION_PREDICTIONS_PATH = ROOT / "validation_predictions.csv"
# Spotter's README asks for predicted_rate to be filled in the December inputs file itself.
DECEMBER_PREDICTIONS_PATH = DECEMBER_INPUTS_PATH

ID_COLUMN = "load_id"
TARGET = "posted_rate"
RANDOM_SEED = 42

EQUIPMENT_TYPES = ["Dry Van", "Flatbed", "Reefer"]

# Weights are capped at 47,500 lb in the source data; negative values are sign errors.
WEIGHT_CAP = 47_500.0

# A load is a price outlier when its log rate is this far from the robust expected rate
# (log(1.8) ~ 0.59, i.e. 1.8x too high or too low). Normal noise is about +/-10%.
OUTLIER_LOG_THRESHOLD = 0.59
