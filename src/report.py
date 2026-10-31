"""Build the written report (reports/report.docx and .pdf) from the saved results and figures.

Run after train, diagnostics, predict, score.py and the EDA notebook:  python -m src.report
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

from src import config, data, split

DOCX_PATH = config.REPORTS_DIR / "report.docx"
PDF_PATH = config.REPORTS_DIR / "report.pdf"
DECEMBER_CHART = config.ROOT / "scorer_results" / "candidate_december.png"
MUTED = RGBColor(0x52, 0x51, 0x4E)


def money(value: float) -> str:
    return f"${value:,.0f}"


def add_table(document, frame: pd.DataFrame, widths=None) -> None:
    table = document.add_table(rows=1, cols=len(frame.columns))
    table.style = "Light List Accent 1"
    for cell, column in zip(table.rows[0].cells, frame.columns):
        cell.text = str(column)
    for _, row in frame.iterrows():
        for cell, value in zip(table.add_row().cells, row):
            cell.text = "" if pd.isna(value) else str(value)
    for row in table.rows:
        for index, cell in enumerate(row.cells):
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(9)
            if widths:
                cell.width = Inches(widths[index])
    document.add_paragraph()


def add_figure(document, path, caption: str, width: float = 6.3) -> None:
    document.add_picture(str(path), width=Inches(width))
    document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    note = document.add_paragraph(caption)
    note.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in note.runs:
        run.font.size = Pt(9)
        run.font.italic = True
        run.font.color.rgb = MUTED


def bullets(document, items) -> None:
    for item in items:
        document.add_paragraph(item, style="List Bullet")


def data_quality_table(train: pd.DataFrame, validation: pd.DataFrame, n_outliers: int) -> pd.DataFrame:
    train_cities = set(train["pickup"]) | set(train["delivery"])
    unseen_rows = (~validation["pickup"].isin(train_cities) | ~validation["delivery"].isin(train_cities)).sum()
    unseen_cities = len((set(validation["pickup"]) | set(validation["delivery"])) - train_cities)
    rows = [
        ("Negative weight (sign error)", (train["weight"] < 0).sum(), (validation["weight"] < 0).sum(), "Absolute value"),
        ("Missing weight", train["weight"].isna().sum(), validation["weight"].isna().sum(), "Median for the equipment type + a missing flag"),
        ("Missing market_index", train["market_index"].isna().sum(), validation["market_index"].isna().sum(), "Median market_index of the same day"),
        ("Broken prices (1.8x+ off expected)", n_outliers, "-", "Excluded from training; reported in raw test scores"),
        ("quote_signal unreliable / leaky", "all rows", "all rows", "Not used as a feature (section 3.4)"),
        (f"Cities not seen in training ({unseen_cities})", "-", unseen_rows, "Model uses coordinates, not city names"),
        ("December file lacks coordinates and market_index", "-", "31 rows", "City coordinate lookup; daily market_index from validation inputs"),
    ]
    return pd.DataFrame(rows, columns=["Issue", "train_test", "validation", "How it was handled"])


def build() -> None:
    metadata = json.loads(config.METADATA_PATH.read_text())
    diagnostics = json.loads((config.REPORTS_DIR / "diagnostics.json").read_text())
    results = pd.read_csv(config.BACKTEST_RESULTS_PATH)
    december = pd.read_csv(config.DECEMBER_PREDICTIONS_PATH)
    train, validation = data.load_train_test(), data.load_validation()

    summary = results.groupby("model").agg(
        MAE=("MAE", "mean"), worst=("MAE", "max"), RMSE=("RMSE", "mean"), MAPE=("MAPE", "mean"),
        bias=("bias_pct", "mean"), raw_MAE=("raw_MAE", "mean"), raw_RMSE=("raw_RMSE", "mean"),
    ).sort_values("MAE")
    selected_bias = results.loc[results["model"] == metadata["selected_model"], "bias_pct"]
    selected = metadata["selected_model"]
    best = summary.loc[selected]
    naive = summary.loc["Naive $/mile"]
    no_level = summary.loc["LightGBM"]
    level = metadata["market_level"]

    document = Document()
    style = document.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10.5)

    document.add_heading("Freight Rate Prediction", level=0)
    byline = document.add_paragraph(f"Spotter Machine Learning Engineer assessment  |  {date.today():%d %B %Y}")
    byline.runs[0].font.color.rgb = MUTED

    document.add_heading("1. Summary", level=1)
    bullets(
        document,
        [
            "Task: predict posted_rate for 12,000 loads in Nov-Dec 2025, learning from 48,000 labelled loads in Jan-Oct 2025.",
            f"Validation: three rolling, time-based folds that each train on the past and predict the next two months, "
            f"the same shape as the real task. The selected model averages {money(best['MAE'])} MAE ({best['MAPE']:.1f}% MAPE) "
            f"on clean test loads, with a worst fold of {money(best['worst'])}; a naive $/mile baseline scores {money(naive['MAE'])}.",
            f"Model: an average of LightGBM (70%) and a log-linear model (30%) on log(rate), plus a market price level shared "
            f"by all loads on a day (recent level + trend + quarter-end ramp). The market level is the biggest gain: LightGBM "
            f"alone scores {money(no_level['MAE'])}.",
            "Key data findings: quote_signal leaks the answer in some months, mirrors it in others and is pure noise in Nov-Dec, "
            f"so it is not used; {metadata['excluded_outliers']:,} broken prices (1.4%) are excluded from training; "
            "eight validation cities never appear in training and are priced from coordinates.",
        ],
    )

    document.add_heading("2. Data and data quality", level=1)
    document.add_paragraph(
        "train_test.csv holds 48,000 loads from 1 January to 31 October 2025 with the posted rate. validation.csv holds "
        "12,000 loads from 1 November to 31 December 2025 without it. Rates are close to multiplicative in their drivers "
        "(shorter trips cost more per mile; Reefer > Flatbed > Dry Van; heavier loads cost slightly more), so all models "
        "work on log(rate)."
    )
    add_table(document, data_quality_table(train, validation, metadata["excluded_outliers"]), widths=[2.0, 0.8, 0.8, 2.7])
    document.add_paragraph(
        "Broken prices were found by comparing each load's log rate with a robust log-linear estimate. Normal loads sit "
        "within about ±15% of it, the broken ones 2-7x away, with no loads in between, so the 1.8x cut-off is not sensitive."
    )
    add_figure(document, config.FIGURES_DIR / "outlier_residuals.png", "Figure 1. Distance of each load's rate from its expected rate (log scale).")

    document.add_heading("3. Validation approach and data split", level=1)
    document.add_heading("3.1 Why the split is by time, not random", level=2)
    document.add_paragraph(
        "The real task is a forecast: train on Jan-Oct and predict Nov-Dec. A random split puts loads from the same days in "
        "both train and test, so the model is tested on market conditions it has already seen, and it rewards features "
        "like quote_signal whose meaning changes over time. The table compares the same LightGBM model (without the market "
        "level) on both splits."
    )
    comparison = pd.DataFrame(diagnostics["split_comparison"]).rename(columns={"split": "Split", "features": "Features", "MAE": "MAE ($)"})
    add_table(document, comparison, widths=[2.2, 2.0, 1.0])
    document.add_paragraph(
        "The random split reports errors 2-3x lower than the time split; it would have selected the wrong features and "
        "overstated accuracy."
    )

    document.add_heading("3.2 Rolling time-based folds", level=2)
    dates = train["date"]
    folds = []
    for fold in split.FOLDS:
        train_mask, test_mask = split.fold_masks(dates, fold)
        folds.append(
            {
                "Fold": fold.name,
                "Train": f"{dates[train_mask].min():%d %b} - {dates[train_mask].max():%d %b}",
                "Test": f"{dates[test_mask].min():%d %b} - {dates[test_mask].max():%d %b}",
                "Train loads": f"{train_mask.sum():,}",
                "Test loads": f"{test_mask.sum():,}",
            }
        )
    add_table(document, pd.DataFrame(folds), widths=[1.6, 1.3, 1.3, 1.0, 1.0])
    bullets(
        document,
        [
            "Each fold trains only on data before the test period and predicts the following two months, so the horizon "
            "matches Nov-Dec. Three folds give three independent tests across different market conditions.",
            "Models are compared on the mean and the worst fold, so a model that is lucky on one period is not preferred.",
            "Broken prices are removed from training only. Test scores are reported on clean loads (the model's real accuracy) "
            "and on all loads ('raw'), because the hidden validation labels may contain the same kind of errors.",
            "MAE is the main metric (dollars per load); RMSE and MAPE are reported too, as the scoring metric is not specified.",
            "After selection, the final model is retrained once on all clean Jan-Oct loads.",
        ],
    )

    document.add_heading("3.3 Leakage checks", level=2)
    bullets(
        document,
        [
            "Lookup tables (daily market_index medians, city coordinates, weight medians) are built from input columns only, "
            "never from posted_rate. Validation inputs are used only to extend the daily market_index table to Nov-Dec, "
            "which needs no labels.",
            "The models and the market price level are fitted inside each fold from that fold's training data only.",
            "Broken prices are flagged once on all Jan-Oct loads. Flagging each fold's training data on its own picks "
            "exactly the same loads (275, 400 and 533), because no load sits between normal noise and the 1.8x cut-off.",
        ],
    )

    document.add_heading("3.4 The quote_signal trap", level=2)
    qs = pd.DataFrame(diagnostics["quote_signal_by_month"])
    august = qs[qs["month"] == "2025-08"].iloc[0]
    document.add_paragraph(
        "quote_signal changes behaviour week by week: in Jan-Mar, Jun and Sep it equals price per mile (correlation +0.996, "
        "a leak of the answer); in Apr, May, Jul and Oct it mirrors it (-0.99); in August it is noise (-0.02). Validation "
        "has no labels, but in every labelled month quote_signal is also strongly correlated with distance (because price "
        "per mile is), so that correlation shows the state without the answer. For November and December it is "
        f"{qs.iloc[-2]['corr_quote_signal_distance']:+.2f} and {qs.iloc[-1]['corr_quote_signal_distance']:+.2f}, the same as "
        f"August ({august['corr_quote_signal_distance']:+.2f}): noise. On August, a model using quote_signal scored "
        f"{money(august['MAE_with_quote_signal'])} MAE against {money(august['MAE_without_quote_signal'])} without it, so the feature is dropped."
    )
    add_figure(document, config.FIGURES_DIR / "quote_signal_regimes.png", "Figure 2. quote_signal by week (top) and the label-free check by month (bottom).")

    document.add_heading("3.5 Robustness to unseen cities", level=2)
    cities = pd.DataFrame(diagnostics["unseen_cities"])
    document.add_paragraph(
        f"Eight training cities ({', '.join(cities['hidden_cities'].iloc[0])}) were hidden from training and the models "
        "scored on loads touching them, mirroring the eight new cities in validation."
    )
    add_table(document, cities[["model", "MAE_on_unseen_city_loads"]].rename(columns={"model": "Model", "MAE_on_unseen_city_loads": "MAE on unseen-city loads ($)"}), widths=[3.0, 2.0])

    document.add_heading("4. Model", level=1)
    document.add_heading("4.1 Load model", level=2)
    document.add_paragraph(
        "Two models predict log(rate) from the load's own attributes, and their predictions are averaged (70/30):"
    )
    bullets(
        document,
        [
            "LightGBM with these inputs: " + ", ".join(metadata["tree_features"]) + ". City names are left out so it "
            "generalises to new cities; coordinates, straight-line distance and the lat/lon deltas carry the lane geography. "
            "Default-style settings (1,500 trees, learning rate 0.03, 31 leaves); tuning them changed backtest MAE by under $1.",
            "A log-linear Ridge model on log distance, log weight, log market_index, equipment, day of week and one-hot pickup "
            "and delivery cities. It captures the multiplicative structure cleanly; for an unseen city it falls back to no city effect.",
        ],
    )
    document.add_heading("4.2 Market price level", level=2)
    document.add_paragraph(
        "Prices for all loads on a given day move together in a way market_index does not explain. The model separates "
        "this daily level from the load model by alternating between the two fits, measures it on out-of-fold predictions "
        "(so the load model cannot have memorised the day), then describes it with three parts:"
    )
    bullets(
        document,
        [
            f"Recent level: the average of the last {metadata['level_window_days']} days of training, carried forward "
            f"({np.expm1(level['recent_level_log']):+.1%} against the Jan-Oct average).",
            f"Trend: prices drifted upward through the year by {np.expm1(level['trend_per_month_log']):+.2%} a month; the trend is continued from the recent level.",
            f"Quarter-end ramp: prices climb {np.expm1(level['ramp_effect_log']):+.1%} over the last four weeks of each quarter "
            "and reset when the next starts (seen at the end of March, June and September). December ends a quarter.",
        ],
    )
    add_figure(document, config.FIGURES_DIR / "market_index_and_level.png", "Figure 3. Daily market_index (top) and the estimated daily price level (bottom).")

    document.add_heading("4.3 Model comparison", level=2)
    table = summary.reset_index().rename(columns={"model": "Model", "worst": "Worst fold MAE", "bias": "Bias", "raw_MAE": "Raw MAE", "raw_RMSE": "Raw RMSE"})
    for column in ["MAE", "Worst fold MAE", "RMSE", "Raw MAE", "Raw RMSE"]:
        table[column] = table[column].map(money)
    table["MAPE"] = table["MAPE"].map("{:.2f}%".format)
    table["Bias"] = table["Bias"].map("{:+.1f}%".format)
    add_table(document, table[["Model", "MAE", "Worst fold MAE", "RMSE", "MAPE", "Bias", "Raw MAE", "Raw RMSE"]], widths=[1.9, 0.65, 0.8, 0.65, 0.6, 0.6, 0.7, 0.75])
    document.add_paragraph(metadata["selection_reason"])
    document.add_paragraph(
        "Bias is the average log error on clean loads (positive means predictions too low). Raw RMSE is dominated by the "
        "broken prices, which no model can predict; on clean loads the error is about 2% of the rate."
    )
    add_figure(document, config.FIGURES_DIR / "backtest_mae.png", "Figure 4. Mean MAE over the three folds (bar) and worst fold (tick).")

    document.add_heading("5. December prediction chart", level=1)
    add_figure(document, DECEMBER_CHART, "Figure 5. Chart produced by the provided score.py for the fixed December inputs.")
    document.add_paragraph(
        f"Only the date changes (Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb). Predictions run from "
        f"{money(december['predicted_rate'].min())} to {money(december['predicted_rate'].max())}. Each day uses that day's "
        "market_index from the validation inputs, which has a weekly cycle (peaking mid-week), so the line rises and falls "
        "weekly. From early December the quarter-end ramp lifts prices towards 31 December. The 32 past loads on this lane "
        "(all equipment types) were priced between about $790 and $920 per month on median. No holiday-specific effect is "
        "assumed: training contains no December and showed no effect around Memorial Day or 4 July."
    )

    document.add_heading("6. Limitations and next steps", level=1)
    bullets(
        document,
        [
            "The scoring metric is not stated. The model targets typical (median-like) error in log space; if raw RMSE is used, "
            "broken prices in the validation labels will dominate any model's score.",
            "Nov-Dec level is a forecast. The trend and quarter-end ramp are extrapolated from three quarters; a year-end "
            "holiday effect, if any, cannot be learned from this data. In production the level would be re-estimated weekly "
            "from the latest posted rates.",
            f"The selected model under-predicted by {selected_bias.min():+.1f}% to {selected_bias.max():+.1f}% across the three "
            "folds: prices rose a little faster than the straight-line trend. No correction factor was added, because three "
            "folds are too few to estimate one reliably; a damped or piecewise trend is the next thing to test.",
            "Next steps: prediction intervals (quantile LightGBM) for pricing risk; lane-level history features once more "
            "data exists; find the source of quote_signal and fix the pipeline that produces it.",
        ],
    )

    document.add_heading("Appendix: reproduce", level=1)
    for command in [
        "python -m pip install -r requirements-dev.txt",
        "python -m src.train",
        "python -m src.diagnostics",
        "python -m src.predict",
        "python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv",
        "python -m pytest",
    ]:
        paragraph = document.add_paragraph(command)
        paragraph.runs[0].font.name = "Consolas"
        paragraph.runs[0].font.size = Pt(9)

    document.save(DOCX_PATH)
    print(f"Saved {DOCX_PATH}")


def export_pdf() -> None:
    from docx2pdf import convert

    convert(str(DOCX_PATH), str(PDF_PATH))
    print(f"Saved {PDF_PATH}")


if __name__ == "__main__":
    build()
    export_pdf()
