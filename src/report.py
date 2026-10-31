"""Build the written report (reports/report.docx and .pdf) from the saved results and figures.

Run after train, diagnostics, predict, score.py and the EDA notebook:  python -m src.report
"""

from __future__ import annotations

import json
import re
from datetime import date

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from src import config, data, split
from src.train import prepare_training_data

DOCX_PATH = config.REPORTS_DIR / "report.docx"
PDF_PATH = config.REPORTS_DIR / "report.pdf"
DECEMBER_CHART = config.ROOT / "scorer_results" / "candidate_december.png"

AUTHOR = "Hammad Amer"
TITLE = "Freight Rate Prediction"
SUBTITLE = "Spotter Machine Learning Engineer Assessment"
FONT = "Calibri"
CODE_FONT = "Consolas"
BLACK = RGBColor(0, 0, 0)
TEXT_WIDTH = 6.5  # inches: US Letter with 1-inch margins
NUMERIC = re.compile(r"^[$+\-−]?[\d,.]+%?$")


class Report:
    """A plain, black-and-white document with numbered tables and figures."""

    def __init__(self):
        self.document = Document()
        self.tables = 0
        self.figures = 0
        self._configure()

    def _configure(self) -> None:
        for section in self.document.sections:
            section.top_margin = section.bottom_margin = Inches(1)
            section.left_margin = section.right_margin = Inches(1)
        styles = self.document.styles
        _set_style(styles["Normal"], FONT, 10.5)
        styles["Normal"].paragraph_format.space_after = Pt(6)
        styles["Normal"].paragraph_format.line_spacing = 1.15
        for name, size, before, after in [("Heading 1", 13, 18, 6), ("Heading 2", 11, 12, 4)]:
            style = styles[name]
            _set_style(style, FONT, size, bold=True)
            style.paragraph_format.space_before = Pt(before)
            style.paragraph_format.space_after = Pt(after)
            style.paragraph_format.keep_with_next = True
        _set_style(styles["Footer"], FONT, 9)
        _add_page_number(self.document.sections[0])

    def title_block(self) -> None:
        today = date.today()
        for text, size, bold, after in [
            (TITLE, 20, True, 2),
            (SUBTITLE, 12, False, 2),
            (f"{AUTHOR}  |  {today.day} {today:%B %Y}", 10.5, False, 10),
        ]:
            paragraph = self.document.add_paragraph()
            paragraph.paragraph_format.space_after = Pt(after)
            run = paragraph.add_run(text)
            run.font.size, run.font.bold = Pt(size), bold
        _bottom_border(paragraph)
        properties = self.document.core_properties
        properties.author = properties.last_modified_by = AUTHOR
        properties.title, properties.subject = TITLE, SUBTITLE
        properties.comments = properties.keywords = ""

    def heading(self, text: str, level: int = 1) -> None:
        self.document.add_heading(text, level=level)

    def paragraph(self, text: str) -> None:
        self.document.add_paragraph(text)

    def equation(self, text: str) -> None:
        paragraph = self.document.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run(text)
        run.font.italic = True

    def table(self, frame: pd.DataFrame, caption: str, widths) -> None:
        """Caption above, then a table ruled top, below the header and at the bottom."""
        self.tables += 1
        self._caption(f"Table {self.tables}.", caption, keep_with_next=True)
        table = self.document.add_table(rows=1, cols=len(frame.columns))
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        _table_rules(table)
        rows = [list(frame.columns)] + [["" if pd.isna(value) else str(value) for value in row] for row in frame.itertuples(index=False)]
        for index, values in enumerate(rows):
            cells = table.rows[0].cells if index == 0 else table.add_row().cells
            for column, (cell, value) in enumerate(zip(cells, values)):
                cell.width = Inches(widths[column])
                paragraph = cell.paragraphs[0]
                paragraph.paragraph_format.space_before = paragraph.paragraph_format.space_after = Pt(2)
                paragraph.paragraph_format.line_spacing = 1.0
                paragraph.paragraph_format.keep_with_next = index < len(rows) - 1
                if column > 0 and all(NUMERIC.match(str(row[column])) or row[column] in ("", "-") for row in rows[1:]):
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                run = paragraph.add_run(value)
                run.font.size = Pt(9)
                run.font.bold = index == 0
            if index == 0:
                for cell in cells:
                    _cell_bottom_rule(cell)
        self.document.add_paragraph().paragraph_format.space_after = Pt(2)

    def figure(self, path, caption: str, width: float = 6.2) -> None:
        self.figures += 1
        self.document.add_picture(str(path), width=Inches(width))
        picture = self.document.paragraphs[-1]
        picture.alignment = WD_ALIGN_PARAGRAPH.CENTER
        picture.paragraph_format.keep_with_next = True
        self._caption(f"Figure {self.figures}.", caption)

    def code(self, lines) -> None:
        for line in lines:
            paragraph = self.document.add_paragraph()
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.left_indent = Inches(0.25)
            run = paragraph.add_run(line)
            run.font.name, run.font.size = CODE_FONT, Pt(9)
        paragraph.paragraph_format.space_after = Pt(6)

    def _caption(self, label: str, text: str, keep_with_next: bool = False) -> None:
        paragraph = self.document.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_after = Pt(8)
        paragraph.paragraph_format.keep_with_next = keep_with_next
        for part, bold in [(label + " ", True), (text, False)]:
            run = paragraph.add_run(part)
            run.font.size, run.font.bold = Pt(9), bold

    def save(self) -> None:
        self.document.save(DOCX_PATH)
        print(f"Saved {DOCX_PATH}")


def _set_style(style, name: str, size: float, bold: bool = False) -> None:
    """Explicit black font, overriding the template's theme fonts and accent colours."""
    style.font.name = name
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.italic = False
    style.font.color.rgb = BLACK
    fonts = style.element.get_or_add_rPr().get_or_add_rFonts()
    for attribute in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        fonts.attrib.pop(qn(attribute), None)
    for attribute in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        fonts.set(qn(attribute), name)


def _border(tag: str, size: int = 0) -> OxmlElement:
    element = OxmlElement(f"w:{tag}")
    element.set(qn("w:val"), "single" if size else "nil")
    if size:
        element.set(qn("w:sz"), str(size))
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), "000000")
    return element


def _table_rules(table) -> None:
    borders = OxmlElement("w:tblBorders")
    for tag, size in [("top", 8), ("left", 0), ("bottom", 8), ("right", 0), ("insideH", 0), ("insideV", 0)]:
        borders.append(_border(tag, size))
    table._tbl.tblPr.insert_element_before(borders, "w:shd", "w:tblLayout", "w:tblCellMar", "w:tblLook", "w:tblCaption", "w:tblDescription")


def _cell_bottom_rule(cell) -> None:
    borders = OxmlElement("w:tcBorders")
    borders.append(_border("bottom", 4))
    cell._tc.get_or_add_tcPr().insert_element_before(borders, "w:shd", "w:noWrap", "w:tcMar", "w:textDirection", "w:tcFitText", "w:vAlign", "w:hideMark")


def _bottom_border(paragraph) -> None:
    borders = OxmlElement("w:pBdr")
    borders.append(_border("bottom", 6))
    # Word requires pPr children in schema order, so the border goes before spacing and justification.
    paragraph._p.get_or_add_pPr().insert_element_before(
        borders, "w:shd", "w:tabs", "w:suppressAutoHyphens", "w:spacing", "w:ind", "w:jc", "w:outlineLvl", "w:rPr"
    )


def _add_page_number(section) -> None:
    paragraph = section.footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    run = OxmlElement("w:r")
    text = OxmlElement("w:t")
    text.text = "1"
    run.append(text)
    field.append(run)
    paragraph._p.append(field)


def money(value: float) -> str:
    return f"${value:,.0f}"


def data_quality_table(train: pd.DataFrame, validation: pd.DataFrame, n_outliers: int) -> pd.DataFrame:
    train_cities = set(train["pickup"]) | set(train["delivery"])
    unseen_rows = (~validation["pickup"].isin(train_cities) | ~validation["delivery"].isin(train_cities)).sum()
    unseen_cities = len((set(validation["pickup"]) | set(validation["delivery"])) - train_cities)
    rows = [
        ("Negative weight", (train["weight"] < 0).sum(), (validation["weight"] < 0).sum(), "Sign error; absolute value taken"),
        ("Missing weight", train["weight"].isna().sum(), validation["weight"].isna().sum(), "Median for the equipment type, plus an indicator"),
        ("Missing market_index", train["market_index"].isna().sum(), validation["market_index"].isna().sum(), "Median market_index of the same day"),
        ("Erroneous rate (1.8x or more from expected)", f"{n_outliers:,}", "-", "Excluded from training; kept in raw test scores"),
        ("quote_signal unstable over time", "All rows", "All rows", "Not used (Section 3.4)"),
        (f"Cities absent from training ({unseen_cities})", "-", f"{unseen_rows:,} loads", "Coordinates used instead of city names"),
        ("December inputs lack coordinates and market_index", "-", "31 rows", "City coordinate lookup; daily market_index from validation inputs"),
    ]
    return pd.DataFrame(rows, columns=["Issue", "Training", "Validation", "Treatment"])


def build() -> None:
    metadata = json.loads(config.METADATA_PATH.read_text())
    diagnostics = json.loads((config.REPORTS_DIR / "diagnostics.json").read_text())
    results = pd.read_csv(config.BACKTEST_RESULTS_PATH)
    december = pd.read_csv(config.DECEMBER_PREDICTIONS_PATH)
    train, validation = data.load_train_test(), data.load_validation()
    cleaned, outlier, _ = prepare_training_data()

    summary = results.groupby("model").agg(
        MAE=("MAE", "mean"), worst=("MAE", "max"), RMSE=("RMSE", "mean"), MAPE=("MAPE", "mean"),
        bias=("bias_pct", "mean"), raw_MAE=("raw_MAE", "mean"), raw_RMSE=("raw_RMSE", "mean"),
    ).sort_values("MAE")
    selected = metadata["selected_model"]
    selected_bias = results.loc[results["model"] == selected, "bias_pct"]
    best, naive = summary.loc[selected], summary.loc["Naive $/mile"]
    lightgbm, lightgbm_level = summary.loc["LightGBM"], summary.loc["LightGBM + market level"]
    level = metadata["market_level"]

    clean = cleaned[~outlier]
    rate_per_mile = clean[config.TARGET] / clean["distance"]
    equipment_premium = rate_per_mile.groupby(clean["equipment"]).median() / rate_per_mile[clean["equipment"] == "Dry Van"].median() - 1
    lane = clean[(clean["pickup"] == "Lexington") & (clean["delivery"] == "Fort Wayne")][config.TARGET]
    fold_flags = []
    for fold in split.FOLDS:
        fit, _ = split.fold_masks(cleaned["date"], fold)
        per_fold = data.flag_price_outliers(cleaned[fit])
        fold_flags.append((int(per_fold.sum()), bool((per_fold == outlier[fit]).all())))

    report = Report()
    report.title_block()

    report.heading("1 Summary")
    report.paragraph(
        f"This report describes a model that predicts the posted rate of a truckload from its lane, distance, equipment, weight "
        f"and date. It was trained on {len(train):,} loads posted between January and October 2025 and used to price "
        f"{len(validation):,} loads from November and December 2025."
    )
    report.paragraph(
        "The model has two components. A load model, a 70/30 average of a LightGBM regressor and a log-linear regression, "
        "prices each load from its own attributes. A market price level, shared by all loads on a given day, captures how "
        "rates move over time: a gradual upward trend and a rise of about 4% over the last four weeks of each quarter."
    )
    report.paragraph(
        f"On three rolling backtests that each forecast two months ahead, the model has a mean absolute error (MAE) of "
        f"{money(best['MAE'])} per load ({best['MAPE']:.1f}% of the rate), with a worst fold of {money(best['worst'])}. "
        f"A median rate-per-mile baseline scores {money(naive['MAE'])}, and LightGBM without the market level "
        f"{money(lightgbm['MAE'])}."
    )
    report.paragraph(
        f"Two data issues shaped the approach. About {outlier.mean():.1%} of training rates ({metadata['excluded_outliers']:,} "
        "loads) are data-entry errors, two to seven times their expected value, and are excluded from training. The "
        "quote_signal column equals the rate per mile in some months, mirrors it in others and carries no information in "
        "November and December, so it is not used."
    )

    report.heading("2 Data and data quality")
    report.paragraph(
        f"train_test.csv contains {len(train):,} loads from 1 January to 31 October 2025, each with pickup and delivery city "
        "and coordinates, distance, equipment type, weight, posting date, a daily market_index, a quote_signal and the posted "
        f"rate. validation.csv contains {len(validation):,} loads from 1 November to 31 December 2025 with the same inputs "
        "and no rate. Table 1 lists the data-quality issues found and how each was handled."
    )
    report.table(
        data_quality_table(train, validation, metadata["excluded_outliers"]),
        "Data-quality issues and their treatment.",
        widths=[2.15, 0.75, 0.9, 2.7],
    )
    report.paragraph(
        "Erroneous rates were identified by comparing each load's log rate per mile with a ridge regression on lane, "
        "equipment, distance, weight, market_index and month, refitted once without the flagged loads so that they do "
        "not distort the estimate. Normal loads lie within about 20% of the estimate and erroneous ones two to seven times "
        "away, with no loads in between (Figure 1), so the 1.8x threshold is not sensitive."
    )
    report.figure(
        config.FIGURES_DIR / "outlier_residuals.png",
        "Absolute log ratio of each load's rate to its expected rate. Note the log-scaled count axis.",
    )

    report.heading("3 Validation approach and data split")
    report.heading("3.1 Splitting by time", level=2)
    report.paragraph(
        "The task is a forecast: the model is trained on January to October and applied to November and December. A random "
        "split places loads from the same days in both the training and test sets, so the test set contains no unseen "
        "market conditions, and it rewards features whose meaning drifts over time. Table 2 compares the two splits for "
        "the same LightGBM model without the market level."
    )
    comparison = pd.DataFrame(diagnostics["split_comparison"])
    comparison["MAE"] = comparison["MAE"].map("${:.1f}".format)
    comparison["split"] = comparison["split"].str.replace("->", "to", regex=False)
    report.table(
        comparison.rename(columns={"split": "Split", "features": "Features", "MAE": "MAE"}),
        "Test MAE of the same model under a random and a time-based split.",
        widths=[2.6, 2.4, 1.0],
    )
    report.paragraph(
        "The random split understates the error by a factor of two to three, and it favours quote_signal, which is harmful "
        "in a true forecast (Section 3.4). All model selection therefore uses time-based folds."
    )

    report.heading("3.2 Rolling time-based folds", level=2)
    dates = train["date"]
    folds = []
    for fold in split.FOLDS:
        train_mask, test_mask = split.fold_masks(dates, fold)
        folds.append(
            {
                "Fold": fold.name.replace("->", "to"),
                "Training period": f"{dates[train_mask].min():%d %b} to {dates[train_mask].max():%d %b}",
                "Test period": f"{dates[test_mask].min():%d %b} to {dates[test_mask].max():%d %b}",
                "Training loads": f"{train_mask.sum():,}",
                "Test loads": f"{test_mask.sum():,}",
            }
        )
    report.table(pd.DataFrame(folds), "Backtest folds.", widths=[1.6, 1.45, 1.45, 1.0, 0.9])
    report.paragraph(
        "Each fold trains on all data up to a cutoff and tests on the following two months, matching the November to "
        "December horizon (Table 3). Erroneous rates are removed from the training data only. Test error is reported on "
        "clean loads, which reflects the model's accuracy, and on all loads ('raw'), since the hidden validation labels "
        "may contain similar errors. Because the scoring metric is not specified, MAE, RMSE and MAPE are all reported. "
        "Models are selected on mean MAE and on worst-fold MAE, to avoid favouring a model that does well in a single "
        f"period. The selected model is then refitted on all {metadata['training_rows']:,} clean January to October loads."
    )

    report.heading("3.3 Leakage controls", level=2)
    counts = ", ".join(str(count) for count, _ in fold_flags[:-1]) + f" and {fold_flags[-1][0]}"
    flags_note = (
        f"flagging each fold's training data separately selects exactly the same loads ({counts}), because there is a wide "
        "gap between normal variation and the threshold."
        if all(same for _, same in fold_flags)
        else f"flagging each fold's training data separately selects {counts} loads, with small differences from the global flags."
    )
    report.paragraph(
        "Lookup tables (daily market_index medians, city coordinates and weight medians) are built from input columns only "
        "and never use posted_rate. Validation inputs are used for two purposes only: extending the daily market_index table "
        "to November and December, and locating cities that appear only in validation. Within each fold, the models and "
        "the market level are fitted on that fold's training period alone. Erroneous rates are flagged once on the full "
        f"January to October data; {flags_note}"
    )

    report.heading("3.4 quote_signal", level=2)
    qs = pd.DataFrame(diagnostics["quote_signal_by_month"])
    august = qs[qs["month"] == "2025-08"].iloc[0]
    labelled = qs[qs["corr_quote_signal_rate_per_mile"].notna() & (qs["month"] != "2025-08")]
    report.paragraph(
        "quote_signal behaves differently from month to month (Figure 2). In January to March, June and September it is "
        "almost exactly the rate per mile (correlation +0.996); in April, May, July and October it is its mirror image "
        "(-0.99); in August it is unrelated (-0.02). Its state in November and December cannot be checked against labels, "
        "but it can be inferred: in every other labelled month quote_signal is also strongly correlated with distance "
        f"(absolute correlation of about {labelled['corr_quote_signal_distance'].abs().min():.1f}), because the rate per mile "
        f"is. That correlation is {qs.iloc[-2]['corr_quote_signal_distance']:+.2f} in November and "
        f"{qs.iloc[-1]['corr_quote_signal_distance']:+.2f} in December, as in August "
        f"({august['corr_quote_signal_distance']:+.2f}). A model trained on the other months and tested on August has an "
        f"MAE of {money(august['MAE_with_quote_signal'])} with quote_signal and {money(august['MAE_without_quote_signal'])} "
        "without it, so the feature is excluded."
    )
    report.figure(
        config.FIGURES_DIR / "quote_signal_regimes.png",
        "Weekly correlation of quote_signal with the rate per mile (top) and the label-free check by month (bottom).",
        width=5.6,
    )

    report.heading("3.5 Unseen cities", level=2)
    cities = pd.DataFrame(diagnostics["unseen_cities"])
    hidden = cities["hidden_cities"].iloc[0]
    report.paragraph(
        "Eight validation cities never appear in training. To test how the models handle new cities, eight training cities "
        f"({', '.join(hidden[:-1])} and {hidden[-1]}) were withheld from training and the models were scored on the loads "
        "touching them (Table 4). The coordinate-based LightGBM model generalises better than the log-linear model with "
        "city indicators, and blending the two costs nothing on these loads."
    )
    unseen = cities[["model", "MAE_on_unseen_city_loads"]].copy()
    unseen["MAE_on_unseen_city_loads"] = unseen["MAE_on_unseen_city_loads"].map("${:.1f}".format)
    report.table(
        unseen.rename(columns={"model": "Model", "MAE_on_unseen_city_loads": "MAE on loads touching a withheld city"}),
        "Error on loads involving cities withheld from training.",
        widths=[3.2, 2.6],
    )

    report.heading("4 Model")
    report.heading("4.1 Load model", level=2)
    report.paragraph(
        "Rates are close to multiplicative in their drivers: shorter trips cost more per mile, Reefer and Flatbed loads cost "
        f"about {equipment_premium['Reefer']:.0%} and {equipment_premium['Flatbed']:.0%} more per mile than Dry Van, and "
        "heavier loads cost slightly more. All models therefore predict the logarithm of the rate."
    )
    report.paragraph(
        "The load model is a 70/30 average of two models. The first is LightGBM (1,500 trees, learning rate 0.03, 31 leaves) "
        "using distance and its logarithm, equipment, weight, a missing-weight indicator, market_index, pickup and delivery "
        "coordinates, straight-line distance, latitude and longitude differences, and day of week. City names are excluded "
        "so that the model can price unseen cities; tuning the hyperparameters changed backtest MAE by less than $1. The "
        "second is a ridge regression on log distance and its square, log weight and log market_index, with indicators for "
        "equipment, pickup city, delivery city and day of week. A city it has not seen receives no city effect."
    )

    report.heading("4.2 Market price level", level=2)
    report.paragraph(
        "After accounting for load attributes, the rates of all loads on a given day still move together in a way that "
        "market_index does not explain. This daily level is estimated jointly with the load model by alternating between the "
        "two fits for five iterations. Its final values come from out-of-fold predictions, with days assigned to five groups, "
        "so that the load model cannot absorb day-specific effects. The daily level shows two regular patterns (Figure 3): "
        f"an upward trend of {np.expm1(level['trend_per_month_log']):+.2%} per month, and a rise of "
        f"{np.expm1(level['ramp_effect_log']):.1%} over the last 28 days of each quarter that reverses on the first day of the "
        "next quarter. For a future date t, the level is forecast as"
    )
    report.equation("level(t) = L + b·(t − t₀) + r·q(t)")
    report.paragraph(
        f"where L is the mean level over the last {metadata['level_window_days']} days of training with the quarter-end "
        f"effect removed ({np.expm1(level['recent_level_log']):+.1%} relative to the January to October average), t₀ is the "
        "middle of that window, b is the trend, r is the quarter-end effect, and q(t) rises linearly from 0 to 1 over the "
        "last 28 days of the quarter. The level is modelled separately because tree models cannot extrapolate in time: a "
        "date beyond the training range would be treated as the last date seen."
    )
    report.figure(
        config.FIGURES_DIR / "market_index_and_level.png",
        "Daily market_index (top) and the estimated daily price level with the fitted and forecast level (bottom).",
    )

    report.heading("4.3 Results", level=2)
    table = summary.reset_index().rename(columns={"model": "Model", "worst": "Worst fold", "bias": "Bias", "raw_MAE": "Raw MAE", "raw_RMSE": "Raw RMSE"})
    for column in ["MAE", "Worst fold", "RMSE", "Raw MAE", "Raw RMSE"]:
        table[column] = table[column].map(money)
    table["MAPE"] = table["MAPE"].map("{:.2f}%".format)
    table["Bias"] = table["Bias"].map("{:+.1f}%".format)
    report.table(
        table[["Model", "MAE", "Worst fold", "RMSE", "MAPE", "Bias", "Raw MAE", "Raw RMSE"]],
        "Backtest results, mean over three folds. MAE, worst fold, RMSE, MAPE and bias are on clean test loads.",
        widths=[1.95, 0.6, 0.75, 0.6, 0.6, 0.55, 0.7, 0.75],
    )
    report.paragraph(
        f"The blend with the market level has the lowest mean and worst-fold MAE (Table 5, Figure 4). Adding the market level "
        f"is the largest single improvement, reducing LightGBM's error from {money(lightgbm['MAE'])} to "
        f"{money(lightgbm_level['MAE'])}. Blending with the log-linear model lowers it further to {money(best['MAE'])} and the "
        f"worst fold from {money(lightgbm_level['worst'])} to {money(best['worst'])}. Bias is the mean log error on clean "
        "loads, where a positive value means under-prediction. Raw RMSE is dominated by erroneous rates, which no model can "
        "predict, and is therefore similar across models."
    )
    report.figure(config.FIGURES_DIR / "backtest_mae.png", "Mean MAE over the three folds (bars) and worst fold (ticks).")

    report.heading("5 December prediction chart")
    report.figure(DECEMBER_CHART, "December predictions for the fixed load, as produced by the provided score.py.", width=5.8)
    report.paragraph(
        f"Only the date varies (Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb). Predictions range from "
        f"{money(december['predicted_rate'].min())} to {money(december['predicted_rate'].max())}. The weekly oscillation follows "
        "market_index, which peaks on Thursdays, and the upward slope reflects the trend and the quarter-end rise that "
        f"begins in early December. For reference, the {len(lane)} earlier loads on this lane, across all equipment types, "
        f"were priced between {money(lane.min())} and {money(lane.max())}. No holiday effect is assumed: the training data "
        "contain no December and show no effect around Memorial Day or 4 July."
    )

    report.heading("6 Limitations and next steps")
    report.paragraph(
        "The scoring metric is not specified. The model is optimised for typical error; under raw RMSE, erroneous rates in "
        "the validation labels would dominate the score of any model."
    )
    report.paragraph(
        "The November and December level is an extrapolation. The trend and the quarter-end rise are estimated from three "
        "quarters, and the data contain no December, so a year-end holiday effect cannot be learned. In production, the "
        "level would be re-estimated weekly from the most recent posted rates."
    )
    report.paragraph(
        f"The selected model under-predicted by {selected_bias.min():.1f}% to {selected_bias.max():.1f}% across the folds, "
        "because rates rose slightly faster than a linear trend. No correction was applied, since three folds are too few "
        "to estimate one reliably; a damped or piecewise trend is the next candidate."
    )
    report.paragraph(
        "Natural extensions are prediction intervals (for example, quantile LightGBM) to quantify pricing risk, lane-history "
        "features as more data accumulates, and tracing the source of quote_signal."
    )

    report.heading("Appendix A  Reproducing the results")
    report.paragraph("With the four provided files in data/, the following commands reproduce every number and figure in this report:")
    report.code(
        [
            "python -m pip install -r requirements-dev.txt",
            "python -m src.train",
            "python -m src.diagnostics",
            "python -m src.predict",
            "python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv",
            "python -m pytest",
            "python -m src.report",
        ]
    )
    report.paragraph("The test suite passes, and a fresh clone of the repository reproduces the submitted predictions exactly.")
    report.save()


def export_pdf() -> None:
    from docx2pdf import convert

    convert(str(DOCX_PATH), str(PDF_PATH))
    print(f"Saved {PDF_PATH}")


if __name__ == "__main__":
    build()
    export_pdf()
