from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from analysis.derived_feature_presets import inspect_derived_presets
from analysis.derived_features import apply_derived_columns
from analysis.input_paths import resolve_v03_failures_path

try:
    from catboost import CatBoostRegressor, Pool
except ModuleNotFoundError:  # pragma: no cover - optional for quick local use
    CatBoostRegressor = None
    Pool = None


RAW_CANDIDATE_COLUMNS = [
    "Месторождение",
    "Принадлежность",
    "Тип УЭЦН",
    "Дебит жидк.",
    "Газовый фактор",
    "ГЖФ",
    "Кпрод.",
    "Обводненность",
    "Рпл.",
    "Рзаб",
    "Частота",
    "Загр, Двиг,",
    "Работа в кривизне",
    "Ном. Произв. м₃/сут",
    "Ном.напор (50Гц)",
    "Номинальная частота, Гц",
    "Кол.ступеней",
    "Мощность, кВт",
    "Ном. ток/ A",
    "Ток x.x",
    "Глубина спуска УЭЦН, по НКТ",
    "ВГ, м",
    "Дав. Нас",
]

SPLIT_COLUMNS = [
    "Kpod",
    "Kpod_freq_adjusted",
    "pressure_ratio",
    "pressure_margin_to_bubble",
    "frequency_to_reference_ratio",
    "frequency_over_reference_hz",
    "qliq_per_kw",
    "qliq_per_current",
    "motor_load_per_hz",
    "nominal_head_per_stage",
    "ГЖФ",
    "Загр, Двиг,",
    "Работа в кривизне",
]


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def load_dataset(path: Path, sheet_name: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    df = pd.read_excel(path, sheet_name=sheet_name)
    notes: list[str] = []
    if "Failure Flag" in df.columns:
        raw_event = _numeric(df["Failure Flag"])
        original_rows = len(df)
        df = df.loc[raw_event.isin([0, 1])].copy()
        notes.append(f"Filtered Failure Flag to 0/1: {original_rows} -> {len(df)} rows.")

    derived_presets = [preset for preset in inspect_derived_presets(df) if preset.available]
    derived_specs = [{"name": preset.name, "formula": preset.formula} for preset in derived_presets]
    if derived_specs:
        df, _ = apply_derived_columns(df, derived_specs)
    derived_names = [preset.name for preset in derived_presets]
    return df, derived_names, notes


def candidate_columns(df: pd.DataFrame, derived_names: list[str]) -> list[str]:
    names = [column for column in [*RAW_CANDIDATE_COLUMNS, *derived_names] if column in df.columns]
    return list(dict.fromkeys(names))


def top_correlations(df: pd.DataFrame, target_column: str, columns: list[str], top_n: int = 12) -> pd.DataFrame:
    target = _numeric(df[target_column])
    rows: list[dict[str, object]] = []
    for column in columns:
        if column == target_column or column not in df.columns:
            continue
        series = _numeric(df[column])
        valid = target.notna() & series.notna()
        if int(valid.sum()) < 40:
            continue
        corr = series[valid].corr(target[valid])
        if pd.notna(corr):
            rows.append({"feature": column, "rows": int(valid.sum()), "correlation": float(corr)})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("correlation", key=lambda item: item.abs(), ascending=False).head(top_n).reset_index(drop=True)


def top_catboost_importance(df: pd.DataFrame, target_column: str, columns: list[str], top_n: int = 12) -> pd.DataFrame:
    if CatBoostRegressor is None or Pool is None:
        return pd.DataFrame(columns=["feature", "importance"])

    target = _numeric(df[target_column])
    working = df.loc[target.notna()].copy()
    target = _numeric(working[target_column])
    if working.empty or target.nunique(dropna=True) < 2:
        return pd.DataFrame(columns=["feature", "importance"])

    prepared = pd.DataFrame(index=working.index)
    categorical_columns: list[str] = []
    for column in columns:
        if column == target_column or column not in working.columns:
            continue
        series = working[column]
        numeric = _numeric(series)
        if int(numeric.notna().sum()) >= max(20, int(series.notna().sum() * 0.5)):
            prepared[column] = numeric
        else:
            prepared[column] = series.astype("string").fillna("__missing__")
            categorical_columns.append(column)

    prepared = prepared[[column for column in prepared.columns if int(prepared[column].notna().sum()) >= 20]]
    if prepared.empty:
        return pd.DataFrame(columns=["feature", "importance"])

    model = CatBoostRegressor(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    cat_indices = [prepared.columns.get_loc(column) for column in categorical_columns if column in prepared.columns]
    model.fit(Pool(prepared, target, cat_features=cat_indices))
    frame = pd.DataFrame({"feature": prepared.columns.tolist(), "importance": model.get_feature_importance()})
    return frame.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


def best_split_summary(df: pd.DataFrame, target_column: str, column: str) -> dict[str, object] | None:
    if column not in df.columns:
        return None
    x = _numeric(df[column])
    y = _numeric(df[target_column])
    valid = x.notna() & y.notna()
    x = x[valid]
    y = y[valid]
    if len(x) < 100:
        return None

    quantiles = sorted(set(float(x.quantile(q)) for q in np.linspace(0.1, 0.9, 17)))
    best: tuple[float, float, str, float, float, int, int] | None = None
    for threshold in quantiles:
        low = y[x <= threshold]
        high = y[x > threshold]
        if len(low) < 20 or len(high) < 20:
            continue
        low_median = float(low.median())
        high_median = float(high.median())
        direction = "positive_excess" if high_median < low_median else "negative_excess"
        item = (abs(high_median - low_median), threshold, direction, low_median, high_median, len(low), len(high))
        if best is None or item[0] > best[0]:
            best = item

    if best is None:
        return None
    score, threshold, direction, low_median, high_median, low_rows, high_rows = best
    return {
        "feature": column,
        "threshold": float(threshold),
        "direction": direction,
        "median_le_threshold": float(low_median),
        "median_gt_threshold": float(high_median),
        "rows_le_threshold": int(low_rows),
        "rows_gt_threshold": int(high_rows),
        "median_gap": float(score),
    }


def print_frame(title: str, frame: pd.DataFrame) -> None:
    print(f"\n## {title}")
    if frame.empty:
        print("(empty)")
        return
    print(frame.to_string(index=False))


def build_subsets(df: pd.DataFrame, target_column: str) -> dict[str, pd.DataFrame]:
    target = _numeric(df[target_column])
    subsets = {
        "ALL": df.copy(),
        "TTF > 30": df.loc[target > 30].copy(),
        "TTF > 90": df.loc[target > 90].copy(),
        "TTF < 30": df.loc[target < 30].copy(),
    }

    if "Месторождение" in df.columns:
        top_fields = df["Месторождение"].astype("string").value_counts(dropna=True).head(3).index.tolist()
        for value in top_fields:
            subsets[f"Field={value}, TTF > 30"] = df.loc[(df["Месторождение"].astype("string") == value) & (target > 30)].copy()

    if "Принадлежность" in df.columns:
        top_contractors = df["Принадлежность"].astype("string").value_counts(dropna=True).head(3).index.tolist()
        for value in top_contractors:
            subsets[f"Contractor={value}, TTF > 30"] = df.loc[(df["Принадлежность"].astype("string") == value) & (target > 30)].copy()

    return subsets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ESP run-life stress proxies on the V03 failures workbook.")
    parser.add_argument(
        "xlsx_path",
        nargs="?",
        default=str(resolve_v03_failures_path()),
        help="Path to the Excel workbook.",
    )
    parser.add_argument("--sheet", default="Свод", help="Sheet name to analyze.")
    parser.add_argument("--target", default="Наработка (сут)", help="Target TTF column.")
    parser.add_argument("--top", type=int, default=12, help="Number of rows to print in each ranking.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workbook_path = Path(args.xlsx_path)
    df, derived_names, notes = load_dataset(workbook_path, args.sheet)
    columns = candidate_columns(df, derived_names)
    subsets = build_subsets(df, args.target)

    print(f"Workbook: {workbook_path}")
    print(f"Sheet: {args.sheet}")
    print(f"Rows after event filtering: {len(df)}")
    print(f"Available derived columns: {len(derived_names)}")
    for note in notes:
        print(f"- {note}")
    if derived_names:
        print("- Derived:", ", ".join(derived_names))

    for subset_name, subset_df in subsets.items():
        print(f"\n# {subset_name}")
        print(f"Rows: {len(subset_df)}")
        corr_frame = top_correlations(subset_df, args.target, columns, top_n=args.top)
        print_frame("Top correlations", corr_frame)

        run_catboost = not subset_name.startswith("Field=") and not subset_name.startswith("Contractor=")
        if run_catboost:
            importance_frame = top_catboost_importance(subset_df, args.target, columns, top_n=args.top)
            if not importance_frame.empty:
                print_frame("Top CatBoost importances", importance_frame)

        split_rows = [best_split_summary(subset_df, args.target, column) for column in SPLIT_COLUMNS]
        split_frame = pd.DataFrame([row for row in split_rows if row is not None])
        if not split_frame.empty:
            split_frame = split_frame.sort_values("median_gap", ascending=False).head(args.top).reset_index(drop=True)
        print_frame("Best one-threshold split heuristics", split_frame)


if __name__ == "__main__":
    main()
