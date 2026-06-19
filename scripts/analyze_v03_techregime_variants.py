from __future__ import annotations

import sqlite3
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

try:
    from catboost import CatBoostRegressor, Pool
except ModuleNotFoundError:  # pragma: no cover
    CatBoostRegressor = None
    Pool = None

from analysis.derived_feature_presets import inspect_derived_presets
from analysis.derived_features import apply_derived_columns
from analysis.input_paths import resolve_v03_all_path, resolve_v03_failures_path
from analysis.sqlite_paths import resolve_techregime_db_path


FAILURES_PATH = resolve_v03_failures_path()
ALL_PATH = resolve_v03_all_path()
TR_DB_PATH = resolve_techregime_db_path()

TR_QUERY_COLUMNS = {
    "freq": "col_0047",
    "load": "col_0048",
    "rpl": "col_0061",
    "rpump_intake": "col_0062",
    "rzab": "col_0063",
    "qliq": "col_0064",
    "watercut": "col_0065",
    "gas_factor": "col_0067",
    "qgas": "col_0068",
    "kprod": "col_0069",
}
TR_FEATURE_COLUMNS = list(TR_QUERY_COLUMNS)


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def print_table(title: str, rows: list[tuple] | pd.DataFrame) -> None:
    print(f"\n## {title}")
    if isinstance(rows, pd.DataFrame):
        if rows.empty:
            print("(empty)")
            return
        print(rows.to_string(index=False))
        return
    if not rows:
        print("(empty)")
        return
    for row in rows:
        print(row)


def load_target(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Свод").reset_index(names="row_id")
    df["event"] = _numeric(df["Failure Flag"])
    df["TTF"] = _numeric(df["Наработка (сут)"])
    if "Дата монтажа" in df.columns:
        df["Дата монтажа"] = pd.to_datetime(df["Дата монтажа"], errors="coerce")
    if "Дата остановки" in df.columns:
        df["Дата остановки"] = pd.to_datetime(df["Дата остановки"], errors="coerce")

    derived_presets = [preset for preset in inspect_derived_presets(df) if preset.available]
    if derived_presets:
        df, _ = apply_derived_columns(df, [{"name": preset.name, "formula": preset.formula} for preset in derived_presets])
    return df


def load_techregime_subset(wells: list[str]) -> pd.DataFrame:
    if not wells:
        return pd.DataFrame(columns=["well_id", "dt", *TR_FEATURE_COLUMNS])
    query_select = ", ".join([f"{storage} as {alias}" for alias, storage in TR_QUERY_COLUMNS.items()])
    frames: list[pd.DataFrame] = []
    with sqlite3.connect(TR_DB_PATH) as connection:
        for start in range(0, len(wells), 400):
            chunk = wells[start : start + 400]
            placeholders = ",".join(["?"] * len(chunk))
            query = f"""
                SELECT col_0003 as well_id, col_0010 as dt, {query_select}
                FROM techregime_records
                WHERE col_0003 IN ({placeholders})
            """
            frames.append(pd.read_sql_query(query, connection, params=chunk))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["well_id", "dt", *TR_FEATURE_COLUMNS])
    df["dt"] = pd.to_datetime(df["dt"], dayfirst=True, errors="coerce")
    for column in TR_FEATURE_COLUMNS:
        df[column] = _numeric(df[column])
    return df.loc[df["well_id"].notna() & df["dt"].notna()].copy()


def aggregate_techregime_features(runs: pd.DataFrame, tr_df: pd.DataFrame) -> pd.DataFrame:
    tr_by_well = {well: frame.sort_values("dt").reset_index(drop=True) for well, frame in tr_df.groupby("well_id")}
    uniq_by_well = {
        well: frame.groupby("dt", as_index=False)[TR_FEATURE_COLUMNS].mean(numeric_only=True).sort_values("dt").reset_index(drop=True)
        for well, frame in tr_by_well.items()
    }

    records: list[dict[str, float | int]] = []
    for row in runs[["row_id", "Скв.", "Дата монтажа", "Дата остановки"]].to_dict(orient="records"):
        well = str(row["Скв."])
        start = row["Дата монтажа"]
        stop = row["Дата остановки"]
        entry: dict[str, float | int] = {"row_id": int(row["row_id"])}

        for label, source in (("rawdup", tr_by_well.get(well)), ("uniqdate", uniq_by_well.get(well))):
            if source is None:
                continue
            interval = source.loc[(source["dt"] >= start) & (source["dt"] <= stop)].copy()
            if interval.empty:
                continue
            last30 = interval.loc[interval["dt"] >= (stop - pd.Timedelta(days=30))].copy()
            entry[f"{label}_record_count"] = int(len(interval))
            entry[f"{label}_date_count"] = int(interval["dt"].nunique())

            for feature in TR_FEATURE_COLUMNS:
                series = _numeric(interval[feature])
                positive = series.loc[series > 0]
                if not positive.empty:
                    mean_value = float(positive.mean())
                    entry[f"{feature}_{label}_mean_pos"] = mean_value
                    entry[f"{feature}_{label}_std_pos"] = float(positive.std(ddof=0)) if len(positive) > 1 else 0.0
                    entry[f"{feature}_{label}_range_pos"] = float(positive.max() - positive.min())
                    entry[f"{feature}_{label}_cv_pos"] = float(positive.std(ddof=0) / mean_value) if abs(mean_value) > 1e-12 else np.nan

                positive_last30 = _numeric(last30[feature])
                positive_last30 = positive_last30.loc[positive_last30 > 0]
                if not positive_last30.empty:
                    mean_last30 = float(positive_last30.mean())
                    entry[f"{feature}_{label}_last30_mean_pos"] = mean_last30
                    entry[f"{feature}_{label}_last30_std_pos"] = float(positive_last30.std(ddof=0)) if len(positive_last30) > 1 else 0.0
                    if not positive.empty and abs(float(positive.mean())) > 1e-12:
                        entry[f"{feature}_{label}_last30_to_mean_ratio"] = float(mean_last30 / float(positive.mean()))

        records.append(entry)
    return pd.DataFrame(records)


def add_variant_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for prefix in ("rawdup_mean_pos", "uniqdate_mean_pos", "rawdup_last30_mean_pos", "uniqdate_last30_mean_pos"):
        qliq = _numeric(result.get(f"qliq_{prefix}", pd.Series(index=result.index, dtype=float)))
        freq = _numeric(result.get(f"freq_{prefix}", pd.Series(index=result.index, dtype=float)))
        rzab = _numeric(result.get(f"rzab_{prefix}", pd.Series(index=result.index, dtype=float)))
        intake = _numeric(result.get(f"rpump_intake_{prefix}", pd.Series(index=result.index, dtype=float)))
        nominal_rate = _numeric(result["Ном. Произв. м₃/сут"])
        nominal_freq = _numeric(result["Номинальная частота, Гц"])
        pbubble = _numeric(result["Дав. Нас"])
        power = _numeric(result["Мощность, кВт"])

        result[f"Kpod_{prefix}"] = qliq / nominal_rate
        result[f"Kpod_freq_{prefix}"] = result[f"Kpod_{prefix}"] * (nominal_freq / freq)
        result[f"pressure_ratio_bhp_{prefix}"] = rzab / pbubble
        result[f"pressure_ratio_intake_{prefix}"] = intake / pbubble
        result[f"qliq_per_kw_{prefix}"] = qliq / power
    return result


def correlation_rows(df: pd.DataFrame, target_column: str, columns: list[str], top_n: int = 8) -> list[tuple[str, int, float]]:
    y = _numeric(df[target_column])
    rows: list[tuple[str, int, float]] = []
    for column in columns:
        if column not in df.columns:
            continue
        x = _numeric(df[column])
        valid = x.notna() & y.notna()
        if int(valid.sum()) < 40:
            continue
        rows.append((column, int(valid.sum()), float(x[valid].corr(y[valid]))))
    rows.sort(key=lambda item: abs(item[2]), reverse=True)
    return rows[:top_n]


def top_catboost_importance(df: pd.DataFrame, target_column: str, features: list[str], top_n: int = 8) -> list[tuple[str, float]]:
    if CatBoostRegressor is None or Pool is None:
        return []
    y = _numeric(df[target_column])
    working = df.loc[y.notna()].copy()
    y = _numeric(working[target_column])
    if len(working) < 100 or int(y.nunique(dropna=True)) < 2:
        return []

    candidate_columns = [column for column in features if column in working.columns and column != target_column]
    prepared = pd.DataFrame(index=working.index)
    categorical_columns: list[str] = []
    for column in candidate_columns:
        series = working[column]
        numeric = _numeric(series)
        if int(numeric.notna().sum()) >= max(20, int(series.notna().sum() * 0.5)):
            prepared[column] = numeric
        else:
            prepared[column] = series.astype("string").fillna("__missing__")
            categorical_columns.append(column)
    prepared = prepared[[column for column in prepared.columns if int(prepared[column].notna().sum()) >= 20]]
    if prepared.empty:
        return []

    model = CatBoostRegressor(
        iterations=250,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    cat_indices = [prepared.columns.get_loc(column) for column in categorical_columns if column in prepared.columns]
    model.fit(Pool(prepared, y, cat_features=cat_indices))
    frame = pd.DataFrame({"feature": prepared.columns.tolist(), "importance": model.get_feature_importance()})
    frame = frame.sort_values("importance", ascending=False).head(top_n)
    return [(str(row.feature), float(row.importance)) for row in frame.itertuples(index=False)]


def compare_workbook_vs_current_extraction(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for workbook_column, aggregated_column in [
        ("Дебит жидк.", "qliq_rawdup_mean_pos"),
        ("Частота", "freq_rawdup_mean_pos"),
        ("Рзаб", "rzab_rawdup_mean_pos"),
        ("Загр, Двиг,", "load_rawdup_mean_pos"),
        ("Рпл.", "rpl_rawdup_mean_pos"),
        ("Кпрод.", "kprod_rawdup_mean_pos"),
        ("Обводненность", "watercut_rawdup_mean_pos"),
    ]:
        x = _numeric(df[workbook_column])
        y = _numeric(df[aggregated_column])
        valid = x.notna() & y.notna()
        if not valid.any():
            continue
        rows.append(
            {
                "workbook_column": workbook_column,
                "pairs": int(valid.sum()),
                "mae": float((x[valid] - y[valid]).abs().mean()),
                "corr": float(x[valid].corr(y[valid])),
            }
        )
    return pd.DataFrame(rows)


def analyze_failures_tr_variants() -> None:
    failures = load_target(FAILURES_PATH)
    failures = failures.loc[
        failures["event"].eq(1)
        & failures["TTF"].notna()
        & failures["Скв."].notna()
        & failures["Дата монтажа"].notna()
        & failures["Дата остановки"].notna()
    ].copy()

    tr_df = load_techregime_subset(failures["Скв."].astype(str).unique().tolist())
    aggregated = aggregate_techregime_features(failures, tr_df)
    modeling = add_variant_derived_features(failures.merge(aggregated, on="row_id", how="left"))

    print(f"# V03 Failures + Techregime Variants")
    print(f"Runs in scope: {len(failures)}")
    print(f"TR rows loaded: {len(tr_df)}")
    print(f"Rows with current-like Qliq aggregate: {int(modeling['qliq_rawdup_mean_pos'].notna().sum())}")
    print_table("Workbook vs current extraction check", compare_workbook_vs_current_extraction(modeling).round(4))

    feature_sets = {
        "current-like full-run means": [
            "qliq_rawdup_mean_pos",
            "freq_rawdup_mean_pos",
            "load_rawdup_mean_pos",
            "rzab_rawdup_mean_pos",
            "rpump_intake_rawdup_mean_pos",
            "rpl_rawdup_mean_pos",
            "watercut_rawdup_mean_pos",
            "gas_factor_rawdup_mean_pos",
            "kprod_rawdup_mean_pos",
            "Kpod_rawdup_mean_pos",
            "Kpod_freq_rawdup_mean_pos",
            "pressure_ratio_bhp_rawdup_mean_pos",
            "pressure_ratio_intake_rawdup_mean_pos",
            "qliq_per_kw_rawdup_mean_pos",
        ],
        "last-30-day means": [
            "qliq_rawdup_last30_mean_pos",
            "freq_rawdup_last30_mean_pos",
            "load_rawdup_last30_mean_pos",
            "rzab_rawdup_last30_mean_pos",
            "rpump_intake_rawdup_last30_mean_pos",
            "rpl_rawdup_last30_mean_pos",
            "watercut_rawdup_last30_mean_pos",
            "gas_factor_rawdup_last30_mean_pos",
            "kprod_rawdup_last30_mean_pos",
            "Kpod_rawdup_last30_mean_pos",
            "Kpod_freq_rawdup_last30_mean_pos",
            "pressure_ratio_bhp_rawdup_last30_mean_pos",
            "pressure_ratio_intake_rawdup_last30_mean_pos",
            "qliq_per_kw_rawdup_last30_mean_pos",
        ],
        "last-30d / full-run ratios": [
            "qliq_rawdup_last30_to_mean_ratio",
            "freq_rawdup_last30_to_mean_ratio",
            "load_rawdup_last30_to_mean_ratio",
            "rzab_rawdup_last30_to_mean_ratio",
            "rpump_intake_rawdup_last30_to_mean_ratio",
            "rpl_rawdup_last30_to_mean_ratio",
            "watercut_rawdup_last30_to_mean_ratio",
            "gas_factor_rawdup_last30_to_mean_ratio",
            "kprod_rawdup_last30_to_mean_ratio",
        ],
        "full-run variability (unsafe for modeling)": [
            "qliq_uniqdate_cv_pos",
            "freq_uniqdate_cv_pos",
            "load_uniqdate_cv_pos",
            "rzab_uniqdate_cv_pos",
            "rpump_intake_uniqdate_cv_pos",
            "rpl_uniqdate_cv_pos",
            "watercut_uniqdate_cv_pos",
            "gas_factor_uniqdate_cv_pos",
            "kprod_uniqdate_cv_pos",
            "qliq_uniqdate_range_pos",
            "freq_uniqdate_range_pos",
            "load_uniqdate_range_pos",
            "rzab_uniqdate_range_pos",
            "rpump_intake_uniqdate_range_pos",
            "rpl_uniqdate_range_pos",
            "watercut_uniqdate_range_pos",
            "gas_factor_uniqdate_range_pos",
            "kprod_uniqdate_range_pos",
        ],
        "fixed-window variability": [
            "qliq_rawdup_last30_std_pos",
            "freq_rawdup_last30_std_pos",
            "load_rawdup_last30_std_pos",
            "rzab_rawdup_last30_std_pos",
            "rpump_intake_rawdup_last30_std_pos",
            "rpl_rawdup_last30_std_pos",
            "watercut_rawdup_last30_std_pos",
            "gas_factor_rawdup_last30_std_pos",
            "kprod_rawdup_last30_std_pos",
        ],
    }

    for scope_name, subset in (
        ("all failures", modeling),
        ("failures with TTF > 30", modeling.loc[modeling["TTF"] > 30].copy()),
    ):
        print(f"\n# Scope: {scope_name} ({len(subset)} rows)")
        for label, columns in feature_sets.items():
            print_table(label, correlation_rows(subset, "TTF", columns, top_n=8))

    dedup_rows = []
    for feature in ("qliq", "freq", "rpump_intake", "Kpod", "Kpod_freq", "pressure_ratio_intake", "qliq_per_kw"):
        raw_col = f"{feature}_rawdup_mean_pos"
        uniq_col = f"{feature}_uniqdate_mean_pos"
        if raw_col not in modeling.columns or uniq_col not in modeling.columns:
            continue
        y = _numeric(modeling["TTF"])
        raw_x = _numeric(modeling[raw_col])
        uniq_x = _numeric(modeling[uniq_col])
        raw_valid = raw_x.notna() & y.notna()
        uniq_valid = uniq_x.notna() & y.notna()
        dedup_rows.append(
            {
                "feature": feature,
                "corr_rawdup": float(raw_x[raw_valid].corr(y[raw_valid])) if raw_valid.any() else np.nan,
                "corr_uniqdate": float(uniq_x[uniq_valid].corr(y[uniq_valid])) if uniq_valid.any() else np.nan,
            }
        )
    print_table("Duplicate-row sensitivity (full-run mean)", pd.DataFrame(dedup_rows).round(4))


def analyze_v03_all_scopes() -> None:
    df = load_target(ALL_PATH)
    usable = df.loc[df["event"].isin([0, 1]) & df["TTF"].notna()].copy()
    duplicate_key_columns = ["Месторождение", "Скв.", "Дата монтажа", "Дата остановки"]
    duplicate_counts = usable.groupby(duplicate_key_columns, dropna=False).size().rename("n").reset_index()
    duplicate_groups = duplicate_counts.loc[duplicate_counts["n"] > 1]

    print(f"\n# V03 All Scope Comparison")
    print(f"Rows with event 0/1 and usable TTF: {len(usable)}")
    print(f"Duplicate usable run groups: {len(duplicate_groups)}")
    print(f"Rows inside duplicate usable groups: {int(duplicate_groups['n'].sum()) if not duplicate_groups.empty else 0}")
    event_summary = usable["event"].value_counts(dropna=False).sort_index()
    print("Usable event distribution:")
    for key, value in event_summary.items():
        print(f"- event={key:g}: {int(value)}")

    features = [
        "Месторождение",
        "Принадлежность",
        "Дебит жидк.",
        "Частота",
        "Загр, Двиг,",
        "Рзаб",
        "Рпл.",
        "Дав. Нас",
        "ГЖФ",
        "Газовый фактор",
        "Кпрод.",
        "Обводненность",
        "Работа в кривизне",
        "Мощность, кВт",
        "Ток x.x",
        "Kpod",
        "Kpod_freq_adjusted",
        "pressure_ratio",
        "frequency_to_reference_ratio",
        "frequency_over_reference_hz",
        "qliq_per_hz",
        "qliq_per_kw",
        "current_to_nominal_ratio",
        "qliq_per_current",
        "bhp_to_reservoir_ratio",
        "pressure_margin_to_bubble",
        "nominal_head_per_stage",
        "motor_load_per_hz",
        "curve_work_per_meter",
    ]
    features = [column for column in features if column in df.columns]

    subsets = {
        "all rows with event 0/1 and TTF": usable,
        "failures only": df.loc[df["event"].eq(1) & df["TTF"].notna()].copy(),
        "censored only": df.loc[df["event"].eq(0) & df["TTF"].notna()].copy(),
        "all rows with event 0/1, TTF > 30": df.loc[df["event"].isin([0, 1]) & df["TTF"].gt(30)].copy(),
        "all rows with event 0/1, deduplicated": usable.sort_values(duplicate_key_columns).drop_duplicates(subset=duplicate_key_columns, keep="first").copy(),
    }

    for label, subset in subsets.items():
        print(f"\n# Scope: {label} ({len(subset)} rows)")
        print_table("Top correlations", correlation_rows(subset, "TTF", features, top_n=10))
        importance = top_catboost_importance(subset, "TTF", features, top_n=8)
        print_table("Top CatBoost importances", importance)


def main() -> None:
    analyze_failures_tr_variants()
    analyze_v03_all_scopes()


if __name__ == "__main__":
    main()
