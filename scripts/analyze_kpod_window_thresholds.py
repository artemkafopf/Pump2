from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from analysis.derived_feature_presets import inspect_derived_presets
from analysis.derived_features import apply_derived_columns
from analysis.input_paths import resolve_v03_all_path, resolve_v03_failures_path
from scripts.analyze_failure_horizon import load_techregime_daily
from scripts.analyze_v03_stress import _numeric


ALL_PATH = resolve_v03_all_path()
FAILURES_PATH = resolve_v03_failures_path()


LOW_THRESHOLDS = [0.40, 0.55, 0.70]
HIGH_THRESHOLDS = [0.85, 1.00]


def threshold_suffix(threshold: float) -> str:
    return f"{int(round(threshold * 100)):03d}"


def load_runs(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Свод").reset_index(names="row_id")
    df["Failure Flag"] = _numeric(df["Failure Flag"])
    df["Наработка (сут)"] = _numeric(df["Наработка (сут)"])
    df["Дата монтажа"] = pd.to_datetime(df["Дата монтажа"], errors="coerce")
    df["Дата остановки"] = pd.to_datetime(df["Дата остановки"], errors="coerce")
    df["Скв."] = df["Скв."].astype("string").str.strip()
    df = df.loc[df["Failure Flag"].isin([0, 1])].copy()
    df = df.loc[df["Наработка (сут)"].notna() & (df["Наработка (сут)"] > 30)].copy()
    df = df.loc[df["Скв."].notna() & df["Дата монтажа"].notna() & df["Дата остановки"].notna()].copy()
    df = df.loc[df["Дата остановки"] > df["Дата монтажа"]].copy()

    derived_presets = [preset for preset in inspect_derived_presets(df) if preset.available]
    if derived_presets:
        df, _ = apply_derived_columns(df, [{"name": preset.name, "formula": preset.formula} for preset in derived_presets])

    dedup_columns = ["Скв.", "Дата монтажа", "Дата остановки", "Failure Flag"]
    before = len(df)
    df = df.drop_duplicates(subset=dedup_columns, keep="first").reset_index(drop=True)
    if before != len(df):
        print(f"Dropped duplicate usable runs: {before} -> {len(df)}")
    return df


def aggregate_last30_kpod_features(runs: pd.DataFrame, tr_daily: pd.DataFrame) -> pd.DataFrame:
    tr_by_well = {well: frame.sort_values("dt").reset_index(drop=True) for well, frame in tr_daily.groupby("well_id")}
    rows: list[dict[str, object]] = []
    for run in runs.to_dict(orient="records"):
        well = str(run["Скв."])
        source = tr_by_well.get(well)
        if source is None:
            continue
        interval = source.loc[(source["dt"] >= run["Дата монтажа"]) & (source["dt"] <= run["Дата остановки"])].copy()
        if interval.empty:
            continue
        last30 = interval.loc[interval["dt"] >= (run["Дата остановки"] - pd.Timedelta(days=30))].copy()
        if last30.empty:
            continue

        nominal_rate = _numeric(pd.Series([run.get("Ном. Произв. м₃/сут")])).iloc[0]
        nominal_freq = _numeric(pd.Series([run.get("Номинальная частота, Гц")])).iloc[0]
        if not np.isfinite(nominal_rate) or nominal_rate <= 0:
            continue

        last30["kpod_daily"] = _numeric(last30["qliq"]) / nominal_rate
        if np.isfinite(nominal_freq) and nominal_freq > 0:
            last30["kpod_freq_daily"] = last30["kpod_daily"] * (nominal_freq / _numeric(last30["freq"]))
        else:
            last30["kpod_freq_daily"] = np.nan

        kpod = _numeric(last30["kpod_daily"])
        valid_kpod = kpod.loc[np.isfinite(kpod) & (kpod > 0)]
        if valid_kpod.empty:
            continue

        entry: dict[str, object] = {
            "row_id": int(run["row_id"]),
            "Месторождение": run.get("Месторождение"),
            "Принадлежность": run.get("Принадлежность"),
            "Тип УЭЦН": run.get("Тип УЭЦН"),
            "TTF": float(run["Наработка (сут)"]),
            "event": int(run["Failure Flag"]),
            "tr_last30_days_total": int(valid_kpod.shape[0]),
            "kpod_last30_mean": float(valid_kpod.mean()),
            "kpod_last30_median": float(valid_kpod.median()),
            "kpod_last30_std": float(valid_kpod.std(ddof=0)) if len(valid_kpod) > 1 else 0.0,
            "kpod_last30_min": float(valid_kpod.min()),
            "kpod_last30_max": float(valid_kpod.max()),
        }

        valid_kpod_freq = _numeric(last30["kpod_freq_daily"]).loc[lambda s: np.isfinite(s) & (s > 0)]
        if not valid_kpod_freq.empty:
            entry["kpod_freq_last30_mean"] = float(valid_kpod_freq.mean())
            entry["kpod_freq_last30_median"] = float(valid_kpod_freq.median())

        for threshold in LOW_THRESHOLDS:
            count = int((valid_kpod < threshold).sum())
            suffix = threshold_suffix(threshold)
            entry[f"kpod_days_lt_{suffix}"] = count
            entry[f"kpod_share_lt_{suffix}"] = float(count / len(valid_kpod))

        for threshold in HIGH_THRESHOLDS:
            count = int((valid_kpod > threshold).sum())
            suffix = threshold_suffix(threshold)
            entry[f"kpod_days_gt_{suffix}"] = count
            entry[f"kpod_share_gt_{suffix}"] = float(count / len(valid_kpod))

        entry["kpod_low70_minus_high85_days"] = int((valid_kpod < 0.70).sum() - (valid_kpod > 0.85).sum())
        entry["kpod_low70_minus_high85_share"] = float(
            ((valid_kpod < 0.70).sum() - (valid_kpod > 0.85).sum()) / len(valid_kpod)
        )
        rows.append(entry)

    return pd.DataFrame(rows)


def correlation_rows(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    target = _numeric(df["TTF"])
    rows: list[dict[str, object]] = []
    for column in columns:
        if column not in df.columns:
            continue
        series = _numeric(df[column])
        valid = target.notna() & series.notna()
        if int(valid.sum()) < 20:
            continue
        corr = series[valid].corr(target[valid])
        rows.append({"feature": column, "rows": int(valid.sum()), "correlation": float(corr)})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("correlation", key=lambda item: item.abs(), ascending=False).reset_index(drop=True)


def threshold_bin(value: int, *, low_cut: int = 5, high_cut: int = 15) -> str:
    if value <= low_cut:
        return f"0-{low_cut}"
    if value <= high_cut:
        return f"{low_cut + 1}-{high_cut}"
    return f">{high_cut}"


def summarize_cross_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = df.copy()
    working["low_bin"] = working["kpod_days_lt_070"].astype(int).map(lambda value: threshold_bin(value, low_cut=5, high_cut=15))
    working["high_bin"] = working["kpod_days_gt_085"].astype(int).map(lambda value: threshold_bin(value, low_cut=5, high_cut=15))

    counts = (
        working.pivot_table(index="low_bin", columns="high_bin", values="TTF", aggfunc="size", fill_value=0)
        .reindex(index=["0-5", "6-15", ">15"], columns=["0-5", "6-15", ">15"])
        .fillna(0)
        .astype(int)
    )
    medians = (
        working.pivot_table(index="low_bin", columns="high_bin", values="TTF", aggfunc="median")
        .reindex(index=["0-5", "6-15", ">15"], columns=["0-5", "6-15", ">15"])
    )
    return counts, medians


def print_group_report(group_name: str, df: pd.DataFrame) -> None:
    print(f"\n# {group_name}")
    print(f"Rows: {len(df)}")
    if len(df) < 20:
        print("(too few rows)")
        return

    target = _numeric(df["TTF"])
    kpod = _numeric(df["kpod_last30_mean"])
    valid = target.notna() & kpod.notna()
    if int(valid.sum()) >= 20:
        print(f"kpod_last30_mean corr with TTF: {float(kpod[valid].corr(target[valid])):.4f}")

    feature_candidates = [
        "kpod_last30_mean",
        "kpod_last30_median",
        "kpod_last30_std",
        "kpod_days_lt_040",
        "kpod_share_lt_040",
        "kpod_days_lt_055",
        "kpod_share_lt_055",
        "kpod_days_lt_070",
        "kpod_share_lt_070",
        "kpod_days_gt_085",
        "kpod_share_gt_085",
        "kpod_days_gt_100",
        "kpod_share_gt_100",
        "kpod_low70_minus_high85_days",
        "kpod_low70_minus_high85_share",
    ]
    corr_frame = correlation_rows(df, feature_candidates).head(8)
    if corr_frame.empty:
        print("(no threshold features with enough rows)")
    else:
        print("\nTop threshold/window correlations")
        print(corr_frame.to_string(index=False))

    low = target.loc[_numeric(df["kpod_last30_mean"]) < 0.40]
    high = target.loc[_numeric(df["kpod_last30_mean"]) >= 0.85]
    print(
        "\nLast-30 Kpod mean extremes:"
        f" rows_lt_0.40={int(len(low))}, median_lt_0.40={None if low.empty else float(low.median()):.1f};"
        f" rows_ge_0.85={int(len(high))}, median_ge_0.85={None if high.empty else float(high.median()):.1f}"
    )

    print("\nDays below 0.70 bins")
    low_bins = (
        df.assign(low_bin=df["kpod_days_lt_070"].astype(int).map(lambda value: threshold_bin(value, low_cut=5, high_cut=15)))
        .groupby("low_bin", sort=False)["TTF"]
        .agg(rows="size", median_ttf="median")
        .reindex(["0-5", "6-15", ">15"])
    )
    print(low_bins.to_string())

    print("\nDays above 0.85 bins")
    high_bins = (
        df.assign(high_bin=df["kpod_days_gt_085"].astype(int).map(lambda value: threshold_bin(value, low_cut=5, high_cut=15)))
        .groupby("high_bin", sort=False)["TTF"]
        .agg(rows="size", median_ttf="median")
        .reindex(["0-5", "6-15", ">15"])
    )
    print(high_bins.to_string())

    counts, medians = summarize_cross_table(df)
    print("\nCross table row counts: days<0.70 vs days>0.85")
    print(counts.to_string())
    print("\nCross table median TTF: days<0.70 vs days>0.85")
    print(medians.round(1).to_string())


def main() -> None:
    dataset_path = ALL_PATH
    if len(sys.argv) > 1 and sys.argv[1].strip().lower() == "failures":
        dataset_path = FAILURES_PATH

    print(f"## Dataset: {dataset_path}")
    runs = load_runs(dataset_path)
    print(f"Usable runs: {len(runs)}")
    print(f"Unique wells: {runs['Скв.'].nunique()}")

    tr_daily = load_techregime_daily(runs["Скв."].dropna().astype(str).unique().tolist())
    print(f"Techregime daily rows: {len(tr_daily)}")
    aggregated = aggregate_last30_kpod_features(runs, tr_daily)
    print(f"Runs with last30 Kpod features: {len(aggregated)}")

    if aggregated.empty:
        print("No aggregated rows were produced.")
        return

    aggregated_features = aggregated.drop(columns=["Месторождение", "Принадлежность", "Тип УЭЦН"], errors="ignore")
    merged = runs.merge(aggregated_features, on="row_id", how="inner")
    print(f"Merged rows: {len(merged)}")

    print_group_report("Ya", merged.loc[merged["Месторождение"].astype("string") == "Ya"].copy())
    print_group_report("Vt", merged.loc[merged["Месторождение"].astype("string") == "Vt"].copy())

    for field in ["Ya", "Vt"]:
        field_df = merged.loc[merged["Месторождение"].astype("string") == field].copy()
        contractor_counts = field_df["Принадлежность"].astype("string").value_counts(dropna=True)
        for contractor, count in contractor_counts.items():
            if int(count) < 60:
                continue
            subgroup = field_df.loc[field_df["Принадлежность"].astype("string") == contractor].copy()
            print_group_report(f"{field} | contractor={contractor}", subgroup)


if __name__ == "__main__":
    main()
