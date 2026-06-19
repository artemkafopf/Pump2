from __future__ import annotations

import re
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

from analysis.weibull_model import fit_weibull_stress_model
from scripts.analyze_failure_horizon import load_techregime_daily
from scripts.analyze_kpod_window_thresholds import ALL_PATH, aggregate_last30_kpod_features, load_runs
from scripts.analyze_v03_stress import _numeric


CYRILLIC_TO_LATIN = str.maketrans(
    {
        "А": "A",
        "В": "B",
        "С": "C",
        "Е": "E",
        "К": "K",
        "М": "M",
        "Н": "H",
        "О": "O",
        "Р": "P",
        "Т": "T",
        "У": "Y",
        "Х": "X",
        "а": "a",
        "в": "b",
        "с": "c",
        "е": "e",
        "к": "k",
        "м": "m",
        "н": "h",
        "о": "o",
        "р": "p",
        "т": "t",
        "у": "y",
        "х": "x",
    }
)


def normalize_pump_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().translate(CYRILLIC_TO_LATIN).upper()
    return re.sub(r"[^A-Z0-9]+", "", text)


def pump_family(value: object) -> str:
    normalized = normalize_pump_text(value)
    if not normalized:
        return "<missing>"
    for prefix in ("MT5A", "5A", "GN", "SN", "DN"):
        if normalized.startswith(prefix):
            return prefix
    match = re.match(r"[A-Z0-9]+", normalized)
    return match.group(0) if match else normalized


def nominal_rate_band(series: pd.Series) -> pd.Series:
    numeric = _numeric(series)
    bins = [-np.inf, 100.0, 250.0, 500.0, 900.0, np.inf]
    labels = ["<=100", "100-250", "250-500", "500-900", ">900"]
    return pd.cut(numeric, bins=bins, labels=labels, right=True, include_lowest=True).astype("string").fillna("<missing>")


def prepare_dataset() -> pd.DataFrame:
    runs = load_runs(ALL_PATH)
    tr_daily = load_techregime_daily(runs["Скв."].dropna().astype(str).unique().tolist())
    aggregated = aggregate_last30_kpod_features(runs, tr_daily)
    aggregated_features = aggregated.drop(columns=["Месторождение", "Принадлежность", "Тип УЭЦН"], errors="ignore")
    merged = runs.merge(aggregated_features, on="row_id", how="inner")
    merged["pump_family"] = merged["Тип УЭЦН"].map(pump_family).astype("string")
    merged["nominal_rate_band"] = nominal_rate_band(merged["Ном. Произв. м₃/сут"])
    merged["kpod_last30_mean"] = _numeric(merged["kpod_last30_mean"])
    merged["Наработка (сут)"] = _numeric(merged["Наработка (сут)"])
    merged["Failure Flag"] = _numeric(merged["Failure Flag"])
    merged = merged.loc[
        merged["kpod_last30_mean"].notna()
        & (merged["kpod_last30_mean"] > 0)
        & merged["Наработка (сут)"].notna()
        & (merged["Наработка (сут)"] > 30)
        & merged["Failure Flag"].isin([0, 1])
    ].copy()
    return merged


def build_models() -> dict[str, list[dict[str, object]]]:
    low_term = {
        "name": "kpod_low",
        "column": "kpod_last30_mean",
        "transform": "negative_excess",
        "reference_mode": "fit",
        "reference_init": 0.70,
        "reference_bounds": [0.10, 1.20],
        "coefficient_mode": "fit",
        "coefficient_value": 0.05,
        "coefficient_non_negative": True,
        "coefficient_bounds": [0.0, None],
    }
    high_term = {
        "name": "kpod_high",
        "column": "kpod_last30_mean",
        "transform": "positive_excess",
        "reference_mode": "fit",
        "reference_init": 0.85,
        "reference_bounds": [0.30, 1.60],
        "coefficient_mode": "fit",
        "coefficient_value": 0.05,
        "coefficient_non_negative": True,
        "coefficient_bounds": [0.0, None],
    }
    return {
        "baseline": [],
        "low_side": [low_term],
        "high_side": [high_term],
        "u_shape": [
            {
                **low_term,
                "reference_init": 0.55,
                "reference_bounds": [0.05, 0.80],
            },
            {
                **high_term,
                "reference_init": 0.85,
                "reference_bounds": [0.60, 1.60],
            },
        ],
    }


def subset_specs(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    specs: list[tuple[str, pd.DataFrame]] = [
        ("ALL", df.copy()),
        ("Field=Ya", df.loc[df["Месторождение"].astype("string") == "Ya"].copy()),
        ("Field=Vt", df.loc[df["Месторождение"].astype("string") == "Vt"].copy()),
    ]

    family_counts = df["pump_family"].astype("string").value_counts(dropna=True)
    for family, count in family_counts.items():
        if int(count) < 80:
            continue
        specs.append((f"Family={family}", df.loc[df["pump_family"].astype("string") == family].copy()))

    band_counts = df["nominal_rate_band"].astype("string").value_counts(dropna=True)
    for band, count in band_counts.items():
        if str(band) == "<missing>" or int(count) < 80:
            continue
        specs.append((f"RateBand={band}", df.loc[df["nominal_rate_band"].astype("string") == band].copy()))

    return specs


def fit_subset_models(name: str, df: pd.DataFrame, model_terms: dict[str, list[dict[str, object]]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if len(df) < 80 or int(_numeric(df["Failure Flag"]).sum()) < 20:
        return pd.DataFrame()

    group_columns = [column for column in ["Месторождение", "Принадлежность"] if column in df.columns]
    baseline_result = fit_weibull_stress_model(
        df,
        duration_column="Наработка (сут)",
        event_column="Failure Flag",
        group_columns=group_columns,
        stress_terms=[],
        min_group_size=20,
    )
    baseline_aic = float(baseline_result.aic)
    baseline_nll = float(baseline_result.nll)

    low_rows = df.loc[df["kpod_last30_mean"] < 0.40, "Наработка (сут)"]
    high_rows = df.loc[df["kpod_last30_mean"] >= 0.85, "Наработка (сут)"]

    for model_name, terms in model_terms.items():
        if model_name == "baseline":
            result = baseline_result
        else:
            result = fit_weibull_stress_model(
                df,
                duration_column="Наработка (сут)",
                event_column="Failure Flag",
                group_columns=group_columns,
                stress_terms=terms,
                min_group_size=20,
            )

        row: dict[str, object] = {
            "subset": name,
            "model": model_name,
            "rows": int(len(result.prepared_df)),
            "failures": int(_numeric(result.prepared_df["Failure Flag"]).sum()),
            "groups": int(len(result.groups)),
            "aic": float(result.aic),
            "delta_aic_vs_baseline": float(result.aic - baseline_aic),
            "nll": float(result.nll),
            "delta_nll_vs_baseline": float(result.nll - baseline_nll),
            "success": bool(result.success),
            "median_ttf_kpod_lt_040": None if low_rows.empty else float(low_rows.median()),
            "median_ttf_kpod_ge_085": None if high_rows.empty else float(high_rows.median()),
        }
        for term in result.stress_terms:
            row[f"{term.name}_coef"] = float(result.stress_coefficients.get(term.name, np.nan))
            row[f"{term.name}_ref"] = float(result.reference_values.get(term.name, np.nan))
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    df = prepare_dataset()
    models = build_models()

    print("## Dataset scope")
    print(f"Rows with last-30 Kpod and TTF>30: {len(df)}")
    print(f"Failures: {int(_numeric(df['Failure Flag']).sum())}")
    print("\n## Pump family counts")
    print(df["pump_family"].astype("string").value_counts(dropna=True).head(12).to_string())
    print("\n## Nominal rate band counts")
    print(df["nominal_rate_band"].astype("string").value_counts(dropna=True).to_string())

    frames: list[pd.DataFrame] = []
    for subset_name, subset_df in subset_specs(df):
        frame = fit_subset_models(subset_name, subset_df, models)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        print("\nNo subset had enough rows for fitting.")
        return

    results = pd.concat(frames, ignore_index=True)

    summary_columns = [
        "subset",
        "model",
        "rows",
        "failures",
        "groups",
        "aic",
        "delta_aic_vs_baseline",
        "nll",
        "delta_nll_vs_baseline",
        "median_ttf_kpod_lt_040",
        "median_ttf_kpod_ge_085",
        "kpod_low_coef",
        "kpod_low_ref",
        "kpod_high_coef",
        "kpod_high_ref",
    ]
    available_columns = [column for column in summary_columns if column in results.columns]

    print("\n## Model comparison")
    print(results[available_columns].sort_values(["subset", "aic", "model"]).to_string(index=False))

    best_rows = (
        results.sort_values(["subset", "aic", "model"])
        .groupby("subset", as_index=False)
        .first()
        .sort_values("aic")
        .reset_index(drop=True)
    )
    print("\n## Best model by subset")
    print(best_rows[available_columns].to_string(index=False))


if __name__ == "__main__":
    main()
