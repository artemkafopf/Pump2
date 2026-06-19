from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_EXCLUDED_COLUMNS = {
    "event",
    "infant_mortality_flag",
}


def _feature_group(column: str) -> str:
    if column.startswith("regime_"):
        return "regime"
    if column.startswith("telemetry_"):
        return "telemetry"
    if column.startswith("design_"):
        return "design"
    if column.startswith("free_gas_"):
        return "gas"
    return "run_level"


def _candidate_numeric_columns(
    df: pd.DataFrame,
    target_column: str,
    excluded_columns: Iterable[str] | None = None,
) -> list[str]:
    excluded = set(excluded_columns or set()) | {target_column}
    candidates: list[str] = []
    for column in df.columns:
        if column in excluded:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().sum() > 0:
            candidates.append(column)
    return candidates


def build_ttf_correlation_frame(
    df: pd.DataFrame,
    target_column: str = "TTF_days",
    feature_columns: Iterable[str] | None = None,
    excluded_columns: Iterable[str] | None = None,
    min_pairs: int = 5,
) -> pd.DataFrame:
    if target_column not in df.columns:
        return pd.DataFrame(
            columns=[
                "feature",
                "feature_group",
                "valid_pairs",
                "pearson_corr",
                "spearman_corr",
                "abs_pearson_corr",
                "abs_spearman_corr",
                "feature_mean",
                "feature_std",
            ]
        )

    target = pd.to_numeric(df[target_column], errors="coerce")
    selected_features = list(feature_columns) if feature_columns is not None else _candidate_numeric_columns(
        df,
        target_column=target_column,
        excluded_columns=set(DEFAULT_EXCLUDED_COLUMNS) | set(excluded_columns or set()),
    )

    rows: list[dict[str, float | int | str]] = []
    for column in selected_features:
        feature = pd.to_numeric(df[column], errors="coerce")
        pair_frame = pd.DataFrame({"feature": feature, "target": target}).dropna()
        if len(pair_frame) < max(2, int(min_pairs)):
            continue
        if pair_frame["feature"].nunique() < 2 or pair_frame["target"].nunique() < 2:
            continue
        pearson_corr = float(pair_frame["feature"].corr(pair_frame["target"], method="pearson"))
        spearman_corr = float(pair_frame["feature"].corr(pair_frame["target"], method="spearman"))
        rows.append(
            {
                "feature": column,
                "feature_group": _feature_group(column),
                "valid_pairs": int(len(pair_frame)),
                "pearson_corr": pearson_corr,
                "spearman_corr": spearman_corr,
                "abs_pearson_corr": abs(pearson_corr),
                "abs_spearman_corr": abs(spearman_corr),
                "feature_mean": float(pair_frame["feature"].mean()),
                "feature_std": float(pair_frame["feature"].std(ddof=1)),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "feature",
                "feature_group",
                "valid_pairs",
                "pearson_corr",
                "spearman_corr",
                "abs_pearson_corr",
                "abs_spearman_corr",
                "feature_mean",
                "feature_std",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        ["abs_spearman_corr", "abs_pearson_corr", "valid_pairs", "feature"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
