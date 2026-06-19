from __future__ import annotations

import numpy as np
import pandas as pd

from .stress_transforms import apply_transform
from .weibull_model import WeibullStressFitResult


def build_stress_term_observation_frame(
    result: WeibullStressFitResult,
    term_name: str,
    target_column: str,
) -> pd.DataFrame:
    if target_column not in result.prepared_df.columns:
        raise ValueError(f"Target column '{target_column}' was not found in the prepared model dataframe.")

    term = result.get_stress_term(term_name)
    rows: list[dict[str, object]] = []
    for _, row in result.prepared_df.iterrows():
        coefficient, reference_value, reference_multiplier = result.resolve_term_parameters(term_name, row=row)
        transformed = apply_transform(
            term.transform,
            [float(row[term.column])],
            [reference_value],
            scale=term.scale,
        )[0]
        if not np.isfinite(transformed):
            continue
        entry = row.to_dict()
        entry["stress_value"] = float(transformed)
        entry["weighted_stress_value"] = float(coefficient) * float(transformed)
        entry["stress_reference_value"] = float(reference_value)
        entry["stress_reference_multiplier"] = float(reference_multiplier)
        entry["stress_coefficient"] = float(coefficient)
        entry["target_value"] = pd.to_numeric(pd.Series([entry[target_column]]), errors="coerce").iloc[0]
        rows.append(entry)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["target_value"] = pd.to_numeric(frame["target_value"], errors="coerce")
    return frame.loc[frame["target_value"].notna()].reset_index(drop=True)


def _correlation_row(df: pd.DataFrame, value_column: str, label: str) -> dict[str, object] | None:
    working = df.copy()
    working[value_column] = pd.to_numeric(working[value_column], errors="coerce")
    working["target_value"] = pd.to_numeric(working["target_value"], errors="coerce")
    working = working.dropna(subset=[value_column, "target_value"])
    if len(working) < 3 or working[value_column].nunique() < 2 or working["target_value"].nunique() < 2:
        return None
    pearson = float(working[value_column].corr(working["target_value"], method="pearson"))
    spearman = float(working[value_column].corr(working["target_value"], method="spearman"))
    return {
        "group": label,
        "rows": int(len(working)),
        "pearson_corr": pearson,
        "spearman_corr": spearman,
        "abs_pearson_corr": abs(pearson),
        "abs_spearman_corr": abs(spearman),
        "mean_stress": float(working[value_column].mean()),
        "mean_target": float(working["target_value"].mean()),
    }


def build_grouped_stress_correlation_frame(
    observation_df: pd.DataFrame,
    value_column: str = "weighted_stress_value",
    group_column: str | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    overall = _correlation_row(observation_df, value_column=value_column, label="ALL")
    if overall is not None:
        rows.append(overall)

    if group_column is not None and group_column in observation_df.columns:
        for group_key, group_df in observation_df.groupby(group_column, dropna=False, sort=True):
            label = "<missing>" if pd.isna(group_key) else str(group_key)
            row = _correlation_row(group_df, value_column=value_column, label=label)
            if row is not None:
                rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=["group", "rows", "pearson_corr", "spearman_corr", "abs_pearson_corr", "abs_spearman_corr", "mean_stress", "mean_target"]
        )
    return pd.DataFrame(rows).sort_values(
        ["abs_spearman_corr", "rows", "group"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
