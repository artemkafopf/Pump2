from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .stress_transforms import TRANSFORM_LIBRARY
from .weibull_model import fit_weibull_stress_model


@dataclass(frozen=True, slots=True)
class TransformSelectionResult:
    transform: str
    nll: float
    aic: float
    bic: float
    delta_nll_vs_baseline: float
    delta_aic_vs_baseline: float
    delta_bic_vs_baseline: float
    coefficient: float | None
    reference_value: float | None
    success: bool
    message: str


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def _reference_defaults(df: pd.DataFrame, column: str) -> tuple[float, float, float]:
    series = _numeric_series(df, column).dropna()
    if series.empty:
        raise ValueError(f"Column '{column}' has no numeric values for transform selection.")
    return float(series.median()), float(series.min()), float(series.max())


def _build_term_payload(column: str, transform: str, coefficient_non_negative: bool, scale: float, reference_init: float, reference_lower: float, reference_upper: float) -> dict:
    lower_bound = 0.0 if coefficient_non_negative else -10.0
    return {
        "name": f"transform_scan_{column}",
        "column": column,
        "transform": transform,
        "reference_mode": "fit",
        "reference_init": float(reference_init),
        "reference_bounds": [float(reference_lower), float(reference_upper)],
        "coefficient_mode": "fit",
        "coefficient_value": 0.05,
        "coefficient_non_negative": bool(coefficient_non_negative),
        "coefficient_bounds": [lower_bound, None if coefficient_non_negative else 10.0],
        "scale": float(scale),
    }


def rank_transform_candidates(
    df: pd.DataFrame,
    *,
    duration_column: str,
    event_column: str,
    column: str,
    group_columns: list[str] | None = None,
    filters: list[dict] | None = None,
    min_group_size: int = 20,
    candidate_transforms: list[str] | None = None,
    coefficient_non_negative: bool = True,
    scale: float = 1.0,
) -> list[TransformSelectionResult]:
    transforms = candidate_transforms or sorted(TRANSFORM_LIBRARY)
    reference_init, reference_lower, reference_upper = _reference_defaults(df, column)

    baseline_result = fit_weibull_stress_model(
        df,
        duration_column=duration_column,
        event_column=event_column,
        group_columns=group_columns,
        stress_terms=[],
        filters=filters,
        min_group_size=min_group_size,
    )

    ranked: list[TransformSelectionResult] = []
    for transform in transforms:
        term = _build_term_payload(
            column=column,
            transform=transform,
            coefficient_non_negative=coefficient_non_negative,
            scale=scale,
            reference_init=reference_init,
            reference_lower=reference_lower,
            reference_upper=reference_upper,
        )
        try:
            result = fit_weibull_stress_model(
                df,
                duration_column=duration_column,
                event_column=event_column,
                group_columns=group_columns,
                stress_terms=[term],
                filters=filters,
                min_group_size=min_group_size,
            )
            coefficient = result.stress_coefficients.get(term["name"])
            reference_value = result.reference_values.get(term["name"])
            ranked.append(
                TransformSelectionResult(
                    transform=transform,
                    nll=float(result.nll),
                    aic=float(result.aic),
                    bic=float(result.bic),
                    delta_nll_vs_baseline=float(result.nll - baseline_result.nll),
                    delta_aic_vs_baseline=float(result.aic - baseline_result.aic),
                    delta_bic_vs_baseline=float(result.bic - baseline_result.bic),
                    coefficient=None if coefficient is None else float(coefficient),
                    reference_value=None if reference_value is None else float(reference_value),
                    success=bool(result.success),
                    message=str(result.message),
                )
            )
        except Exception as exc:
            ranked.append(
                TransformSelectionResult(
                    transform=transform,
                    nll=float("inf"),
                    aic=float("inf"),
                    bic=float("inf"),
                    delta_nll_vs_baseline=float("inf"),
                    delta_aic_vs_baseline=float("inf"),
                    delta_bic_vs_baseline=float("inf"),
                    coefficient=None,
                    reference_value=None,
                    success=False,
                    message=str(exc),
                )
            )

    return sorted(
        ranked,
        key=lambda item: (
            0 if item.success else 1,
            item.aic,
            item.nll,
            item.transform,
        ),
    )
