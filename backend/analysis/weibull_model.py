from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .data_utils import build_representative_row, prepare_modeling_dataframe
from .stress_transforms import EPSILON, apply_transform


LARGE_PENALTY = 1e12


def _softplus(value: float) -> float:
    if value > 50:
        return value
    return float(np.log1p(np.exp(value)))


def _inverse_softplus(value: float) -> float:
    clipped = max(float(value), 1e-6)
    if clipped > 50:
        return clipped
    return float(np.log(np.expm1(clipped)))


def _normalize_bounds(bounds: list[float | None] | tuple[float | None, float | None] | None) -> tuple[float | None, float | None]:
    if bounds is None:
        return (None, None)
    lower, upper = bounds
    return lower, upper


def evaluate_weibull_nll(durations: np.ndarray, events: np.ndarray, beta: float, eta: float) -> float:
    if not np.isfinite(beta) or not np.isfinite(eta) or beta <= 0.0 or eta <= 0.0:
        return LARGE_PENALTY

    log_eta = np.log(eta)
    log_t = np.log(np.clip(durations, 1e-12, None))
    power_term = np.clip(beta * (log_t - log_eta), -700.0, 700.0)
    z = np.exp(power_term)
    log_pdf = np.log(beta) - log_eta + (beta - 1.0) * (log_t - log_eta) - z
    log_survival = -z
    log_likelihood = np.where(events == 1, log_pdf, log_survival)
    if not np.isfinite(log_likelihood).all():
        return LARGE_PENALTY
    nll = float(-np.sum(log_likelihood))
    return nll if np.isfinite(nll) else LARGE_PENALTY


def fit_basic_weibull(durations: np.ndarray, events: np.ndarray, beta_init: float = 1.5, eta_init: float | None = None) -> dict[str, float | bool | str]:
    clean_durations = np.asarray(durations, dtype=float)
    clean_events = np.asarray(events, dtype=int)
    clean_mask = np.isfinite(clean_durations) & np.isfinite(clean_events)
    clean_durations = clean_durations[clean_mask]
    clean_events = clean_events[clean_mask]
    if len(clean_durations) == 0:
        raise ValueError("No valid rows were available for Weibull fitting.")

    eta_guess = float(np.median(clean_durations)) if eta_init is None else float(eta_init)
    eta_guess = max(eta_guess, 1e-3)
    x0 = np.asarray([np.log(max(beta_init, 1e-3)), np.log(eta_guess)], dtype=float)
    bounds = [(np.log(1e-3), np.log(1e3)), (np.log(1e-3), np.log(1e9))]

    def objective(params: np.ndarray) -> float:
        beta = float(np.exp(params[0]))
        eta = float(np.exp(params[1]))
        return evaluate_weibull_nll(clean_durations, clean_events, beta, eta)

    optimization = minimize(objective, x0=x0, method="L-BFGS-B", bounds=bounds)
    beta = float(np.exp(optimization.x[0]))
    eta = float(np.exp(optimization.x[1]))
    nll = float(objective(optimization.x))
    parameter_count = 2
    return {
        "beta": beta,
        "eta": eta,
        "nll": nll,
        "aic": float((2 * parameter_count) + (2 * nll)),
        "bic": float((parameter_count * np.log(len(clean_durations))) + (2 * nll)),
        "success": bool(optimization.success),
        "message": str(optimization.message),
    }


@dataclass(slots=True)
class StressTermConfig:
    name: str
    column: str
    transform: str
    reference_mode: str = "fixed"
    reference_value: float | None = None
    reference_column: str | None = None
    reference_init: float | None = None
    reference_bounds: tuple[float | None, float | None] = (None, None)
    reference_multiplier_mode: str = "fixed"
    reference_multiplier_value: float = 1.0
    reference_multiplier_init: float = 1.0
    reference_multiplier_bounds: tuple[float | None, float | None] = (0.1, 10.0)
    coefficient_mode: str = "fit"
    coefficient_value: float = 0.05
    coefficient_bounds: tuple[float | None, float | None] = (0.0, None)
    coefficient_non_negative: bool = True
    scale: float = 1.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StressTermConfig":
        if not payload.get("name"):
            raise ValueError("Each stress term must have a name.")
        if not payload.get("column"):
            raise ValueError(f"Stress term '{payload['name']}' must define a source column.")
        if not payload.get("transform"):
            raise ValueError(f"Stress term '{payload['name']}' must define a transform.")
        return cls(
            name=str(payload["name"]),
            column=str(payload["column"]),
            transform=str(payload["transform"]),
            reference_mode=str(payload.get("reference_mode", "fixed")),
            reference_value=payload.get("reference_value"),
            reference_column=payload.get("reference_column"),
            reference_init=payload.get("reference_init"),
            reference_bounds=_normalize_bounds(payload.get("reference_bounds")),
            reference_multiplier_mode=str(payload.get("reference_multiplier_mode", "fixed")),
            reference_multiplier_value=float(payload.get("reference_multiplier_value", 1.0)),
            reference_multiplier_init=float(payload.get("reference_multiplier_init", 1.0)),
            reference_multiplier_bounds=_normalize_bounds(payload.get("reference_multiplier_bounds", (0.1, 10.0))),
            coefficient_mode=str(payload.get("coefficient_mode", "fit")),
            coefficient_value=float(payload.get("coefficient_value", 0.05)),
            coefficient_bounds=_normalize_bounds(payload.get("coefficient_bounds", (0.0, None))),
            coefficient_non_negative=bool(payload.get("coefficient_non_negative", True)),
            scale=float(payload.get("scale", 1.0) or 1.0),
        )

    def initial_reference_value(self, df: pd.DataFrame) -> float:
        if self.reference_mode == "fixed":
            if self.reference_value is None:
                raise ValueError(f"Stress term '{self.name}' requires reference_value for fixed mode.")
            return float(self.reference_value)
        if self.reference_mode == "fit":
            if self.reference_init is not None:
                return float(self.reference_init)
            median_value = pd.to_numeric(df[self.column], errors="coerce").median()
            return float(median_value)
        if self.reference_mode == "column":
            if not self.reference_column:
                raise ValueError(f"Stress term '{self.name}' requires reference_column for column mode.")
            series = pd.to_numeric(df[self.reference_column], errors="coerce")
            return float(series.median())
        raise ValueError(f"Unsupported reference mode '{self.reference_mode}' for term '{self.name}'.")


@dataclass(slots=True)
class FitStageSummary:
    name: str
    success: bool
    message: str
    nll: float
    aic: float
    bic: float
    parameter_count: int


@dataclass(slots=True)
class WeibullStressFitResult:
    beta_by_group: dict[str, float]
    eta_by_group: dict[str, float]
    stress_coefficients: dict[str, float]
    reference_values: dict[str, float]
    reference_multipliers: dict[str, float]
    nll: float
    aic: float
    bic: float
    success: bool
    message: str
    groups: list[str]
    eta_group_by_original_group: dict[str, str]
    group_stats: dict[str, dict[str, int]]
    stage_summaries: list[FitStageSummary]
    prepared_df: pd.DataFrame = field(repr=False)
    duration_column: str = ""
    event_column: str = ""
    group_columns: list[str] = field(default_factory=list)
    stress_terms: list[StressTermConfig] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "beta_by_group": self.beta_by_group,
            "eta_by_group": self.eta_by_group,
            "stress_coefficients": self.stress_coefficients,
            "reference_values": self.reference_values,
            "reference_multipliers": self.reference_multipliers,
            "nll": self.nll,
            "aic": self.aic,
            "bic": self.bic,
            "success": self.success,
            "message": self.message,
            "groups": self.groups,
            "eta_group_by_original_group": self.eta_group_by_original_group,
            "group_stats": self.group_stats,
            "stage_summaries": [
                {
                    "name": stage.name,
                    "success": stage.success,
                    "message": stage.message,
                    "nll": stage.nll,
                    "aic": stage.aic,
                    "bic": stage.bic,
                    "parameter_count": stage.parameter_count,
                }
                for stage in self.stage_summaries
            ],
        }

    def representative_row(self, group_key: str) -> pd.Series:
        subset = self.prepared_df.loc[self.prepared_df["analysis_original_group"].astype(str) == str(group_key)]
        return build_representative_row(subset if not subset.empty else self.prepared_df)

    def beta_for_group(self, group_key: str) -> float:
        if group_key not in self.beta_by_group:
            raise KeyError(f"Unknown group '{group_key}'.")
        return float(self.beta_by_group[group_key])

    def get_stress_term(self, term_name: str) -> StressTermConfig:
        term = next((item for item in self.stress_terms if item.name == term_name), None)
        if term is None:
            raise KeyError(f"Unknown stress term '{term_name}'.")
        return term

    def eta_group_for_original_group(self, group_key: str) -> str:
        if group_key not in self.eta_group_by_original_group:
            raise KeyError(f"Unknown group '{group_key}'.")
        return str(self.eta_group_by_original_group[group_key])

    def group_source_rows(self, group_key: str) -> pd.DataFrame:
        return self.prepared_df.loc[self.prepared_df["analysis_original_group"].astype(str) == str(group_key)].copy()

    def group_bucket_rows(self, group_key: str) -> pd.DataFrame:
        eta_group = self.eta_group_for_original_group(group_key)
        return self.prepared_df.loc[self.prepared_df["analysis_group_key"].astype(str) == eta_group].copy()

    def evaluate_group_parameters(self, group_key: str, beta: float, eta: float, use_source_rows: bool = True) -> dict[str, float]:
        group_df = self.group_source_rows(group_key) if use_source_rows else self.group_bucket_rows(group_key)
        durations = pd.to_numeric(group_df[self.duration_column], errors="coerce").to_numpy(dtype=float)
        events = pd.to_numeric(group_df[self.event_column], errors="coerce").to_numpy(dtype=int)
        nll = evaluate_weibull_nll(durations, events, float(beta), float(eta))
        parameter_count = 2
        return {
            "nll": nll,
            "aic": float((2 * parameter_count) + (2 * nll)),
            "bic": float((parameter_count * np.log(len(group_df))) + (2 * nll)) if len(group_df) else float("nan"),
            "rows": float(len(group_df)),
        }

    def fit_group_only_weibull(self, group_key: str, use_source_rows: bool = True) -> dict[str, float | bool | str]:
        group_df = self.group_source_rows(group_key) if use_source_rows else self.group_bucket_rows(group_key)
        durations = pd.to_numeric(group_df[self.duration_column], errors="coerce").to_numpy(dtype=float)
        events = pd.to_numeric(group_df[self.event_column], errors="coerce").to_numpy(dtype=int)
        return fit_basic_weibull(durations, events, beta_init=self.beta_for_group(group_key), eta_init=self.adjusted_eta(group_key))

    def _row_stress(self, row: pd.Series | dict[str, Any]) -> float:
        row_series = row if isinstance(row, pd.Series) else pd.Series(row)
        total = 0.0
        for term in self.stress_terms:
            coefficient, reference, _ = self.resolve_term_parameters(term.name, row=row_series)
            transformed = apply_transform(
                term.transform,
                [float(row_series[term.column])],
                [reference],
                scale=term.scale,
            )[0]
            if not np.isfinite(transformed):
                raise ValueError(f"Stress term '{term.name}' produced an invalid value for the selected row.")
            total += coefficient * float(transformed)
        return total

    def resolve_term_parameters(
        self,
        term_name: str,
        row: pd.Series | dict[str, Any] | None = None,
        overrides: dict[str, float] | None = None,
    ) -> tuple[float, float, float]:
        term = self.get_stress_term(term_name)
        row_series = None if row is None else (row if isinstance(row, pd.Series) else pd.Series(row))
        overrides = overrides or {}

        coefficient = float(overrides.get("coefficient", self.stress_coefficients.get(term.name, float(term.coefficient_value))))
        reference_multiplier = float(
            overrides.get("reference_multiplier", self.reference_multipliers.get(term.name, term.reference_multiplier_value))
        )

        if "reference_value" in overrides:
            reference = float(overrides["reference_value"])
        elif term.reference_mode == "fixed":
            reference = float(self.reference_values.get(term.name, float(term.reference_value or 0.0)))
        elif term.reference_mode == "fit":
            reference = float(self.reference_values.get(term.name, term.initial_reference_value(self.prepared_df)))
        else:
            if row_series is None:
                raise ValueError(f"Row context is required to resolve column-based reference for term '{term_name}'.")
            reference = float(row_series[term.reference_column]) * reference_multiplier

        return coefficient, reference, reference_multiplier

    def adjusted_eta_with_term_overrides(
        self,
        group_key: str,
        row: pd.Series | dict[str, Any] | None = None,
        term_overrides: dict[str, dict[str, float]] | None = None,
    ) -> float:
        eta_group = self.eta_group_for_original_group(group_key)
        if eta_group not in self.eta_by_group:
            raise KeyError(f"Unknown eta group '{eta_group}'.")
        if row is None:
            row = self.representative_row(group_key)
        row_series = row if isinstance(row, pd.Series) else pd.Series(row)

        stress = 0.0
        for term in self.stress_terms:
            coefficient, reference, _ = self.resolve_term_parameters(
                term.name,
                row=row_series,
                overrides=(term_overrides or {}).get(term.name),
            )
            transformed = apply_transform(
                term.transform,
                [float(row_series[term.column])],
                [reference],
                scale=term.scale,
            )[0]
            if not np.isfinite(transformed):
                raise ValueError(f"Stress term '{term.name}' produced an invalid value for the selected row.")
            stress += coefficient * float(transformed)
        return float(self.eta_by_group[eta_group] * np.exp(-np.clip(stress, -50.0, 50.0)))

    def adjusted_eta(self, group_key: str, row: pd.Series | dict[str, Any] | None = None) -> float:
        eta_group = self.eta_group_for_original_group(group_key)
        if eta_group not in self.eta_by_group:
            raise KeyError(f"Unknown eta group '{eta_group}'.")
        if row is None:
            row = self.representative_row(group_key)
        stress = self._row_stress(row)
        return float(self.eta_by_group[eta_group] * np.exp(-np.clip(stress, -50.0, 50.0)))

    def curve_frame(self, group_key: str, row: pd.Series | dict[str, Any] | None = None, num_points: int = 200) -> pd.DataFrame:
        group_rows = self.group_source_rows(group_key)
        if group_rows.empty:
            group_rows = self.prepared_df
        beta = self.beta_for_group(group_key)
        eta = self.adjusted_eta(group_key, row=row)
        max_duration = float(max(group_rows[self.duration_column].max(), eta * 3.0))
        times = np.linspace(1e-6, max_duration * 1.1, num_points)
        cdf = 1.0 - np.exp(-np.power(times / eta, beta))
        survival = 1.0 - cdf
        pdf = (beta / eta) * np.power(times / eta, beta - 1.0) * np.exp(-np.power(times / eta, beta))
        return pd.DataFrame({"time": times, "pdf": pdf, "survival": survival, "cdf": cdf})


@dataclass(slots=True)
class _PreparedTerm:
    config: StressTermConfig
    x_values: np.ndarray
    reference_column_values: np.ndarray | None


@dataclass(slots=True)
class _OptimizationResult:
    beta_by_group: dict[str, float]
    eta_by_group: dict[str, float]
    stress_coefficients: dict[str, float]
    reference_values: dict[str, float]
    reference_multipliers: dict[str, float]
    nll: float
    aic: float
    bic: float
    success: bool
    message: str
    parameter_count: int


def _group_stats(df: pd.DataFrame, event_column: str) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}
    for original_group, source_df in df.groupby("analysis_original_group", sort=True):
        source_failures = int(source_df[event_column].sum())
        eta_group = str(source_df["analysis_group_key"].iloc[0])
        bucket_df = df.loc[df["analysis_group_key"].astype(str) == eta_group].copy()
        bucket_failures = int(bucket_df[event_column].sum())
        stats[str(original_group)] = {
            "bucket_rows": int(len(bucket_df)),
            "bucket_failures": bucket_failures,
            "bucket_censored": int(len(bucket_df) - bucket_failures),
            "source_rows": int(len(source_df)),
            "source_failures": source_failures,
            "source_censored": int(len(source_df) - source_failures),
            "group_level": int(source_df["analysis_group_level"].iloc[0]) if "analysis_group_level" in source_df.columns else 0,
            "eta_group": eta_group,
        }
    return stats


def _prepare_terms(df: pd.DataFrame, stress_terms: list[StressTermConfig]) -> list[_PreparedTerm]:
    prepared: list[_PreparedTerm] = []
    for term in stress_terms:
        reference_column_values = None
        if term.reference_mode == "column":
            if term.reference_column not in df.columns:
                raise ValueError(f"Reference column '{term.reference_column}' was not found for term '{term.name}'.")
            reference_column_values = pd.to_numeric(df[term.reference_column], errors="coerce").to_numpy(dtype=float)
        prepared.append(
            _PreparedTerm(
                config=term,
                x_values=pd.to_numeric(df[term.column], errors="coerce").to_numpy(dtype=float),
                reference_column_values=reference_column_values,
            )
        )
    return prepared


def _validate_initial_terms(df: pd.DataFrame, stress_terms: list[StressTermConfig]) -> pd.DataFrame:
    validated = df.copy()
    if not stress_terms:
        return validated

    valid_mask = pd.Series(True, index=validated.index)
    for term in stress_terms:
        source = pd.to_numeric(validated[term.column], errors="coerce").to_numpy(dtype=float)
        multiplier = term.reference_multiplier_init if term.reference_multiplier_mode == "fit" else term.reference_multiplier_value
        if term.reference_mode == "fixed":
            reference = np.full(len(validated), float(term.reference_value), dtype=float)
        elif term.reference_mode == "fit":
            reference = np.full(len(validated), term.initial_reference_value(validated), dtype=float)
        else:
            reference = pd.to_numeric(validated[term.reference_column], errors="coerce").to_numpy(dtype=float) * float(multiplier)

        values = apply_transform(term.transform, source, reference, scale=term.scale)
        valid_mask &= pd.Series(np.isfinite(values), index=validated.index)

    return validated.loc[valid_mask].copy()


def _build_initial_guess(
    df: pd.DataFrame,
    duration_column: str,
    beta_group_keys: list[str],
    eta_group_keys: list[str],
    stress_terms: list[StressTermConfig],
    previous: _OptimizationResult | None = None,
) -> tuple[np.ndarray, list[tuple[float | None, float | None]]]:
    durations = pd.to_numeric(df[duration_column], errors="coerce")
    global_eta = float(np.median(durations))
    if global_eta <= 0:
        global_eta = 100.0

    initial_values: list[float] = []
    bounds: list[tuple[float | None, float | None]] = []

    for group_key in beta_group_keys:
        beta_guess = 1.5
        if previous and group_key in previous.beta_by_group:
            beta_guess = previous.beta_by_group[group_key]
        elif previous and previous.beta_by_group:
            beta_guess = float(np.median(list(previous.beta_by_group.values())))
        initial_values.append(np.log(max(beta_guess, 1e-3)))
        bounds.append((np.log(1e-3), np.log(1e3)))

    for group_key in eta_group_keys:
        eta_guess = global_eta
        if previous and group_key in previous.eta_by_group:
            eta_guess = previous.eta_by_group[group_key]
        elif previous and previous.eta_by_group:
            eta_guess = float(np.median(list(previous.eta_by_group.values())))
        initial_values.append(np.log(max(eta_guess, 1e-3)))
        bounds.append((np.log(1e-3), np.log(1e9)))

    for term in stress_terms:
        if term.coefficient_mode == "fit":
            if previous and term.name in previous.stress_coefficients:
                coefficient_guess = previous.stress_coefficients[term.name]
            else:
                coefficient_guess = max(term.coefficient_value, 0.0) if term.coefficient_non_negative else term.coefficient_value
            if term.coefficient_non_negative:
                initial_values.append(_inverse_softplus(coefficient_guess))
                bounds.append((None, None))
            else:
                initial_values.append(coefficient_guess)
                bounds.append(term.coefficient_bounds)

        if term.reference_mode == "fit":
            reference_guess = previous.reference_values.get(term.name) if previous else None
            if reference_guess is None:
                reference_guess = term.initial_reference_value(df)
            initial_values.append(float(reference_guess))
            bounds.append(term.reference_bounds)

        if term.reference_mode == "column" and term.reference_multiplier_mode == "fit":
            multiplier_guess = previous.reference_multipliers.get(term.name) if previous else None
            if multiplier_guess is None:
                multiplier_guess = term.reference_multiplier_init
            initial_values.append(float(multiplier_guess))
            bounds.append(term.reference_multiplier_bounds)

    return np.asarray(initial_values, dtype=float), bounds


def _fit_core(
    df: pd.DataFrame,
    duration_column: str,
    event_column: str,
    beta_group_column_name: str,
    eta_group_column_name: str,
    stress_terms: list[StressTermConfig],
    previous: _OptimizationResult | None = None,
) -> _OptimizationResult:
    modeling = df.copy()
    beta_group_keys = sorted(modeling[beta_group_column_name].astype(str).unique().tolist())
    beta_group_index_map = {group_key: index for index, group_key in enumerate(beta_group_keys)}
    beta_group_indices = modeling[beta_group_column_name].astype(str).map(beta_group_index_map).to_numpy(dtype=int)
    eta_group_keys = sorted(modeling[eta_group_column_name].astype(str).unique().tolist())
    eta_group_index_map = {group_key: index for index, group_key in enumerate(eta_group_keys)}
    eta_group_indices = modeling[eta_group_column_name].astype(str).map(eta_group_index_map).to_numpy(dtype=int)
    durations = pd.to_numeric(modeling[duration_column], errors="coerce").to_numpy(dtype=float)
    events = pd.to_numeric(modeling[event_column], errors="coerce").to_numpy(dtype=int)
    prepared_terms = _prepare_terms(modeling, stress_terms)

    x0, bounds = _build_initial_guess(
        modeling,
        duration_column,
        beta_group_keys,
        eta_group_keys,
        stress_terms,
        previous=previous,
    )

    def unpack_parameters(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float], dict[str, float], dict[str, float]]:
        offset = 0
        beta_logs = params[offset : offset + len(beta_group_keys)]
        beta_values = np.exp(beta_logs)
        offset += len(beta_group_keys)
        eta_logs = params[offset : offset + len(eta_group_keys)]
        eta_values = np.exp(eta_logs)
        offset += len(eta_group_keys)

        coefficients: dict[str, float] = {}
        reference_values: dict[str, float] = {}
        multipliers: dict[str, float] = {}

        for prepared_term in prepared_terms:
            term = prepared_term.config
            if term.coefficient_mode == "fit":
                raw_value = float(params[offset])
                offset += 1
                coefficients[term.name] = _softplus(raw_value) if term.coefficient_non_negative else raw_value
            else:
                coefficients[term.name] = float(term.coefficient_value)

            if term.reference_mode == "fit":
                reference_values[term.name] = float(params[offset])
                offset += 1
            elif term.reference_mode == "fixed":
                reference_values[term.name] = float(term.reference_value)

            if term.reference_mode == "column":
                if term.reference_multiplier_mode == "fit":
                    multipliers[term.name] = float(params[offset])
                    offset += 1
                else:
                    multipliers[term.name] = float(term.reference_multiplier_value)

        return beta_values, eta_values, coefficients, reference_values, multipliers

    def objective(params: np.ndarray) -> float:
        beta_values, eta_values, coefficients, reference_values, multipliers = unpack_parameters(params)
        if not np.isfinite(beta_values).all() or np.any(beta_values <= 0.0):
            return LARGE_PENALTY

        stress = np.zeros(len(modeling), dtype=float)
        for prepared_term in prepared_terms:
            term = prepared_term.config
            if term.reference_mode == "fixed":
                reference = np.full(len(modeling), reference_values[term.name], dtype=float)
            elif term.reference_mode == "fit":
                reference = np.full(len(modeling), reference_values[term.name], dtype=float)
            else:
                reference = prepared_term.reference_column_values * multipliers[term.name]
            transformed = apply_transform(term.transform, prepared_term.x_values, reference, scale=term.scale)
            if not np.isfinite(transformed).all():
                return LARGE_PENALTY
            stress += coefficients[term.name] * transformed

        beta_per_row = beta_values[beta_group_indices]
        log_eta = np.log(eta_values[eta_group_indices]) - np.clip(stress, -50.0, 50.0)
        log_t = np.log(np.clip(durations, 1e-12, None))
        power_term = np.clip(beta_per_row * (log_t - log_eta), -700.0, 700.0)
        z = np.exp(power_term)
        log_pdf = np.log(beta_per_row) - log_eta + (beta_per_row - 1.0) * (log_t - log_eta) - z
        log_survival = -z
        log_likelihood = np.where(events == 1, log_pdf, log_survival)
        if not np.isfinite(log_likelihood).all():
            return LARGE_PENALTY
        nll = float(-np.sum(log_likelihood))
        return nll if np.isfinite(nll) else LARGE_PENALTY

    optimization = minimize(
        objective,
        x0=x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": 5000,
            "maxfun": 50000,
        },
    )
    beta_values, eta_values, coefficients, reference_values, multipliers = unpack_parameters(optimization.x)
    parameter_count = len(x0)
    nll = float(objective(optimization.x))
    aic = float((2 * parameter_count) + (2 * nll))
    bic = float((parameter_count * np.log(len(modeling))) + (2 * nll))
    beta_by_group = {group_key: float(beta_values[index]) for group_key, index in beta_group_index_map.items()}
    eta_by_group = {group_key: float(eta_values[index]) for group_key, index in eta_group_index_map.items()}

    return _OptimizationResult(
        beta_by_group=beta_by_group,
        eta_by_group=eta_by_group,
        stress_coefficients=coefficients,
        reference_values=reference_values,
        reference_multipliers=multipliers,
        nll=nll,
        aic=aic,
        bic=bic,
        success=bool(optimization.success),
        message=str(optimization.message),
        parameter_count=parameter_count,
    )


def fit_weibull_stress_model(
    df: pd.DataFrame,
    duration_column: str,
    event_column: str,
    group_columns: list[str] | None = None,
    stress_terms: list[dict] | list[StressTermConfig] | None = None,
    filters: list[dict] | None = None,
    min_group_size: int = 20,
) -> WeibullStressFitResult:
    parsed_terms = [
        term if isinstance(term, StressTermConfig) else StressTermConfig.from_dict(term)
        for term in (stress_terms or [])
    ]
    prepared_df, _ = prepare_modeling_dataframe(
        df,
        duration_column=duration_column,
        event_column=event_column,
        group_columns=group_columns,
        stress_terms=[
            {
                "name": term.name,
                "column": term.column,
                "transform": term.transform,
                "reference_mode": term.reference_mode,
                "reference_value": term.reference_value,
                "reference_column": term.reference_column,
            }
            for term in parsed_terms
        ],
        filters=filters,
        min_group_size=min_group_size,
    )
    prepared_df = _validate_initial_terms(prepared_df, parsed_terms)
    if prepared_df.empty:
        raise ValueError("No rows remain after validating stress transform inputs.")

    stage_summaries: list[FitStageSummary] = []

    stage0_df = prepared_df.copy()
    stage0_df["__global_group__"] = "GLOBAL"
    stage0 = _fit_core(
        stage0_df,
        duration_column=duration_column,
        event_column=event_column,
        beta_group_column_name="__global_group__",
        eta_group_column_name="__global_group__",
        stress_terms=[],
        previous=None,
    )
    stage_summaries.append(
        FitStageSummary(
            name="stage_0_global_weibull",
            success=stage0.success,
            message=stage0.message,
            nll=stage0.nll,
            aic=stage0.aic,
            bic=stage0.bic,
            parameter_count=stage0.parameter_count,
        )
    )

    stage1 = _fit_core(
        prepared_df,
        duration_column=duration_column,
        event_column=event_column,
        beta_group_column_name="analysis_original_group",
        eta_group_column_name="analysis_group_key",
        stress_terms=[],
        previous=stage0,
    )
    stage_summaries.append(
        FitStageSummary(
            name="stage_1_grouped_weibull",
            success=stage1.success,
            message=stage1.message,
            nll=stage1.nll,
            aic=stage1.aic,
            bic=stage1.bic,
            parameter_count=stage1.parameter_count,
        )
    )

    final_result = stage1
    if parsed_terms:
        fixed_reference_terms = [
            term
            for term in parsed_terms
            if term.reference_mode != "fit" and not (term.reference_mode == "column" and term.reference_multiplier_mode == "fit")
        ]
        if fixed_reference_terms:
            stage2 = _fit_core(
                prepared_df,
                duration_column=duration_column,
                event_column=event_column,
                beta_group_column_name="analysis_original_group",
                eta_group_column_name="analysis_group_key",
                stress_terms=fixed_reference_terms,
                previous=stage1,
            )
            stage_summaries.append(
                FitStageSummary(
                    name="stage_2_grouped_weibull_fixed_reference_stress",
                    success=stage2.success,
                    message=stage2.message,
                    nll=stage2.nll,
                    aic=stage2.aic,
                    bic=stage2.bic,
                    parameter_count=stage2.parameter_count,
                )
            )
            final_result = stage2

        stage3 = _fit_core(
            prepared_df,
            duration_column=duration_column,
            event_column=event_column,
            beta_group_column_name="analysis_original_group",
            eta_group_column_name="analysis_group_key",
            stress_terms=parsed_terms,
            previous=final_result,
        )
        stage_summaries.append(
            FitStageSummary(
                name="stage_3_grouped_weibull_full_stress",
                success=stage3.success,
                message=stage3.message,
                nll=stage3.nll,
                aic=stage3.aic,
                bic=stage3.bic,
                parameter_count=stage3.parameter_count,
            )
        )
        final_result = stage3

    return WeibullStressFitResult(
        beta_by_group=final_result.beta_by_group,
        eta_by_group=final_result.eta_by_group,
        stress_coefficients=final_result.stress_coefficients,
        reference_values=final_result.reference_values,
        reference_multipliers=final_result.reference_multipliers,
        nll=final_result.nll,
        aic=final_result.aic,
        bic=final_result.bic,
        success=final_result.success,
        message=final_result.message,
        groups=sorted(final_result.beta_by_group),
        eta_group_by_original_group={
            str(original_group): str(group_df["analysis_group_key"].iloc[0])
            for original_group, group_df in prepared_df.groupby("analysis_original_group", sort=True)
        },
        group_stats=_group_stats(prepared_df, event_column),
        stage_summaries=stage_summaries,
        prepared_df=prepared_df,
        duration_column=duration_column,
        event_column=event_column,
        group_columns=[column for column in (group_columns or []) if column in df.columns],
        stress_terms=parsed_terms,
    )
