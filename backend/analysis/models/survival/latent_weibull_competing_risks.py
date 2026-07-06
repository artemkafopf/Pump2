from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize


EPSILON = 1e-12
LATENT_BETA_1_MAX = 1.0
LATENT_BETA_2_MIN = 1.0
DEFAULT_MIN_WEIGHT_1 = 0.05
DEFAULT_WEIGHT_SCHEME = "blended_tail"
WEIGHT_SCHEMES = {
    "risk_weighted",
    "uniform",
    "sqrt_risk",
    "blended_tail",
}


def _validate_positive(name: str, value: float) -> float:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be a finite positive number.")
    return numeric


def _validate_probability(name: str, value: float) -> float:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
    return numeric


def _clip_probability_open_interval(value: float, epsilon: float = 1e-9) -> float:
    return float(np.clip(float(value), epsilon, 1.0 - epsilon))


def _clip_latent_component_betas(beta_1: float, beta_2: float) -> tuple[float, float]:
    return (
        float(np.clip(float(beta_1), 0.05, LATENT_BETA_1_MAX)),
        float(np.clip(float(beta_2), LATENT_BETA_2_MIN, 8.0)),
    )


def _clip_min_weight_1(min_weight_1: float) -> float:
    return float(np.clip(float(min_weight_1), 1e-4, 0.95))


def _validate_weight_scheme(weight_scheme: str) -> str:
    scheme = str(weight_scheme).strip().lower()
    if scheme not in WEIGHT_SCHEMES:
        raise ValueError(f"weight_scheme must be one of {sorted(WEIGHT_SCHEMES)}.")
    return scheme


def _as_time_array(times: float | Iterable[float] | np.ndarray) -> tuple[np.ndarray, bool]:
    values = np.asarray(times, dtype=float)
    is_scalar = values.ndim == 0
    values = np.atleast_1d(values).astype(float)
    if not np.isfinite(values).all():
        raise ValueError("Time values must be finite.")
    if (values < 0.0).any():
        raise ValueError("Time values must be non-negative.")
    return values, is_scalar


def _restore_shape(values: np.ndarray, is_scalar: bool) -> float | np.ndarray:
    if is_scalar:
        return float(values[0])
    return values


@dataclass(frozen=True, slots=True)
class WeibullParameters:
    beta: float
    eta: float
    label: str = "Weibull"

    def __post_init__(self) -> None:
        object.__setattr__(self, "beta", _validate_positive("beta", self.beta))
        object.__setattr__(self, "eta", _validate_positive("eta", self.eta))


@dataclass(frozen=True, slots=True)
class TwoComponentLatentWeibullModel:
    weight_1: float
    component_1: WeibullParameters
    component_2: WeibullParameters

    def __post_init__(self) -> None:
        object.__setattr__(self, "weight_1", _validate_probability("weight_1", self.weight_1))

    @property
    def weight_2(self) -> float:
        return 1.0 - float(self.weight_1)


@dataclass(frozen=True, slots=True)
class ThreeComponentLatentWeibullModel:
    weight_1: float
    weight_2: float
    component_1: WeibullParameters
    component_2: WeibullParameters
    component_3: WeibullParameters

    def __post_init__(self) -> None:
        weight_1 = _validate_probability("weight_1", self.weight_1)
        weight_2 = _validate_probability("weight_2", self.weight_2)
        if weight_1 + weight_2 >= 1.0:
            raise ValueError("weight_1 + weight_2 must be less than 1 for a three-component model.")
        object.__setattr__(self, "weight_1", weight_1)
        object.__setattr__(self, "weight_2", weight_2)

    @property
    def weight_3(self) -> float:
        return 1.0 - float(self.weight_1) - float(self.weight_2)


@dataclass(frozen=True, slots=True)
class CompetingRiskModel:
    causes: tuple[WeibullParameters, ...]

    def __post_init__(self) -> None:
        if len(self.causes) < 2:
            raise ValueError("CompetingRiskModel requires at least two causes.")


@dataclass(frozen=True, slots=True)
class LatentWeibullCurveFitResult:
    model: TwoComponentLatentWeibullModel | ThreeComponentLatentWeibullModel
    success: bool
    message: str
    objective_value: float
    rmse: float
    weighted_rmse: float
    n_iter: int
    nfev: int
    n_starts: int = 1
    best_start_index: int = 0
    method: str = "weighted_least_squares_to_km"


@dataclass(frozen=True, slots=True)
class ClassicalWeibullCurveFitResult:
    params: WeibullParameters
    success: bool
    message: str
    objective_value: float
    rmse: float
    weighted_rmse: float
    n_iter: int
    nfev: int
    n_starts: int = 1
    best_start_index: int = 0
    method: str = "weighted_least_squares_to_km"


def _latent_component_rows(
    model: TwoComponentLatentWeibullModel | ThreeComponentLatentWeibullModel,
) -> list[tuple[WeibullParameters, float]]:
    if isinstance(model, ThreeComponentLatentWeibullModel):
        return [
            (model.component_1, float(model.weight_1)),
            (model.component_2, float(model.weight_2)),
            (model.component_3, float(model.weight_3)),
        ]
    return [
        (model.component_1, float(model.weight_1)),
        (model.component_2, float(model.weight_2)),
    ]


def _coerce_three_component_model_by_eta(model: ThreeComponentLatentWeibullModel) -> ThreeComponentLatentWeibullModel:
    rows = [
        (float(model.component_1.eta), model.component_1, float(model.weight_1)),
        (float(model.component_2.eta), model.component_2, float(model.weight_2)),
        (float(model.component_3.eta), model.component_3, float(model.weight_3)),
    ]
    rows.sort(key=lambda row: row[0])
    total = sum(row[2] for row in rows)
    weights = [float(np.clip(row[2] / max(total, EPSILON), 1e-9, 1.0)) for row in rows]
    weight_1 = weights[0]
    weight_2 = weights[1]
    if weight_1 + weight_2 >= 1.0:
        weight_1 = min(weight_1, 0.999998)
        weight_2 = min(weight_2, 0.999999 - weight_1)
    return ThreeComponentLatentWeibullModel(
        weight_1=weight_1,
        weight_2=weight_2,
        component_1=rows[0][1],
        component_2=rows[1][1],
        component_3=rows[2][1],
    )


def weibull_cumulative_hazard(times: float | Iterable[float] | np.ndarray, params: WeibullParameters) -> float | np.ndarray:
    t, is_scalar = _as_time_array(times)
    values = np.power(t / params.eta, params.beta)
    return _restore_shape(values, is_scalar)


def weibull_survival(times: float | Iterable[float] | np.ndarray, params: WeibullParameters) -> float | np.ndarray:
    cumulative_hazard = np.asarray(weibull_cumulative_hazard(times, params), dtype=float)
    values = np.exp(-np.clip(cumulative_hazard, 0.0, 700.0))
    is_scalar = np.asarray(times).ndim == 0
    return _restore_shape(np.atleast_1d(values), is_scalar)


def weibull_cdf(times: float | Iterable[float] | np.ndarray, params: WeibullParameters) -> float | np.ndarray:
    survival = np.asarray(weibull_survival(times, params), dtype=float)
    values = 1.0 - survival
    is_scalar = np.asarray(times).ndim == 0
    return _restore_shape(np.atleast_1d(values), is_scalar)


def weibull_hazard(times: float | Iterable[float] | np.ndarray, params: WeibullParameters) -> float | np.ndarray:
    t, is_scalar = _as_time_array(times)
    safe_t = np.clip(t, EPSILON, None)
    values = (params.beta / params.eta) * np.power(safe_t / params.eta, params.beta - 1.0)
    return _restore_shape(values, is_scalar)


def weibull_density(times: float | Iterable[float] | np.ndarray, params: WeibullParameters) -> float | np.ndarray:
    hazard = np.asarray(weibull_hazard(times, params), dtype=float)
    survival = np.asarray(weibull_survival(times, params), dtype=float)
    values = hazard * survival
    is_scalar = np.asarray(times).ndim == 0
    return _restore_shape(np.atleast_1d(values), is_scalar)


def latent_survival(
    times: float | Iterable[float] | np.ndarray,
    model: TwoComponentLatentWeibullModel | ThreeComponentLatentWeibullModel,
) -> float | np.ndarray:
    values = None
    for component, weight in _latent_component_rows(model):
        component_survival = np.asarray(weibull_survival(times, component), dtype=float)
        values = (weight * component_survival) if values is None else (values + (weight * component_survival))
    assert values is not None
    is_scalar = np.asarray(times).ndim == 0
    return _restore_shape(np.atleast_1d(values), is_scalar)


def latent_density(
    times: float | Iterable[float] | np.ndarray,
    model: TwoComponentLatentWeibullModel | ThreeComponentLatentWeibullModel,
) -> float | np.ndarray:
    values = None
    for component, weight in _latent_component_rows(model):
        component_density = np.asarray(weibull_density(times, component), dtype=float)
        values = (weight * component_density) if values is None else (values + (weight * component_density))
    assert values is not None
    is_scalar = np.asarray(times).ndim == 0
    return _restore_shape(np.atleast_1d(values), is_scalar)


def latent_hazard(
    times: float | Iterable[float] | np.ndarray,
    model: TwoComponentLatentWeibullModel | ThreeComponentLatentWeibullModel,
) -> float | np.ndarray:
    density = np.asarray(latent_density(times, model), dtype=float)
    survival = np.asarray(latent_survival(times, model), dtype=float)
    values = density / np.clip(survival, EPSILON, None)
    is_scalar = np.asarray(times).ndim == 0
    return _restore_shape(np.atleast_1d(values), is_scalar)


def latent_posterior_given_survival(
    times: float | Iterable[float] | np.ndarray,
    model: TwoComponentLatentWeibullModel | ThreeComponentLatentWeibullModel,
) -> tuple[float | np.ndarray, ...]:
    component_terms = [
        float(weight) * np.asarray(weibull_survival(times, component), dtype=float)
        for component, weight in _latent_component_rows(model)
    ]
    denominator = np.clip(np.sum(component_terms, axis=0), EPSILON, None)
    posterior_terms = [term / denominator for term in component_terms]
    is_scalar = np.asarray(times).ndim == 0
    return tuple(_restore_shape(np.atleast_1d(term), is_scalar) for term in posterior_terms)


def latent_posterior_given_failure(
    times: float | Iterable[float] | np.ndarray,
    model: TwoComponentLatentWeibullModel | ThreeComponentLatentWeibullModel,
) -> tuple[float | np.ndarray, ...]:
    component_terms = [
        float(weight) * np.asarray(weibull_density(times, component), dtype=float)
        for component, weight in _latent_component_rows(model)
    ]
    denominator = np.clip(np.sum(component_terms, axis=0), EPSILON, None)
    posterior_terms = [term / denominator for term in component_terms]
    is_scalar = np.asarray(times).ndim == 0
    return tuple(_restore_shape(np.atleast_1d(term), is_scalar) for term in posterior_terms)


def latent_next_window_failure_probability(
    current_age: float,
    horizon: float,
    model: TwoComponentLatentWeibullModel | ThreeComponentLatentWeibullModel,
) -> float:
    age = _validate_positive("current_age or zero-compatible age", max(float(current_age), 0.0) + EPSILON) - EPSILON
    window = _validate_positive("horizon", horizon)
    s_now = float(latent_survival(age, model))
    s_future = float(latent_survival(age + window, model))
    return float(np.clip(1.0 - (s_future / np.clip(s_now, EPSILON, None)), 0.0, 1.0))


def latent_life_quantile(probability: float, model: TwoComponentLatentWeibullModel | ThreeComponentLatentWeibullModel) -> float:
    p = _validate_probability("probability", probability)
    if p <= 0.0 or p >= 1.0:
        raise ValueError("probability must be strictly between 0 and 1.")
    target_survival = 1.0 - p
    upper = max(model.component_1.eta, model.component_2.eta)
    for _ in range(60):
        if float(latent_survival(upper, model)) <= target_survival:
            break
        upper *= 2.0
    else:
        return float("nan")
    return float(brentq(lambda t: float(latent_survival(t, model)) - target_survival, 0.0, upper))


def latent_remaining_life_quantile(
    current_age: float,
    probability: float,
    model: TwoComponentLatentWeibullModel | ThreeComponentLatentWeibullModel,
) -> float:
    age = max(float(current_age), 0.0)
    p = _validate_probability("probability", probability)
    if p <= 0.0 or p >= 1.0:
        raise ValueError("probability must be strictly between 0 and 1.")
    survival_now = float(latent_survival(age, model))
    target_survival = (1.0 - p) * survival_now
    upper = max(model.component_1.eta, model.component_2.eta, age + 1.0)
    for _ in range(60):
        if float(latent_survival(age + upper, model)) <= target_survival:
            break
        upper *= 2.0
    else:
        return float("nan")
    return float(brentq(lambda r: float(latent_survival(age + r, model)) - target_survival, 0.0, upper))


def latent_curve_frame(
    model: TwoComponentLatentWeibullModel | ThreeComponentLatentWeibullModel,
    max_time: float,
    num_points: int = 400,
) -> pd.DataFrame:
    time_grid = np.linspace(0.0, _validate_positive("max_time", max_time), int(num_points))
    survival = np.asarray(latent_survival(time_grid, model), dtype=float)
    density = np.asarray(latent_density(time_grid, model), dtype=float)
    hazard = np.asarray(latent_hazard(time_grid, model), dtype=float)
    posterior_terms = latent_posterior_given_survival(time_grid, model)
    frame_dict: dict[str, np.ndarray] = {
        "time": time_grid,
        "survival": survival,
        "failure": 1.0 - survival,
        "density": density,
        "hazard": hazard,
    }
    for index, (component, _) in enumerate(_latent_component_rows(model), start=1):
        frame_dict[f"component_{index}_survival"] = np.asarray(weibull_survival(time_grid, component), dtype=float)
    for index, posterior in enumerate(posterior_terms, start=1):
        frame_dict[f"posterior_{index}_survival"] = np.asarray(posterior, dtype=float)
    return pd.DataFrame(frame_dict)


def _normalize_weight_pair(weight_1: float, weight_2: float) -> tuple[float, float]:
    weights = np.asarray([float(weight_1), float(weight_2)], dtype=float)
    weights = np.clip(weights, EPSILON, None)
    weights = weights / weights.sum()
    return float(weights[0]), float(weights[1])


def _logit(probability: float) -> float:
    p = _clip_probability_open_interval(probability)
    return float(np.log(p / (1.0 - p)))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = np.exp(-value)
        return float(1.0 / (1.0 + z))
    z = np.exp(value)
    return float(z / (1.0 + z))


def _coerce_latent_model(model: TwoComponentLatentWeibullModel) -> TwoComponentLatentWeibullModel:
    beta_1, beta_2 = _clip_latent_component_betas(model.component_1.beta, model.component_2.beta)
    return TwoComponentLatentWeibullModel(
        weight_1=_clip_probability_open_interval(model.weight_1),
        component_1=WeibullParameters(beta=beta_1, eta=model.component_1.eta, label=model.component_1.label),
        component_2=WeibullParameters(beta=beta_2, eta=model.component_2.eta, label=model.component_2.label),
    )


def _van_der_corput(index: int, base: int) -> float:
    value = 0.0
    denominator = 1.0
    current = int(index)
    while current > 0:
        current, remainder = divmod(current, base)
        denominator *= float(base)
        value += float(remainder) / denominator
    return value


def _halton_points(num_points: int, dimension: int) -> np.ndarray:
    if num_points <= 0 or dimension <= 0:
        return np.zeros((0, max(int(dimension), 0)), dtype=float)
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    if dimension > len(primes):
        raise ValueError(f"Halton helper only supports up to {len(primes)} dimensions.")
    return np.asarray(
        [
            [_van_der_corput(point_index, primes[dim_index]) for dim_index in range(dimension)]
            for point_index in range(1, num_points + 1)
        ],
        dtype=float,
    )


def filter_km_frame_for_fit(km_frame: pd.DataFrame, min_n_risk: int = 0) -> pd.DataFrame:
    if min_n_risk <= 0 or "n_risk" not in km_frame.columns:
        return km_frame.copy()
    filtered = km_frame.loc[(km_frame["time"].eq(0.0)) | (km_frame["n_risk"] >= int(min_n_risk))].copy()
    if len(filtered) < 3:
        return km_frame.copy()
    return filtered


def build_survival_curve_weights(
    times: Iterable[float] | np.ndarray,
    *,
    n_risk: Iterable[float] | np.ndarray | None = None,
    weight_scheme: str = DEFAULT_WEIGHT_SCHEME,
    gamma: float = 0.5,
    tail_lambda: float = 1.0,
    tail_power: float = 1.0,
) -> np.ndarray:
    scheme = _validate_weight_scheme(weight_scheme)
    time_values = np.asarray(times, dtype=float)
    if time_values.ndim != 1:
        raise ValueError("times must be one-dimensional.")
    if time_values.size == 0:
        return np.asarray([], dtype=float)
    time_values = np.where(np.isfinite(time_values) & (time_values >= 0.0), time_values, 0.0)
    time_max = float(np.max(time_values)) if time_values.size else 0.0
    time_norm = time_values / max(time_max, 1.0)

    if n_risk is None:
        risk_norm = np.ones_like(time_values, dtype=float)
    else:
        risk_values = np.asarray(n_risk, dtype=float)
        if risk_values.shape != time_values.shape:
            raise ValueError("n_risk must match times shape.")
        risk_values = np.where(np.isfinite(risk_values) & (risk_values > 0.0), risk_values, 1.0)
        risk_norm = risk_values / max(float(np.max(risk_values)), 1.0)

    gamma_value = max(float(gamma), 0.0)
    tail_lambda_value = max(float(tail_lambda), 0.0)
    tail_power_value = max(float(tail_power), 0.0)

    if scheme == "uniform":
        weights = np.ones_like(time_values, dtype=float)
    elif scheme == "risk_weighted":
        weights = risk_norm
    elif scheme == "sqrt_risk":
        weights = np.sqrt(risk_norm)
    else:
        risk_component = np.power(risk_norm, gamma_value)
        tail_component = 1.0 + (tail_lambda_value * np.power(time_norm, tail_power_value))
        weights = risk_component * tail_component

    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, EPSILON)
    weights = weights / max(float(np.max(weights)), EPSILON)
    return weights.astype(float)


def fit_latent_weibull_to_survival_curve(
    times: Iterable[float] | np.ndarray,
    survival: Iterable[float] | np.ndarray,
    *,
    initial_model: TwoComponentLatentWeibullModel,
    curve_weights: Iterable[float] | np.ndarray | None = None,
    optimize_beta_1: bool = True,
    optimize_eta_1: bool = True,
    optimize_beta_2: bool = True,
    optimize_eta_2: bool = True,
    optimize_weight_1: bool = True,
    optimize_weight_2: bool = True,
    min_weight_1: float = DEFAULT_MIN_WEIGHT_1,
    num_starts: int = 1,
    max_iter: int = 400,
) -> LatentWeibullCurveFitResult:
    time_values = np.asarray(times, dtype=float)
    survival_values = np.asarray(survival, dtype=float)
    if time_values.ndim != 1 or survival_values.ndim != 1 or time_values.shape[0] != survival_values.shape[0]:
        raise ValueError("times and survival must be one-dimensional arrays of equal length.")
    valid = np.isfinite(time_values) & np.isfinite(survival_values) & (time_values >= 0.0)
    time_values = time_values[valid]
    survival_values = survival_values[valid]
    if time_values.size < 3:
        raise ValueError("At least three valid survival points are required.")
    order = np.argsort(time_values)
    time_values = time_values[order]
    survival_values = np.clip(survival_values[order], 0.0, 1.0)

    if curve_weights is None:
        weight_values = np.ones_like(time_values, dtype=float)
    else:
        weight_values = np.asarray(curve_weights, dtype=float)
        weight_values = weight_values[valid][order]
        weight_values = np.where(np.isfinite(weight_values) & (weight_values > 0.0), weight_values, 1.0)
    weight_values = weight_values / np.clip(weight_values.max(), EPSILON, None)

    if time_values[0] > 0.0:
        time_values = np.insert(time_values, 0, 0.0)
        survival_values = np.insert(survival_values, 0, 1.0)
        weight_values = np.insert(weight_values, 0, 1.0)
    elif not np.isclose(survival_values[0], 1.0):
        survival_values[0] = 1.0

    initial_model = _coerce_latent_model(initial_model)

    optimize_weight = bool(optimize_weight_1 or optimize_weight_2)
    min_weight_1 = _clip_min_weight_1(min_weight_1)

    base_theta = np.asarray(
        [
            np.log(initial_model.component_1.beta),
            np.log(initial_model.component_1.eta),
            np.log(initial_model.component_2.beta),
            np.log(initial_model.component_2.eta),
            _logit(initial_model.weight_1),
        ],
        dtype=float,
    )
    optimize_mask = np.asarray(
        [
            optimize_beta_1,
            optimize_eta_1,
            optimize_beta_2,
            optimize_eta_2,
            optimize_weight,
        ],
        dtype=bool,
    )

    def _full_theta(free_theta: np.ndarray) -> np.ndarray:
        theta = base_theta.copy()
        theta[optimize_mask] = np.asarray(free_theta, dtype=float)
        return theta

    def _build_model(theta: np.ndarray) -> TwoComponentLatentWeibullModel:
        full_theta = _full_theta(theta)
        beta_1 = float(np.exp(full_theta[0]))
        eta_1 = float(np.exp(full_theta[1]))
        beta_2 = float(np.exp(full_theta[2]))
        eta_2 = float(np.exp(full_theta[3]))
        beta_1, beta_2 = _clip_latent_component_betas(beta_1, beta_2)
        weight_1 = _sigmoid(float(full_theta[4]))
        model = TwoComponentLatentWeibullModel(
            weight_1=weight_1,
            component_1=WeibullParameters(beta=beta_1, eta=eta_1, label=initial_model.component_1.label),
            component_2=WeibullParameters(beta=beta_2, eta=eta_2, label=initial_model.component_2.label),
        )
        return _coerce_latent_model(model)

    def _objective(theta: np.ndarray) -> float:
        model = _build_model(theta)
        fitted = np.asarray(latent_survival(time_values, model), dtype=float)
        residual = fitted - survival_values
        return float(np.mean(weight_values * np.square(residual)))

    full_bounds = [
        (np.log(0.05), np.log(LATENT_BETA_1_MAX)),
        (np.log(10.0), np.log(5000.0)),
        (np.log(LATENT_BETA_2_MIN), np.log(8.0)),
        (np.log(10.0), np.log(5000.0)),
        (_logit(min_weight_1), _logit(1.0 - 1e-4)),
    ]
    theta0 = base_theta[optimize_mask]
    bounds = [bound for bound, optimize in zip(full_bounds, optimize_mask, strict=False) if optimize]

    if theta0.size == 0:
        fitted_model = _build_model(np.asarray([], dtype=float))
        fitted_survival = np.asarray(latent_survival(time_values, fitted_model), dtype=float)
        residual = fitted_survival - survival_values
        rmse = float(np.sqrt(np.mean(np.square(residual))))
        weighted_rmse = float(np.sqrt(np.mean(weight_values * np.square(residual))))
        return LatentWeibullCurveFitResult(
            model=fitted_model,
            success=True,
            message="No free parameters selected; returned fixed initial model.",
            objective_value=float(np.mean(weight_values * np.square(residual))),
            rmse=rmse,
            weighted_rmse=weighted_rmse,
            n_iter=0,
            nfev=0,
            n_starts=1,
            best_start_index=0,
        )

    start_count = max(int(num_starts), 1)
    lower = np.asarray([bound[0] for bound in bounds], dtype=float)
    upper = np.asarray([bound[1] for bound in bounds], dtype=float)
    starts = [np.clip(theta0, lower, upper)]
    if start_count > 1:
        for point in _halton_points(start_count - 1, len(bounds)):
            starts.append(lower + (upper - lower) * point)

    best_result = None
    best_index = 0
    total_nit = 0
    total_nfev = 0
    for start_index, start in enumerate(starts):
        result = minimize(
            _objective,
            np.asarray(start, dtype=float),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(max_iter)},
        )
        total_nit += int(getattr(result, "nit", 0))
        total_nfev += int(getattr(result, "nfev", 0))
        if best_result is None:
            best_result = result
            best_index = start_index
            continue
        if bool(result.success) and not bool(best_result.success):
            best_result = result
            best_index = start_index
            continue
        if float(result.fun) < float(best_result.fun):
            best_result = result
            best_index = start_index

    assert best_result is not None
    fitted_model = _build_model(np.asarray(best_result.x, dtype=float))
    fitted_survival = np.asarray(latent_survival(time_values, fitted_model), dtype=float)
    residual = fitted_survival - survival_values
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    weighted_rmse = float(np.sqrt(np.mean(weight_values * np.square(residual))))
    return LatentWeibullCurveFitResult(
        model=fitted_model,
        success=bool(best_result.success),
        message=str(best_result.message),
        objective_value=float(best_result.fun),
        rmse=rmse,
        weighted_rmse=weighted_rmse,
        n_iter=total_nit,
        nfev=total_nfev,
        n_starts=len(starts),
        best_start_index=best_index,
    )


def fit_latent_weibull_to_km_frame(
    km_frame: pd.DataFrame,
    *,
    initial_model: TwoComponentLatentWeibullModel,
    optimize_beta_1: bool = True,
    optimize_eta_1: bool = True,
    optimize_beta_2: bool = True,
    optimize_eta_2: bool = True,
    optimize_weight_1: bool = True,
    optimize_weight_2: bool = True,
    min_weight_1: float = DEFAULT_MIN_WEIGHT_1,
    weight_scheme: str = DEFAULT_WEIGHT_SCHEME,
    weight_gamma: float = 0.5,
    weight_tail_lambda: float = 1.0,
    weight_tail_power: float = 1.0,
    min_n_risk: int = 0,
    num_starts: int = 1,
    max_iter: int = 400,
) -> LatentWeibullCurveFitResult:
    required = {"time", "survival"}
    missing = required.difference(km_frame.columns)
    if missing:
        raise KeyError(f"KM frame is missing required columns: {sorted(missing)}")
    fit_frame = filter_km_frame_for_fit(km_frame, min_n_risk=min_n_risk)
    n_risk = fit_frame["n_risk"].to_numpy(dtype=float) if "n_risk" in fit_frame.columns else None
    weights = build_survival_curve_weights(
        fit_frame["time"].to_numpy(dtype=float),
        n_risk=n_risk,
        weight_scheme=weight_scheme,
        gamma=weight_gamma,
        tail_lambda=weight_tail_lambda,
        tail_power=weight_tail_power,
    )
    return fit_latent_weibull_to_survival_curve(
        fit_frame["time"].to_numpy(dtype=float),
        fit_frame["survival"].to_numpy(dtype=float),
        initial_model=initial_model,
        curve_weights=weights,
        optimize_beta_1=optimize_beta_1,
        optimize_eta_1=optimize_eta_1,
        optimize_beta_2=optimize_beta_2,
        optimize_eta_2=optimize_eta_2,
        optimize_weight_1=optimize_weight_1,
        optimize_weight_2=optimize_weight_2,
        min_weight_1=min_weight_1,
        num_starts=num_starts,
        max_iter=max_iter,
    )


def fit_three_component_latent_weibull_to_survival_curve(
    times: Iterable[float] | np.ndarray,
    survival: Iterable[float] | np.ndarray,
    *,
    initial_model: ThreeComponentLatentWeibullModel,
    curve_weights: Iterable[float] | np.ndarray | None = None,
    optimize_beta_1: bool = True,
    optimize_eta_1: bool = True,
    optimize_beta_2: bool = True,
    optimize_eta_2: bool = True,
    optimize_beta_3: bool = True,
    optimize_eta_3: bool = True,
    optimize_weight_1: bool = True,
    optimize_weight_2: bool = True,
    num_starts: int = 1,
    max_iter: int = 400,
) -> LatentWeibullCurveFitResult:
    time_values = np.asarray(times, dtype=float)
    survival_values = np.asarray(survival, dtype=float)
    if time_values.ndim != 1 or survival_values.ndim != 1 or time_values.shape[0] != survival_values.shape[0]:
        raise ValueError("times and survival must be one-dimensional arrays of equal length.")
    valid = np.isfinite(time_values) & np.isfinite(survival_values) & (time_values >= 0.0)
    time_values = time_values[valid]
    survival_values = survival_values[valid]
    if time_values.size < 3:
        raise ValueError("At least three valid survival points are required.")
    order = np.argsort(time_values)
    time_values = time_values[order]
    survival_values = np.clip(survival_values[order], 0.0, 1.0)

    if curve_weights is None:
        weight_values = np.ones_like(time_values, dtype=float)
    else:
        weight_values = np.asarray(curve_weights, dtype=float)
        weight_values = weight_values[valid][order]
        weight_values = np.where(np.isfinite(weight_values) & (weight_values > 0.0), weight_values, 1.0)
    weight_values = weight_values / np.clip(weight_values.max(), EPSILON, None)

    if time_values[0] > 0.0:
        time_values = np.insert(time_values, 0, 0.0)
        survival_values = np.insert(survival_values, 0, 1.0)
        weight_values = np.insert(weight_values, 0, 1.0)
    elif not np.isclose(survival_values[0], 1.0):
        survival_values[0] = 1.0

    initial_model = _coerce_three_component_model_by_eta(initial_model)
    epsilon = 1e-6

    def _weights_to_stick_params(model: ThreeComponentLatentWeibullModel) -> tuple[float, float]:
        available_1 = 1.0 - (3.0 * epsilon)
        raw_1 = np.clip((model.weight_1 - epsilon) / max(available_1, EPSILON), 1e-6, 1.0 - 1e-6)
        remaining_after_w1 = 1.0 - model.weight_1 - (2.0 * epsilon)
        raw_2 = np.clip((model.weight_2 - epsilon) / max(remaining_after_w1, EPSILON), 1e-6, 1.0 - 1e-6)
        return _logit(float(raw_1)), _logit(float(raw_2))

    base_weight_param_1, base_weight_param_2 = _weights_to_stick_params(initial_model)
    base_theta = np.asarray(
        [
            np.log(initial_model.component_1.beta),
            np.log(initial_model.component_1.eta),
            np.log(initial_model.component_2.beta),
            np.log(initial_model.component_2.eta),
            np.log(initial_model.component_3.beta),
            np.log(initial_model.component_3.eta),
            base_weight_param_1,
            base_weight_param_2,
        ],
        dtype=float,
    )
    optimize_mask = np.asarray(
        [
            optimize_beta_1,
            optimize_eta_1,
            optimize_beta_2,
            optimize_eta_2,
            optimize_beta_3,
            optimize_eta_3,
            optimize_weight_1,
            optimize_weight_2,
        ],
        dtype=bool,
    )

    def _full_theta(free_theta: np.ndarray) -> np.ndarray:
        theta = base_theta.copy()
        theta[optimize_mask] = np.asarray(free_theta, dtype=float)
        return theta

    def _weights_from_theta(theta: np.ndarray) -> tuple[float, float]:
        full_theta = _full_theta(theta)
        if optimize_weight_1 and optimize_weight_2:
            raw_1 = _sigmoid(float(full_theta[6]))
            available_1 = 1.0 - (3.0 * epsilon)
            weight_1 = epsilon + (available_1 * raw_1)
            raw_2 = _sigmoid(float(full_theta[7]))
            remaining_after_w1 = 1.0 - weight_1 - (2.0 * epsilon)
            weight_2 = epsilon + (remaining_after_w1 * raw_2)
            return float(weight_1), float(weight_2)
        if optimize_weight_1 and not optimize_weight_2:
            fixed_weight_2 = float(initial_model.weight_2)
            available_1 = max(1.0 - fixed_weight_2 - (2.0 * epsilon), epsilon)
            raw_1 = _sigmoid(float(full_theta[6]))
            weight_1 = epsilon + ((available_1 - epsilon) * raw_1)
            return float(weight_1), fixed_weight_2
        if optimize_weight_2 and not optimize_weight_1:
            fixed_weight_1 = float(initial_model.weight_1)
            available_2 = max(1.0 - fixed_weight_1 - (2.0 * epsilon), epsilon)
            raw_2 = _sigmoid(float(full_theta[7]))
            weight_2 = epsilon + ((available_2 - epsilon) * raw_2)
            return fixed_weight_1, float(weight_2)
        return float(initial_model.weight_1), float(initial_model.weight_2)

    def _build_model(theta: np.ndarray) -> ThreeComponentLatentWeibullModel:
        full_theta = _full_theta(theta)
        beta_1 = float(np.exp(full_theta[0]))
        eta_1 = float(np.exp(full_theta[1]))
        beta_2 = float(np.exp(full_theta[2]))
        eta_2 = float(np.exp(full_theta[3]))
        beta_3 = float(np.exp(full_theta[4]))
        eta_3 = float(np.exp(full_theta[5]))
        weight_1, weight_2 = _weights_from_theta(theta)
        model = ThreeComponentLatentWeibullModel(
            weight_1=weight_1,
            weight_2=weight_2,
            component_1=WeibullParameters(beta=beta_1, eta=eta_1, label=initial_model.component_1.label),
            component_2=WeibullParameters(beta=beta_2, eta=eta_2, label=initial_model.component_2.label),
            component_3=WeibullParameters(beta=beta_3, eta=eta_3, label=initial_model.component_3.label),
        )
        return _coerce_three_component_model_by_eta(model)

    def _objective(theta: np.ndarray) -> float:
        model = _build_model(theta)
        fitted = np.asarray(latent_survival(time_values, model), dtype=float)
        residual = fitted - survival_values
        return float(np.mean(weight_values * np.square(residual)))

    full_bounds = [
        (np.log(0.05), np.log(8.0)),
        (np.log(10.0), np.log(5000.0)),
        (np.log(0.05), np.log(8.0)),
        (np.log(10.0), np.log(5000.0)),
        (np.log(0.05), np.log(8.0)),
        (np.log(10.0), np.log(5000.0)),
        (_logit(1e-4), _logit(1.0 - 1e-4)),
        (_logit(1e-4), _logit(1.0 - 1e-4)),
    ]
    theta0 = base_theta[optimize_mask]
    bounds = [bound for bound, optimize in zip(full_bounds, optimize_mask, strict=False) if optimize]

    if theta0.size == 0:
        fitted_model = _build_model(np.asarray([], dtype=float))
        fitted_survival = np.asarray(latent_survival(time_values, fitted_model), dtype=float)
        residual = fitted_survival - survival_values
        rmse = float(np.sqrt(np.mean(np.square(residual))))
        weighted_rmse = float(np.sqrt(np.mean(weight_values * np.square(residual))))
        return LatentWeibullCurveFitResult(
            model=fitted_model,
            success=True,
            message="No free parameters selected; returned fixed initial model.",
            objective_value=float(np.mean(weight_values * np.square(residual))),
            rmse=rmse,
            weighted_rmse=weighted_rmse,
            n_iter=0,
            nfev=0,
            n_starts=1,
            best_start_index=0,
        )

    start_count = max(int(num_starts), 1)
    lower = np.asarray([bound[0] for bound in bounds], dtype=float)
    upper = np.asarray([bound[1] for bound in bounds], dtype=float)
    starts = [np.clip(theta0, lower, upper)]
    if start_count > 1:
        for point in _halton_points(start_count - 1, len(bounds)):
            starts.append(lower + (upper - lower) * point)

    best_result = None
    best_index = 0
    total_nit = 0
    total_nfev = 0
    for start_index, start in enumerate(starts):
        result = minimize(
            _objective,
            np.asarray(start, dtype=float),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(max_iter)},
        )
        total_nit += int(getattr(result, "nit", 0))
        total_nfev += int(getattr(result, "nfev", 0))
        if best_result is None:
            best_result = result
            best_index = start_index
            continue
        if bool(result.success) and not bool(best_result.success):
            best_result = result
            best_index = start_index
            continue
        if float(result.fun) < float(best_result.fun):
            best_result = result
            best_index = start_index

    assert best_result is not None
    fitted_model = _build_model(np.asarray(best_result.x, dtype=float))
    fitted_survival = np.asarray(latent_survival(time_values, fitted_model), dtype=float)
    residual = fitted_survival - survival_values
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    weighted_rmse = float(np.sqrt(np.mean(weight_values * np.square(residual))))
    return LatentWeibullCurveFitResult(
        model=fitted_model,
        success=bool(best_result.success),
        message=str(best_result.message),
        objective_value=float(best_result.fun),
        rmse=rmse,
        weighted_rmse=weighted_rmse,
        n_iter=total_nit,
        nfev=total_nfev,
        n_starts=len(starts),
        best_start_index=best_index,
    )


def fit_three_component_latent_weibull_to_km_frame(
    km_frame: pd.DataFrame,
    *,
    initial_model: ThreeComponentLatentWeibullModel,
    optimize_beta_1: bool = True,
    optimize_eta_1: bool = True,
    optimize_beta_2: bool = True,
    optimize_eta_2: bool = True,
    optimize_beta_3: bool = True,
    optimize_eta_3: bool = True,
    optimize_weight_1: bool = True,
    optimize_weight_2: bool = True,
    weight_scheme: str = DEFAULT_WEIGHT_SCHEME,
    weight_gamma: float = 0.5,
    weight_tail_lambda: float = 1.0,
    weight_tail_power: float = 1.0,
    min_n_risk: int = 0,
    num_starts: int = 1,
    max_iter: int = 400,
) -> LatentWeibullCurveFitResult:
    required = {"time", "survival"}
    missing = required.difference(km_frame.columns)
    if missing:
        raise KeyError(f"KM frame is missing required columns: {sorted(missing)}")
    fit_frame = filter_km_frame_for_fit(km_frame, min_n_risk=min_n_risk)
    n_risk = fit_frame["n_risk"].to_numpy(dtype=float) if "n_risk" in fit_frame.columns else None
    weights = build_survival_curve_weights(
        fit_frame["time"].to_numpy(dtype=float),
        n_risk=n_risk,
        weight_scheme=weight_scheme,
        gamma=weight_gamma,
        tail_lambda=weight_tail_lambda,
        tail_power=weight_tail_power,
    )
    return fit_three_component_latent_weibull_to_survival_curve(
        fit_frame["time"].to_numpy(dtype=float),
        fit_frame["survival"].to_numpy(dtype=float),
        initial_model=initial_model,
        curve_weights=weights,
        optimize_beta_1=optimize_beta_1,
        optimize_eta_1=optimize_eta_1,
        optimize_beta_2=optimize_beta_2,
        optimize_eta_2=optimize_eta_2,
        optimize_beta_3=optimize_beta_3,
        optimize_eta_3=optimize_eta_3,
        optimize_weight_1=optimize_weight_1,
        optimize_weight_2=optimize_weight_2,
        num_starts=num_starts,
        max_iter=max_iter,
    )


def fit_weibull_to_survival_curve(
    times: Iterable[float] | np.ndarray,
    survival: Iterable[float] | np.ndarray,
    *,
    initial_params: WeibullParameters,
    curve_weights: Iterable[float] | np.ndarray | None = None,
    optimize_beta: bool = True,
    optimize_eta: bool = True,
    num_starts: int = 1,
    max_iter: int = 400,
) -> ClassicalWeibullCurveFitResult:
    time_values = np.asarray(times, dtype=float)
    survival_values = np.asarray(survival, dtype=float)
    if time_values.ndim != 1 or survival_values.ndim != 1 or time_values.shape[0] != survival_values.shape[0]:
        raise ValueError("times and survival must be one-dimensional arrays of equal length.")
    valid = np.isfinite(time_values) & np.isfinite(survival_values) & (time_values >= 0.0)
    time_values = time_values[valid]
    survival_values = survival_values[valid]
    if time_values.size < 3:
        raise ValueError("At least three valid survival points are required.")
    order = np.argsort(time_values)
    time_values = time_values[order]
    survival_values = np.clip(survival_values[order], 0.0, 1.0)

    if curve_weights is None:
        weight_values = np.ones_like(time_values, dtype=float)
    else:
        weight_values = np.asarray(curve_weights, dtype=float)
        weight_values = weight_values[valid][order]
        weight_values = np.where(np.isfinite(weight_values) & (weight_values > 0.0), weight_values, 1.0)
    weight_values = weight_values / np.clip(weight_values.max(), EPSILON, None)

    if time_values[0] > 0.0:
        time_values = np.insert(time_values, 0, 0.0)
        survival_values = np.insert(survival_values, 0, 1.0)
        weight_values = np.insert(weight_values, 0, 1.0)
    elif not np.isclose(survival_values[0], 1.0):
        survival_values[0] = 1.0

    base_theta = np.log(np.asarray([initial_params.beta, initial_params.eta], dtype=float))
    optimize_mask = np.asarray([optimize_beta, optimize_eta], dtype=bool)

    def _full_theta(free_theta: np.ndarray) -> np.ndarray:
        theta = base_theta.copy()
        theta[optimize_mask] = np.asarray(free_theta, dtype=float)
        return theta

    def _build_params(theta: np.ndarray) -> WeibullParameters:
        full_theta = _full_theta(theta)
        return WeibullParameters(
            beta=float(np.exp(full_theta[0])),
            eta=float(np.exp(full_theta[1])),
            label=initial_params.label,
        )

    def _objective(theta: np.ndarray) -> float:
        params = _build_params(theta)
        fitted = np.asarray(weibull_survival(time_values, params), dtype=float)
        residual = fitted - survival_values
        return float(np.mean(weight_values * np.square(residual)))

    full_bounds = [
        (np.log(0.05), np.log(8.0)),
        (np.log(10.0), np.log(5000.0)),
    ]
    theta0 = base_theta[optimize_mask]
    bounds = [bound for bound, optimize in zip(full_bounds, optimize_mask, strict=False) if optimize]

    if theta0.size == 0:
        fitted_params = _build_params(np.asarray([], dtype=float))
        fitted_survival = np.asarray(weibull_survival(time_values, fitted_params), dtype=float)
        residual = fitted_survival - survival_values
        rmse = float(np.sqrt(np.mean(np.square(residual))))
        weighted_rmse = float(np.sqrt(np.mean(weight_values * np.square(residual))))
        return ClassicalWeibullCurveFitResult(
            params=fitted_params,
            success=True,
            message="No free parameters selected; returned fixed initial parameters.",
            objective_value=float(np.mean(weight_values * np.square(residual))),
            rmse=rmse,
            weighted_rmse=weighted_rmse,
            n_iter=0,
            nfev=0,
            n_starts=1,
            best_start_index=0,
        )

    start_count = max(int(num_starts), 1)
    lower = np.asarray([bound[0] for bound in bounds], dtype=float)
    upper = np.asarray([bound[1] for bound in bounds], dtype=float)
    starts = [np.clip(theta0, lower, upper)]
    if start_count > 1:
        for point in _halton_points(start_count - 1, len(bounds)):
            starts.append(lower + (upper - lower) * point)

    best_result = None
    best_index = 0
    total_nit = 0
    total_nfev = 0
    for start_index, start in enumerate(starts):
        result = minimize(
            _objective,
            np.asarray(start, dtype=float),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(max_iter)},
        )
        total_nit += int(getattr(result, "nit", 0))
        total_nfev += int(getattr(result, "nfev", 0))
        if best_result is None:
            best_result = result
            best_index = start_index
            continue
        if bool(result.success) and not bool(best_result.success):
            best_result = result
            best_index = start_index
            continue
        if float(result.fun) < float(best_result.fun):
            best_result = result
            best_index = start_index

    assert best_result is not None
    fitted_params = _build_params(np.asarray(best_result.x, dtype=float))
    fitted_survival = np.asarray(weibull_survival(time_values, fitted_params), dtype=float)
    residual = fitted_survival - survival_values
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    weighted_rmse = float(np.sqrt(np.mean(weight_values * np.square(residual))))
    return ClassicalWeibullCurveFitResult(
        params=fitted_params,
        success=bool(best_result.success),
        message=str(best_result.message),
        objective_value=float(best_result.fun),
        rmse=rmse,
        weighted_rmse=weighted_rmse,
        n_iter=total_nit,
        nfev=total_nfev,
        n_starts=len(starts),
        best_start_index=best_index,
    )


def fit_weibull_to_km_frame(
    km_frame: pd.DataFrame,
    *,
    initial_params: WeibullParameters,
    optimize_beta: bool = True,
    optimize_eta: bool = True,
    weight_scheme: str = DEFAULT_WEIGHT_SCHEME,
    weight_gamma: float = 0.5,
    weight_tail_lambda: float = 1.0,
    weight_tail_power: float = 1.0,
    min_n_risk: int = 0,
    num_starts: int = 1,
    max_iter: int = 400,
) -> ClassicalWeibullCurveFitResult:
    required = {"time", "survival"}
    missing = required.difference(km_frame.columns)
    if missing:
        raise KeyError(f"KM frame is missing required columns: {sorted(missing)}")
    fit_frame = filter_km_frame_for_fit(km_frame, min_n_risk=min_n_risk)
    n_risk = fit_frame["n_risk"].to_numpy(dtype=float) if "n_risk" in fit_frame.columns else None
    weights = build_survival_curve_weights(
        fit_frame["time"].to_numpy(dtype=float),
        n_risk=n_risk,
        weight_scheme=weight_scheme,
        gamma=weight_gamma,
        tail_lambda=weight_tail_lambda,
        tail_power=weight_tail_power,
    )
    return fit_weibull_to_survival_curve(
        fit_frame["time"].to_numpy(dtype=float),
        fit_frame["survival"].to_numpy(dtype=float),
        initial_params=initial_params,
        curve_weights=weights,
        optimize_beta=optimize_beta,
        optimize_eta=optimize_eta,
        num_starts=num_starts,
        max_iter=max_iter,
    )


def competing_overall_survival(
    times: float | Iterable[float] | np.ndarray,
    model: CompetingRiskModel,
) -> float | np.ndarray:
    t, is_scalar = _as_time_array(times)
    total_hazard = np.zeros_like(t)
    for cause in model.causes:
        total_hazard += np.asarray(weibull_cumulative_hazard(t, cause), dtype=float)
    values = np.exp(-np.clip(total_hazard, 0.0, 700.0))
    return _restore_shape(values, is_scalar)


def competing_overall_cdf(
    times: float | Iterable[float] | np.ndarray,
    model: CompetingRiskModel,
) -> float | np.ndarray:
    survival = np.asarray(competing_overall_survival(times, model), dtype=float)
    values = 1.0 - survival
    is_scalar = np.asarray(times).ndim == 0
    return _restore_shape(np.atleast_1d(values), is_scalar)


def competing_cause_hazard(
    times: float | Iterable[float] | np.ndarray,
    cause: WeibullParameters,
) -> float | np.ndarray:
    return weibull_hazard(times, cause)


def competing_overall_hazard(
    times: float | Iterable[float] | np.ndarray,
    model: CompetingRiskModel,
) -> float | np.ndarray:
    t, is_scalar = _as_time_array(times)
    values = np.zeros_like(t)
    for cause in model.causes:
        values += np.asarray(competing_cause_hazard(t, cause), dtype=float)
    return _restore_shape(values, is_scalar)


def competing_curve_frame(model: CompetingRiskModel, max_time: float, num_points: int = 400) -> pd.DataFrame:
    time_grid = np.linspace(0.0, _validate_positive("max_time", max_time), int(num_points))
    survival = np.asarray(competing_overall_survival(time_grid, model), dtype=float)
    overall_hazard = np.asarray(competing_overall_hazard(time_grid, model), dtype=float)
    frame = pd.DataFrame(
        {
            "time": time_grid,
            "survival": survival,
            "all_cause_failure": 1.0 - survival,
            "overall_hazard": overall_hazard,
        }
    )

    # CIF via analytic cumulative-hazard increments: dCIF_k = S(u)·dH_k(u).
    # H_k(t) = (t/η_k)^{β_k} is finite at 0 for every β>0, unlike the density
    # S·h_k, which is singular at t=0 when β_k<1 — a trapezoid on the density
    # diverges there (CIF ≫ 1). Same scheme as analysis.models.survival.cif.
    cif_total = np.zeros_like(time_grid)
    survival_mid = 0.5 * (survival[:-1] + survival[1:])
    for index, cause in enumerate(model.causes, start=1):
        cause_hazard = np.asarray(competing_cause_hazard(time_grid, cause), dtype=float)
        cause_density = survival * cause_hazard
        cause_cum_hazard = np.asarray(weibull_cumulative_hazard(time_grid, cause), dtype=float)
        cause_cif = np.concatenate(([0.0], np.cumsum(survival_mid * np.diff(cause_cum_hazard))))
        cif_total += cause_cif
        frame[f"hazard_{index}"] = cause_hazard
        frame[f"density_{index}"] = cause_density
        frame[f"cif_{index}"] = cause_cif
    frame["identity_error"] = frame["survival"] + cif_total - 1.0
    return frame


def competing_next_window_probabilities(
    current_age: float,
    horizon: float,
    model: CompetingRiskModel,
    num_points: int = 800,
) -> pd.DataFrame:
    age = max(float(current_age), 0.0)
    window = _validate_positive("horizon", horizon)
    grid_end = age + window
    frame = competing_curve_frame(model, max_time=max(grid_end, 1.0), num_points=max(int(num_points), 50))
    survival_now = float(np.interp(age, frame["time"], frame["survival"]))
    survival_future = float(np.interp(age + window, frame["time"], frame["survival"]))
    rows: list[dict[str, float | str]] = []
    total_probability = 0.0
    for index, cause in enumerate(model.causes, start=1):
        cif_now = float(np.interp(age, frame["time"], frame[f"cif_{index}"]))
        cif_future = float(np.interp(age + window, frame["time"], frame[f"cif_{index}"]))
        window_probability = float(
            np.clip((cif_future - cif_now) / np.clip(survival_now, EPSILON, None), 0.0, 1.0)
        )
        total_probability += window_probability
        rows.append(
            {
                "cause": cause.label,
                "window_probability": window_probability,
                "cumulative_incidence_now": cif_now,
                "cumulative_incidence_future": cif_future,
            }
        )
    all_cause_window = float(np.clip(1.0 - (survival_future / np.clip(survival_now, EPSILON, None)), 0.0, 1.0))
    output = pd.DataFrame(rows)
    output["share_of_predicted_failures"] = output["window_probability"] / np.clip(total_probability, EPSILON, None)
    output.attrs["all_cause_window_probability"] = all_cause_window
    output.attrs["survival_now"] = survival_now
    output.attrs["survival_future"] = survival_future
    return output
