from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import gammaln, logsumexp

from analysis.data.data_utils import normalize_event_series


EPSILON = 1e-12


def _as_float_array(values: Iterable[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("Expected a one-dimensional numeric array.")
    return array


def _log_weibull_survival(durations: np.ndarray, beta: float, eta: float) -> np.ndarray:
    safe_durations = np.clip(durations, EPSILON, None)
    log_ratio = np.log(safe_durations) - np.log(eta)
    power = np.clip(beta * log_ratio, -700.0, 700.0)
    return -np.exp(power)


def _log_weibull_density(durations: np.ndarray, beta: float, eta: float) -> np.ndarray:
    safe_durations = np.clip(durations, EPSILON, None)
    log_ratio = np.log(safe_durations) - np.log(eta)
    power = np.clip(beta * log_ratio, -700.0, 700.0)
    return np.log(beta) - np.log(eta) + ((beta - 1.0) * log_ratio) - np.exp(power)


def weibull_survival(durations: Iterable[float] | np.ndarray, beta: float, eta: float) -> np.ndarray:
    return np.exp(_log_weibull_survival(_as_float_array(durations), float(beta), float(eta)))


def weibull_density(durations: Iterable[float] | np.ndarray, beta: float, eta: float) -> np.ndarray:
    return np.exp(_log_weibull_density(_as_float_array(durations), float(beta), float(eta)))


def weibull_hazard(durations: Iterable[float] | np.ndarray, beta: float, eta: float) -> np.ndarray:
    durations_array = _as_float_array(durations)
    safe_durations = np.clip(durations_array, EPSILON, None)
    return (float(beta) / float(eta)) * np.power(safe_durations / float(eta), float(beta) - 1.0)


def load_survival_file(path: str | Path, sheet_name: str | int | None = None) -> pd.DataFrame:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        engine = "openpyxl" if suffix == ".xlsx" else "xlrd"
        sheet = 0 if sheet_name is None else sheet_name
        return pd.read_excel(source, sheet_name=sheet, engine=engine)
    if suffix == ".csv":
        return pd.read_csv(source)
    raise ValueError(f"Unsupported file type '{suffix}'. Use CSV or Excel.")


@dataclass(slots=True)
class SurvivalValidationReport:
    rows_before_validation: int
    rows_after_validation: int
    removed_rows: int
    failure_count: int
    censored_count: int
    notes: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {"item": "rows_before_validation", "value": self.rows_before_validation},
            {"item": "rows_after_validation", "value": self.rows_after_validation},
            {"item": "removed_rows", "value": self.removed_rows},
            {"item": "failure_count", "value": self.failure_count},
            {"item": "censored_count", "value": self.censored_count},
        ]
        for note in self.notes:
            rows.append({"item": "note", "value": note})
        return pd.DataFrame(rows)


def validate_survival_dataframe(
    df: pd.DataFrame,
    *,
    duration_column: str | None,
    event_column: str,
    id_column: str | None = None,
    start_date_column: str | None = None,
    end_date_column: str | None = None,
) -> tuple[pd.DataFrame, SurvivalValidationReport]:
    frame = df.copy()
    notes: list[str] = []

    if duration_column is None:
        if not start_date_column or not end_date_column:
            raise ValueError("Provide either duration_column or both start_date_column and end_date_column.")
        if start_date_column not in frame.columns or end_date_column not in frame.columns:
            raise ValueError("Start/end date columns were not found.")
        start_dates = pd.to_datetime(frame[start_date_column], errors="coerce")
        end_dates = pd.to_datetime(frame[end_date_column], errors="coerce")
        frame["duration"] = (end_dates - start_dates).dt.total_seconds() / 86400.0
        duration_column = "duration"
        invalid_dates = start_dates.isna() | end_dates.isna()
        if invalid_dates.any():
            notes.append(f"Found {int(invalid_dates.sum())} rows with invalid start/end dates.")
        invalid_order = (~invalid_dates) & (end_dates < start_dates)
        if invalid_order.any():
            notes.append(f"Found {int(invalid_order.sum())} rows with end_date earlier than start_date.")
    elif duration_column not in frame.columns:
        raise ValueError(f"Duration column '{duration_column}' was not found.")

    if event_column not in frame.columns:
        raise ValueError(f"Event column '{event_column}' was not found.")

    if id_column and id_column not in frame.columns:
        raise ValueError(f"ID column '{id_column}' was not found.")

    frame[duration_column] = pd.to_numeric(frame[duration_column], errors="coerce")
    frame[event_column] = normalize_event_series(frame[event_column])

    if id_column:
        duplicate_ids = frame[id_column].notna() & frame[id_column].duplicated(keep=False)
        if duplicate_ids.any():
            notes.append(f"Removed {int(duplicate_ids.sum())} rows with duplicate IDs.")
            frame = frame.loc[~duplicate_ids].copy()
    else:
        frame["id"] = [f"row_{index}" for index in range(len(frame))]
        id_column = "id"

    missing_required = frame[duration_column].isna() | frame[event_column].isna()
    if missing_required.any():
        notes.append(f"Removed {int(missing_required.sum())} rows with missing duration or event.")
        frame = frame.loc[~missing_required].copy()

    invalid_duration = frame[duration_column] <= 0.0
    if invalid_duration.any():
        notes.append(f"Removed {int(invalid_duration.sum())} rows with non-positive duration.")
        frame = frame.loc[~invalid_duration].copy()

    invalid_event = ~frame[event_column].isin([0, 1])
    if invalid_event.any():
        notes.append(f"Removed {int(invalid_event.sum())} rows with event values outside 0/1.")
        frame = frame.loc[~invalid_event].copy()

    infinite_duration = ~np.isfinite(frame[duration_column].to_numpy(dtype=float))
    if infinite_duration.any():
        notes.append(f"Removed {int(infinite_duration.sum())} rows with non-finite duration.")
        frame = frame.loc[~infinite_duration].copy()

    frame = frame.reset_index(drop=True)
    report = SurvivalValidationReport(
        rows_before_validation=int(len(df)),
        rows_after_validation=int(len(frame)),
        removed_rows=int(len(df) - len(frame)),
        failure_count=int(frame[event_column].sum()),
        censored_count=int(len(frame) - frame[event_column].sum()),
        notes=notes,
    )
    if frame.empty:
        raise ValueError("No rows remain after validation.")
    return frame, report


def kaplan_meier_frame(durations: Iterable[float] | np.ndarray, events: Iterable[int] | np.ndarray) -> pd.DataFrame:
    durations_array = _as_float_array(durations)
    events_array = np.asarray(events, dtype=int)
    order = np.argsort(durations_array)
    sorted_durations = durations_array[order]
    sorted_events = events_array[order]
    unique_times = np.unique(sorted_durations)

    rows: list[dict[str, float | int]] = []
    survival = 1.0
    cumulative_hazard = 0.0
    for time in unique_times:
        at_risk = int(np.sum(sorted_durations >= time))
        failures = int(np.sum((sorted_durations == time) & (sorted_events == 1)))
        censored = int(np.sum((sorted_durations == time) & (sorted_events == 0)))
        if at_risk <= 0:
            continue
        if failures > 0:
            survival *= max(1.0 - (failures / at_risk), 0.0)
            cumulative_hazard += failures / at_risk
        rows.append(
            {
                "time": float(time),
                "n_risk": at_risk,
                "n_events": failures,
                "n_censored": censored,
                "survival": survival,
                "cumulative_hazard": cumulative_hazard,
            }
        )
    return pd.DataFrame(rows)


def mixture_log_likelihood(
    durations: Iterable[float] | np.ndarray,
    events: Iterable[int] | np.ndarray,
    weights: Iterable[float] | np.ndarray,
    betas: Iterable[float] | np.ndarray,
    etas: Iterable[float] | np.ndarray,
) -> float:
    durations_array = _as_float_array(durations)
    events_array = np.asarray(events, dtype=int)
    weights_array = np.asarray(weights, dtype=float)
    betas_array = np.asarray(betas, dtype=float)
    etas_array = np.asarray(etas, dtype=float)
    if not np.isclose(weights_array.sum(), 1.0, atol=1e-8):
        raise ValueError("Mixture weights must sum to 1.")
    log_weights = np.log(np.clip(weights_array, EPSILON, None))
    log_density = np.column_stack(
        [_log_weibull_density(durations_array, beta, eta) for beta, eta in zip(betas_array, etas_array, strict=False)]
    )
    log_survival = np.column_stack(
        [_log_weibull_survival(durations_array, beta, eta) for beta, eta in zip(betas_array, etas_array, strict=False)]
    )
    log_failure = logsumexp(log_weights + log_density, axis=1)
    log_censor = logsumexp(log_weights + log_survival, axis=1)
    contributions = np.where(events_array == 1, log_failure, log_censor)
    return float(np.sum(contributions))


@dataclass(slots=True)
class BayesianLatentWeibullConfig:
    n_components: int = 2
    n_iter: int = 2000
    burn_in: int = 1000
    thin: int = 5
    n_chains: int = 2
    alpha_dirichlet: float = 1.0
    mu_log_eta: float = 5.0
    sigma_log_eta: float = 1.5
    mu_log_beta: float = 0.0
    sigma_log_beta: float = 0.7
    proposal_sd_log_eta: float = 0.10
    proposal_sd_log_beta: float = 0.08
    random_seed: int = 42
    uncertainty_threshold: float = 0.75
    use_unknown_k: bool = False
    k_max: int = 6
    lambda_k: float = 3.0
    birth_death_probability: float = 0.25

    def __post_init__(self) -> None:
        if self.n_components < 1:
            raise ValueError("n_components must be at least 1.")
        if self.n_iter <= self.burn_in:
            raise ValueError("n_iter must be greater than burn_in.")
        if self.thin < 1:
            raise ValueError("thin must be at least 1.")
        if self.n_chains < 1:
            raise ValueError("n_chains must be at least 1.")
        if self.k_max < 1:
            raise ValueError("k_max must be at least 1.")
        if self.n_components > self.k_max:
            raise ValueError("n_components cannot exceed k_max.")
        if not 0.0 <= self.birth_death_probability <= 1.0:
            raise ValueError("birth_death_probability must be between 0 and 1.")
        if self.lambda_k <= 0.0:
            raise ValueError("lambda_k must be positive.")


@dataclass(slots=True)
class _ChainState:
    weights: np.ndarray
    log_eta: np.ndarray
    log_beta: np.ndarray
    allocations: np.ndarray

    @property
    def n_components(self) -> int:
        return int(len(self.weights))


@dataclass(slots=True)
class BayesianLatentWeibullChainResult:
    chain_id: int
    k_draws: np.ndarray
    weights_raw: np.ndarray
    eta_raw: np.ndarray
    beta_raw: np.ndarray
    weights_relabeled: np.ndarray
    eta_relabeled: np.ndarray
    beta_relabeled: np.ndarray
    log_likelihood: np.ndarray
    log_posterior: np.ndarray
    acceptance_rate_eta: np.ndarray
    acceptance_rate_beta: np.ndarray
    birth_attempts: int
    birth_accepts: int
    death_attempts: int
    death_accepts: int
    seed: int

    @property
    def max_k(self) -> int:
        return int(self.weights_relabeled.shape[1])


def _component_median_lifetimes(eta_values: np.ndarray, beta_values: np.ndarray) -> np.ndarray:
    return eta_values * np.power(np.log(2.0), 1.0 / np.clip(beta_values, EPSILON, None))


def _relabel_saved_draws(
    weights_draws: list[np.ndarray],
    eta_draws: list[np.ndarray],
    beta_draws: list[np.ndarray],
    max_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_draws = len(weights_draws)
    weights_raw = np.zeros((n_draws, max_k), dtype=float)
    eta_raw = np.full((n_draws, max_k), np.nan, dtype=float)
    beta_raw = np.full((n_draws, max_k), np.nan, dtype=float)
    weights_relabeled = np.zeros((n_draws, max_k), dtype=float)
    eta_relabeled = np.full((n_draws, max_k), np.nan, dtype=float)
    beta_relabeled = np.full((n_draws, max_k), np.nan, dtype=float)

    for draw_index, (weights, etas, betas) in enumerate(zip(weights_draws, eta_draws, beta_draws, strict=False)):
        k = len(weights)
        weights_raw[draw_index, :k] = weights
        eta_raw[draw_index, :k] = etas
        beta_raw[draw_index, :k] = betas
        order = np.argsort(_component_median_lifetimes(etas, betas))
        weights_relabeled[draw_index, :k] = weights[order]
        eta_relabeled[draw_index, :k] = etas[order]
        beta_relabeled[draw_index, :k] = betas[order]
    return weights_raw, eta_raw, beta_raw, weights_relabeled, eta_relabeled, beta_relabeled


def _log_normal_prior(value: float, mu: float, sigma: float) -> float:
    z_score = (value - mu) / sigma
    return float(-0.5 * np.square(z_score) - np.log(sigma) - 0.5 * np.log(2.0 * np.pi))


def _log_dirichlet_symmetric_prior(weights: np.ndarray, alpha: float) -> float:
    k = len(weights)
    return float(gammaln(k * alpha) - (k * gammaln(alpha)) + ((alpha - 1.0) * np.sum(np.log(np.clip(weights, EPSILON, None)))))


def _log_truncated_poisson_prior(k: int, lambda_k: float, k_max: int) -> float:
    support = np.arange(1, k_max + 1, dtype=float)
    log_unnormalized = support * np.log(lambda_k) - gammaln(support + 1.0)
    normalization = logsumexp(log_unnormalized)
    return float((k * np.log(lambda_k) - gammaln(k + 1.0)) - normalization)


def _component_complete_log_likelihood(
    durations: np.ndarray,
    events: np.ndarray,
    mask: np.ndarray,
    *,
    log_eta: float,
    log_beta: float,
) -> float:
    if not mask.any():
        return 0.0
    eta = float(np.exp(log_eta))
    beta = float(np.exp(log_beta))
    selected_durations = durations[mask]
    selected_events = events[mask]
    contributions = np.where(
        selected_events == 1,
        _log_weibull_density(selected_durations, beta, eta),
        _log_weibull_survival(selected_durations, beta, eta),
    )
    return float(np.sum(contributions))


def _component_complete_log_posterior(
    durations: np.ndarray,
    events: np.ndarray,
    mask: np.ndarray,
    *,
    log_eta: float,
    log_beta: float,
    config: BayesianLatentWeibullConfig,
) -> float:
    log_likelihood = _component_complete_log_likelihood(durations, events, mask, log_eta=log_eta, log_beta=log_beta)
    log_eta_prior = _log_normal_prior(log_eta, config.mu_log_eta, config.sigma_log_eta)
    log_beta_prior = _log_normal_prior(log_beta, config.mu_log_beta, config.sigma_log_beta)
    return float(log_likelihood + log_eta_prior + log_beta_prior)


def _observed_log_posterior(
    durations: np.ndarray,
    events: np.ndarray,
    state: _ChainState,
    config: BayesianLatentWeibullConfig,
) -> tuple[float, float]:
    eta_values = np.exp(state.log_eta)
    beta_values = np.exp(state.log_beta)
    log_likelihood = mixture_log_likelihood(durations, events, state.weights, beta_values, eta_values)
    log_weight_prior = _log_dirichlet_symmetric_prior(state.weights, config.alpha_dirichlet)
    log_eta_prior = float(np.sum([_log_normal_prior(value, config.mu_log_eta, config.sigma_log_eta) for value in state.log_eta]))
    log_beta_prior = float(np.sum([_log_normal_prior(value, config.mu_log_beta, config.sigma_log_beta) for value in state.log_beta]))
    log_k_prior = _log_truncated_poisson_prior(state.n_components, config.lambda_k, config.k_max) if config.use_unknown_k else 0.0
    return log_likelihood, float(log_likelihood + log_weight_prior + log_eta_prior + log_beta_prior + log_k_prior)


def _initialize_chain(
    durations: np.ndarray,
    events: np.ndarray,
    config: BayesianLatentWeibullConfig,
    rng: np.random.Generator,
) -> _ChainState:
    n_obs = len(durations)
    initial_k = config.n_components
    if initial_k == 1:
        allocations = np.zeros(n_obs, dtype=int)
    else:
        base = np.log(np.clip(durations, EPSILON, None))
        quantiles = np.quantile(base, np.linspace(0.0, 1.0, initial_k + 1))
        splits = quantiles[1:-1]
        allocations = np.digitize(base, splits, right=True)
    log_eta = np.zeros(initial_k, dtype=float)
    log_beta = np.zeros(initial_k, dtype=float)

    global_median = float(np.median(durations))
    for component in range(initial_k):
        mask = allocations == component
        component_durations = durations[mask] if mask.any() else durations
        eta_guess = max(float(np.median(component_durations)), EPSILON)
        beta_guess = 0.8 if component == 0 and initial_k > 1 else 1.5 + (0.1 * component)
        log_eta[component] = np.log(eta_guess) + rng.normal(0.0, 0.10)
        log_beta[component] = np.log(beta_guess) + rng.normal(0.0, 0.05)
    if not np.isfinite(log_eta).all():
        log_eta[:] = np.log(max(global_median, 1.0))
    counts = np.bincount(allocations, minlength=initial_k).astype(float) + config.alpha_dirichlet
    weights = counts / counts.sum()
    return _ChainState(weights=weights, log_eta=log_eta, log_beta=log_beta, allocations=allocations)


def _sample_allocations(
    durations: np.ndarray,
    events: np.ndarray,
    state: _ChainState,
    rng: np.random.Generator,
) -> np.ndarray:
    eta_values = np.exp(state.log_eta)
    beta_values = np.exp(state.log_beta)
    log_weights = np.log(np.clip(state.weights, EPSILON, None))
    log_kernels = np.zeros((len(durations), len(state.weights)), dtype=float)
    for component, (beta, eta) in enumerate(zip(beta_values, eta_values, strict=False)):
        log_kernel = np.where(
            events == 1,
            _log_weibull_density(durations, beta, eta),
            _log_weibull_survival(durations, beta, eta),
        )
        log_kernels[:, component] = log_weights[component] + log_kernel
    normalized = log_kernels - logsumexp(log_kernels, axis=1, keepdims=True)
    probabilities = np.exp(normalized)
    draws = rng.uniform(size=len(durations))
    cumulative = np.cumsum(probabilities, axis=1)
    return np.argmax(draws[:, None] <= cumulative, axis=1).astype(int)


def _mh_update_component_parameters(
    durations: np.ndarray,
    events: np.ndarray,
    state: _ChainState,
    component: int,
    config: BayesianLatentWeibullConfig,
    rng: np.random.Generator,
) -> tuple[bool, bool]:
    mask = state.allocations == component
    current_log_eta = float(state.log_eta[component])
    current_log_beta = float(state.log_beta[component])
    current_target = _component_complete_log_posterior(
        durations,
        events,
        mask,
        log_eta=current_log_eta,
        log_beta=current_log_beta,
        config=config,
    )

    proposal_log_eta = current_log_eta + rng.normal(0.0, config.proposal_sd_log_eta)
    proposed_target_eta = _component_complete_log_posterior(
        durations,
        events,
        mask,
        log_eta=proposal_log_eta,
        log_beta=current_log_beta,
        config=config,
    )
    accepted_eta = np.log(rng.uniform()) < (proposed_target_eta - current_target)
    if accepted_eta:
        state.log_eta[component] = proposal_log_eta
        current_log_eta = proposal_log_eta
        current_target = proposed_target_eta

    proposal_log_beta = current_log_beta + rng.normal(0.0, config.proposal_sd_log_beta)
    proposed_target_beta = _component_complete_log_posterior(
        durations,
        events,
        mask,
        log_eta=current_log_eta,
        log_beta=proposal_log_beta,
        config=config,
    )
    accepted_beta = np.log(rng.uniform()) < (proposed_target_beta - current_target)
    if accepted_beta:
        state.log_beta[component] = proposal_log_beta
    return bool(accepted_eta), bool(accepted_beta)


def _birth_probability(k: int, config: BayesianLatentWeibullConfig) -> float:
    if not config.use_unknown_k:
        return 0.0
    if k <= 1:
        return 1.0 if k < config.k_max else 0.0
    if k >= config.k_max:
        return 0.0
    return 0.5


def _death_probability(k: int, config: BayesianLatentWeibullConfig) -> float:
    if not config.use_unknown_k or k <= 1:
        return 0.0
    if k >= config.k_max:
        return 1.0
    return 0.5


def _sample_prior_component(config: BayesianLatentWeibullConfig, rng: np.random.Generator) -> tuple[float, float]:
    return (
        float(rng.normal(config.mu_log_eta, config.sigma_log_eta)),
        float(rng.normal(config.mu_log_beta, config.sigma_log_beta)),
    )


def _try_birth_move(
    durations: np.ndarray,
    events: np.ndarray,
    state: _ChainState,
    config: BayesianLatentWeibullConfig,
    rng: np.random.Generator,
) -> tuple[bool, _ChainState]:
    current_k = state.n_components
    if current_k >= config.k_max:
        return False, state

    current_ll, current_post = _observed_log_posterior(durations, events, state, config)
    u = float(rng.beta(1.0, current_k))
    new_log_eta, new_log_beta = _sample_prior_component(config, rng)
    proposed_state = _ChainState(
        weights=np.concatenate([(1.0 - u) * state.weights, [u]]),
        log_eta=np.concatenate([state.log_eta, [new_log_eta]]),
        log_beta=np.concatenate([state.log_beta, [new_log_beta]]),
        allocations=state.allocations.copy(),
    )
    proposed_ll, proposed_post = _observed_log_posterior(durations, events, proposed_state, config)
    proposed_k = proposed_state.n_components
    log_acceptance = (
        proposed_post
        - current_post
        + np.log(max(_death_probability(proposed_k, config), EPSILON))
        - np.log(proposed_k)
        - np.log(max(_birth_probability(current_k, config), EPSILON))
        - np.log(current_k)
    )
    if np.log(rng.uniform()) < log_acceptance:
        proposed_state.allocations = _sample_allocations(durations, events, proposed_state, rng)
        return True, proposed_state
    return False, state


def _try_death_move(
    durations: np.ndarray,
    events: np.ndarray,
    state: _ChainState,
    config: BayesianLatentWeibullConfig,
    rng: np.random.Generator,
) -> tuple[bool, _ChainState]:
    current_k = state.n_components
    if current_k <= 1:
        return False, state

    remove_index = int(rng.integers(0, current_k))
    remove_weight = float(state.weights[remove_index])
    if remove_weight >= 1.0:
        return False, state

    current_ll, current_post = _observed_log_posterior(durations, events, state, config)
    kept_weights = np.delete(state.weights, remove_index)
    proposed_weights = kept_weights / np.clip(1.0 - remove_weight, EPSILON, None)
    proposed_state = _ChainState(
        weights=proposed_weights,
        log_eta=np.delete(state.log_eta, remove_index),
        log_beta=np.delete(state.log_beta, remove_index),
        allocations=np.zeros_like(state.allocations),
    )
    proposed_ll, proposed_post = _observed_log_posterior(durations, events, proposed_state, config)
    proposed_k = proposed_state.n_components
    log_acceptance = (
        proposed_post
        - current_post
        + np.log(max(_birth_probability(proposed_k, config), EPSILON))
        - np.log(proposed_k)
        - np.log(max(_death_probability(current_k, config), EPSILON))
        + np.log(current_k)
    )
    if np.log(rng.uniform()) < log_acceptance:
        proposed_state.allocations = _sample_allocations(durations, events, proposed_state, rng)
        return True, proposed_state
    return False, state


def _run_single_chain(
    durations: np.ndarray,
    events: np.ndarray,
    config: BayesianLatentWeibullConfig,
    *,
    chain_id: int,
) -> BayesianLatentWeibullChainResult:
    seed = int(config.random_seed + (1009 * chain_id))
    rng = np.random.default_rng(seed)
    state = _initialize_chain(durations, events, config, rng)
    saved_k: list[int] = []
    saved_weights: list[np.ndarray] = []
    saved_eta: list[np.ndarray] = []
    saved_beta: list[np.ndarray] = []
    saved_log_likelihood: list[float] = []
    saved_log_posterior: list[float] = []
    accepted_eta = np.zeros(config.k_max, dtype=int)
    accepted_beta = np.zeros(config.k_max, dtype=int)
    birth_attempts = 0
    birth_accepts = 0
    death_attempts = 0
    death_accepts = 0

    for iteration in range(config.n_iter):
        state.allocations = _sample_allocations(durations, events, state, rng)
        counts = np.bincount(state.allocations, minlength=state.n_components).astype(float)
        state.weights = rng.dirichlet(counts + config.alpha_dirichlet)

        for component in range(state.n_components):
            accepted_eta_component, accepted_beta_component = _mh_update_component_parameters(
                durations,
                events,
                state,
                component,
                config,
                rng,
            )
            accepted_eta[component] += int(accepted_eta_component)
            accepted_beta[component] += int(accepted_beta_component)

        if config.use_unknown_k and rng.uniform() < config.birth_death_probability:
            birth_prob = _birth_probability(state.n_components, config)
            death_prob = _death_probability(state.n_components, config)
            if birth_prob > 0.0 and (death_prob <= 0.0 or rng.uniform() < birth_prob):
                birth_attempts += 1
                accepted, state = _try_birth_move(durations, events, state, config, rng)
                birth_accepts += int(accepted)
            elif death_prob > 0.0:
                death_attempts += 1
                accepted, state = _try_death_move(durations, events, state, config, rng)
                death_accepts += int(accepted)

        if iteration >= config.burn_in and ((iteration - config.burn_in) % config.thin == 0):
            eta_values = np.exp(state.log_eta)
            beta_values = np.exp(state.log_beta)
            log_likelihood, log_posterior = _observed_log_posterior(durations, events, state, config)
            saved_k.append(state.n_components)
            saved_weights.append(state.weights.copy())
            saved_eta.append(eta_values.copy())
            saved_beta.append(beta_values.copy())
            saved_log_likelihood.append(log_likelihood)
            saved_log_posterior.append(log_posterior)

    max_k = config.k_max if config.use_unknown_k else config.n_components
    weights_raw, eta_raw, beta_raw, weights_relabeled, eta_relabeled, beta_relabeled = _relabel_saved_draws(
        saved_weights,
        saved_eta,
        saved_beta,
        max_k=max_k,
    )
    denominator = max(config.n_iter, 1)
    return BayesianLatentWeibullChainResult(
        chain_id=chain_id,
        k_draws=np.asarray(saved_k, dtype=int),
        weights_raw=weights_raw,
        eta_raw=eta_raw,
        beta_raw=beta_raw,
        weights_relabeled=weights_relabeled,
        eta_relabeled=eta_relabeled,
        beta_relabeled=beta_relabeled,
        log_likelihood=np.asarray(saved_log_likelihood, dtype=float),
        log_posterior=np.asarray(saved_log_posterior, dtype=float),
        acceptance_rate_eta=accepted_eta / denominator,
        acceptance_rate_beta=accepted_beta / denominator,
        birth_attempts=birth_attempts,
        birth_accepts=birth_accepts,
        death_attempts=death_attempts,
        death_accepts=death_accepts,
        seed=seed,
    )


def _autocorrelation(values: np.ndarray, lag: int) -> float:
    centered = values - values.mean()
    denominator = np.dot(centered, centered)
    if denominator <= 0.0:
        return 0.0
    numerator = np.dot(centered[:-lag], centered[lag:]) if lag < len(values) else 0.0
    return float(numerator / denominator)


def _effective_sample_size(values: np.ndarray) -> float:
    if len(values) < 4:
        return float(len(values))
    rho_sum = 0.0
    for lag in range(1, min(len(values) - 1, 200)):
        rho = _autocorrelation(values, lag)
        if rho <= 0.0:
            break
        rho_sum += rho
    return float(len(values) / max(1.0 + (2.0 * rho_sum), EPSILON))


def _split_rhat(chains: np.ndarray) -> float:
    if chains.shape[0] < 2 or chains.shape[1] < 4:
        return float("nan")
    split_chains: list[np.ndarray] = []
    for chain in chains:
        half = len(chain) // 2
        if half < 2:
            continue
        split_chains.append(chain[:half])
        split_chains.append(chain[-half:])
    if len(split_chains) < 2:
        return float("nan")
    split = np.asarray(split_chains, dtype=float)
    chain_means = split.mean(axis=1)
    chain_variances = split.var(axis=1, ddof=1)
    n = split.shape[1]
    between = n * chain_means.var(ddof=1)
    within = chain_variances.mean()
    if within <= 0.0:
        return 1.0
    var_hat = ((n - 1.0) / n) * within + (between / n)
    return float(np.sqrt(var_hat / within))


def _posterior_trace_frame(chains: list[BayesianLatentWeibullChainResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for chain in chains:
        sample_count = len(chain.k_draws)
        for sample_index in range(sample_count):
            rows.append(
                {
                    "chain": chain.chain_id,
                    "sample": sample_index,
                    "parameter": "K",
                    "component": "all",
                    "value": float(chain.k_draws[sample_index]),
                }
            )
            rows.append(
                {
                    "chain": chain.chain_id,
                    "sample": sample_index,
                    "parameter": "log_likelihood",
                    "component": "all",
                    "value": float(chain.log_likelihood[sample_index]),
                }
            )
            rows.append(
                {
                    "chain": chain.chain_id,
                    "sample": sample_index,
                    "parameter": "log_posterior",
                    "component": "all",
                    "value": float(chain.log_posterior[sample_index]),
                }
            )
            current_k = int(chain.k_draws[sample_index])
            for component in range(current_k):
                rows.extend(
                    [
                        {
                            "chain": chain.chain_id,
                            "sample": sample_index,
                            "parameter": "weight",
                            "component": f"component_{component + 1}",
                            "value": float(chain.weights_relabeled[sample_index, component]),
                        },
                        {
                            "chain": chain.chain_id,
                            "sample": sample_index,
                            "parameter": "eta",
                            "component": f"component_{component + 1}",
                            "value": float(chain.eta_relabeled[sample_index, component]),
                        },
                        {
                            "chain": chain.chain_id,
                            "sample": sample_index,
                            "parameter": "beta",
                            "component": f"component_{component + 1}",
                            "value": float(chain.beta_relabeled[sample_index, component]),
                        },
                    ]
                )
    return pd.DataFrame(rows)


def _mixture_survival_draw(times: np.ndarray, weights: np.ndarray, betas: np.ndarray, etas: np.ndarray) -> np.ndarray:
    matrix = np.column_stack(
        [weights[idx] * weibull_survival(times, betas[idx], etas[idx]) for idx in range(len(weights))]
    )
    return matrix.sum(axis=1)


def _mixture_density_draw(times: np.ndarray, weights: np.ndarray, betas: np.ndarray, etas: np.ndarray) -> np.ndarray:
    matrix = np.column_stack(
        [weights[idx] * weibull_density(times, betas[idx], etas[idx]) for idx in range(len(weights))]
    )
    return matrix.sum(axis=1)


@dataclass(slots=True)
class BayesianLatentWeibullFitResult:
    cleaned_df: pd.DataFrame = field(repr=False)
    duration_column: str
    event_column: str
    id_column: str
    validation_report: SurvivalValidationReport
    config: BayesianLatentWeibullConfig
    chains: list[BayesianLatentWeibullChainResult]
    limitations: list[str] = field(default_factory=list)

    @property
    def weights_draws(self) -> np.ndarray:
        return np.vstack([chain.weights_relabeled for chain in self.chains])

    @property
    def eta_draws(self) -> np.ndarray:
        return np.vstack([chain.eta_relabeled for chain in self.chains])

    @property
    def beta_draws(self) -> np.ndarray:
        return np.vstack([chain.beta_relabeled for chain in self.chains])

    @property
    def k_draws(self) -> np.ndarray:
        return np.concatenate([chain.k_draws for chain in self.chains])

    @property
    def max_components(self) -> int:
        return int(self.weights_draws.shape[1])

    def posterior_k_frame(self) -> pd.DataFrame:
        support = np.arange(1, self.max_components + 1, dtype=int)
        rows: list[dict[str, Any]] = []
        for k in support:
            probability = float(np.mean(self.k_draws == k))
            rows.append({"K": int(k), "posterior_probability": probability})
        return pd.DataFrame(rows)

    def _iter_draws(self):
        for chain in self.chains:
            for draw_index, k in enumerate(chain.k_draws):
                current_k = int(k)
                yield (
                    chain.weights_relabeled[draw_index, :current_k],
                    chain.beta_relabeled[draw_index, :current_k],
                    chain.eta_relabeled[draw_index, :current_k],
                    current_k,
                )

    def _component_presence(self, component_index: int) -> np.ndarray:
        return self.k_draws >= (component_index + 1)

    def _safe_component_chain_tensor(self, attribute: str, component_index: int) -> np.ndarray | None:
        chain_arrays: list[np.ndarray] = []
        for chain in self.chains:
            values = getattr(chain, attribute)[:, component_index]
            if np.isnan(values).any():
                return None
            chain_arrays.append(values)
        return np.stack(chain_arrays, axis=0)

    def posterior_summary_frame(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        rows.append(
            {
                "component": "all",
                "parameter": "K",
                "mean": float(np.mean(self.k_draws)),
                "sd": float(np.std(self.k_draws, ddof=1)) if len(self.k_draws) > 1 else 0.0,
                "q2_5": float(np.quantile(self.k_draws, 0.025)),
                "q50": float(np.quantile(self.k_draws, 0.50)),
                "q97_5": float(np.quantile(self.k_draws, 0.975)),
                "ess": _effective_sample_size(self.k_draws.astype(float)),
                "rhat": _split_rhat(np.stack([chain.k_draws.astype(float) for chain in self.chains], axis=0)) if len(self.chains) > 1 else float("nan"),
                "presence_probability": 1.0,
            }
        )
        weight_chain_tensor = np.stack([chain.weights_relabeled for chain in self.chains], axis=0)
        for component in range(self.max_components):
            component_name = f"component_{component + 1}"
            presence = self._component_presence(component)
            arrays = {
                "weight": self.weights_draws[:, component],
                "eta": self.eta_draws[presence, component],
                "beta": self.beta_draws[presence, component],
                "median_life": _component_median_lifetimes(self.eta_draws[presence, component], self.beta_draws[presence, component]),
            }
            for parameter, values in arrays.items():
                clean_values = np.asarray(values, dtype=float)
                clean_values = clean_values[np.isfinite(clean_values)]
                if len(clean_values) == 0:
                    continue
                if parameter == "weight":
                    chain_array = weight_chain_tensor[:, :, component]
                    rhat = _split_rhat(chain_array) if len(self.chains) > 1 else float("nan")
                else:
                    attribute_name = {
                        "eta": "eta_relabeled",
                        "beta": "beta_relabeled",
                        "median_life": "eta_relabeled",
                    }[parameter]
                    chain_array = self._safe_component_chain_tensor(attribute_name, component)
                    if parameter == "median_life" and chain_array is not None:
                        beta_chain = self._safe_component_chain_tensor("beta_relabeled", component)
                        chain_array = None if beta_chain is None else _component_median_lifetimes(chain_array, beta_chain)
                    rhat = _split_rhat(chain_array) if chain_array is not None and len(self.chains) > 1 else float("nan")
                rows.append(
                    {
                        "component": component_name,
                        "parameter": parameter,
                        "mean": float(np.mean(clean_values)),
                        "sd": float(np.std(clean_values, ddof=1)) if len(clean_values) > 1 else 0.0,
                        "q2_5": float(np.quantile(clean_values, 0.025)),
                        "q50": float(np.quantile(clean_values, 0.50)),
                        "q97_5": float(np.quantile(clean_values, 0.975)),
                        "ess": _effective_sample_size(clean_values),
                        "rhat": rhat,
                        "presence_probability": float(np.mean(presence)),
                    }
                )
        return pd.DataFrame(rows)

    def trace_frame(self) -> pd.DataFrame:
        return _posterior_trace_frame(self.chains)

    def posterior_survival_frame(self, max_time: float | None = None, num_points: int = 250) -> pd.DataFrame:
        durations = self.cleaned_df[self.duration_column].to_numpy(dtype=float)
        max_duration = float(np.max(durations))
        horizon = max_duration if max_time is None else float(max_time)
        time_grid = np.linspace(0.0, max(horizon, max_duration), int(num_points))
        survival_draws = []
        density_draws = []
        hazard_draws = []
        component_survival_sum = np.zeros((self.max_components, len(time_grid)), dtype=float)
        component_hazard_sum = np.zeros((self.max_components, len(time_grid)), dtype=float)
        component_counts = np.zeros(self.max_components, dtype=float)
        for weights, betas, etas, current_k in self._iter_draws():
            survival = _mixture_survival_draw(time_grid, weights, betas, etas)
            density = _mixture_density_draw(time_grid, weights, betas, etas)
            hazard = density / np.clip(survival, EPSILON, None)
            survival_draws.append(survival)
            density_draws.append(density)
            hazard_draws.append(hazard)
            for component in range(current_k):
                component_survival_sum[component] += weibull_survival(time_grid, betas[component], etas[component])
                component_hazard_sum[component] += weibull_hazard(time_grid, betas[component], etas[component])
                component_counts[component] += 1.0
        survival_stack = np.asarray(survival_draws, dtype=float)
        density_stack = np.asarray(density_draws, dtype=float)
        hazard_stack = np.asarray(hazard_draws, dtype=float)
        frame = pd.DataFrame(
            {
                "time": time_grid,
                "survival_mean": survival_stack.mean(axis=0),
                "survival_q2_5": np.quantile(survival_stack, 0.025, axis=0),
                "survival_q50": np.quantile(survival_stack, 0.50, axis=0),
                "survival_q97_5": np.quantile(survival_stack, 0.975, axis=0),
                "density_mean": density_stack.mean(axis=0),
                "hazard_mean": hazard_stack.mean(axis=0),
                "cum_hazard_mean": -np.log(np.clip(survival_stack.mean(axis=0), EPSILON, None)),
            }
        )
        for component in range(self.max_components):
            denominator = max(component_counts[component], 1.0)
            frame[f"component_{component + 1}_survival_mean"] = component_survival_sum[component] / denominator
            frame[f"component_{component + 1}_hazard_mean"] = component_hazard_sum[component] / denominator
        return frame

    def posterior_life_quantiles_frame(
        self,
        quantiles: tuple[float, ...] = (0.10, 0.25, 0.50, 0.90),
    ) -> pd.DataFrame:
        """Posterior B10, B25, B50, B90 life quantiles for the mixture.

        B_q satisfies S(B_q) = 1 - q.  For each posterior draw the quantile is
        found via root-finding; the result is summarised across draws.
        """
        durations = self.cleaned_df[self.duration_column].to_numpy(dtype=float)
        upper_search = float(np.quantile(durations, 0.99)) * 20.0

        rows: list[dict[str, Any]] = []
        for q in quantiles:
            target = 1.0 - q
            q_lives: list[float] = []

            for weights, betas, etas, _ in self._iter_draws():
                def _mix_surv(t: float, w: np.ndarray = weights, b: np.ndarray = betas, e: np.ndarray = etas) -> float:
                    return float(_mixture_survival_draw(np.asarray([t]), w, b, e)[0])

                upper = upper_search
                for _ in range(60):
                    if _mix_surv(upper) <= target:
                        break
                    upper *= 2.0
                else:
                    q_lives.append(np.nan)
                    continue
                try:
                    val = float(brentq(lambda t: _mix_surv(t) - target, EPSILON, upper, xtol=1e-4))
                except Exception:
                    val = np.nan
                q_lives.append(val)

            arr = np.asarray(q_lives, dtype=float)
            arr_ok = arr[np.isfinite(arr)]
            rows.append(
                {
                    "quantile": q,
                    "label": f"B{int(round(q * 100))}",
                    "mean": float(np.mean(arr_ok)) if len(arr_ok) else np.nan,
                    "sd": float(np.std(arr_ok, ddof=1)) if len(arr_ok) > 1 else np.nan,
                    "q2_5": float(np.quantile(arr_ok, 0.025)) if len(arr_ok) else np.nan,
                    "q50": float(np.quantile(arr_ok, 0.50)) if len(arr_ok) else np.nan,
                    "q97_5": float(np.quantile(arr_ok, 0.975)) if len(arr_ok) else np.nan,
                }
            )
        return pd.DataFrame(rows)

    def posterior_mixture_mean_life_draws(self) -> np.ndarray:
        """Posterior draw-level mixture mean life E[T] = Σ_k w_k η_k Γ(1 + 1/β_k)."""
        mean_lives: list[float] = []
        for weights, betas, etas, _ in self._iter_draws():
            mean_k = etas * np.exp(gammaln(1.0 + 1.0 / np.clip(betas, EPSILON, None)))
            mean_lives.append(float(np.dot(weights, mean_k)))
        return np.asarray(mean_lives, dtype=float)

    def observation_posterior_probabilities(self) -> pd.DataFrame:
        durations = self.cleaned_df[self.duration_column].to_numpy(dtype=float)
        events = self.cleaned_df[self.event_column].to_numpy(dtype=int)
        probability_sum = np.zeros((len(durations), self.max_components), dtype=float)
        for weights, betas, etas, current_k in self._iter_draws():
            log_weights = np.log(np.clip(weights, EPSILON, None))
            log_kernels = np.full((len(durations), current_k), -np.inf, dtype=float)
            for component in range(current_k):
                log_kernel = np.where(
                    events == 1,
                    _log_weibull_density(durations, betas[component], etas[component]),
                    _log_weibull_survival(durations, betas[component], etas[component]),
                )
                log_kernels[:, component] = log_weights[component] + log_kernel
            log_kernels -= logsumexp(log_kernels, axis=1, keepdims=True)
            probability_sum[:, :current_k] += np.exp(log_kernels)
        averaged = probability_sum / len(self.k_draws)
        output = self.cleaned_df[[self.id_column, self.duration_column, self.event_column]].copy()
        for component in range(self.max_components):
            output[f"component_{component + 1}_probability"] = averaged[:, component]
        output["most_probable_component"] = 1 + np.argmax(averaged, axis=1)
        output["max_probability"] = averaged.max(axis=1)
        output["classification_entropy"] = -np.sum(averaged * np.log(np.clip(averaged, EPSILON, None)), axis=1)
        output["uncertainty_flag"] = output["max_probability"] < self.config.uncertainty_threshold
        return output

    def active_unit_predictions(self, horizons: tuple[int, ...] = (30, 60, 90)) -> pd.DataFrame:
        active = self.cleaned_df.loc[self.cleaned_df[self.event_column].astype(int) == 0].copy()
        if active.empty:
            return pd.DataFrame(columns=[self.id_column, self.duration_column])
        ages = active[self.duration_column].to_numpy(dtype=float)
        output = active[[self.id_column, self.duration_column]].copy()
        for horizon in horizons:
            draw_probabilities = np.zeros((len(active), len(self.k_draws)), dtype=float)
            for draw_index, (weights, betas, etas, _) in enumerate(self._iter_draws()):
                survival_now = _mixture_survival_draw(ages, weights, betas, etas)
                survival_future = _mixture_survival_draw(ages + float(horizon), weights, betas, etas)
                draw_probabilities[:, draw_index] = 1.0 - (survival_future / np.clip(survival_now, EPSILON, None))
            output[f"next_{horizon}_day_failure_mean"] = draw_probabilities.mean(axis=1)
            output[f"next_{horizon}_day_failure_q2_5"] = np.quantile(draw_probabilities, 0.025, axis=1)
            output[f"next_{horizon}_day_failure_q97_5"] = np.quantile(draw_probabilities, 0.975, axis=1)

        component_probs = self.observation_posterior_probabilities().set_index(self.id_column)
        for component in range(self.max_components):
            output[f"component_{component + 1}_probability"] = output[self.id_column].map(
                component_probs[f"component_{component + 1}_probability"]
            )
        output["median_remaining_life"] = [self._posterior_mean_remaining_life(age) for age in ages]
        return output

    def _posterior_mean_remaining_life(self, current_age: float) -> float:
        def posterior_mean_survival(time_value: float) -> float:
            survival_values = [
                float(_mixture_survival_draw(np.asarray([time_value], dtype=float), weights, betas, etas)[0])
                for weights, betas, etas, _ in self._iter_draws()
            ]
            return float(np.mean(survival_values))

        base_survival = posterior_mean_survival(float(current_age))
        target = 0.5 * base_survival
        finite_etas = self.eta_draws[np.isfinite(self.eta_draws)]
        upper = max(float(current_age) + 30.0, float(np.quantile(finite_etas, 0.95)) if len(finite_etas) else 365.0)
        for _ in range(50):
            if posterior_mean_survival(upper) <= target:
                break
            upper *= 1.5
        else:
            return float("nan")
        return float(
            brentq(
                lambda extra: posterior_mean_survival(float(current_age) + extra) - target,
                0.0,
                max(upper - float(current_age), 1.0),
            )
        )

    def diagnostics_frame(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for chain in self.chains:
            rows.append(
                {
                    "chain": chain.chain_id,
                    "seed": chain.seed,
                    "saved_draws": int(len(chain.k_draws)),
                    "mean_K": float(np.mean(chain.k_draws)),
                    "min_K": int(np.min(chain.k_draws)),
                    "max_K": int(np.max(chain.k_draws)),
                    "mean_log_likelihood": float(chain.log_likelihood.mean()),
                    "mean_log_posterior": float(chain.log_posterior.mean()),
                    "acceptance_eta_mean": float(chain.acceptance_rate_eta.mean()),
                    "acceptance_beta_mean": float(chain.acceptance_rate_beta.mean()),
                    "birth_attempts": int(chain.birth_attempts),
                    "birth_accept_rate": float(chain.birth_accepts / max(chain.birth_attempts, 1)),
                    "death_attempts": int(chain.death_attempts),
                    "death_accept_rate": float(chain.death_accepts / max(chain.death_attempts, 1)),
                }
            )
        return pd.DataFrame(rows)

    def export_bundle(self, output_dir: str | Path) -> None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        self.validation_report.to_frame().to_csv(destination / "validation_report.csv", index=False)
        self.posterior_summary_frame().to_csv(destination / "posterior_summary.csv", index=False)
        self.posterior_k_frame().to_csv(destination / "posterior_k.csv", index=False)
        self.trace_frame().to_csv(destination / "trace_draws.csv", index=False)
        self.observation_posterior_probabilities().to_csv(destination / "observation_latent_probabilities.csv", index=False)
        self.active_unit_predictions().to_csv(destination / "active_unit_predictions.csv", index=False)
        self.posterior_survival_frame().to_csv(destination / "posterior_survival.csv", index=False)
        diagnostics = {
            "config": asdict(self.config),
            "limitations": self.limitations,
            "chains": self.diagnostics_frame().to_dict(orient="records"),
        }
        (destination / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        report_lines = [
            "# Bayesian Latent Weibull Mixture Report",
            "",
            f"- Model variant: {'unknown-K birth/death Bayesian Weibull mixture' if self.config.use_unknown_k else f'fixed-K Bayesian Weibull mixture (`K={self.config.n_components}`)'}",
            f"- Rows after validation: {self.validation_report.rows_after_validation}",
            f"- Failures: {self.validation_report.failure_count}",
            f"- Right-censored: {self.validation_report.censored_count}",
            "",
            "## Limitations",
            "",
        ]
        for limitation in self.limitations:
            report_lines.append(f"- {limitation}")
        report_lines.append("")
        report_lines.append("## Notes")
        report_lines.append("")
        for note in self.validation_report.notes:
            report_lines.append(f"- {note}")
        (destination / "report.md").write_text("\n".join(report_lines), encoding="utf-8")


def fit_bayesian_latent_weibull(
    df: pd.DataFrame,
    *,
    duration_column: str | None,
    event_column: str,
    id_column: str | None = None,
    start_date_column: str | None = None,
    end_date_column: str | None = None,
    config: BayesianLatentWeibullConfig | None = None,
) -> BayesianLatentWeibullFitResult:
    effective_config = BayesianLatentWeibullConfig() if config is None else config
    cleaned_df, validation_report = validate_survival_dataframe(
        df,
        duration_column=duration_column,
        event_column=event_column,
        id_column=id_column,
        start_date_column=start_date_column,
        end_date_column=end_date_column,
    )
    effective_duration_column = "duration" if duration_column is None else duration_column
    effective_id_column = "id" if id_column is None else id_column
    durations = cleaned_df[effective_duration_column].to_numpy(dtype=float)
    events = cleaned_df[event_column].to_numpy(dtype=int)
    chains = [
        _run_single_chain(durations, events, effective_config, chain_id=chain_id)
        for chain_id in range(effective_config.n_chains)
    ]
    limitations = [
        "Latent mixture components are probabilistic clusters, not confirmed physical failure causes.",
    ]
    if effective_config.use_unknown_k:
        limitations.append("Unknown-K inference uses birth/death trans-dimensional moves with a truncated Poisson prior on K.")
    else:
        limitations.append("This run uses a fixed-K Bayesian Weibull mixture.")
    return BayesianLatentWeibullFitResult(
        cleaned_df=cleaned_df,
        duration_column=effective_duration_column,
        event_column=event_column,
        id_column=effective_id_column,
        validation_report=validation_report,
        config=effective_config,
        chains=chains,
        limitations=limitations,
    )
