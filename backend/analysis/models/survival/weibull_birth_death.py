"""
Unknown-K Bayesian Weibull mixture via birth-death MCMC.

Implements the methodology of:
    Marín, Rodríguez-Bernal & Wiper (2005). Using Weibull Mixture Distributions
    to Model Heterogeneous Survival Data. Commun. Stat. Simul. Comput. 34(3): 673-684.

The population survival function is:
    S(t) = Σ_k w_k exp(-(t/η_k)^β_k),  Σ_k w_k = 1

The number of components K is treated as unknown with a truncated Poisson prior.
A birth-death MCMC process samples over variable-dimensional spaces.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import gammaln, logsumexp

from .bayesian_latent_weibull import (
    EPSILON,
    SurvivalValidationReport,
    _effective_sample_size,
    _log_weibull_density,
    _log_weibull_survival,
    _split_rhat,
    kaplan_meier_frame,
    load_survival_file,
    validate_survival_dataframe,
    weibull_density,
    weibull_hazard,
    weibull_survival,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class BirthDeathWeibullConfig:
    """Configuration for the unknown-K birth-death MCMC sampler.

    Attributes
    ----------
    lambda_k:
        Poisson rate parameter for the prior on K.  P(K=k) ∝ λ^k / k!
    k_max:
        Maximum allowed number of components.
    birth_rate:
        Fixed birth rate β for the continuous-time birth-death process.
    bd_time:
        Duration t₀ of the birth-death process at each Gibbs step (paper uses t₀=1).
    n_iter:
        Total Gibbs iterations (includes burn-in).
    burn_in:
        Number of iterations to discard.
    thin:
        Save one draw every `thin` post-burn-in iterations.
    n_chains:
        Number of independent chains.
    alpha_dirichlet:
        Symmetric Dirichlet concentration for the weight prior.
    mu_log_eta, sigma_log_eta:
        Log-normal prior on the Weibull scale η (characteristic life).
    mu_log_beta, sigma_log_beta:
        Log-normal prior on the Weibull shape β.
    proposal_sd_log_eta, proposal_sd_log_beta:
        Metropolis-Hastings random-walk proposal standard deviations.
    random_seed:
        Base seed; chain i uses seed + 1009*i.
    uncertainty_threshold:
        Max-probability threshold below which an observation is flagged as uncertain.
    """

    lambda_k: float = 3.0
    k_max: int = 10
    birth_rate: float = 3.0
    bd_time: float = 1.0
    n_iter: int = 10_000
    burn_in: int = 2_000
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

    def __post_init__(self) -> None:
        if self.k_max < 1:
            raise ValueError("k_max must be at least 1.")
        if self.lambda_k <= 0.0:
            raise ValueError("lambda_k must be positive.")
        if self.birth_rate <= 0.0:
            raise ValueError("birth_rate must be positive.")
        if self.n_iter <= self.burn_in:
            raise ValueError("n_iter must exceed burn_in.")
        if self.thin < 1:
            raise ValueError("thin must be at least 1.")
        if self.n_chains < 1:
            raise ValueError("n_chains must be at least 1.")


# ---------------------------------------------------------------------------
# Internal state (mutable, variable-K)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _BDState:
    k: int
    weights: np.ndarray    # shape (k,), sums to 1
    log_eta: np.ndarray    # shape (k,)
    log_beta: np.ndarray   # shape (k,)
    allocations: np.ndarray  # shape (n,), values in {0,...,k-1}


# ---------------------------------------------------------------------------
# Saved posterior draw (variable K)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class BirthDeathDraw:
    """One thinned posterior draw from the birth-death chain."""

    k: int
    weights: np.ndarray   # shape (k,)
    eta: np.ndarray       # shape (k,), relabeled by median lifetime
    beta: np.ndarray      # shape (k,)
    log_likelihood: float


# ---------------------------------------------------------------------------
# Chain-level result
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class BirthDeathChainResult:
    """Posterior samples from a single birth-death MCMC chain."""

    chain_id: int
    draws: list[BirthDeathDraw]
    k_trace: np.ndarray          # one entry per iteration (including burn-in)
    acceptance_count_eta: np.ndarray   # per-iteration count (variable length)
    acceptance_count_beta: np.ndarray
    birth_count: int
    death_count: int
    bd_event_count: int
    seed: int

    @property
    def k_posterior_trace(self) -> np.ndarray:
        return np.asarray([d.k for d in self.draws], dtype=int)


# ---------------------------------------------------------------------------
# Core numerical helpers
# ---------------------------------------------------------------------------

def _component_median_lifetimes(eta: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return eta * np.power(np.log(2.0), 1.0 / np.clip(beta, EPSILON, None))


def _log_prior_k(k: int, lambda_k: float, k_max: int) -> float:
    """Log of the truncated Poisson prior P(K=k) ∝ λ^k / k!."""
    if k < 1 or k > k_max:
        return -np.inf
    return float(k * np.log(lambda_k) - gammaln(k + 1.0))


def _compute_log_kernels(
    durations: np.ndarray,
    log_eta: np.ndarray,
    log_beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (log_f, log_S) matrices of shape (n, k).

    log_f[i, j] = log Weibull density   of observation i under component j.
    log_S[i, j] = log Weibull survival  of observation i under component j.
    """
    n = len(durations)
    k = len(log_eta)
    log_f = np.empty((n, k), dtype=float)
    log_S = np.empty((n, k), dtype=float)
    for j in range(k):
        eta_j = float(np.exp(log_eta[j]))
        beta_j = float(np.exp(log_beta[j]))
        log_f[:, j] = _log_weibull_density(durations, beta_j, eta_j)
        log_S[:, j] = _log_weibull_survival(durations, beta_j, eta_j)
    return log_f, log_S


def _compute_full_log_lik(
    events: np.ndarray,
    log_weights: np.ndarray,
    log_f: np.ndarray,
    log_S: np.ndarray,
) -> float:
    """Mixture log-likelihood using logsumexp for numerical stability.

    Failures (event=1) contribute the mixture density;
    censored (event=0) contribute the mixture survival.
    """
    log_mix_f = logsumexp(log_weights + log_f, axis=1)   # (n,)
    log_mix_S = logsumexp(log_weights + log_S, axis=1)   # (n,)
    contributions = np.where(events == 1, log_mix_f, log_mix_S)
    return float(np.sum(contributions))


def _compute_death_rates(
    events: np.ndarray,
    log_weights: np.ndarray,
    log_f: np.ndarray,
    log_S: np.ndarray,
    full_log_lik: float,
    config: BirthDeathWeibullConfig,
) -> np.ndarray:
    """Compute the death rate δ_j for each component j.

    From the paper (Section 4):
        δ_j = β · [l(k, w, θ, a \\ j | data) / l(k, w, θ, a | data)] · P(k-1) / (k · P(k))

    With the truncated Poisson prior P(k) ∝ λ^k/k!:
        P(k-1) / (k · P(k)) = 1 / λ

    So: δ_j = β_birth · exp(log_lik_excl - full_log_lik) / λ_k
    """
    k = len(log_weights)
    death_rates = np.zeros(k, dtype=float)
    if k <= 1:
        return death_rates

    for j in range(k):
        excl_idx = [l for l in range(k) if l != j]
        log_w_excl = log_weights[excl_idx]
        log_f_excl = log_f[:, excl_idx]
        log_S_excl = log_S[:, excl_idx]

        # log(1 - w_j) from the remaining log-weights
        log_one_minus_wj = logsumexp(log_w_excl)

        # Rescaled mixture: divide by (1 - w_j)
        log_mix_f_excl = logsumexp(log_w_excl + log_f_excl, axis=1) - log_one_minus_wj
        log_mix_S_excl = logsumexp(log_w_excl + log_S_excl, axis=1) - log_one_minus_wj

        log_lik_excl = float(np.sum(np.where(events == 1, log_mix_f_excl, log_mix_S_excl)))
        log_ratio = log_lik_excl - full_log_lik

        log_death_j = np.log(config.birth_rate) + log_ratio - np.log(config.lambda_k)
        death_rates[j] = float(np.exp(np.clip(log_death_j, -700.0, 700.0)))

    return death_rates


# ---------------------------------------------------------------------------
# Birth / death moves
# ---------------------------------------------------------------------------

def _birth_move(
    state: _BDState,
    config: BirthDeathWeibullConfig,
    rng: np.random.Generator,
) -> _BDState:
    """Add a new component sampled from the prior (Section 4 of the paper).

    w_{k+1} ~ Beta(1, k)
    log_eta_{k+1} ~ N(mu_log_eta, sigma_log_eta^2)
    log_beta_{k+1} ~ N(mu_log_beta, sigma_log_beta^2)

    Existing weights rescaled: w_j → w_j / (1 + w_{k+1}) for j=1,...,k.
    """
    w_new_raw = float(rng.beta(1.0, float(state.k)))
    new_log_eta = float(rng.normal(config.mu_log_eta, config.sigma_log_eta))
    new_log_beta = float(rng.normal(config.mu_log_beta, config.sigma_log_beta))

    new_weights = np.append(
        state.weights / (1.0 + w_new_raw),
        w_new_raw / (1.0 + w_new_raw),
    )
    new_log_eta_arr = np.append(state.log_eta, new_log_eta)
    new_log_beta_arr = np.append(state.log_beta, new_log_beta)

    return _BDState(
        k=state.k + 1,
        weights=new_weights,
        log_eta=new_log_eta_arr,
        log_beta=new_log_beta_arr,
        allocations=state.allocations.copy(),
    )


def _death_move(state: _BDState, j: int) -> _BDState:
    """Remove component j and rescale remaining weights.

    Remaining weights rescaled: w_l → w_l / (1 - w_j) for l ≠ j.
    """
    w_j = float(state.weights[j])
    scale = max(1.0 - w_j, EPSILON)

    excl_idx = np.array([l for l in range(state.k) if l != j], dtype=int)
    new_weights = state.weights[excl_idx] / scale
    new_weights /= new_weights.sum()   # re-normalize for floating-point safety
    new_log_eta = state.log_eta[excl_idx]
    new_log_beta = state.log_beta[excl_idx]

    # Remap allocations: indices > j shift down by 1; those at j go to 0
    new_allocs = state.allocations.copy()
    new_allocs[new_allocs == j] = 0
    for l in range(j + 1, state.k):
        new_allocs[new_allocs == l] = l - 1

    return _BDState(
        k=state.k - 1,
        weights=new_weights,
        log_eta=new_log_eta,
        log_beta=new_log_beta,
        allocations=new_allocs,
    )


# ---------------------------------------------------------------------------
# Birth-death process (continuous-time step)
# ---------------------------------------------------------------------------

def _run_bd_step(
    durations: np.ndarray,
    events: np.ndarray,
    state: _BDState,
    config: BirthDeathWeibullConfig,
    rng: np.random.Generator,
    counters: dict[str, int],
) -> _BDState:
    """Simulate the birth-death process for t₀ = config.bd_time time units.

    This corresponds to Step 6 in the paper's Gibbs algorithm.
    """
    current_time = 0.0
    max_events = 100   # safety cap to avoid infinite loops

    for _ in range(max_events):
        log_f, log_S = _compute_log_kernels(durations, state.log_eta, state.log_beta)
        log_w = np.log(np.clip(state.weights, EPSILON, None))

        full_log_lik = _compute_full_log_lik(events, log_w, log_f, log_S)
        death_rates = _compute_death_rates(events, log_w, log_f, log_S, full_log_lik, config)
        total_death = float(np.sum(death_rates))

        b_rate = config.birth_rate if state.k < config.k_max else 0.0
        total_rate = b_rate + total_death
        if total_rate <= EPSILON:
            break

        time_to_event = float(rng.exponential(1.0 / total_rate))
        if current_time + time_to_event >= config.bd_time:
            break

        current_time += time_to_event
        counters["events"] += 1

        if rng.uniform() < (b_rate / total_rate):
            state = _birth_move(state, config, rng)
            counters["births"] += 1
        else:
            if state.k > 1 and total_death > EPSILON:
                probs = death_rates / total_death
                j = int(rng.choice(state.k, p=probs))
                state = _death_move(state, j)
                counters["deaths"] += 1

    return state


# ---------------------------------------------------------------------------
# Gibbs sampler sub-steps (variable K)
# ---------------------------------------------------------------------------

def _sample_allocations_vk(
    durations: np.ndarray,
    events: np.ndarray,
    state: _BDState,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample Z_i ~ Pr(z_i = j | k, w, θ, a, x_i) for each observation."""
    log_f, log_S = _compute_log_kernels(durations, state.log_eta, state.log_beta)
    log_w = np.log(np.clip(state.weights, EPSILON, None))

    log_kernel = np.where(events[:, None] == 1, log_f, log_S)  # (n, k)
    log_unnorm = log_w + log_kernel                              # (n, k)
    log_norm = log_unnorm - logsumexp(log_unnorm, axis=1, keepdims=True)
    probs = np.exp(log_norm)

    draws = rng.uniform(size=len(durations))
    cumul = np.cumsum(probs, axis=1)
    return np.argmax(draws[:, None] <= cumul, axis=1).astype(int)


def _mh_update_vk(
    durations: np.ndarray,
    events: np.ndarray,
    state: _BDState,
    j: int,
    config: BirthDeathWeibullConfig,
    rng: np.random.Generator,
) -> tuple[bool, bool]:
    """Metropolis-Hastings update for (log_eta_j, log_beta_j) given allocations."""
    mask = state.allocations == j
    cur_log_eta = float(state.log_eta[j])
    cur_log_beta = float(state.log_beta[j])

    def _log_target(le: float, lb: float) -> float:
        if not mask.any():
            return 0.0
        eta_j = float(np.exp(le))
        beta_j = float(np.exp(lb))
        t_sel = durations[mask]
        e_sel = events[mask]
        contrib = np.where(
            e_sel == 1,
            _log_weibull_density(t_sel, beta_j, eta_j),
            _log_weibull_survival(t_sel, beta_j, eta_j),
        )
        log_lik = float(np.sum(contrib))
        log_prior = (
            -0.5 * ((le - config.mu_log_eta) / config.sigma_log_eta) ** 2
            - 0.5 * ((lb - config.mu_log_beta) / config.sigma_log_beta) ** 2
        )
        return float(log_lik + log_prior)

    cur_target = _log_target(cur_log_eta, cur_log_beta)

    prop_le = cur_log_eta + rng.normal(0.0, config.proposal_sd_log_eta)
    prop_target_eta = _log_target(prop_le, cur_log_beta)
    accepted_eta = float(np.log(rng.uniform())) < (prop_target_eta - cur_target)
    if accepted_eta:
        state.log_eta[j] = prop_le
        cur_log_eta = prop_le
        cur_target = prop_target_eta

    prop_lb = cur_log_beta + rng.normal(0.0, config.proposal_sd_log_beta)
    prop_target_beta = _log_target(cur_log_eta, prop_lb)
    accepted_beta = float(np.log(rng.uniform())) < (prop_target_beta - cur_target)
    if accepted_beta:
        state.log_beta[j] = prop_lb

    return bool(accepted_eta), bool(accepted_beta)


# ---------------------------------------------------------------------------
# State initialization
# ---------------------------------------------------------------------------

def _initialize_bd_state(
    durations: np.ndarray,
    events: np.ndarray,
    config: BirthDeathWeibullConfig,
    rng: np.random.Generator,
) -> _BDState:
    """Start with K=2 using quantile-based initial allocations."""
    k_init = min(2, config.k_max)
    n = len(durations)

    log_dur = np.log(np.clip(durations, EPSILON, None))
    if k_init > 1:
        median_log = float(np.median(log_dur))
        allocations = (log_dur >= median_log).astype(int)
    else:
        allocations = np.zeros(n, dtype=int)

    log_eta = np.zeros(k_init, dtype=float)
    log_beta = np.zeros(k_init, dtype=float)
    global_median = float(np.median(durations))

    for j in range(k_init):
        mask = allocations == j
        comp_dur = durations[mask] if mask.any() else durations
        eta_guess = max(float(np.median(comp_dur)), EPSILON)
        beta_guess = 0.8 if (j == 0 and k_init > 1) else 1.5
        log_eta[j] = np.log(eta_guess) + rng.normal(0.0, 0.05)
        log_beta[j] = np.log(beta_guess) + rng.normal(0.0, 0.05)

    counts = np.bincount(allocations, minlength=k_init).astype(float) + config.alpha_dirichlet
    weights = counts / counts.sum()

    return _BDState(k=k_init, weights=weights, log_eta=log_eta, log_beta=log_beta, allocations=allocations)


# ---------------------------------------------------------------------------
# Single chain runner
# ---------------------------------------------------------------------------

def _relabel_draw(weights: np.ndarray, eta: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort components by median lifetime (ascending) to address label switching."""
    order = np.argsort(_component_median_lifetimes(eta, beta))
    return weights[order], eta[order], beta[order]


def _run_bd_chain(
    durations: np.ndarray,
    events: np.ndarray,
    config: BirthDeathWeibullConfig,
    chain_id: int,
) -> BirthDeathChainResult:
    seed = int(config.random_seed + 1009 * chain_id)
    rng = np.random.default_rng(seed)

    state = _initialize_bd_state(durations, events, config, rng)
    draws: list[BirthDeathDraw] = []
    k_trace: list[int] = []
    acceptance_count_eta: list[int] = []
    acceptance_count_beta: list[int] = []
    counters: dict[str, int] = {"births": 0, "deaths": 0, "events": 0}

    for iteration in range(config.n_iter):
        # 1. Sample component allocations
        state.allocations = _sample_allocations_vk(durations, events, state, rng)

        # 2. Sample mixture weights from Dirichlet conditional
        counts = np.bincount(state.allocations, minlength=state.k).astype(float)
        state.weights = rng.dirichlet(counts + config.alpha_dirichlet)

        # 3. MH for Weibull parameters of each component
        acc_eta_iter = 0
        acc_beta_iter = 0
        for j in range(state.k):
            ae, ab = _mh_update_vk(durations, events, state, j, config, rng)
            acc_eta_iter += int(ae)
            acc_beta_iter += int(ab)
        acceptance_count_eta.append(acc_eta_iter)
        acceptance_count_beta.append(acc_beta_iter)

        # 4. Birth-death step (changes K)
        state = _run_bd_step(durations, events, state, config, rng, counters)

        k_trace.append(state.k)

        # 5. Save thinned post-burn-in draws
        if iteration >= config.burn_in and (iteration - config.burn_in) % config.thin == 0:
            log_f, log_S = _compute_log_kernels(durations, state.log_eta, state.log_beta)
            log_w = np.log(np.clip(state.weights, EPSILON, None))
            log_lik = _compute_full_log_lik(events, log_w, log_f, log_S)

            eta_vals = np.exp(state.log_eta)
            beta_vals = np.exp(state.log_beta)
            w_rl, eta_rl, beta_rl = _relabel_draw(state.weights.copy(), eta_vals, beta_vals)

            draws.append(
                BirthDeathDraw(
                    k=state.k,
                    weights=w_rl,
                    eta=eta_rl,
                    beta=beta_rl,
                    log_likelihood=log_lik,
                )
            )

    return BirthDeathChainResult(
        chain_id=chain_id,
        draws=draws,
        k_trace=np.asarray(k_trace, dtype=int),
        acceptance_count_eta=np.asarray(acceptance_count_eta, dtype=int),
        acceptance_count_beta=np.asarray(acceptance_count_beta, dtype=int),
        birth_count=counters["births"],
        death_count=counters["deaths"],
        bd_event_count=counters["events"],
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Fit result with analysis methods
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class BirthDeathFitResult:
    """Full result from a birth-death Weibull mixture fit."""

    cleaned_df: pd.DataFrame = field(repr=False)
    duration_column: str
    event_column: str
    id_column: str
    validation_report: SurvivalValidationReport
    config: BirthDeathWeibullConfig
    chains: list[BirthDeathChainResult]
    limitations: list[str] = field(default_factory=list)

    @property
    def all_draws(self) -> list[BirthDeathDraw]:
        return [d for chain in self.chains for d in chain.draws]

    def posterior_k_frame(self) -> pd.DataFrame:
        """Posterior distribution of K."""
        k_values = np.asarray([d.k for d in self.all_draws], dtype=int)
        unique, counts = np.unique(k_values, return_counts=True)
        return pd.DataFrame({"k": unique, "probability": counts / counts.sum()})

    def p_single_weibull(self) -> float:
        """P(K=1 | data)."""
        row = self.posterior_k_frame()
        match = row.loc[row["k"] == 1, "probability"]
        return float(match.iloc[0]) if not match.empty else 0.0

    def posterior_survival_frame(self, max_time: float | None = None, num_points: int = 250) -> pd.DataFrame:
        """Posterior mean and credible intervals for survival, density, hazard."""
        durations = self.cleaned_df[self.duration_column].to_numpy(dtype=float)
        max_dur = float(np.max(durations))
        horizon = max_dur if max_time is None else float(max_time)
        t_grid = np.linspace(0.0, max(horizon, max_dur), int(num_points))

        draws = self.all_draws
        surv_mat = np.empty((len(draws), len(t_grid)), dtype=float)
        dens_mat = np.empty_like(surv_mat)
        haz_mat = np.empty_like(surv_mat)

        for i, draw in enumerate(draws):
            s = np.zeros(len(t_grid), dtype=float)
            f = np.zeros(len(t_grid), dtype=float)
            for j in range(draw.k):
                s += draw.weights[j] * np.exp(_log_weibull_survival(t_grid, float(draw.beta[j]), float(draw.eta[j])))
                f += draw.weights[j] * np.exp(_log_weibull_density(t_grid, float(draw.beta[j]), float(draw.eta[j])))
            h = f / np.clip(s, EPSILON, None)
            surv_mat[i] = s
            dens_mat[i] = f
            haz_mat[i] = h

        return pd.DataFrame(
            {
                "time": t_grid,
                "survival_mean": surv_mat.mean(axis=0),
                "survival_q2_5": np.quantile(surv_mat, 0.025, axis=0),
                "survival_q50": np.quantile(surv_mat, 0.50, axis=0),
                "survival_q97_5": np.quantile(surv_mat, 0.975, axis=0),
                "density_mean": dens_mat.mean(axis=0),
                "hazard_mean": haz_mat.mean(axis=0),
                "cum_hazard_mean": -np.log(np.clip(surv_mat.mean(axis=0), EPSILON, None)),
            }
        )

    def posterior_mixture_survival(self, times: np.ndarray) -> np.ndarray:
        """Return posterior mean survival at given times."""
        draws = self.all_draws
        surv_sum = np.zeros(len(times), dtype=float)
        for draw in draws:
            s = np.zeros(len(times), dtype=float)
            for j in range(draw.k):
                s += draw.weights[j] * np.exp(
                    _log_weibull_survival(times, float(draw.beta[j]), float(draw.eta[j]))
                )
            surv_sum += s
        return surv_sum / max(len(draws), 1)

    def posterior_life_quantiles_frame(
        self,
        quantiles: tuple[float, ...] = (0.10, 0.25, 0.50, 0.90),
    ) -> pd.DataFrame:
        """Posterior B10, B25, B50, B90 life quantiles.

        B_q is the time at which the cumulative failure probability reaches q,
        i.e. S(B_q) = 1 - q.  The posterior distribution is summarized by its
        mean, median and 95% credible interval.
        """
        durations = self.cleaned_df[self.duration_column].to_numpy(dtype=float)
        upper_search = float(np.quantile(durations, 0.99)) * 20.0

        rows: list[dict[str, Any]] = []
        for q in quantiles:
            target = 1.0 - q
            q_lives: list[float] = []
            for draw in self.all_draws:
                def _mix_surv(t: float, d: BirthDeathDraw = draw) -> float:
                    s = 0.0
                    for jj in range(d.k):
                        s += d.weights[jj] * float(
                            np.exp(_log_weibull_survival(np.asarray([t]), float(d.beta[jj]), float(d.eta[jj]))[0])
                        )
                    return s

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

    def posterior_summary_frame(self) -> pd.DataFrame:
        """Per-component posterior summaries for draws with the modal K.

        Because K varies across draws, this summary is conditioned on the
        modal (most common) K for interpretability.
        """
        draws = self.all_draws
        k_values = [d.k for d in draws]
        modal_k = int(np.bincount(k_values).argmax())

        modal_draws = [d for d in draws if d.k == modal_k]
        if not modal_draws:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for j in range(modal_k):
            weights_j = np.asarray([d.weights[j] for d in modal_draws])
            eta_j = np.asarray([d.eta[j] for d in modal_draws])
            beta_j = np.asarray([d.beta[j] for d in modal_draws])
            median_j = _component_median_lifetimes(eta_j, beta_j)
            mean_j = eta_j * np.exp(gammaln(1.0 + 1.0 / np.clip(beta_j, EPSILON, None)))

            for param, arr in [("weight", weights_j), ("eta", eta_j), ("beta", beta_j),
                                ("median_life", median_j), ("mean_life", mean_j)]:
                rows.append(
                    {
                        "component": f"component_{j + 1}",
                        "parameter": param,
                        "mean": float(np.mean(arr)),
                        "sd": float(np.std(arr, ddof=1)) if len(arr) > 1 else np.nan,
                        "q2_5": float(np.quantile(arr, 0.025)),
                        "q50": float(np.quantile(arr, 0.50)),
                        "q97_5": float(np.quantile(arr, 0.975)),
                        "n_draws": len(modal_draws),
                    }
                )
        return pd.DataFrame(rows)

    def observation_posterior_probabilities(self) -> pd.DataFrame:
        """Posterior component membership probability for each observation.

        Failures use the density-based posterior; censored use the survival-based posterior.
        """
        durations = self.cleaned_df[self.duration_column].to_numpy(dtype=float)
        events = self.cleaned_df[self.event_column].to_numpy(dtype=int)
        n = len(durations)

        draws = self.all_draws
        k_values = [d.k for d in draws]
        modal_k = int(np.bincount(k_values).argmax())

        prob_sum = np.zeros((n, modal_k), dtype=float)
        count = 0

        for draw in draws:
            if draw.k != modal_k:
                continue
            count += 1
            log_f, log_S = _compute_log_kernels(durations, np.log(draw.beta), np.log(draw.eta))
            log_w = np.log(np.clip(draw.weights, EPSILON, None))
            log_kernel = np.where(events[:, None] == 1, log_f, log_S)
            log_unnorm = log_w + log_kernel
            log_norm = log_unnorm - logsumexp(log_unnorm, axis=1, keepdims=True)
            prob_sum += np.exp(log_norm)

        if count == 0:
            return self.cleaned_df[[self.id_column, self.duration_column, self.event_column]].copy()

        averaged = prob_sum / count
        output = self.cleaned_df[[self.id_column, self.duration_column, self.event_column]].copy()
        for j in range(modal_k):
            output[f"component_{j + 1}_probability"] = averaged[:, j]
        output["most_probable_component"] = 1 + np.argmax(averaged, axis=1)
        output["max_probability"] = averaged.max(axis=1)
        output["classification_entropy"] = -np.sum(
            averaged * np.log(np.clip(averaged, EPSILON, None)), axis=1
        )
        output["uncertainty_flag"] = output["max_probability"] < self.config.uncertainty_threshold
        return output

    def active_unit_predictions(self, horizons: tuple[int, ...] = (30, 60, 90)) -> pd.DataFrame:
        """Conditional failure probability over future windows for censored units."""
        active = self.cleaned_df.loc[self.cleaned_df[self.event_column].astype(int) == 0].copy()
        if active.empty:
            return pd.DataFrame(columns=[self.id_column, self.duration_column])

        ages = active[self.duration_column].to_numpy(dtype=float)
        draws = self.all_draws
        output = active[[self.id_column, self.duration_column]].copy()

        for horizon in horizons:
            draw_probs = np.zeros((len(active), len(draws)), dtype=float)
            for idx, draw in enumerate(draws):
                s_now = np.zeros(len(ages), dtype=float)
                s_fut = np.zeros(len(ages), dtype=float)
                for j in range(draw.k):
                    s_now += draw.weights[j] * np.exp(
                        _log_weibull_survival(ages, float(draw.beta[j]), float(draw.eta[j]))
                    )
                    s_fut += draw.weights[j] * np.exp(
                        _log_weibull_survival(ages + float(horizon), float(draw.beta[j]), float(draw.eta[j]))
                    )
                draw_probs[:, idx] = 1.0 - s_fut / np.clip(s_now, EPSILON, None)
            output[f"next_{horizon}_day_failure_mean"] = draw_probs.mean(axis=1)
            output[f"next_{horizon}_day_failure_q2_5"] = np.quantile(draw_probs, 0.025, axis=1)
            output[f"next_{horizon}_day_failure_q97_5"] = np.quantile(draw_probs, 0.975, axis=1)

        return output

    def diagnostics_frame(self) -> pd.DataFrame:
        """Chain-level diagnostics summary."""
        rows: list[dict[str, Any]] = []
        for chain in self.chains:
            post_iters = config.n_iter - config.burn_in if (config := self.config) else 0
            rows.append(
                {
                    "chain": chain.chain_id,
                    "seed": chain.seed,
                    "saved_draws": len(chain.draws),
                    "birth_count": chain.birth_count,
                    "death_count": chain.death_count,
                    "bd_event_count": chain.bd_event_count,
                    "mean_k_post_burnin": float(chain.k_posterior_trace.mean()) if chain.draws else np.nan,
                    "mean_log_lik": float(np.mean([d.log_likelihood for d in chain.draws])) if chain.draws else np.nan,
                }
            )
        return pd.DataFrame(rows)

    def k_trace_frame(self) -> pd.DataFrame:
        """K trace for each chain (all iterations including burn-in)."""
        rows: list[dict[str, Any]] = []
        for chain in self.chains:
            for iteration, k_val in enumerate(chain.k_trace):
                rows.append({"chain": chain.chain_id, "iteration": iteration, "k": int(k_val)})
        return pd.DataFrame(rows)

    def export_bundle(self, output_dir: str | Path) -> None:
        dest = Path(output_dir)
        dest.mkdir(parents=True, exist_ok=True)

        self.validation_report.to_frame().to_csv(dest / "validation_report.csv", index=False)
        self.posterior_k_frame().to_csv(dest / "posterior_k_distribution.csv", index=False)
        self.posterior_summary_frame().to_csv(dest / "posterior_summary.csv", index=False)
        self.posterior_life_quantiles_frame().to_csv(dest / "posterior_life_quantiles.csv", index=False)
        self.posterior_survival_frame().to_csv(dest / "posterior_survival.csv", index=False)
        self.observation_posterior_probabilities().to_csv(dest / "observation_latent_probabilities.csv", index=False)
        self.active_unit_predictions().to_csv(dest / "active_unit_predictions.csv", index=False)
        self.k_trace_frame().to_csv(dest / "k_trace.csv", index=False)
        self.diagnostics_frame().to_csv(dest / "diagnostics.csv", index=False)

        meta = {
            "model": "birth_death_weibull_mixture",
            "config": self.config.__dict__,
            "p_single_weibull": self.p_single_weibull(),
            "limitations": self.limitations,
            "chain_diagnostics": self.diagnostics_frame().to_dict(orient="records"),
        }
        (dest / "diagnostics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        report = [
            "# Birth-Death Bayesian Weibull Mixture Report",
            "",
            f"- Model: unknown-K Weibull mixture (birth-death MCMC)",
            f"- Rows after validation: {self.validation_report.rows_after_validation}",
            f"- Failures: {self.validation_report.failure_count}",
            f"- Right-censored: {self.validation_report.censored_count}",
            f"- P(K=1 | data): {self.p_single_weibull():.3f}",
            "",
            "## Posterior K Distribution",
            "",
        ]
        for _, row in self.posterior_k_frame().iterrows():
            report.append(f"- K={int(row['k'])}: {row['probability']:.3f}")
        report += ["", "## Limitations", ""]
        for lim in self.limitations:
            report.append(f"- {lim}")
        (dest / "report.md").write_text("\n".join(report), encoding="utf-8")


# ---------------------------------------------------------------------------
# Top-level fit function
# ---------------------------------------------------------------------------

def fit_birth_death_weibull(
    df: pd.DataFrame,
    *,
    duration_column: str | None,
    event_column: str,
    id_column: str | None = None,
    start_date_column: str | None = None,
    end_date_column: str | None = None,
    config: BirthDeathWeibullConfig | None = None,
) -> BirthDeathFitResult:
    """Fit a Bayesian Weibull mixture with unknown K via birth-death MCMC.

    Parameters
    ----------
    df:
        Input DataFrame.  Must contain duration/event columns or date columns.
    duration_column:
        Column with observed duration (positive, in any consistent time unit).
        Pass ``None`` together with *start_date_column* and *end_date_column* to
        compute duration from dates.
    event_column:
        Binary column: 1 = observed failure, 0 = right-censored.
    id_column:
        Unique identifier column.  Auto-generated if ``None``.
    start_date_column, end_date_column:
        Used only when *duration_column* is ``None``.
    config:
        Sampler configuration.  Defaults to :class:`BirthDeathWeibullConfig`.

    Returns
    -------
    BirthDeathFitResult
    """
    cfg = BirthDeathWeibullConfig() if config is None else config

    cleaned_df, validation_report = validate_survival_dataframe(
        df,
        duration_column=duration_column,
        event_column=event_column,
        id_column=id_column,
        start_date_column=start_date_column,
        end_date_column=end_date_column,
    )
    eff_dur_col = "duration" if duration_column is None else duration_column
    eff_id_col = "id" if id_column is None else id_column

    durations = cleaned_df[eff_dur_col].to_numpy(dtype=float)
    events = cleaned_df[event_column].to_numpy(dtype=int)

    chains = [
        _run_bd_chain(durations, events, cfg, chain_id=cid)
        for cid in range(cfg.n_chains)
    ]

    limitations = [
        "Latent mixture components represent probabilistic heterogeneity, not confirmed physical failure causes.",
        "With unknown K, label switching is partially addressed by relabeling draws by median lifetime.",
        "Birth-death acceptance rate depends on the birth_rate and lambda_k hyperparameters.",
        "Posterior credible intervals for K may be wide with small sample sizes.",
    ]

    return BirthDeathFitResult(
        cleaned_df=cleaned_df,
        duration_column=eff_dur_col,
        event_column=event_column,
        id_column=eff_id_col,
        validation_report=validation_report,
        config=cfg,
        chains=chains,
        limitations=limitations,
    )
