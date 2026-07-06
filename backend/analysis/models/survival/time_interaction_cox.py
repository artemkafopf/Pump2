"""Time-interaction (extended) Cox via exact event-time episode splitting.

Factored out of ``scripts/run/phase_a_extended_cox_fix.py`` (Phase A T6) so the
**corrected** estimator is reusable and tested.  It fits

    h(t) = h₀(t) · exp(β·X + γ·X·log t)

as a genuine partial likelihood by episode-splitting subject-level survival data
at **every distinct failure time** and handing the long format to lifelines'
``CoxTimeVaryingFitter``.

Why the split convention matters (2026-07-06 correction, do not regress):
cutting at every distinct failure time makes each episode's ``stop`` a risk-set
evaluation time, so ``X·log(stop)`` equals ``X·log(t)`` for the failing subject
*and* every subject at risk at that time — the exact time-interaction partial
likelihood.  The first shipped version cut at fixed 30-day intervals with the
covariate at episode *end*; cases then carried ``log(own failure time)`` while
at-risk controls carried ``log(grid edge)`` — a mechanical negative bias on γ
(shipped γ = −0.851; corrected γ ≈ +0.2).  The regression test in
``tests/test_time_interaction_cox.py`` fits a synthetic dataset with a known flat
HR and asserts γ ≈ 0 — the check that would have caught the −0.851 bug.

The module is covariate-agnostic: pass any binary/continuous ``covariate`` column
(``sour`` in the Vt H₂S application; a mode indicator in Phase B B4).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class TimeInteractionCoxResult:
    """β (proportional part), γ (log-time interaction) and their model SEs/p.

    The citable uncertainty for γ is the cluster (well) bootstrap, not the model
    SE — ``CoxTimeVaryingFitter`` in lifelines 0.30.x has no robust/cluster SE.
    """

    beta: float
    beta_se: float
    beta_p: float
    gamma: float
    gamma_se: float
    gamma_p: float
    n_subjects: int
    n_events: int
    n_intervals: int

    @property
    def hr_beta(self) -> float:
        return float(np.exp(self.beta))

    def hr_at(self, t: float) -> float:
        """Implied hazard ratio at time ``t`` for a unit change in the covariate."""
        return float(np.exp(self.beta + self.gamma * np.log(max(float(t), 1.0))))


def episode_split(
    subjects: pd.DataFrame,
    *,
    duration_col: str = "duration",
    event_col: str = "event",
    covariate_col: str = "covariate",
    id_col: str = "subject_id",
    cluster_col: str = "well_key",
) -> pd.DataFrame:
    """Long-format split at every distinct failure time (exact risk-set covariate).

    For a subject with duration ``T`` the cut points are all failure times
    ``< T`` plus ``T`` itself, so every episode's ``stop`` is a risk-set
    evaluation time and ``covariate·log(stop)`` is the value shared by everyone at
    risk at that time.  Vectorised (called ~300× by the well bootstrap).

    Returns columns ``[id_col, cluster_col, start, stop, event, cov, cov_logt]``.
    """
    df = subjects.reset_index(drop=True)
    ev = np.sort(df.loc[df[event_col] == 1, duration_col].unique())
    T = df[duration_col].to_numpy(dtype=float)
    n_cuts = np.searchsorted(ev, T, side="left")   # failure times strictly < T_i
    counts = n_cuts + 1                             # + the subject's own terminal episode

    stops_per_subj = [np.append(ev[:k], t) for k, t in zip(n_cuts, T)]
    stops = np.concatenate(stops_per_subj)
    starts = np.concatenate([np.concatenate(([0.0], s[:-1])) for s in stops_per_subj])

    subj_pos = np.repeat(np.arange(len(df)), counts)
    events = np.zeros(len(stops), dtype=int)
    last_rows = np.cumsum(counts) - 1
    events[last_rows] = df[event_col].to_numpy(dtype=int)

    cov = df[covariate_col].to_numpy(dtype=float)[subj_pos]
    has_cluster = cluster_col in df.columns
    out = {
        id_col: df[id_col].to_numpy()[subj_pos],
        "start": starts,
        "stop": stops,
        "event": events,
        "cov": cov,
        "cov_logt": cov * np.log(np.maximum(stops, 1.0)),
    }
    if has_cluster:
        out[cluster_col] = df[cluster_col].to_numpy()[subj_pos]
    return pd.DataFrame(out)


def fit_time_interaction_cox(
    subjects: pd.DataFrame,
    *,
    duration_col: str = "duration",
    event_col: str = "event",
    covariate_col: str = "covariate",
    id_col: str = "subject_id",
    penalizer: float = 0.0,
) -> TimeInteractionCoxResult:
    """Fit ``h(t)=h₀(t)·exp(β·X + γ·X·log t)`` by exact event-time episode split.

    ``subjects`` is one row per subject with ``duration_col``, ``event_col`` and
    ``covariate_col``.  An ``id_col`` is created if absent.  Returns β, γ with
    model SEs/p; use :func:`bootstrap_gamma_ci` for the citable γ interval.
    """
    from lifelines import CoxTimeVaryingFitter

    df = subjects.copy()
    if id_col not in df.columns:
        df[id_col] = np.arange(len(df))
    split = episode_split(
        df, duration_col=duration_col, event_col=event_col,
        covariate_col=covariate_col, id_col=id_col,
    )
    ctv = CoxTimeVaryingFitter(penalizer=penalizer)
    ctv.fit(
        split[[id_col, "start", "stop", "event", "cov", "cov_logt"]],
        id_col=id_col, event_col="event", start_col="start", stop_col="stop",
    )
    s = ctv.summary
    return TimeInteractionCoxResult(
        beta=float(s.loc["cov", "coef"]),
        beta_se=float(s.loc["cov", "se(coef)"]),
        beta_p=float(s.loc["cov", "p"]),
        gamma=float(s.loc["cov_logt", "coef"]),
        gamma_se=float(s.loc["cov_logt", "se(coef)"]),
        gamma_p=float(s.loc["cov_logt", "p"]),
        n_subjects=int(len(df)),
        n_events=int(df[event_col].sum()),
        n_intervals=int(len(split)),
    )


def bootstrap_gamma_ci(
    subjects: pd.DataFrame,
    *,
    duration_col: str = "duration",
    event_col: str = "event",
    covariate_col: str = "covariate",
    cluster_col: str = "well_key",
    n_boot: int = 300,
    seed: int = 42,
) -> dict:
    """Cluster (well) bootstrap percentile CI for γ — the citable uncertainty.

    Resamples clusters with replacement, re-episode-splits and refits each draw.
    Returns point γ plus median / 2.5% / 97.5% percentile bounds and the number
    of successful refits.
    """
    df = subjects.copy()
    if cluster_col not in df.columns:
        df[cluster_col] = np.arange(len(df))
    point = fit_time_interaction_cox(
        df, duration_col=duration_col, event_col=event_col, covariate_col=covariate_col,
    ).gamma

    clusters = df[cluster_col].unique()
    rng = np.random.default_rng(seed)
    gammas: list[float] = []
    for _ in range(n_boot):
        drawn = rng.choice(clusters, size=len(clusters), replace=True)
        parts = []
        for k, c in enumerate(drawn):
            g = df[df[cluster_col] == c].copy()
            g[cluster_col] = f"{c}_{k}"
            parts.append(g)
        boot = pd.concat(parts, ignore_index=True)
        boot["subject_id"] = np.arange(len(boot))
        try:
            gammas.append(
                fit_time_interaction_cox(
                    boot, duration_col=duration_col, event_col=event_col,
                    covariate_col=covariate_col,
                ).gamma
            )
        except Exception:
            continue

    arr = np.asarray(gammas, dtype=float)
    ok = len(arr) >= max(20, int(0.1 * n_boot))
    return {
        "gamma_point": round(float(point), 4),
        "gamma_boot_median": round(float(np.median(arr)), 4) if ok else None,
        "gamma_ci_lo": round(float(np.percentile(arr, 2.5)), 4) if ok else None,
        "gamma_ci_hi": round(float(np.percentile(arr, 97.5)), 4) if ok else None,
        "n_boot_ok": int(len(arr)),
    }


__all__ = [
    "TimeInteractionCoxResult",
    "episode_split",
    "fit_time_interaction_cox",
    "bootstrap_gamma_ci",
]
