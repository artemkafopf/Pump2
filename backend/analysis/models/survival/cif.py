"""Aalen–Johansen cumulative incidence for competing risks (Phase B B2).

Absolute per-cause risk under competing risks is the **Aalen–Johansen** CIF, not
``1 − KM_cause`` (which overstates incidence — with ~40% of failures belonging to
"the other group" the overstatement is material).  This module implements AJ
directly so the CIF can be evaluated at arbitrary times, combined across causes
(they partition all-cause incidence: ``Σ_k CIF_k(t) = 1 − S_all(t)``), and
bootstrapped cheaply for Gray-style band comparisons.

Event coding convention: an integer ``event_code`` per subject where ``0`` =
censored and ``1, 2, …`` code the competing causes.  :func:`event_code_series`
builds it from a ``mode_group`` column and an all-cause event flag.

The manual estimator (KM all-cause survival × cause-specific increments) is
cross-checked against lifelines' ``AalenJohansenFitter`` in the unit tests.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class CIFCurve:
    """Step-function CIF for one cause: right-continuous, jumps at event times."""

    times: np.ndarray          # distinct all-cause event times (ascending)
    cif: np.ndarray            # CIF value just after each event time
    cause_events: int          # number of events of this cause
    all_events: int            # number of all-cause events in the risk set

    def at(self, t: float | np.ndarray) -> float | np.ndarray:
        """CIF evaluated at time(s) ``t`` (0 before the first event time)."""
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        idx = np.searchsorted(self.times, t_arr, side="right") - 1
        out = np.where(idx >= 0, self.cif[np.clip(idx, 0, len(self.cif) - 1)], 0.0)
        return float(out[0]) if np.isscalar(t) or np.ndim(t) == 0 else out

    @property
    def final(self) -> float:
        return float(self.cif[-1]) if len(self.cif) else 0.0


def event_code_series(
    mode_group: pd.Series,
    event: pd.Series,
    cause_order: tuple[str, ...],
) -> pd.Series:
    """Integer event code: 0 censored; 1.. per ``cause_order`` position.

    A run with ``event == 0`` is censored (code 0) regardless of ``mode_group``.
    """
    code_map = {g: i + 1 for i, g in enumerate(cause_order)}
    codes = mode_group.map(code_map).fillna(0).astype(int)
    return np.where(event.to_numpy() == 1, codes.to_numpy(), 0)


def aalen_johansen_cif(
    durations: np.ndarray,
    event_code: np.ndarray,
    cause_code: int,
) -> CIFCurve:
    """Aalen–Johansen CIF for ``cause_code``.

    CIF_k(t) = Σ_{t_i ≤ t} S(t_{i-1}) · d_{k,i} / n_i, where S is the KM all-cause
    survival, d_{k,i} the cause-k events at t_i, n_i the number at risk at t_i.
    """
    t = np.asarray(durations, dtype=float)
    c = np.asarray(event_code, dtype=int)
    order = np.argsort(t, kind="mergesort")
    t, c = t[order], c[order]
    n = len(t)

    times = np.unique(t[c > 0])           # distinct event times (any cause)
    if len(times) == 0:
        return CIFCurve(np.asarray([]), np.asarray([]), 0, 0)

    cif = np.empty(len(times), dtype=float)
    surv_prev = 1.0                        # S(t_{i-1})
    cum = 0.0
    for i, ti in enumerate(times):
        at_risk = int(np.sum(t >= ti))
        d_all = int(np.sum((t == ti) & (c > 0)))
        d_k = int(np.sum((t == ti) & (c == cause_code)))
        if at_risk > 0:
            cum += surv_prev * (d_k / at_risk)
            surv_prev *= (1.0 - d_all / at_risk)
        cif[i] = cum

    return CIFCurve(
        times=times,
        cif=cif,
        cause_events=int(np.sum(c == cause_code)),
        all_events=int(np.sum(c > 0)),
    )


def all_cause_km(durations: np.ndarray, event: np.ndarray) -> CIFCurve:
    """KM all-cause *failure* function (1 − S) as a CIFCurve for convenience."""
    code = np.where(np.asarray(event) == 1, 1, 0)
    return aalen_johansen_cif(durations, code, 1)


def cif_by_cause(
    durations: np.ndarray,
    event_code: np.ndarray,
    cause_order: tuple[str, ...],
) -> dict[str, CIFCurve]:
    """One :class:`CIFCurve` per cause (keyed by group name)."""
    return {
        g: aalen_johansen_cif(durations, event_code, i + 1)
        for i, g in enumerate(cause_order)
    }


def mode_mix_at(
    cifs: dict[str, CIFCurve],
    t: float,
) -> dict[str, float]:
    """Share of *incidence* by cause at time ``t`` (CIF_k(t) / Σ_j CIF_j(t)).

    Uses the CIF, not raw counts, so censoring is handled correctly.  Returns 0
    shares when no incidence has accrued by ``t``.
    """
    vals = {g: float(curve.at(t)) for g, curve in cifs.items()}
    total = sum(vals.values())
    if total <= 0:
        return {g: 0.0 for g in cifs}
    return {g: vals[g] / total for g in cifs}


def parametric_competing_cif(
    betas: list[float],
    etas: list[float],
    max_time: float,
    n_grid: int = 4000,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Competing-risks CIF from cause-specific Weibull hazards (B5).

    Returns ``(u, S_all, cifs)`` where each ``cif_k(t) = ∫₀ᵗ S_all·h_k`` is built
    from **analytic cumulative hazards** ``H_k(t)=(t/η_k)^{β_k}`` — finite at 0
    even when β<1 — via ``ΔCIF_k = S_all(u_{i-1})·ΔH_k``.  This avoids the density
    singularity that makes a naïve ``∫ S·h`` trapezoid (with h(0)=∞ for β<1)
    diverge.  By construction ``Σ_k CIF_k(t) = 1 − S_all(t)``.
    """
    u = np.linspace(0.0, float(max_time), int(n_grid))
    Hs = [np.power(np.clip(u, 0.0, None) / eta, beta) for beta, eta in zip(betas, etas)]
    H_all = np.sum(Hs, axis=0)
    s_all = np.exp(-np.clip(H_all, 0.0, 700.0))
    cifs = []
    for Hk in Hs:
        incr = s_all[:-1] * np.diff(Hk)
        cifs.append(np.concatenate([[0.0], np.cumsum(incr)]))
    return u, s_all, cifs


def time_to_incidence(u: np.ndarray, cif: np.ndarray, p: float) -> float:
    """First time the (monotone) CIF reaches probability ``p``; NaN if never."""
    if len(cif) == 0 or cif[-1] < p:
        return float("nan")
    return float(np.interp(p, cif, u))


def bootstrap_cif_difference(
    df: pd.DataFrame,
    *,
    group_col: str,
    group_a,
    group_b,
    duration_col: str,
    event_code_col: str,
    cause_code: int,
    at_times: tuple[float, ...],
    cluster_col: str = "well_key",
    n_boot: int = 400,
    seed: int = 42,
) -> pd.DataFrame:
    """Gray-style bootstrap comparison of one cause's CIF between two subgroups.

    Resamples clusters (wells) with replacement; at each requested time computes
    ``CIF_a − CIF_b`` and returns the point difference, percentile CI, and a
    two-sided bootstrap p-value (2·min(share above 0, share below 0)).
    """
    a = df[df[group_col] == group_a]
    b = df[df[group_col] == group_b]

    def _cif(sub: pd.DataFrame) -> CIFCurve:
        return aalen_johansen_cif(
            sub[duration_col].to_numpy(float),
            sub[event_code_col].to_numpy(int),
            cause_code,
        )

    point = {t: _cif(a).at(t) - _cif(b).at(t) for t in at_times}

    rng = np.random.default_rng(seed)
    diffs = {t: [] for t in at_times}
    # Resample wells within each arm independently.
    clusters_a = a[cluster_col].unique()
    clusters_b = b[cluster_col].unique()
    a_by = {c: a[a[cluster_col] == c] for c in clusters_a}
    b_by = {c: b[b[cluster_col] == c] for c in clusters_b}

    for _ in range(n_boot):
        da = pd.concat([a_by[c] for c in rng.choice(clusters_a, len(clusters_a), replace=True)],
                       ignore_index=True)
        db = pd.concat([b_by[c] for c in rng.choice(clusters_b, len(clusters_b), replace=True)],
                       ignore_index=True)
        ca, cb = _cif(da), _cif(db)
        for t in at_times:
            diffs[t].append(ca.at(t) - cb.at(t))

    rows = []
    for t in at_times:
        arr = np.asarray(diffs[t], dtype=float)
        frac_pos = float(np.mean(arr > 0))
        frac_neg = float(np.mean(arr < 0))
        p = float(min(1.0, 2.0 * min(frac_pos, frac_neg)))
        rows.append({
            "time": t,
            f"cif_{group_a}": round(float(_cif(a).at(t)), 4),
            f"cif_{group_b}": round(float(_cif(b).at(t)), 4),
            "diff_point": round(float(point[t]), 4),
            "diff_ci_lo": round(float(np.percentile(arr, 2.5)), 4),
            "diff_ci_hi": round(float(np.percentile(arr, 97.5)), 4),
            "boot_p_two_sided": round(p, 4),
            "n_boot": int(len(arr)),
        })
    return pd.DataFrame(rows)


__all__ = [
    "CIFCurve",
    "event_code_series",
    "aalen_johansen_cif",
    "all_cause_km",
    "cif_by_cause",
    "mode_mix_at",
    "parametric_competing_cif",
    "time_to_incidence",
    "bootstrap_cif_difference",
]
