"""Cause-specific stratified Cox helper for Phase B (B3/B4).

A cause-specific hazard model fits cause *k* treating all other-cause failures as
**censored** at their failure time — the estimand for *etiology* ("does this
covariate act on this component?").  Operationally that is just an ordinary Cox
model whose event column is the cause-*k* indicator (``event_<group>`` from the
competing-risks loader), with the Phase-A design inherited unchanged:

    strata = ['stratum_key']   global β across strata (shared coefficients,
                               stratum-specific baseline hazards)
    cluster_col = 'well_key'   robust cluster-robust SEs for repeated runs
    duration   = 'tte'         (ttf_mix operating-time clock)

Returns a tidy per-covariate summary (HR + 95% CI + robust p) so the B3/B4 scripts
can assemble forest plots without touching lifelines internals.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class CauseCoxResult:
    summary: pd.DataFrame       # one row per covariate (coef, HR, ci, p)
    n: int
    n_events: int
    n_strata: int
    success: bool
    message: str


def fit_cause_specific_cox(
    df: pd.DataFrame,
    *,
    covariates: list[str],
    event_col: str,
    duration_col: str = "tte",
    strata: list[str] | None = None,
    cluster_col: str = "well_key",
    penalizer: float = 0.0,
    min_events: int = 15,
) -> CauseCoxResult:
    """Fit one cause-specific stratified Cox.

    Complete-case on the requested ``covariates`` (rows with any NaN covariate are
    dropped).  Strata levels with zero cause-*k* events are dropped (they carry no
    likelihood but can make lifelines complain).  Returns ``success=False`` with a
    message rather than raising on a singular / non-converging fit, so a whole
    forest sweep does not abort on one degenerate cause-stratum.
    """
    from lifelines import CoxPHFitter

    strata = list(strata) if strata is not None else ["stratum_key"]
    keep_cols = list(dict.fromkeys(
        covariates + [event_col, duration_col, *strata, cluster_col]
    ))
    work = df[keep_cols].copy()
    work[duration_col] = pd.to_numeric(work[duration_col], errors="coerce")
    work = work[work[duration_col] > 0]
    work = work.dropna(subset=covariates + [duration_col, event_col])

    # Drop strata with no events of this cause (uninformative, can break the fit).
    ev_by_stratum = work.groupby(strata, observed=True)[event_col].transform("sum")
    work = work[ev_by_stratum > 0]

    n_events = int(work[event_col].sum())
    if n_events < min_events:
        return CauseCoxResult(
            summary=pd.DataFrame(),
            n=int(len(work)), n_events=n_events, n_strata=0,
            success=False,
            message=f"only {n_events} events (< min_events={min_events})",
        )

    cph = CoxPHFitter(penalizer=penalizer)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cph.fit(
                work, duration_col=duration_col, event_col=event_col,
                strata=strata, cluster_col=cluster_col, robust=True,
            )
    except Exception as exc:  # singular matrix, non-convergence, etc.
        return CauseCoxResult(
            summary=pd.DataFrame(),
            n=int(len(work)), n_events=n_events, n_strata=0,
            success=False, message=f"fit failed: {exc}",
        )

    s = cph.summary
    out = pd.DataFrame({
        "covariate": s.index,
        "coef": s["coef"].to_numpy(),
        "se": s["se(coef)"].to_numpy(),
        "hr": np.exp(s["coef"].to_numpy()),
        "hr_ci_lo": np.exp(s["coef lower 95%"].to_numpy()),
        "hr_ci_hi": np.exp(s["coef upper 95%"].to_numpy()),
        "p": s["p"].to_numpy(),
    }).reset_index(drop=True)

    return CauseCoxResult(
        summary=out,
        n=int(len(work)),
        n_events=n_events,
        n_strata=int(work.groupby(strata, observed=True).ngroups),
        success=True,
        message="ok",
    )


__all__ = ["CauseCoxResult", "fit_cause_specific_cox"]
