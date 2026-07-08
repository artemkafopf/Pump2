"""Temporal-holdout evaluation for Phase C C5 — the stack's first out-of-sample test.

Splits runs by install date (train = installs ≤ cutoff, test = installs > cutoff),
fits on train only, and scores on test with censoring-aware metrics:

* **C-index** at a horizon (lifelines ``concordance_index`` on the predicted survival
  S(h) as the score), with an optional well-cluster bootstrap CI;
* **IPCW Brier score** at 90 / 180 / 365 d, using a Kaplan–Meier estimate of the
  *censoring* distribution G(·) for inverse-probability-of-censoring weights;
* a **calibration** table (predicted-risk bins vs KM-observed failure fraction).

Two models are compared out of sample:
  A. **baseline** — stratum-only KM survival at the horizon (the K=2/Weibull-
     equivalent null: predictions differ only by stratum);
  B. **θ** — a stratified Cox on the merged covariates.

The deliverable is the *honest delta* B − A: how much the merged θ adds over strata
alone on unseen runs.  If that delta is small, that is a headline finding, not a
failure.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class Split:
    train: pd.DataFrame
    test: pd.DataFrame
    cutoff: str


def temporal_split(df: pd.DataFrame, cutoff: str, *, date_col: str = "install_dt") -> Split:
    """Split into train (install ≤ cutoff) / test (install > cutoff)."""
    d = df.copy()
    dt = pd.to_datetime(d[date_col], errors="coerce")
    cut = pd.Timestamp(cutoff)
    return Split(train=d[dt <= cut].copy(), test=d[dt > cut].copy(), cutoff=cutoff)


def _km_survival_at(times: np.ndarray, events: np.ndarray, horizon: float) -> float:
    """KM survival S(horizon) from (times, events); right-continuous step."""
    from lifelines import KaplanMeierFitter
    if len(times) == 0:
        return np.nan
    kmf = KaplanMeierFitter()
    kmf.fit(times, events)
    return float(kmf.predict(horizon))


def stratum_baseline_surv(train: pd.DataFrame, test: pd.DataFrame, horizon: float,
                          *, strata_col: str = "stratum_key",
                          dur: str = "tte", ev: str = "event") -> np.ndarray:
    """Per-test-subject S(horizon) from the train stratum KM (global KM fallback)."""
    global_s = _km_survival_at(train[dur].to_numpy(float), train[ev].to_numpy(int), horizon)
    by_stratum = {}
    for s, g in train.groupby(strata_col, observed=True):
        if int(g[ev].sum()) >= 5:
            by_stratum[s] = _km_survival_at(g[dur].to_numpy(float), g[ev].to_numpy(int), horizon)
    return test[strata_col].map(lambda s: by_stratum.get(s, global_s)).to_numpy(float)


def _censoring_km(train: pd.DataFrame, *, dur: str = "tte", ev: str = "event"):
    """KM of the *censoring* distribution G(t) fit on train (event = 1 − failure)."""
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()
    kmf.fit(train[dur].to_numpy(float), 1 - train[ev].to_numpy(int))
    return kmf


def ipcw_brier(test: pd.DataFrame, surv_at_h: np.ndarray, horizon: float, cens_km,
               *, dur: str = "tte", ev: str = "event", eps: float = 1e-3) -> tuple[float, int]:
    """IPCW Brier score at ``horizon``; returns (brier, n_contributing).

    Weight construction (Graf 1999):
      * failed by h (T≤h, δ=1):      (0 − S(h))² / G(T)
      * survived past h (T>h):       (1 − S(h))² / G(h)
      * censored before h:           weight 0 (no contribution)
    """
    t = test[dur].to_numpy(float)
    e = test[ev].to_numpy(int)
    s = np.asarray(surv_at_h, float)
    g_h = max(float(cens_km.predict(horizon)), eps)
    contrib, n = [], 0
    for ti, ei, si in zip(t, e, s):
        if not np.isfinite(si):
            continue
        if ti <= horizon and ei == 1:
            g_t = max(float(cens_km.predict(ti)), eps)
            contrib.append((0.0 - si) ** 2 / g_t); n += 1
        elif ti > horizon:
            contrib.append((1.0 - si) ** 2 / g_h); n += 1
        # censored before horizon → skip
    if not contrib:
        return float("nan"), 0
    return float(np.mean(contrib)), n


def cindex_at(test: pd.DataFrame, surv_at_h: np.ndarray,
              *, dur: str = "tte", ev: str = "event") -> float:
    """Concordance using S(h) as the score (higher survival ⇒ later event)."""
    from lifelines.utils import concordance_index
    mask = np.isfinite(surv_at_h)
    if mask.sum() < 10 or int(test[ev].to_numpy()[mask].sum()) < 5:
        return float("nan")
    try:
        return float(concordance_index(test[dur].to_numpy(float)[mask],
                                       surv_at_h[mask], test[ev].to_numpy(int)[mask]))
    except Exception:
        return float("nan")


def cindex_bootstrap_ci(test: pd.DataFrame, surv_at_h: np.ndarray, *, cluster="well_key",
                        n_boot: int = 200, seed: int = 7,
                        dur: str = "tte", ev: str = "event") -> tuple[float, float]:
    """Well-cluster bootstrap percentile CI for the horizon C-index."""
    rng = np.random.default_rng(seed)
    t = test.reset_index(drop=True)
    s = np.asarray(surv_at_h, float)
    clusters = t[cluster].unique()
    vals = []
    for _ in range(n_boot):
        drawn = rng.choice(clusters, size=len(clusters), replace=True)
        idx = np.concatenate([t.index[t[cluster] == c].to_numpy() for c in drawn])
        ci = cindex_at(t.iloc[idx], s[idx], dur=dur, ev=ev)
        if np.isfinite(ci):
            vals.append(ci)
    if len(vals) < 20:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def n_at_risk(test: pd.DataFrame, horizon: float, *, dur: str = "tte", ev: str = "event") -> int:
    """Test runs that are informative at ``horizon`` (failed by h, or followed past h)."""
    t = test[dur].to_numpy(float); e = test[ev].to_numpy(int)
    return int(np.sum((t > horizon) | ((t <= horizon) & (e == 1))))


def calibration_table(test: pd.DataFrame, surv_at_h: np.ndarray, horizon: float,
                      *, q: int = 5, dur: str = "tte", ev: str = "event") -> pd.DataFrame:
    """Predicted-risk-bin vs KM-observed failure fraction at ``horizon``."""
    risk = 1.0 - np.asarray(surv_at_h, float)
    d = test.copy()
    d["_risk"] = risk
    d = d[np.isfinite(d["_risk"])]
    try:
        d["_bin"] = pd.qcut(d["_risk"], q=q, duplicates="drop")
    except Exception:
        return pd.DataFrame()
    rows = []
    for b, g in d.groupby("_bin", observed=True):
        obs = 1.0 - _km_survival_at(g[dur].to_numpy(float), g[ev].to_numpy(int), horizon)
        rows.append({"risk_bin": str(b), "n": len(g),
                     "pred_risk_mean": round(float(g["_risk"].mean()), 4),
                     "obs_failure_km": round(float(obs), 4) if np.isfinite(obs) else None})
    return pd.DataFrame(rows)


__all__ = [
    "Split", "temporal_split", "stratum_baseline_surv", "ipcw_brier", "cindex_at",
    "cindex_bootstrap_ci", "n_at_risk", "calibration_table", "_censoring_km",
]
