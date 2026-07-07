"""Phase D — D4 alarm / operational metrics (the phase verdict).

The deliverable Phase D is judged on is not a hazard ratio but *alarm performance on
unseen installs*.  Given a per-landmark risk score and the guard-gapped binary target
(``y`` = 1 failed-in-window, 0 survived-window, NaN unobservable), these functions
compute what ops actually cares about:

* :func:`precision_recall_at_capacity` — flag the top-k% highest-risk landmarks and
  report precision / recall at that flag rate (top-5% and top-10% per §0.1);
* :func:`lead_time_distribution` — operating days from a run's *first* alarm to its
  failure (how much warning the alarm buys);
* :func:`false_alarms_per_pump_year` — false alarms normalised by observed pump-years;
* :func:`dynamic_auc` — landmark-level AUC of the score against ``y``;
* :func:`cluster_bootstrap_delta` — well-cluster bootstrap CI on any scalar metric's
  delta between the dynamic model and the baseline.

All operate on a tidy per-landmark frame; they never touch lifelines or the DB, so
they are unit-testable and reused across horizons / guards / splits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _flag_topk(score: np.ndarray, k_frac: float) -> np.ndarray:
    """Boolean flag for the top ``k_frac`` fraction of finite scores (ties→threshold)."""
    s = np.asarray(score, float)
    finite = np.isfinite(s)
    if finite.sum() == 0:
        return np.zeros(len(s), bool)
    n_flag = max(1, int(np.ceil(finite.sum() * k_frac)))
    thr = np.sort(s[finite])[::-1][min(n_flag, finite.sum()) - 1]
    return finite & (s >= thr)


def precision_recall_at_capacity(df: pd.DataFrame, score_col: str, *,
                                 y_col: str = "y", k_fracs=(0.05, 0.10)) -> pd.DataFrame:
    """Precision / recall / flag counts at each top-k capacity.

    Only landmarks with an observable target (``y`` ∈ {0,1}) and a finite score
    contribute — censored-in-window landmarks are excluded from precision/recall.
    """
    d = df[np.isfinite(df[score_col]) & df[y_col].isin([0, 1])].copy()
    y = d[y_col].to_numpy(int)
    s = d[score_col].to_numpy(float)
    n_pos = int(y.sum())
    rows = []
    for k in k_fracs:
        flag = _flag_topk(s, k)
        tp = int(((flag) & (y == 1)).sum())
        fp = int(((flag) & (y == 0)).sum())
        n_flag = int(flag.sum())
        rows.append({
            "k_frac": k, "n_scored": len(d), "n_flagged": n_flag, "n_pos": n_pos,
            "tp": tp, "fp": fp,
            "precision": round(tp / n_flag, 4) if n_flag else np.nan,
            "recall": round(tp / n_pos, 4) if n_pos else np.nan,
        })
    return pd.DataFrame(rows)


def lead_time_distribution(df: pd.DataFrame, score_col: str, k_frac: float, *,
                           y_col: str = "y", id_col: str = "row_id",
                           age_col: str = "op_age", terminal_col: str = "terminal_op_age"
                           ) -> dict:
    """Lead time (operating days) from a run's first alarm to its failure.

    For each failing run that is ever flagged (score in the top-``k_frac`` at any of
    its landmarks), lead time = terminal operating age − operating age at the run's
    first flagged landmark.  Returns median / IQR / n and the raw array.
    """
    d = df.copy()
    d["_flag"] = _flag_topk(d[score_col].to_numpy(float), k_frac)
    leads = []
    for rid, g in d.groupby(id_col):
        # failing runs only: a landmark with y==1 means this run fails in-window
        if not (g[y_col] == 1).any():
            continue
        flagged = g[g["_flag"]]
        if flagged.empty:
            continue
        first_alarm_age = float(flagged[age_col].min())
        terminal = float(g[terminal_col].iloc[0]) if terminal_col in g else float(
            g[age_col].max() + g["tte_land"].iloc[-1])
        leads.append(terminal - first_alarm_age)
    leads = np.array(leads, float)
    if len(leads) == 0:
        return {"n": 0, "median": np.nan, "q1": np.nan, "q3": np.nan, "values": leads}
    return {"n": len(leads), "median": float(np.median(leads)),
            "q1": float(np.percentile(leads, 25)), "q3": float(np.percentile(leads, 75)),
            "values": leads}


def false_alarms_per_pump_year(df: pd.DataFrame, score_col: str, k_frac: float, *,
                               y_col: str = "y", id_col: str = "row_id",
                               cadence: int = 30) -> float:
    """False alarms per observed pump-year at the top-``k_frac`` flag rate.

    A false alarm is a flagged landmark whose target is ``y == 0`` (survived window).
    Pump-years = (number of scored landmarks × cadence operating-days) / 365.
    """
    d = df[np.isfinite(df[score_col]) & df[y_col].isin([0, 1])].copy()
    if d.empty:
        return float("nan")
    flag = _flag_topk(d[score_col].to_numpy(float), k_frac)
    fp = int((flag & (d[y_col].to_numpy(int) == 0)).sum())
    pump_years = len(d) * cadence / 365.0
    return round(fp / pump_years, 3) if pump_years > 0 else float("nan")


def dynamic_auc(df: pd.DataFrame, score_col: str, *, y_col: str = "y") -> float:
    """Landmark-level AUC of ``score`` vs the binary window target (Mann–Whitney)."""
    d = df[np.isfinite(df[score_col]) & df[y_col].isin([0, 1])]
    y = d[y_col].to_numpy(int); s = d[score_col].to_numpy(float)
    n1 = int(y.sum()); n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    auc = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return round(float(auc), 4)


def cluster_bootstrap_delta(df: pd.DataFrame, metric_fn, *, cluster: str = "well_key",
                            n_boot: int = 300, seed: int = 7) -> tuple[float, float, float]:
    """Well-cluster bootstrap of a scalar ``metric_fn(frame) -> float``.

    Returns (point, ci_lo, ci_hi).  ``metric_fn`` should compute a *delta* (dynamic −
    baseline) so the CI answers "does the dynamic model beat the null out of sample".
    """
    rng = np.random.default_rng(seed)
    point = metric_fn(df)
    clusters = df[cluster].dropna().unique()
    if len(clusters) < 5:
        return point, float("nan"), float("nan")
    idx_by_cluster = {c: df.index[df[cluster] == c].to_numpy() for c in clusters}
    vals = []
    for _ in range(n_boot):
        drawn = rng.choice(clusters, size=len(clusters), replace=True)
        idx = np.concatenate([idx_by_cluster[c] for c in drawn])
        v = metric_fn(df.loc[idx])
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < 20:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


__all__ = [
    "precision_recall_at_capacity", "lead_time_distribution",
    "false_alarms_per_pump_year", "dynamic_auc", "cluster_bootstrap_delta",
]
