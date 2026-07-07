"""Phase D — landmark survival models (D2 inference + D3 prediction).

Three pieces the Phase D scripts share, all on the **operating-day** clock:

* :func:`install_cohort_split` — the C5 out-of-sample convention (train = installs ≤
  cutoff, test = installs after), applied to the landmark frame;
* :func:`build_episode_frame` + :func:`fit_time_varying_cox` — the **D2** inference
  view: episode-split the landmark frame (start = landmark, stop = next
  landmark/event/censor) and fit a stratified time-varying Cox
  (``CoxTimeVaryingFitter``), whose coefficients say which dynamics carry hazard;
* :func:`fit_landmark_cox` + :func:`predict_window_risk` + :func:`baseline_window_risk`
  — the **D3** prediction view: a landmark "super-model" Cox on the stacked
  landmark rows (duration = time-from-landmark), and the **age + stratum** null it
  must beat (conditional KM risk over the guard-gapped window).

Leakage discipline: the D3 split is fixed *before* screening; screening runs on the
train frame only (the caller passes ``split.train`` into :func:`fit_landmark_cox`).

Clustering note (D2): ``CoxTimeVaryingFitter`` derives its robust variance from
``id_col``; we use ``row_id`` (run-level) rather than ``well_key`` because two runs
of one well can occupy overlapping operating-age intervals, which the counting-process
model forbids.  Run-level robust SEs are marginally anti-conservative vs well-level;
the D2 view is diagnostic (the verdict lives in D4), so this is acceptable and stated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis.models.survival.temporal_holdout import _km_survival_at

C5_CUTOFF = "2023-12-31"


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CohortSplit:
    train: pd.DataFrame
    test: pd.DataFrame
    cutoff: str
    key: str


def install_cohort_split(frame: pd.DataFrame, cutoff: str = C5_CUTOFF,
                         *, date_col: str = "install_dt") -> CohortSplit:
    """Split landmark frame by **install** date (train ≤ cutoff, test after)."""
    dt = pd.to_datetime(frame[date_col], errors="coerce")
    cut = pd.Timestamp(cutoff)
    return CohortSplit(train=frame[dt <= cut].copy(), test=frame[dt > cut].copy(),
                       cutoff=cutoff, key="install_cohort")


def landmark_calendar_split(frame: pd.DataFrame, cutoff: str = "2025-06-30") -> CohortSplit:
    """Secondary robustness split by the landmark's **calendar date**.

    The landmark calendar date = install_dt + op_age is approximate (operating age
    ≠ calendar age), so this is a robustness read, not the primary split.
    """
    land_dt = pd.to_datetime(frame["install_dt"], errors="coerce") + pd.to_timedelta(
        frame["op_age"], unit="D")
    cut = pd.Timestamp(cutoff)
    return CohortSplit(train=frame[land_dt <= cut].copy(), test=frame[land_dt > cut].copy(),
                       cutoff=cutoff, key="landmark_calendar")


# ---------------------------------------------------------------------------
# D2 — episode-split time-varying Cox (inference view)
# ---------------------------------------------------------------------------

def build_episode_frame(frame: pd.DataFrame, features: list[str],
                        *, id_col: str = "row_id", strata: str = "stratum_key",
                        event_source: str = "event_land") -> pd.DataFrame:
    """Counting-process episode frame from the landmark frame.

    One interval per landmark: ``[op_age, next op_age)``; the final interval of a run
    ends at its terminal operating age (``op_age + tte_land``) and carries the event.
    Covariates are the landmark features (piecewise-constant over the interval).

    ``event_source`` selects which terminal-event indicator ends the run: ``event_land``
    (all-cause) or a cause-specific indicator (``event_hydraulic`` …) — the latter
    realises the cause-specific time-varying hazard (other-cause failures censored).
    """
    cols = [id_col, strata, "well_key", "op_age", "tte_land", event_source, *features]
    d = frame[cols].sort_values([id_col, "op_age"]).reset_index(drop=True)
    rows = []
    for rid, g in d.groupby(id_col, sort=False):
        g = g.reset_index(drop=True)
        terminal = float(g["op_age"].iloc[-1] + g["tte_land"].iloc[-1])
        for i in range(len(g)):
            start = float(g["op_age"].iloc[i])
            stop = float(g["op_age"].iloc[i + 1]) if i + 1 < len(g) else terminal
            if stop <= start:
                continue
            is_last = i == len(g) - 1
            rec = {id_col: rid, strata: g[strata].iloc[i], "well_key": g["well_key"].iloc[i],
                   "start": start, "stop": stop,
                   "event": int(g[event_source].iloc[i]) if is_last else 0}
            for f in features:
                rec[f] = g[f].iloc[i]
            rows.append(rec)
    return pd.DataFrame(rows)


@dataclass
class TVCoxResult:
    summary: pd.DataFrame       # covariate | coef | hr | ci | p
    n_intervals: int
    n_events: int
    success: bool
    message: str


def fit_time_varying_cox(episodes: pd.DataFrame, features: list[str],
                         *, id_col: str = "row_id", strata: str = "stratum_key",
                         event_col: str = "event", penalizer: float = 0.0) -> TVCoxResult:
    """Fit a stratified time-varying Cox on the episode frame (complete-case)."""
    from lifelines import CoxTimeVaryingFitter

    keep = [id_col, strata, "start", "stop", event_col, *features]
    d = episodes[keep].dropna(subset=features).copy()
    # drop zero-variance features on this complete-case sample
    features = [f for f in features if d[f].nunique(dropna=True) >= 2]
    if not features:
        return TVCoxResult(pd.DataFrame(), len(d), int(d[event_col].sum()), False,
                           "no non-constant features after complete-case")
    n_events = int(d[event_col].sum())
    if n_events < 15:
        return TVCoxResult(pd.DataFrame(), len(d), n_events, False,
                           f"only {n_events} events")

    # NB lifelines' CoxTimeVaryingFitter does not implement robust/cluster SEs
    # ("Not available yet."), so SEs here are **model-based**, not cluster-robust —
    # anti-conservative given repeated intervals per run.  D2 is the diagnostic view
    # (the out-of-sample verdict is D4), so model-based p-values are read as
    # screening signals, not confirmatory inference.
    def _fit(pen):
        m = CoxTimeVaryingFitter(penalizer=pen)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(d[[id_col, strata, "start", "stop", event_col, *features]],
                  id_col=id_col, event_col=event_col, start_col="start",
                  stop_col="stop", strata=[strata], show_progress=False)
        return m
    try:
        m = _fit(penalizer)
        used = penalizer
    except Exception:
        try:
            m = _fit(max(penalizer, 0.1)); used = max(penalizer, 0.1)
        except Exception as exc:
            return TVCoxResult(pd.DataFrame(), len(d), n_events, False, f"fit failed: {exc}")

    s = m.summary
    out = pd.DataFrame({
        "covariate": s.index, "coef": s["coef"].to_numpy(), "se": s["se(coef)"].to_numpy(),
        "hr": np.exp(s["coef"].to_numpy()), "hr_ci_lo": np.exp(s["coef lower 95%"].to_numpy()),
        "hr_ci_hi": np.exp(s["coef upper 95%"].to_numpy()), "p": s["p"].to_numpy(),
    }).reset_index(drop=True)
    return TVCoxResult(out, len(d), n_events, True,
                       "ok" if used == penalizer else f"ridge {used}")


# ---------------------------------------------------------------------------
# D3 — landmark super-model Cox (prediction view)
# ---------------------------------------------------------------------------

@dataclass
class LandmarkCoxFit:
    cph: object
    features: list[str]
    strata: str
    success: bool
    message: str
    strata_levels: frozenset = frozenset()


def fit_landmark_cox(train: pd.DataFrame, features: list[str],
                     *, strata: str = "stratum_key", cluster_col: str = "well_key",
                     duration_col: str = "tte_land", event_col: str = "event_land",
                     penalizer: float = 0.01) -> LandmarkCoxFit:
    """Fit a stratified cluster-robust Cox on stacked landmark rows.

    Duration = operating days from landmark to the run's terminal age; features are
    the landmark trailing-window features.  A small ridge stabilises the many
    correlated features.  Complete-case on ``features``.
    """
    from lifelines import CoxPHFitter
    import warnings

    keep = [cluster_col, strata, duration_col, event_col, *features]
    d = train[list(dict.fromkeys(keep))].copy()
    d[duration_col] = pd.to_numeric(d[duration_col], errors="coerce")
    d = d[d[duration_col] > 0].dropna(subset=features + [duration_col, event_col])
    features = [f for f in features if d[f].nunique(dropna=True) >= 2]
    if not features or int(d[event_col].sum()) < 15:
        return LandmarkCoxFit(None, features, strata, False, "insufficient events/features")
    # keep only strata with events (avoids degenerate baselines)
    ev = d.groupby(strata, observed=True)[event_col].transform("sum")
    d = d[ev >= 5]
    formula = " + ".join(features)
    try:
        cph = CoxPHFitter(penalizer=penalizer)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cph.fit(d, duration_col=duration_col, event_col=event_col, strata=[strata],
                    cluster_col=cluster_col, formula=formula, robust=True)
    except Exception as exc:
        return LandmarkCoxFit(None, features, strata, False, f"fit failed: {exc}")
    return LandmarkCoxFit(cph, features, strata, True, "ok",
                          strata_levels=frozenset(d[strata].unique()))


def predict_window_risk(fit: LandmarkCoxFit, test: pd.DataFrame, u: float,
                        *, strata: str = "stratum_key") -> np.ndarray:
    """Predicted risk of failing within ``u`` operating days of the landmark.

    Returns ``1 − S(u | features, stratum)`` per test row.  Test rows whose stratum
    or features are unavailable to the fitted model yield ``NaN`` (the caller falls
    back to the baseline for those).
    """
    if fit is None or not fit.success:
        return np.full(len(test), np.nan)
    d = test.copy().reset_index(drop=True)
    risk = np.full(len(d), np.nan)
    # score only rows with complete features AND a stratum the model saw in training
    # (lifelines raises on an unseen stratum for the whole batch; those rows stay NaN
    # and the caller falls back to the age+stratum baseline for them).
    ok = (d[fit.features].notna().all(axis=1)
          & d[strata].isin(fit.strata_levels)).to_numpy()
    if ok.sum() == 0:
        return risk
    sub = d.loc[ok]
    try:
        sf = fit.cph.predict_survival_function(sub, times=[u])
        risk[np.flatnonzero(ok)] = 1.0 - sf.iloc[0].to_numpy()   # one column per row
    except Exception:
        return np.full(len(test), np.nan)
    return risk


# ---------------------------------------------------------------------------
# Age + stratum baseline (the null to beat)
# ---------------------------------------------------------------------------

def _run_level(frame: pd.DataFrame, *, strata: str = "stratum_key") -> pd.DataFrame:
    """Collapse the landmark frame to one row per run (terminal op-age + event)."""
    g = frame.sort_values("op_age").groupby("row_id", observed=True)
    out = g.agg(op_age_last=("op_age", "last"), tte_land_last=("tte_land", "last"),
                event=("event_land", "last"),
                **{strata: (strata, "last"), "well_key": ("well_key", "last")})
    out["terminal_op_age"] = out["op_age_last"] + out["tte_land_last"]
    return out.reset_index()


def baseline_window_risk(train_frame: pd.DataFrame, test_frame: pd.DataFrame,
                         g: float, H: float, *, strata: str = "stratum_key",
                         min_events: int = 5) -> np.ndarray:
    """Age + stratum conditional window risk for every test landmark row.

    For a landmark at operating age ``L``: risk = ``1 − S_str(L+g+H) / S_str(L)`` where
    ``S_str`` is the train KM of the run-level operating-time survival in the row's
    stratum (global KM fallback for thin strata).  This is the C5 "age + stratum
    only" null — predictions differ only by stratum and current age.
    """
    runs = _run_level(train_frame, strata=strata)
    t_all = runs["terminal_op_age"].to_numpy(float)
    e_all = runs["event"].to_numpy(int)
    global_km = lambda age: _km_survival_at(t_all, e_all, age)  # noqa: E731

    strata_km: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for s, gg in runs.groupby(strata, observed=True):
        if int(gg["event"].sum()) >= min_events:
            strata_km[s] = (gg["terminal_op_age"].to_numpy(float), gg["event"].to_numpy(int))

    def s_at(stratum, age):
        if stratum in strata_km:
            t, e = strata_km[stratum]
            return _km_survival_at(t, e, age)
        return global_km(age)

    # cache S at the age-grid we need (L and L+g+H per row)
    risk = np.full(len(test_frame), np.nan)
    for i, (_, r) in enumerate(test_frame.reset_index(drop=True).iterrows()):
        L = float(r["op_age"]); s = r[strata]
        s_L = s_at(s, L); s_end = s_at(s, L + g + H)
        if np.isfinite(s_L) and s_L > 0 and np.isfinite(s_end):
            risk[i] = float(np.clip(1.0 - s_end / s_L, 0.0, 1.0))
    return risk


__all__ = [
    "C5_CUTOFF", "CohortSplit", "install_cohort_split", "landmark_calendar_split",
    "build_episode_frame", "fit_time_varying_cox", "TVCoxResult",
    "fit_landmark_cox", "predict_window_risk", "baseline_window_risk", "LandmarkCoxFit",
]
