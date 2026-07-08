"""Reusable stratified-Cox screening cascade for the Phase C covariate blocks.

Generalises the Block-1 chemistry cascade (``workflows/chemistry_cox/phase_chem``)
into a block-agnostic estimator used by C1 (completion), C2 (operational) and C4
(merged θ).  The cascade is the one the whole Cox program agreed on:

    univariate screen  →  Spearman correlation prune  →  joint stratified Cox
    →  VIF prune → refit  →  Schoenfeld PH diagnostic

Design inherited from Phase A (not re-litigated):

* one joint stratified Cox, ``strata=['stratum_key']``, **global β**,
  ``cluster_col='well_key'``, ``robust=True``, duration ``tte`` (ttf_mix clock);
* **mandatory adjusters** (``install_period``, ``run_seq``,
  ``days_since_prev_failure``, ``has_telemetry``) are *forced* into every joint
  model — they are never screened out;
* time-varying coefficients are **not** fitted here via the banned X·log(t)
  product term.  Instead this module fits the STANDARD (proportional) joint model
  and runs a Schoenfeld test to *flag* PH violators; the corrected γ for any
  flagged covariate is estimated separately with
  :mod:`analysis.models.survival.time_interaction_cox` (event-time episode split).

Everything is complete-case on the requested numeric covariates; categorical
adjusters enter via a Patsy ``C(col)`` term and are excluded from VIF.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Cascade thresholds (Phase A / chemistry_cox defaults).
P_UNIVARIATE = 0.10
SPEARMAN_DROP_THR = 0.65
VIF_DROP_THR = 5.0
P_PH = 0.05
MIN_STRATUM_EVENTS = 5
MIN_TOTAL_EVENTS = 15

# The four mandatory cohort adjusters (Phase C §"Mandatory adjusters").
MANDATORY_NUMERIC = ["log_run_seq", "log_days_since_prev_failure", "has_telemetry"]
MANDATORY_CATEGORICAL = ["install_period"]


def _valid_strata_subset(sub: pd.DataFrame, event_col: str, strata: list[str],
                         min_events: int) -> pd.DataFrame:
    ev = sub.groupby(strata, observed=True)[event_col].transform("sum")
    return sub[ev >= min_events]


# ---------------------------------------------------------------------------
# Step 1 — univariate screen
# ---------------------------------------------------------------------------

def univariate_screen(
    df: pd.DataFrame,
    candidates: list[str],
    *,
    duration_col: str = "tte",
    event_col: str = "event",
    strata: list[str] | None = None,
    cluster_col: str = "well_key",
    p_thr: float = P_UNIVARIATE,
) -> pd.DataFrame:
    """One stratified cluster-robust Cox per candidate; flag ``p < p_thr``."""
    from lifelines import CoxPHFitter

    strata = strata or ["stratum_key"]
    rows = []
    for col in candidates:
        if col not in df.columns:
            continue
        sub = df[[cluster_col, duration_col, event_col, *strata, col]].dropna()
        sub = sub[sub[duration_col] > 0]
        sub = _valid_strata_subset(sub, event_col, strata, MIN_STRATUM_EVENTS)
        n_ev = int(sub[event_col].sum())
        if n_ev < MIN_TOTAL_EVENTS:
            rows.append({"covariate": col, "n_runs": len(sub), "n_events": n_ev,
                         "HR": None, "HR_lo": None, "HR_hi": None, "p": None,
                         "c_index": None, "pass_screen": False, "note": "too few events"})
            continue
        try:
            cph = CoxPHFitter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cph.fit(sub, duration_col=duration_col, event_col=event_col,
                        strata=strata, cluster_col=cluster_col, formula=col, robust=True)
            s = cph.summary
            p = float(s.loc[col, "p"])
            rows.append({
                "covariate": col, "n_runs": len(sub), "n_events": n_ev,
                "HR": round(float(np.exp(s.loc[col, "coef"])), 4),
                "HR_lo": round(float(np.exp(s.loc[col, "coef lower 95%"])), 4),
                "HR_hi": round(float(np.exp(s.loc[col, "coef upper 95%"])), 4),
                "p": round(p, 6), "c_index": round(float(cph.concordance_index_), 4),
                "pass_screen": p < p_thr, "note": "",
            })
        except Exception as exc:
            rows.append({"covariate": col, "n_runs": len(sub), "n_events": n_ev,
                         "HR": None, "HR_lo": None, "HR_hi": None, "p": None,
                         "c_index": None, "pass_screen": False, "note": str(exc)[:60]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 2 — Spearman correlation prune
# ---------------------------------------------------------------------------

def correlation_prune(
    df: pd.DataFrame,
    candidates: list[str],
    cindex: dict[str, float],
    thr: float = SPEARMAN_DROP_THR,
) -> tuple[list[str], list[dict]]:
    """Drop one from each |ρ|>thr pair, keeping the higher univariate C-index."""
    sub = df[candidates].apply(pd.to_numeric, errors="coerce").dropna()
    if sub.shape[1] < 2:
        return list(candidates), []
    corr = sub.corr(method="spearman").abs()
    drop: set[str] = set()
    dropped_pairs: list[dict] = []
    for i, a in enumerate(candidates):
        if a in drop:
            continue
        for b in candidates[i + 1:]:
            if b in drop:
                continue
            r = float(corr.loc[a, b]) if a in corr.index and b in corr.columns else 0.0
            if r > thr:
                keep = a if cindex.get(a, 0) >= cindex.get(b, 0) else b
                dropped = b if keep == a else a
                drop.add(dropped)
                dropped_pairs.append({"kept": keep, "dropped": dropped, "rho": round(r, 3)})
    return [c for c in candidates if c not in drop], dropped_pairs


# ---------------------------------------------------------------------------
# Step 3 — VIF prune (numeric terms only)
# ---------------------------------------------------------------------------

def vif_prune(
    df: pd.DataFrame,
    cols: list[str],
    thr: float = VIF_DROP_THR,
) -> tuple[list[str], pd.DataFrame]:
    """Iteratively drop the highest-VIF numeric term until all VIF ≤ thr."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    keep = list(cols)
    records = []
    while len(keep) >= 2:
        sub = df[keep].apply(pd.to_numeric, errors="coerce").dropna()
        X = (sub - sub.mean()).values
        vifs = {c: float(variance_inflation_factor(X, i)) for i, c in enumerate(keep)}
        records.append(dict(vifs))
        worst = max(vifs, key=vifs.get)
        if vifs[worst] <= thr:
            break
        keep.remove(worst)
    vif_df = pd.DataFrame({"term": list(vifs.keys()),
                           "VIF": [round(vifs[c], 2) for c in vifs]}) if keep else pd.DataFrame()
    return keep, vif_df


# ---------------------------------------------------------------------------
# Step 4 — joint stratified Cox
# ---------------------------------------------------------------------------

def _build_formula(numeric: list[str], categorical: list[str]) -> str:
    parts = list(numeric) + [f"C({c})" for c in categorical]
    return " + ".join(parts)


def fit_joint_cox(
    df: pd.DataFrame,
    numeric_terms: list[str],
    categorical_terms: list[str] | None = None,
    *,
    duration_col: str = "tte",
    event_col: str = "event",
    strata: list[str] | None = None,
    cluster_col: str = "well_key",
    min_stratum_events: int = MIN_STRATUM_EVENTS,
):
    """Fit one joint stratified cluster-robust Cox; return (cph, df_used, summary).

    Complete-case on ``numeric_terms`` and ``categorical_terms``.  Strata with
    fewer than ``min_stratum_events`` events are dropped.  On failure returns
    ``(None, df_used, None)`` rather than raising.
    """
    from lifelines import CoxPHFitter

    strata = strata or ["stratum_key"]
    categorical_terms = categorical_terms or []
    need = ([cluster_col, duration_col, event_col, *strata]
            + numeric_terms + categorical_terms)
    sub = df[list(dict.fromkeys(need))].copy()
    sub[duration_col] = pd.to_numeric(sub[duration_col], errors="coerce")
    sub = sub[sub[duration_col] > 0].dropna(subset=numeric_terms + categorical_terms
                                            + [duration_col, event_col])
    sub = _valid_strata_subset(sub, event_col, strata, min_stratum_events)
    if int(sub[event_col].sum()) < MIN_TOTAL_EVENTS:
        return None, sub, None

    # Drop zero-variance terms on this complete-case sample — a constant column
    # (e.g. has_telemetry == 1 in the telemetry-covered Block-2 subpopulation)
    # sends the Cox delta to NaN and halts convergence.
    numeric_terms = [c for c in numeric_terms if sub[c].nunique(dropna=True) >= 2]
    categorical_terms = [c for c in categorical_terms if sub[c].nunique(dropna=True) >= 2]
    if not numeric_terms and not categorical_terms:
        return None, sub, None

    formula = _build_formula(numeric_terms, categorical_terms)

    def _fit(pen: float):
        cph = CoxPHFitter(penalizer=pen)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cph.fit(sub, duration_col=duration_col, event_col=event_col,
                    strata=strata, cluster_col=cluster_col, formula=formula, robust=True)
        return cph

    try:
        cph = _fit(0.0)
    except Exception:
        try:
            cph = _fit(0.1)   # ridge fallback on near-singular / non-convergence
        except Exception:
            return None, sub, None

    s = cph.summary
    summary = pd.DataFrame({
        "term": s.index,
        "coef": s["coef"].to_numpy(), "se": s["se(coef)"].to_numpy(),
        "hr": np.exp(s["coef"].to_numpy()),
        "hr_ci_lo": np.exp(s["coef lower 95%"].to_numpy()),
        "hr_ci_hi": np.exp(s["coef upper 95%"].to_numpy()),
        "p": s["p"].to_numpy(),
    }).reset_index(drop=True)
    return cph, sub, summary


# ---------------------------------------------------------------------------
# Step 5 — Schoenfeld PH diagnostic on the fitted joint model
# ---------------------------------------------------------------------------

def ph_diagnostic(cph, df_used: pd.DataFrame) -> pd.DataFrame:
    """Per-term Schoenfeld PH test on a fitted joint model (log-time transform).

    Returns term | schoenfeld_stat | p_ph | ph_violation (p ≤ 0.05 → EXTENDED
    candidate).  Never raises — a failed test yields NaN and ``ph_violation=False``.
    """
    from lifelines.statistics import proportional_hazard_test

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = proportional_hazard_test(cph, df_used, time_transform="log")
        s = res.summary
        out = pd.DataFrame({
            "term": s.index,
            "schoenfeld_stat": s["test_statistic"].round(4).to_numpy(),
            "p_ph": s["p"].round(6).to_numpy(),
        })
        out["ph_violation"] = out["p_ph"] <= P_PH
        return out.reset_index(drop=True)
    except Exception as exc:
        return pd.DataFrame([{"term": "ALL", "schoenfeld_stat": None,
                              "p_ph": None, "ph_violation": False, "note": str(exc)[:60]}])


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

@dataclass
class ScreeningResult:
    univariate: pd.DataFrame
    correlation_kept: list[str]
    correlation_dropped: list[dict]
    vif: pd.DataFrame
    joint_summary: pd.DataFrame | None
    ph: pd.DataFrame | None
    cph: object
    df_used: pd.DataFrame | None
    final_numeric: list[str]
    final_categorical: list[str]
    n: int
    n_events: int
    notes: list[str] = field(default_factory=list)


def screen_and_fit(
    df: pd.DataFrame,
    candidates: list[str],
    *,
    forced_numeric: list[str] | None = None,
    forced_categorical: list[str] | None = None,
    duration_col: str = "tte",
    event_col: str = "event",
    strata: list[str] | None = None,
    cluster_col: str = "well_key",
    run_vif: bool = True,
) -> ScreeningResult:
    """Full cascade: univariate → correlation → joint(+VIF refit) → PH diagnostic.

    ``forced_numeric`` / ``forced_categorical`` are adjusters always kept in the
    joint model (defaults to the four mandatory cohort adjusters).  Correlation and
    VIF prune only the *screened* candidates, never the forced adjusters.
    """
    strata = strata or ["stratum_key"]
    forced_numeric = MANDATORY_NUMERIC if forced_numeric is None else forced_numeric
    forced_categorical = MANDATORY_CATEGORICAL if forced_categorical is None else forced_categorical
    notes: list[str] = []

    uni = univariate_screen(df, candidates, duration_col=duration_col,
                            event_col=event_col, strata=strata, cluster_col=cluster_col)
    survivors = uni.loc[uni["pass_screen"], "covariate"].tolist()
    cindex = uni.set_index("covariate")["c_index"].to_dict()
    notes.append(f"{len(survivors)}/{len(candidates)} candidates passed univariate (p<{P_UNIVARIATE})")

    kept, dropped_pairs = correlation_prune(df, survivors, cindex)
    if dropped_pairs:
        notes.append(f"correlation prune dropped {len(dropped_pairs)}: {dropped_pairs}")

    # Numeric terms in the joint model = surviving candidates + forced numeric adjusters.
    numeric = list(dict.fromkeys(kept + list(forced_numeric)))
    vif_df = pd.DataFrame()
    if run_vif and len(numeric) >= 2:
        vif_keep, vif_df = vif_prune(df, numeric)
        # never drop a forced adjuster on VIF grounds
        dropped_by_vif = [c for c in numeric if c not in vif_keep and c not in forced_numeric]
        numeric = [c for c in numeric if c in vif_keep or c in forced_numeric]
        if dropped_by_vif:
            notes.append(f"VIF prune dropped {dropped_by_vif}")

    cph, df_used, summary = fit_joint_cox(
        df, numeric, forced_categorical, duration_col=duration_col,
        event_col=event_col, strata=strata, cluster_col=cluster_col)

    ph = ph_diagnostic(cph, df_used) if cph is not None else None
    n = int(len(df_used)) if df_used is not None else 0
    n_events = int(df_used[event_col].sum()) if df_used is not None else 0

    return ScreeningResult(
        univariate=uni, correlation_kept=kept, correlation_dropped=dropped_pairs,
        vif=vif_df, joint_summary=summary, ph=ph, cph=cph, df_used=df_used,
        final_numeric=numeric, final_categorical=list(forced_categorical),
        n=n, n_events=n_events, notes=notes)


def population_ref_coeffs(
    summary: pd.DataFrame,
    df: pd.DataFrame,
    numeric_terms: list[str],
    *,
    block_label: str = "",
    clock: str = "ttf_mix",
    window: str = "early",
) -> pd.DataFrame:
    """θ-handoff rows for numeric terms: β + population-weighted ref (θ=1 average).

    Categorical terms are returned with ``ref_value=NaN`` (they carry their own
    contrast coding); the caller stamps the block/clock/window columns.
    """
    ref = {c: float(pd.to_numeric(df[c], errors="coerce").mean()) for c in numeric_terms}
    rows = []
    for _, r in summary.iterrows():
        term = r["term"]
        is_num = term in numeric_terms
        rows.append({
            "covariate": term,
            "beta": round(float(r["coef"]), 6),
            "gamma": 0.0,
            "ref_value": round(ref[term], 6) if is_num else np.nan,
            "type": "STANDARD",
            "block": block_label,
            "window": window,
            "clock": clock,
        })
    return pd.DataFrame(rows)


__all__ = [
    "univariate_screen", "correlation_prune", "vif_prune", "fit_joint_cox",
    "ph_diagnostic", "screen_and_fit", "population_ref_coeffs",
    "ScreeningResult", "MANDATORY_NUMERIC", "MANDATORY_CATEGORICAL",
    "P_UNIVARIATE", "SPEARMAN_DROP_THR", "VIF_DROP_THR", "P_PH",
]
