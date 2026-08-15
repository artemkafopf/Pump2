"""Workstream C — refit the hazard overlay on the accepted production baseline.

The training table is emitted *from the production replay itself*
(:func:`failure_rate._hist_predicted_failures_by_field` with ``emit_rows``), so the
baseline expected-failure offset ``mu_baseline`` is byte-consistent with what the
accepted bundle ships (validated: Σμ over 2024-01..2026-06 == 707, the D/E predicted
total, matching per reporting field).  The refit is a Poisson GLM with

    E[event | run-month] = mu_baseline * exp(Σ_j beta_j (x_j - ref_j))

i.e. ``offset = log(mu_baseline)`` and a proportional-hazard multiplier
``theta = exp(Σ beta_j (x_j - ref_j))``.  This is deliberately the exact algebra of
``survival.HazardLayer.theta`` / ``apply_theta`` (fit link == serve mechanic, guardrail 1),
and Ql enters as ``ql_z`` (coef ~ beta) plus ``ql_z * log(age)`` (coef ~ gamma), the exact
decomposition ``survival.ql_hazard_theta`` serves (guardrail 2).

Honest-null framing: the layer is promoted beyond stress/sensitivity only if it improves
OOS monthly-count MAE or Poisson deviance without breaking Mc/Ya/Vt/global fact/model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk import crosswalk

# Static run-level covariates carried forward from the shipped hazard layer, with the
# reference values used for centering (so the fitted multiplier reproduces HazardLayer.theta).
STATIC_COVARIATES = [
    "log_glf_mean_opdays",
    "load_std_early",
    "load_mean",
    "frac_kpod_below_0p7",
    "kpod_freq_mean",
    "freq_above_55hz_pct_early",
    "n_freq_steps_per_100d",
    "curvature_deg10m",
    "log_run_seq",
    "log_days_since_prev_failure",
    "install_pre2020",
    "install_2023plus",
]

QL_CAP = C.QL_HAZARD_CAP_LOG_RATIO  # log(5)
FACT_FIRST = "2024-01"
FACT_LAST = "2026-06"
OOS_FIT_LAST = "2024-12"
OOS_SCORE_FIRST = "2025-01"
FOCUS_FIELDS = ["Мирнинский УН", "Ярактинский УН", "Верхнетирский УН"]
GLOBAL_LABEL = "ГЛОБАЛЬНО"


def ql_z_transform(ql_m3d: float, field_ref_log: float, cap: float = QL_CAP) -> float:
    """Fitting-side Ql covariate: ``clip(log1p(Ql) - field_ref_log, +/- cap)``.

    Identical to the deployed serve transform inside ``survival.ql_hazard_theta``
    (``z = clip(log_ql - ref, +/-cap)`` with ``log_ql = log1p(Ql_m3d)``), so a coefficient
    fitted on this covariate can be served without train/serve skew.
    """
    if ql_m3d is None or not np.isfinite(ql_m3d) or ql_m3d < 0:
        return float("nan")
    z = math.log1p(float(ql_m3d)) - float(field_ref_log)
    return float(np.clip(z, -cap, cap))


def load_reference_values(bundle_date: str = C.BUNDLE_DATE) -> dict[str, float]:
    """Centering references for the static covariates, from the shipped esp_cox_coeffs.csv."""
    path = C.hazard_coeffs_path(bundle_date)
    df = pd.read_csv(path, encoding="utf-8-sig")
    return {
        str(r["covariate"]): float(r["reference_value"])
        for _, r in df.iterrows()
        if str(r["covariate"]) in STATIC_COVARIATES and pd.notna(r["reference_value"])
    }


def attach_ql_covariate(train: pd.DataFrame, plan) -> pd.DataFrame:
    """Add ``ql_z`` and ``ql_z_logage`` from *monthly* liquid (plan.liquid_volume / op_days).

    Monthly Ql is genuinely time-varying here (an improvement over the deployed per-run-mean
    fit).  Months with no plan liquid or non-positive op-days get a reference-neutral 0.
    """
    out = train.copy()
    liq = getattr(plan, "liquid_volume", None)
    op = getattr(plan, "op_days", None)
    ref_by_field = C.QL_HAZARD_FIELD_REF_LOG
    zs: list[float] = []
    for _, row in out.iterrows():
        code, month, mf = row["code"], row["month"], row.get("model_field")
        ql = np.nan
        if liq is not None and op is not None and code in liq.index and month in liq.columns:
            try:
                lv = float(liq.at[code, month])
                od = float(op.at[code, month])
                if np.isfinite(lv) and np.isfinite(od) and lv > 0 and od > 0:
                    ql = lv / od
            except Exception:
                ql = np.nan
        ref = ref_by_field.get(str(mf), C.QL_HAZARD_GLOBAL_REF_LOG)
        zs.append(ql_z_transform(ql, ref) if np.isfinite(ql) else np.nan)
    out["ql_z"] = zs
    out["ql_available"] = out["ql_z"].notna()
    # reference-neutral fill (serve falls back to no Ql effect when Ql is missing)
    z_filled = out["ql_z"].fillna(0.0)
    out["ql_z_fit"] = z_filled
    out["ql_z_logage"] = z_filled * np.log(np.maximum(out["age_start"].astype(float), 1.0))
    return out


def attach_kpod_covariate(train: pd.DataFrame, plan, equipment_big_path=None) -> pd.DataFrame:
    """Add raw-Kpod U-shape covariates from monthly Ql / resolved Qnominal.

    This deliberately uses raw Kpod, not ``kpod_freq_mean``.  Existing wells use Big
    installed ``q_nom_m3d``; planned/new wells fall back to the field-typical median.
    """
    out = train.copy()
    liq = getattr(plan, "liquid_volume", None)
    op = getattr(plan, "op_days", None)
    vals: list[float] = []
    for _, row in out.iterrows():
        code, month, mf = row["code"], row["month"], row.get("model_field")
        kpod = np.nan
        qnom = crosswalk.resolve_qnominal(code, mf, equipment_big_path)
        if qnom is not None and np.isfinite(qnom) and qnom > 0 and liq is not None and op is not None:
            if code in liq.index and month in liq.columns:
                try:
                    lv = float(liq.at[code, month])
                    od = float(op.at[code, month])
                    if np.isfinite(lv) and np.isfinite(od) and lv > 0 and od > 0:
                        kpod = (lv / od) / float(qnom)
                except Exception:
                    kpod = np.nan
        vals.append(kpod)
    out["kpod_raw"] = vals
    out["kpod_available"] = out["kpod_raw"].notna()
    k = out["kpod_raw"].astype(float)
    under = (float(C.KPOD_HAZARD_K_LO) - k).clip(lower=0.0, upper=float(C.KPOD_HAZARD_CAP_UNDER))
    over = (k - float(C.KPOD_HAZARD_K_HI)).clip(lower=0.0, upper=float(C.KPOD_HAZARD_CAP_OVER))
    under = under.fillna(0.0)
    over = over.fillna(0.0)
    logage = np.log(np.maximum(out["age_start"].astype(float), 1.0))
    out["kpod_under_fit"] = under
    out["kpod_over_fit"] = over
    out["kpod_under_logage"] = under * logage
    out["kpod_over_logage"] = over * logage
    return out


def _design_matrix(train: pd.DataFrame, refs: dict[str, float], covariates: list[str],
                   use_ql: bool, use_kpod: bool = False) -> tuple[np.ndarray, list[str]]:
    cols: list[str] = []
    mats: list[np.ndarray] = []
    for cov in covariates:
        src = f"cov_{cov}"
        if src not in train.columns:
            continue
        ref = refs.get(cov, float(pd.to_numeric(train[src], errors="coerce").mean()))
        # missing covariate -> reference (centered 0 == reference-neutral, matches serve)
        x = pd.to_numeric(train[src], errors="coerce").fillna(ref).astype(float) - ref
        mats.append(x.to_numpy())
        cols.append(cov)
    if use_ql:
        mats.append(train["ql_z_fit"].astype(float).to_numpy()); cols.append("ql_z")
        mats.append(train["ql_z_logage"].astype(float).to_numpy()); cols.append("ql_z_logage")
    if use_kpod:
        for cov in ("kpod_under_fit", "kpod_over_fit", "kpod_under_logage", "kpod_over_logage"):
            mats.append(train[cov].astype(float).to_numpy()); cols.append(cov)
    X = np.column_stack(mats) if mats else np.zeros((len(train), 0))
    return X, cols


@dataclass
class PoissonFit:
    columns: list[str]
    beta: np.ndarray
    se: np.ndarray
    pvals: np.ndarray
    refs: dict[str, float]
    converged: bool
    llf: float
    n_obs: int
    n_events: int

    def theta(self, train: pd.DataFrame) -> np.ndarray:
        X, cols = _design_matrix_from_columns(train, self)
        eta = X @ self.beta
        return np.exp(np.clip(eta, -8.0, 8.0))


def _design_matrix_from_columns(train: pd.DataFrame, fit: "PoissonFit") -> tuple[np.ndarray, list[str]]:
    mats: list[np.ndarray] = []
    for cov in fit.columns:
        if cov == "ql_z":
            mats.append(train["ql_z_fit"].astype(float).to_numpy())
        elif cov == "ql_z_logage":
            mats.append(train["ql_z_logage"].astype(float).to_numpy())
        elif cov in ("kpod_under_fit", "kpod_over_fit", "kpod_under_logage", "kpod_over_logage"):
            mats.append(train[cov].astype(float).to_numpy())
        else:
            ref = fit.refs.get(cov, 0.0)
            src = f"cov_{cov}"
            x = pd.to_numeric(train.get(src), errors="coerce").fillna(ref).astype(float) - ref
            mats.append(x.to_numpy())
    return (np.column_stack(mats) if mats else np.zeros((len(train), 0))), list(fit.columns)


def fit_poisson_offset(train: pd.DataFrame, refs: dict[str, float], *,
                      covariates: list[str] = STATIC_COVARIATES, use_ql: bool = True,
                      use_kpod: bool = False) -> PoissonFit:
    """Poisson GLM: event ~ offset(log mu) + centered covariates.  Multiplier == HazardLayer.theta."""
    import statsmodels.api as sm

    fit_df = train[train["mu_baseline"] > 0].copy()
    X, cols = _design_matrix(fit_df, refs, covariates, use_ql, use_kpod)
    offset = np.log(fit_df["mu_baseline"].astype(float).to_numpy())
    y = fit_df["event"].astype(float).to_numpy()
    # NO intercept: the served multiplier theta = exp(sum beta_j (x_j - ref_j)) has no
    # constant term (HazardLayer.theta / apply_theta), so the fit must go through the
    # baseline offset with no free scale.  This is guardrail 1 (fit link == serve mechanic).
    model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
    res = model.fit(maxiter=200)
    return PoissonFit(
        columns=cols, beta=np.asarray(res.params), se=np.asarray(res.bse),
        pvals=np.asarray(res.pvalues), refs=refs,
        converged=bool(res.converged), llf=float(res.llf),
        n_obs=int(len(fit_df)), n_events=int(y.sum()),
    )


def predict_mu(train: pd.DataFrame, fit: PoissonFit | None) -> np.ndarray:
    """Predicted expected failures per run-month: mu_baseline * theta (theta=1 if no fit)."""
    mu = train["mu_baseline"].astype(float).to_numpy()
    if fit is None:
        return mu
    X, _ = _design_matrix_from_columns(train, fit)
    eta = X @ fit.beta
    return mu * np.exp(np.clip(eta, -8.0, 8.0))


# ── OOS metrics ──────────────────────────────────────────────────────────────

def _poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    mu = np.clip(mu, 1e-12, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0) - (y - mu)
    return float(2.0 * np.sum(term))


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y)
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * (len(y) - n_pos)))


def fact_model_by_field(scored: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    rows = []
    for field, g in scored.groupby("field"):
        obs = float(g["event"].sum()); pred = float(g[pred_col].sum())
        rows.append({"field": field, "observed": obs, "predicted": pred,
                     "fact_model_ratio": obs / pred if pred > 0 else np.nan})
    obs = float(scored["event"].sum()); pred = float(scored[pred_col].sum())
    rows.append({"field": GLOBAL_LABEL, "observed": obs, "predicted": pred,
                 "fact_model_ratio": obs / pred if pred > 0 else np.nan})
    return pd.DataFrame(rows)


def monthly_count_mae(scored: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    rows = []
    for field, g in scored.groupby("field"):
        m = g.groupby("month").agg(obs=("event", "sum"), pred=(pred_col, "sum"))
        rows.append({"field": field, "monthly_count_mae": float((m["obs"] - m["pred"]).abs().mean())})
    m = scored.groupby("month").agg(obs=("event", "sum"), pred=(pred_col, "sum"))
    rows.append({"field": GLOBAL_LABEL, "monthly_count_mae": float((m["obs"] - m["pred"]).abs().mean())})
    return pd.DataFrame(rows)


def calibration_by_quantile(scored: pd.DataFrame, pred_col: str, q: int = 10) -> pd.DataFrame:
    d = scored.copy()
    p = np.clip(d[pred_col].to_numpy(), 0, 1)
    d["_p"] = p
    try:
        d["bin"] = pd.qcut(pd.Series(p).rank(method="first"), q, labels=False)
    except ValueError:
        d["bin"] = 0
    return d.groupby("bin").agg(n=("event", "size"), pred=("_p", "mean"),
                                obs=("event", "mean")).reset_index()
