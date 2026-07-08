"""Operational + completion + cohort hazard layer for the VBA workbook.

Applies the model report §2/§2.7 hazards as a **physical regime-sensitivity overlay**
on the v2 stratum + K=2 baseline — GLF, load, kpod, frequency, curvature (value),
well history and vintage — grouped so each factor's marginal delta on RUL/TTF can be
shown separately.

**USER OVERRIDE / NOT OOS-VALIDATED.** The subset failed the out-of-sample gate (it
forecasts worse than the stratum baseline OOS — see the holdout table below and model
report §3.8/§8.5). It is shipped `enabled=TRUE` at the user's explicit request as a
what-if / sensitivity tool, NOT a validated forecaster. The cohort groups
(well_history, vintage) carry the largest in-sample coefficients AND generalize the
least — they are the primary transport-failure drivers. The baseline columns are never
altered; every adjusted output is a separate `_sens` column.

Clock = ttf_mix. Operational covariates use the first-30-operating-day early window;
curvature/history/vintage are install-knowable (t0/cohort).

Curvature rule (user): use the value (degrees / 10 m); missing → 0.0 (a straight,
low-curvature well, assumed < 0.3). `curvature_deg10m = curvature.fillna(0.0)`.

Outputs (under ``results_dir("esp_survival_vba_models")``):
  esp_cox_coeffs.csv       covariate | ship_name | hazard_group | beta | hr | ci_lo |
                           ci_hi | p | reference_value | expected_dir | dir_ok |
                           window | mode | block | enabled | ship_reason | clock
  esp_run_covariates.csv   per-run raw covariates keyed by well_key_norm + run_seq
  hazard_manifest.txt      ship_mode, holdout deltas, git commit, clock, date
And a card at ``results_dir("vba_model_v2_hazards")/README-data.md``.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from analysis.data.competing_risks_loader import build_competing_risks_df
from analysis.data.failure_modes import MODE_GROUPS
from analysis.models.survival.cause_specific_cox import fit_cause_specific_cox
from analysis.models.survival.temporal_holdout import (
    temporal_split, stratum_baseline_surv, ipcw_brier, cindex_at,
    cindex_bootstrap_ci, n_at_risk, _censoring_km,
)

CLOCK = "ttf_mix"

# (mart column, ship_name, block, window, mode, hazard_group, expected_dir)
# expected_dir: "+" = larger value shortens life (hazard up); "-" = lengthens;
#               "?" = ambiguous / confounding-by-indication (flag, don't assert).
SUBSET: tuple[tuple[str, str, str, str, str, str, str], ...] = (
    ("log_glf_mean_opdays",       "GLF",                "operational", "early",  "hydraulic",       "GLF",          "-"),
    ("load_std_early",            "load_variability",   "operational", "early",  "hydraulic",       "load",         "+"),
    ("load_mean",                 "load_level",         "operational", "early",  "electro-thermal", "load",         "+"),
    ("frac_kpod_below_0p7",       "chronic_underload",  "operational", "early",  "electro-thermal", "kpod",         "+"),
    ("kpod_freq_mean",            "delivery_coef",      "operational", "early",  "electro-thermal", "kpod",         "?"),
    ("freq_above_55hz_pct_early", "freq_above_55hz",    "operational", "early",  "electro-thermal", "frequency",    "+"),
    ("n_freq_steps_per_100d",     "freq_instability",   "operational", "early",  "electro-thermal", "frequency",    "+"),
    ("curvature_deg10m",          "curvature",          "completion",  "t0",     "all-cause",       "curvature",    "+"),
    ("log_run_seq",               "run_seq",            "history",     "cohort", "all-cause",       "well_history", "+"),
    ("log_days_since_prev_failure", "days_since_fail",  "history",     "cohort", "all-cause",       "well_history", "+"),
    ("install_pre2020",           "vintage_pre2020",    "vintage",     "cohort", "all-cause",       "vintage",      "+"),
    ("install_2023plus",          "vintage_2023plus",   "vintage",     "cohort", "all-cause",       "vintage",      "+"),
)
COVARS = [c[0] for c in SUBSET]
EARLY_COVARS = [c[0] for c in SUBSET if c[3] == "early"]
GROUPS = list(dict.fromkeys(c[5] for c in SUBSET))

# First runs have no prior failure -> impute log_days_since_prev_failure to its own
# mean so the joint fit keeps them and that term is reference-neutral for them (NOT a
# physical pause claim). At scoring, VBA falls back to the reference the same way.
NEUTRAL_IMPUTE = ["log_days_since_prev_failure"]

HORIZONS = [90, 180, 365]
CUTOFFS = ["2023-12-31", "2022-12-31"]


# ── frame prep ───────────────────────────────────────────────────────────────

def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived covariates the layer needs (curvature value, imputed 0)."""
    df = df.copy()
    df["curvature_deg10m"] = pd.to_numeric(df["curvature"], errors="coerce").fillna(0.0)
    for c in ("install_pre2020", "install_2023plus", "log_run_seq"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _fit_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Copy with cohort-missing covariates mean-imputed (reference-neutral)."""
    df = df.copy()
    for c in NEUTRAL_IMPUTE:
        m = pd.to_numeric(df[c], errors="coerce")
        df[c] = m.fillna(m.mean())
    return df


# ── in-sample stratified Cox on the subset (the coefficient card) ────────────

def _fit_stratified(df: pd.DataFrame, covariates: list[str],
                    event_col: str = "event") -> tuple[CoxPHFitter | None, list[str], pd.DataFrame]:
    keep = ["well_key", "tte", event_col, "stratum_key"] + covariates
    tr = df[keep].copy()
    tr["tte"] = pd.to_numeric(tr["tte"], errors="coerce")
    tr = tr[tr["tte"] > 0].dropna(subset=covariates + ["tte", event_col])
    ev = tr.groupby("stratum_key", observed=True)[event_col].transform("sum")
    tr = tr[ev >= 5]
    cols = [c for c in covariates if tr[c].nunique(dropna=True) >= 2]
    tr = tr[["well_key", "tte", event_col, "stratum_key"] + cols]
    for pen in (0.0, 0.1):
        try:
            cph = CoxPHFitter(penalizer=pen)
            cph.fit(tr, "tte", event_col, strata=["stratum_key"],
                    cluster_col="well_key", formula=" + ".join(cols), robust=True)
            return cph, cols, tr
        except Exception:
            continue
    return None, cols, tr


def coefficient_card(df: pd.DataFrame) -> pd.DataFrame:
    """In-sample stratified all-cause Cox → per-covariate β/HR/CI/p + reference +
    direction sanity vs the expected sign."""
    fitdf = _fit_frame(df)
    cph, cols, tr = _fit_stratified(fitdf, COVARS)
    refs = {c: float(pd.to_numeric(tr[c], errors="coerce").mean()) for c in cols}
    summ = cph.summary if cph is not None else pd.DataFrame()
    rows = []
    for col, ship, block, window, mode, group, exp in SUBSET:
        if col in summ.index:
            s = summ.loc[col]
            beta = float(s["coef"]); hr = float(np.exp(beta))
            lo = float(np.exp(s["coef lower 95%"])); hi = float(np.exp(s["coef upper 95%"]))
            p = float(s["p"])
        else:
            beta = hr = lo = hi = p = float("nan")
        # direction check: "+" expects beta>0, "-" expects beta<0, "?" not asserted
        if not np.isfinite(beta) or exp == "?":
            dir_ok = "n/a"
        else:
            dir_ok = "TRUE" if ((beta > 0) == (exp == "+")) else "FALSE"
        rows.append({
            "covariate": col, "ship_name": ship, "hazard_group": group,
            "beta": round(beta, 6) if np.isfinite(beta) else "",
            "hr": round(hr, 4) if np.isfinite(hr) else "",
            "ci_lo": round(lo, 4) if np.isfinite(lo) else "",
            "ci_hi": round(hi, 4) if np.isfinite(hi) else "",
            "p": round(p, 5) if np.isfinite(p) else "",
            "reference_value": round(refs.get(col, float("nan")), 6),
            "expected_dir": exp, "dir_ok": dir_ok,
            "window": window, "mode": mode, "block": block, "clock": CLOCK,
        })
    return pd.DataFrame(rows)


# ── mode attribution (cause-specific Cox per group) ──────────────────────────

def mode_attribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in MODE_GROUPS:
        res = fit_cause_specific_cox(
            _fit_frame(df), covariates=COVARS, event_col=f"event_{group}",
            strata=["stratum_key"],
        )
        if not res.success or res.summary.empty:
            continue
        for _, s in res.summary.iterrows():
            rows.append({
                "covariate": str(s["covariate"]), "mode": group,
                "hr": round(float(s["hr"]), 4), "p": round(float(s["p"]), 5),
                "n_events": res.n_events,
            })
    return pd.DataFrame(rows)


def field_heterogeneity(df: pd.DataFrame, covariate: str) -> pd.DataFrame:
    rows = []
    df = _fit_frame(df).copy()
    df["field"] = df["stratum_key"].astype(str).str.split("_").str[0]
    for field, g in df.groupby("field"):
        if int(g["event"].sum()) < 15 or g[covariate].nunique(dropna=True) < 2:
            continue
        try:
            cph = CoxPHFitter(penalizer=0.1)
            sub = g[["well_key", "tte", "event", covariate]].dropna()
            sub = sub[sub["tte"] > 0]
            cph.fit(sub, "tte", "event", cluster_col="well_key",
                    formula=covariate, robust=True)
            s = cph.summary.loc[covariate]
            rows.append({
                "covariate": covariate, "field": field,
                "n": len(sub), "n_events": int(sub["event"].sum()),
                "hr": round(float(np.exp(s["coef"])), 4), "p": round(float(s["p"]), 4),
            })
        except Exception:
            continue
    return pd.DataFrame(rows).sort_values("n_events", ascending=False)


# ── the out-of-sample gate (informational; verdict overridden by the user) ───

@dataclass
class HoldoutVerdict:
    metrics: pd.DataFrame
    gate_passes: bool
    rationale: str


def _theta_surv_at(cph, test, cols, horizon, valid) -> np.ndarray:
    out = np.full(len(test), np.nan)
    mask = test["stratum_key"].isin(valid).to_numpy()
    if mask.any():
        try:
            sf = cph.predict_survival_function(
                test.loc[mask, cols + ["stratum_key"]], times=[horizon])
            out[mask] = sf.iloc[0].to_numpy()
        except Exception:
            pass
    return out


def holdout_gate(df: pd.DataFrame) -> HoldoutVerdict:
    rows = []
    for cutoff in CUTOFFS:
        split = temporal_split(df, cutoff)
        cc = COVARS + ["tte", "event"]
        train = _fit_frame(split.train[split.train["tte"] > 0]).dropna(subset=cc)
        test = _fit_frame(split.test[split.test["tte"] > 0]).dropna(subset=cc).reset_index(drop=True)
        cph, cols, tr_used = _fit_stratified(train, COVARS)
        valid = set(tr_used["stratum_key"].unique())
        test = test[test["stratum_key"].isin(valid)].reset_index(drop=True)
        if len(test) == 0 or cph is None:
            continue
        cens = _censoring_km(tr_used)
        for h in HORIZONS:
            s_base = stratum_baseline_surv(tr_used, test, h)
            s_theta = _theta_surv_at(cph, test, cols, h, valid)
            ci_b = cindex_at(test, s_base); ci_t = cindex_at(test, s_theta)
            lo, hi = cindex_bootstrap_ci(test, s_theta, n_boot=150)
            br_b, _ = ipcw_brier(test, s_base, h, cens)
            br_t, _ = ipcw_brier(test, s_theta, h, cens)
            rows.append({
                "cutoff": cutoff, "horizon_d": h, "n_test": len(test),
                "n_at_risk": n_at_risk(test, h),
                "cindex_baseline": round(ci_b, 4), "cindex_theta": round(ci_t, 4),
                "cindex_delta": round(ci_t - ci_b, 4) if np.isfinite(ci_t) and np.isfinite(ci_b) else None,
                "cindex_theta_ci_lo": round(lo, 4) if np.isfinite(lo) else None,
                "brier_delta": round(br_t - br_b, 4) if np.isfinite(br_t) and np.isfinite(br_b) else None,
            })
    metrics = pd.DataFrame(rows)
    prim = metrics[metrics["cutoff"] == "2023-12-31"]
    deltas = pd.to_numeric(prim.get("cindex_delta"), errors="coerce").dropna()
    passes = bool(len(deltas) > 0 and (deltas > 0).sum() > len(deltas) / 2)
    rationale = (f"OOS gate {'passes' if passes else 'FAILS'}: "
                 f"{int((deltas > 0).sum())}/{len(deltas)} primary horizons improve "
                 f"C-index. Shipped enabled anyway by USER OVERRIDE (sensitivity tool).")
    return HoldoutVerdict(metrics, passes, rationale)


# ── per-run covariate export ─────────────────────────────────────────────────

def _norm_well(s: object) -> str:
    return str(s).strip().lower() if pd.notna(s) else ""


def run_covariates_table(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "well_key_norm": df["well_key"].map(_norm_well),
        "run_seq": pd.to_numeric(df.get("run_seq"), errors="coerce").astype("Int64"),
        "install_date": df.get("install_date").astype(str),
        "stratum_key": df["stratum_key"].astype(str),
    })
    for c in COVARS:
        # curvature_deg10m is already 0-filled (physical); cohort/operational raw
        # (NaN -> empty cell -> VBA falls back to the covariate's reference = neutral).
        out[c] = pd.to_numeric(df[c], errors="coerce").round(6)
    avail = df[EARLY_COVARS].notna().all(axis=1)
    out["covariate_available"] = avail.map(lambda b: "TRUE" if b else "FALSE")
    out["clock"] = CLOCK
    return out


# ── orchestration ────────────────────────────────────────────────────────────

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[4], text=True).strip()
    except Exception:
        return "unknown"


def build(out_dir: Path, card_dir: Path, *, fit_date: str | None = None) -> dict:
    fit_date = fit_date or date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    card_dir.mkdir(parents=True, exist_ok=True)

    print(f"[hazard] loading competing-risks frame (clock={CLOCK})…")
    df = _prepare(build_competing_risks_df(tte_col=CLOCK))

    print("[hazard] fitting in-sample subset θ (coefficient card, 7 groups)…")
    card = coefficient_card(df)
    print("[hazard] mode attribution (cause-specific Cox)…")
    modes = mode_attribution(df)
    print("[hazard] OOS temporal-holdout gate (informational)…")
    verdict = holdout_gate(df)
    print(f"[hazard] OOS gate passes={verdict.gate_passes} — shipping enabled=TRUE (USER OVERRIDE)")
    print("[hazard] per-field heterogeneity (curvature, run_seq)…")
    het = pd.concat([field_heterogeneity(df, "curvature_deg10m"),
                     field_heterogeneity(df, "log_run_seq")], ignore_index=True)

    # USER OVERRIDE: ship enabled regardless of the (failing) OOS gate.
    card["enabled"] = "TRUE"
    card["ship_reason"] = "physical_sensitivity_not_oos_validated"
    runcov = run_covariates_table(df)

    coeffs_path = out_dir / "esp_cox_coeffs.csv"
    runcov_path = out_dir / "esp_run_covariates.csv"
    card[["covariate", "ship_name", "hazard_group", "beta", "hr", "ci_lo", "ci_hi",
          "p", "reference_value", "expected_dir", "dir_ok", "window", "mode",
          "block", "enabled", "ship_reason", "clock"]].to_csv(
        coeffs_path, index=False, encoding="utf-8-sig")
    runcov.to_csv(runcov_path, index=False, encoding="utf-8-sig")

    commit = _git_commit()
    n_bad = int((card["dir_ok"] == "FALSE").sum())
    manifest = (
        "ESP hazard layer (operational + completion + cohort)\n"
        f"fit_date   : {fit_date}\n"
        f"git_commit : {commit}\n"
        f"clock      : {CLOCK}\n"
        f"ship_mode  : multiplier (USER-OVERRIDE, not OOS-validated)\n"
        f"enabled    : True\n"
        f"groups     : {', '.join(GROUPS)}\n"
        f"n_covariates : {len(card)}\n"
        f"oos_gate_passes : {verdict.gate_passes}\n"
        f"dir_mismatches : {n_bad}\n"
        f"rationale  : {verdict.rationale}\n"
    )
    (out_dir / "hazard_manifest.txt").write_text(manifest, encoding="utf-8")

    verdict.metrics.to_csv(card_dir / "holdout_metrics.csv", index=False, encoding="utf-8-sig")
    modes.to_csv(card_dir / "mode_attribution.csv", index=False, encoding="utf-8-sig")
    het.to_csv(card_dir / "field_heterogeneity.csv", index=False, encoding="utf-8-sig")
    _write_card(card_dir / "README-data.md", card, modes, het, verdict, fit_date, commit)

    print("[hazard] wrote:")
    for p in (coeffs_path, runcov_path, out_dir / "hazard_manifest.txt",
              card_dir / "README-data.md"):
        print(f"  {p}")
    print("\n" + manifest)
    return {"card": card, "modes": modes, "verdict": verdict, "runcov": runcov}


def _tbl(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_none_"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


def _write_card(path, card, modes, het, verdict, fit_date, commit) -> None:
    lines = [
        "# Operational + completion + cohort hazard layer — data card",
        "",
        f"**Date:** {fit_date} · git `{commit}` · clock `ttf_mix` · "
        f"**ship_mode: multiplier (USER-OVERRIDE, NOT OOS-validated)**",
        "",
        "Applies the model report §2/§2.7 hazards (GLF, load, kpod, frequency, "
        "curvature-value, well_history, vintage) as a physical regime-sensitivity "
        "overlay on the v2 stratum + K=2 baseline. Shipped `enabled=TRUE` at the "
        "user's request. **This is a what-if / sensitivity tool, not a validated "
        "forecast.** The baseline columns are never changed; adjusted outputs are "
        "separate `_sens` columns.",
        "",
        "Curvature uses the value (degrees / 10 m); missing → 0.0 (straight well, "
        "assumed < 0.3). First runs get a reference-neutral days-since-failure.",
        "",
        "## Out-of-sample gate (kept for honesty — the layer does NOT improve it)",
        "",
        "> " + verdict.rationale,
        "",
        _tbl(verdict.metrics),
        "",
        "**Cohort caveat:** `well_history` (log_run_seq, the largest coefficient) and "
        "`vintage` move `ESP_RUL_sens` the most and generalize the least — they are the "
        "primary reasons the merged θ failed to transport (§3.8). Treat their deltas as "
        "descriptive of the fitted cohort, not predictive.",
        "",
        "## Coefficient card (in-sample stratified Cox, all-cause) + direction sanity",
        "",
        _tbl(card),
        "",
        "## Mode attribution (cause-specific Cox — where each hazard lands)",
        "",
        _tbl(modes),
        "",
        "## Per-field heterogeneity — curvature value & run history",
        "",
        _tbl(het),
        "",
        "**Feasibility:** operational covariates use the first-30-operating-day early "
        "window (absent for pumps with <30 telemetry days → those groups' deltas are 0 "
        "there). Curvature, well_history and vintage are t0/cohort — their deltas apply "
        "to essentially every run. Delivered via `esp_run_covariates.csv`, joined on "
        "`well_key_norm` + `run_seq`.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


__all__ = ["build", "SUBSET", "COVARS", "GROUPS", "coefficient_card", "holdout_gate",
           "mode_attribution", "run_covariates_table", "HoldoutVerdict"]
