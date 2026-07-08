"""Shared Cox reporting helpers for Phase C (cause-specific sweeps, forests,
per-field heterogeneity).

These are the pieces C1 (completion), C2 (operational) and C3 (mode-differentiation
forests) all reuse, so the "expected mode?" confounding audit is computed and drawn
one way across the phase.  All functions are path-agnostic: they take a fitted /
built frame and return tidy frames or write a figure to an explicit path.

The cause-specific hazard is realised through
:func:`analysis.models.survival.cause_specific_cox.fit_cause_specific_cox` — a
covariate acting on its physically expected mode (and not on the others) is
corroborated; one elevated equally in every mode is flagged as residual confounding.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.models.survival.cause_specific_cox import fit_cause_specific_cox

COX_MODE_GROUPS = ("hydraulic", "electro-thermal", "protector")
GROUP_COLORS = {
    "hydraulic": "#4C78A8", "electro-thermal": "#F58518", "protector": "#54A24B",
}


def cause_specific_sweep(
    df: pd.DataFrame,
    covariates: list[str],
    *,
    groups: tuple[str, ...] = COX_MODE_GROUPS,
    duration_col: str = "tte",
    strata: list[str] | None = None,
    cluster_col: str = "well_key",
    min_events: int = 15,
) -> pd.DataFrame:
    """Fit one cause-specific stratified Cox per group; stack the per-covariate HRs.

    Returns cause_group | covariate | coef | se | hr | hr_ci_lo | hr_ci_hi | p |
    n_events (empty frame if no group cleared ``min_events``).
    """
    strata = strata or ["stratum_key"]
    frames = []
    for group in groups:
        res = fit_cause_specific_cox(
            df, covariates=covariates, event_col=f"event_{group}",
            duration_col=duration_col, strata=strata, cluster_col=cluster_col,
            min_events=min_events)
        if not res.success:
            continue
        s = res.summary.copy()
        s.insert(0, "cause_group", group)
        s["n_events"] = res.n_events
        frames.append(s)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def forest_across_causes(
    coeffs: pd.DataFrame,
    covariate: str,
    path: Path,
    *,
    expected_mode: str | None = None,
    title_prefix: str = "",
) -> str:
    """Forest of HR (log axis) + 95% CI for one covariate across cause groups.

    Draws the ``expected_mode`` label if given and returns an "expected mode?"
    verdict string (``match`` / ``mismatch`` / ``diffuse`` / ``n/a``) computed from
    which group the covariate lands on significantly.
    """
    if coeffs is None or coeffs.empty or "covariate" not in coeffs.columns:
        return "n/a"
    rows = coeffs[coeffs["covariate"] == covariate]
    if rows.empty:
        return "n/a"
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 2.4 + 0.4 * len(rows)))
    sig_groups = []
    for yi, (_, r) in zip(y, rows.iterrows()):
        c = GROUP_COLORS.get(r["cause_group"], "#555")
        ax.plot([r["hr_ci_lo"], r["hr_ci_hi"]], [yi, yi], color=c, lw=2)
        ax.plot(r["hr"], yi, "o", color=c, ms=7)
        star = " *" if r["p"] < 0.05 else ""
        ax.text(r["hr_ci_hi"], yi + 0.12,
                f" HR={r['hr']:.2f} [{r['hr_ci_lo']:.2f},{r['hr_ci_hi']:.2f}] p={r['p']:.3f}{star}",
                va="bottom", fontsize=7.5)
        if r["p"] < 0.05:
            sig_groups.append(r["cause_group"])
    ax.axvline(1.0, color="black", lw=0.9, ls="--")
    ax.set_yticks(y); ax.set_yticklabels(rows["cause_group"])
    ax.set_xscale("log")
    ax.set_xlabel("hazard ratio (log scale)")
    ttl = f"{title_prefix}{covariate}  (cause-specific, clock=ttf_mix)"
    if expected_mode:
        ttl += f"\nexpected mode: {expected_mode}"
    ax.set_title(ttl, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

    if not sig_groups:
        return "null (no mode significant)"
    if len(sig_groups) >= 3:
        return "diffuse (all modes — confounding suspect)"
    if expected_mode is None:
        return "significant: " + "+".join(sig_groups)
    if expected_mode in sig_groups and len(sig_groups) == 1:
        return f"match ({expected_mode})"
    if expected_mode in sig_groups:
        return f"partial (hits {expected_mode} + " + "+".join(g for g in sig_groups if g != expected_mode) + ")"
    return f"mismatch (expected {expected_mode}, hit " + "+".join(sig_groups) + ")"


def per_field_heterogeneity(
    df: pd.DataFrame,
    covariate: str,
    *,
    event_col: str = "event",
    duration_col: str = "tte",
    strata: list[str] | None = None,
    cluster_col: str = "well_key",
    min_events: int = 30,
    adjusters: list[str] | None = None,
) -> pd.DataFrame:
    """Per-field HR of ``covariate`` (field-subset fits); flags sign flips.

    A ``covariate`` adjusted by ``adjusters`` is refit within each field that has
    ≥ ``min_events`` events; the pooled sign is taken from the field with the most
    events and any field flipping sign is flagged.
    """
    strata = strata or ["stratum_key"]
    adjusters = adjusters or []
    cov_list = [covariate] + [a for a in adjusters if a != covariate]
    rows = []
    for fld, g in df.groupby("field", observed=True):
        if int(g[event_col].sum()) < min_events:
            continue
        res = fit_cause_specific_cox(
            g, covariates=cov_list, event_col=event_col, duration_col=duration_col,
            strata=strata, cluster_col=cluster_col, min_events=min_events)
        if not res.success:
            continue
        r = res.summary[res.summary["covariate"] == covariate]
        if r.empty:
            continue
        rows.append({
            "field": fld, "covariate": covariate,
            "hr": round(float(r["hr"].iloc[0]), 3),
            "hr_ci_lo": round(float(r["hr_ci_lo"].iloc[0]), 3),
            "hr_ci_hi": round(float(r["hr_ci_hi"].iloc[0]), 3),
            "coef": round(float(r["coef"].iloc[0]), 4),
            "p": round(float(r["p"].iloc[0]), 4),
            "n_events": res.n_events,
        })
    out = pd.DataFrame(rows).sort_values("n_events", ascending=False) if rows else pd.DataFrame()
    if not out.empty:
        lead_sign = np.sign(out.iloc[0]["coef"])
        out["sign_flip"] = np.sign(out["coef"]) != lead_sign
    return out


__all__ = [
    "cause_specific_sweep", "forest_across_causes", "per_field_heterogeneity",
    "COX_MODE_GROUPS", "GROUP_COLORS",
]
