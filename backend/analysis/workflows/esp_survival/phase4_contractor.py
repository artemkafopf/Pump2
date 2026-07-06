"""Phase 4 — Contractor effect.

For each stratum:
  1. KM per contractor — visual separation
  2. Cox PH with contractor as covariate (strata = field × H2S)
  3. Decision rule:
     - If Schoenfeld p < 0.05 → Extended Cox term
     - If w₁ differs > 10pp across contractors within same field → promote to stratum
     - Otherwise → standard Cox covariate
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

from analysis.models.survival.latent_weibull_competing_risks import (
    TwoComponentLatentWeibullModel,
    latent_life_quantile,
)
from analysis.models.survival.weibull_em import fit_latent_weibull_em


# Contractors with enough cases to analyse
MIN_CONTRACTOR_FAILURES = 10
CONTRACTOR_ORDER = ["Шлюмберже", "Борец", "Новые технологии", "Новомет"]
CONTRACTOR_COLORS = {
    "Шлюмберже": "steelblue",
    "Борец": "darkorange",
    "Новые технологии": "crimson",
    "Новомет": "seagreen",
}


def _km_for_contractors(
    g: pd.DataFrame, stratum: str, ax: plt.Axes
) -> dict[str, float]:
    """Plot KM per contractor on ax. Return median TTE dict."""
    medians: dict[str, float] = {}
    contractors = [c for c in CONTRACTOR_ORDER if c in g["contractor"].values]
    for contractor in contractors:
        sub = g[g["contractor"] == contractor]
        if int(sub["event"].sum()) < MIN_CONTRACTOR_FAILURES:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(sub["tte"], sub["event"], label=contractor)
        kmf.plot_survival_function(ax=ax, color=CONTRACTOR_COLORS.get(contractor, "gray"),
                                   ci_show=True, ci_alpha=0.1)
        med = kmf.median_survival_time_
        if med not in (None, np.inf):
            medians[contractor] = float(med)
    ax.set_title(stratum, fontsize=8)
    ax.set_xlabel("Days")
    ax.set_ylabel("Survival")
    return medians


def _logrank_contractors(g: pd.DataFrame) -> float:
    """Multi-group log-rank test across contractors with ≥ MIN_CONTRACTOR_FAILURES."""
    durations, events, groups = [], [], []
    for contractor, sub in g.groupby("contractor"):
        if int(sub["event"].sum()) < MIN_CONTRACTOR_FAILURES:
            continue
        durations.extend(sub["tte"].tolist())
        events.extend(sub["event"].tolist())
        groups.extend([contractor] * len(sub))
    if len(set(groups)) < 2:
        return float("nan")
    result = multivariate_logrank_test(durations, groups, events)
    return float(result.p_value)


def _cox_contractor(df: pd.DataFrame, stratum: str) -> dict:
    """Fit stratified Cox with contractor as covariate."""
    g = df.copy()
    # Encode contractor as dummy (reference = Шлюмберже if present, else first alphabetically)
    contractors = [c for c in CONTRACTOR_ORDER if c in g["contractor"].values]
    viable = [c for c in contractors if int(g.loc[g["contractor"] == c, "event"].sum()) >= MIN_CONTRACTOR_FAILURES]
    if len(viable) < 2:
        return {"schoenfeld_p": float("nan"), "note": "too few viable contractors"}

    ref = viable[0]
    g["contractor_coded"] = pd.Categorical(g["contractor"], categories=viable).codes.astype(float)
    g = g[g["contractor"].isin(viable)].copy()
    g["stratum_col"] = g["stratum"]

    try:
        cph = CoxPHFitter()
        cph.fit(
            g[["tte", "event", "contractor_coded", "stratum_col"]],
            duration_col="tte",
            event_col="event",
            strata=["stratum_col"],
            formula="contractor_coded",
        )
        summary = cph.summary
        hr = float(np.exp(summary.loc["contractor_coded", "coef"]))
        p_coef = float(summary.loc["contractor_coded", "p"])

        # Schoenfeld test
        from lifelines.statistics import proportional_hazard_test
        sch = proportional_hazard_test(cph, g[["tte", "event", "contractor_coded", "stratum_col"]],
                                       time_transform="log")
        sch_p = float(sch.summary.loc["contractor_coded", "p"])
        return {"hr": hr, "p_coef": p_coef, "schoenfeld_p": sch_p, "reference": ref}
    except Exception as exc:
        return {"schoenfeld_p": float("nan"), "note": str(exc)}


def _w1_per_contractor(
    g: pd.DataFrame,
    stratum: str,
    models: dict[str, TwoComponentLatentWeibullModel],
) -> dict[str, float]:
    """Fit independent K=2 mixture per contractor within a stratum."""
    w1_by_contractor: dict[str, float] = {}
    base_model = models.get(stratum)
    if base_model is None:
        return w1_by_contractor

    fix_b1 = float(base_model.component_1.beta)
    fix_b2 = float(base_model.component_2.beta)

    for contractor, sub in g.groupby("contractor"):
        n_fail = int(sub["event"].sum())
        if n_fail < MIN_CONTRACTOR_FAILURES:
            continue
        dur = sub["tte"].to_numpy(dtype=float)
        ev = sub["event"].to_numpy(dtype=int)
        try:
            # Fix stratum shapes; fit only w₁, η₁, η₂ per contractor
            res = fit_latent_weibull_em(
                dur, ev,
                initial_model=base_model,
                num_starts=6,
                max_iter=200,
                fix_beta1=fix_b1,
                fix_beta2=fix_b2,
            )
            w1_by_contractor[contractor] = round(float(res.model.weight_1), 4)
        except Exception:
            pass
    return w1_by_contractor


def run(
    df: pd.DataFrame,
    out_dir: Path,
    models: dict[str, TwoComponentLatentWeibullModel],
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Focus on strata where contractor variation is meaningful
    focus_strata = ["Vt_sour", "Vt_nonsour", "Ya_nonsour"]
    focus_strata = [s for s in focus_strata if s in df["stratum"].unique()]

    ncols = min(3, len(focus_strata))
    nrows = int(np.ceil(len(focus_strata) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    results = []
    for idx, stratum in enumerate(focus_strata):
        g = df[df["stratum"] == stratum].copy()
        ax = axes_flat[idx]
        medians = _km_for_contractors(g, stratum, ax)
        lr_p = _logrank_contractors(g)
        cox_res = _cox_contractor(g, stratum)
        w1_dict = _w1_per_contractor(g, stratum, models)

        w1_range = max(w1_dict.values()) - min(w1_dict.values()) if len(w1_dict) > 1 else float("nan")

        decision = "standard_cox_covariate"
        if cox_res.get("schoenfeld_p", 1.0) < 0.05:
            decision = "extended_cox_with_logt"
        elif not np.isnan(w1_range) and w1_range > 0.10:
            decision = "promote_to_stratum"

        row = {
            "stratum": stratum,
            "logrank_p": round(lr_p, 4) if not np.isnan(lr_p) else None,
            "cox_hr": round(cox_res.get("hr", float("nan")), 3),
            "cox_p": round(cox_res.get("p_coef", float("nan")), 4),
            "schoenfeld_p": round(cox_res.get("schoenfeld_p", float("nan")), 4),
            "w1_range_across_contractors": round(w1_range, 4) if not np.isnan(w1_range) else None,
            "decision": decision,
            **{f"median_tte_{c.replace(' ', '_')}": round(v, 1) for c, v in medians.items()},
            **{f"w1_{c.replace(' ', '_')}": v for c, v in w1_dict.items()},
        }
        results.append(row)
        print(
            f"  [Phase 4] {stratum}: logrank_p={lr_p:.4f}  "
            f"schoenfeld_p={cox_res.get('schoenfeld_p', float('nan')):.4f}  "
            f"w1_range={w1_range:.3f}  decision={decision}"
        )

    for idx in range(len(focus_strata), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Phase 4 — Contractor KM Curves per Stratum", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_dir / "phase4_contractor_km.png", dpi=150)
    plt.close(fig)

    result_df = pd.DataFrame(results)
    result_df.to_csv(out_dir / "phase4_contractor_analysis.csv", index=False, encoding="utf-8-sig")

    print(f"\n[Phase 4] Complete. Decision summary:")
    for r in results:
        print(f"  {r['stratum']}: {r['decision']}")

    return {"results": result_df}
