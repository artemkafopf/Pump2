"""Phase 4 — Weibull Shape Analysis."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import WeibullFitter

from .config import (
    CATEGORY_BETA_PRIOR, FAILURE_CATEGORIES, FREQ_GROUP_COLORS,
    MAJOR_FIELDS, MIN_FAILURES_WEIBULL, VT_FIELD,
)


def _fit_weibull(T: np.ndarray, E: np.ndarray) -> dict:
    """Fit two-parameter Weibull via lifelines. Returns beta, eta, CIs."""
    if E.sum() < MIN_FAILURES_WEIBULL:
        return {}
    wf = WeibullFitter()
    try:
        wf.fit(T, E)
        beta = float(wf.rho_)      # shape (lifelines calls it rho)
        eta  = float(wf.lambda_)   # scale
        # 95% CI via profile likelihood (lifelines provides summary)
        ci = wf.summary[["coef lower 95%", "coef upper 95%"]]
        rho_row = ci.loc["rho_"] if "rho_" in ci.index else None
        if rho_row is not None:
            beta_lo = float(rho_row["coef lower 95%"])
            beta_hi = float(rho_row["coef upper 95%"])
        else:
            beta_lo = beta_hi = np.nan
        return {
            "beta": round(beta, 3),
            "eta_days": round(eta, 1),
            "beta_ci_lo": round(beta_lo, 3),
            "beta_ci_hi": round(beta_hi, 3),
            "pattern": "decreasing hazard (infant)" if beta < 0.9 else
                       "constant hazard (random)" if beta < 1.1 else
                       "increasing hazard (wear-out)",
        }
    except Exception as exc:
        return {"fit_error": str(exc)}


def run(df: pd.DataFrame, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    rows = []

    # ------------------------------------------------------------------
    # 4.1  By frequency group — Global and per major field
    # ------------------------------------------------------------------
    grp_order = ["Low (≤50 Hz)", "Normal (50–55 Hz)", "High (>55 Hz)"]
    for field in ["GLOBAL"] + MAJOR_FIELDS:
        sub = df if field == "GLOBAL" else df[df["field"] == field]
        for grp in grp_order:
            g = sub[sub["freq_group"] == grp].dropna(subset=["duration"])
            T = g["duration"].values.astype(float)
            E = g["event"].values.astype(float)
            fit = _fit_weibull(T, E)
            rows.append({
                "scope": field,
                "group_type": "freq_group",
                "group": grp,
                "n_runs": len(T),
                "n_fail": int(E.sum()),
                "prior_beta": "—",
                **fit,
            })

    # ------------------------------------------------------------------
    # 4.2  By failure category — Global
    # ------------------------------------------------------------------
    for cat in FAILURE_CATEGORIES:
        g = df[df["failure_category"] == cat].dropna(subset=["duration"])
        T = g["duration"].values.astype(float)
        E = g["event"].values.astype(float)
        fit = _fit_weibull(T, E)
        rows.append({
            "scope": "GLOBAL",
            "group_type": "failure_category",
            "group": cat,
            "n_runs": len(T),
            "n_fail": int(E.sum()),
            "prior_beta": CATEGORY_BETA_PRIOR.get(cat, "—"),
            **fit,
        })

    # ------------------------------------------------------------------
    # 4.3  Vt by failure category
    # ------------------------------------------------------------------
    vt = df[df["is_vt"]]
    for cat in FAILURE_CATEGORIES:
        g = vt[vt["failure_category"] == cat].dropna(subset=["duration"])
        T = g["duration"].values.astype(float)
        E = g["event"].values.astype(float)
        fit = _fit_weibull(T, E)
        rows.append({
            "scope": "Vt",
            "group_type": "failure_category",
            "group": cat,
            "n_runs": len(T),
            "n_fail": int(E.sum()),
            "prior_beta": CATEGORY_BETA_PRIOR.get(cat, "—"),
            **fit,
        })

    weibull_table = pd.DataFrame(rows)
    weibull_table.to_csv(out / "p4_weibull_table.csv", index=False, encoding="utf-8-sig")
    results["weibull_table"] = weibull_table

    # ------------------------------------------------------------------
    # 4.4  Forest plot: beta with CI by group
    # ------------------------------------------------------------------
    for group_type in ["freq_group", "failure_category"]:
        sub = weibull_table[
            (weibull_table["group_type"] == group_type) &
            (weibull_table["scope"] == "GLOBAL") &
            weibull_table["beta"].notna()
        ].copy()
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(9, max(4, len(sub) * 0.6)))
        y = range(len(sub))
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="β = 1 (constant hazard)")
        ax.axvline(0.9, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)
        ax.axvline(1.1, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)

        colors = []
        for _, row in sub.iterrows():
            b = row["beta"]
            if b < 0.9:
                colors.append("#4878CF")
            elif b > 1.1:
                colors.append("#D65F5F")
            else:
                colors.append("#6ACC65")

        for i, (_, row) in enumerate(sub.iterrows()):
            b = row["beta"]
            lo = row.get("beta_ci_lo", np.nan)
            hi = row.get("beta_ci_hi", np.nan)
            ax.scatter(b, i, color=colors[i], s=60, zorder=3)
            if np.isfinite(lo) and np.isfinite(hi):
                ax.hlines(i, lo, hi, color=colors[i], linewidth=2.5, alpha=0.6)
            label = f"{row['group']}  (n={row['n_fail']})"
            ax.text(max(float(hi) if np.isfinite(hi) else b, b) + 0.03, i,
                    label, va="center", fontsize=8)

        ax.set_yticks(list(y))
        ax.set_yticklabels([""] * len(sub))
        ax.set_xlabel("Weibull β (shape parameter)")
        ax.set_title(f"Phase 4 — Weibull β forest plot: {group_type} (Global)\nblue=infant-dominated  green=random  red=wear-out")
        ax.legend(fontsize=8)
        ax.grid(True, axis="x", alpha=0.2)
        fig.tight_layout()
        fig.savefig(out / f"p4_weibull_forest_{group_type}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # 4.5  Weibull curves overlaid — freq groups global
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    t_max = df["duration"].quantile(0.95)
    t_grid = np.linspace(0.01, max(float(t_max), 30), 300)

    for grp, color in FREQ_GROUP_COLORS.items():
        g = df[(df["freq_group"] == grp)].dropna(subset=["duration"])
        T = g["duration"].values.astype(float)
        E = g["event"].values.astype(float)
        if E.sum() < MIN_FAILURES_WEIBULL:
            continue
        wf = WeibullFitter()
        try:
            wf.fit(T, E)
            s = np.exp(-(t_grid / wf.lambda_) ** wf.rho_)
            ax2.plot(t_grid, s, color=color, linewidth=2.0,
                     label=f"{grp} β={wf.rho_:.2f} η={wf.lambda_:.0f}d (n_ev={int(E.sum())})")
        except Exception:
            continue

    ax2.set_xlabel("Time (days)")
    ax2.set_ylabel("S(t) — Weibull fitted")
    ax2.set_title("Phase 4 — Weibull fitted survival by frequency group (Global)")
    ax2.set_ylim(-0.02, 1.05)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)
    fig2.tight_layout()
    fig2.savefig(out / "p4_weibull_curves_freq.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # ------------------------------------------------------------------
    # Print
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 4 — WEIBULL SHAPE ANALYSIS")
    print("=" * 60)
    show_cols = ["scope", "group_type", "group", "n_fail", "prior_beta", "beta", "beta_ci_lo", "beta_ci_hi", "eta_days", "pattern"]
    show_cols = [c for c in show_cols if c in weibull_table.columns]
    print(weibull_table[weibull_table["n_fail"] >= MIN_FAILURES_WEIBULL][show_cols].to_string(index=False))

    return results
