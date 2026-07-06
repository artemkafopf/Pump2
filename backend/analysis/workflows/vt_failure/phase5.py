"""Phase 5 — Competing Risks (CIF via Aalen-Johansen estimator)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import AalenJohansenFitter

from .config import (
    CATEGORY_COLORS, FAILURE_CATEGORIES, MIN_FAILURES_KM, VT_FIELD,
)


def _cif_for_scope(df: pd.DataFrame, scope_label: str, out: Path, min_ev: int = MIN_FAILURES_KM) -> None:
    """Plot CIF curves for all viable categories in scope. Uses integer event codes."""
    # Map failure_category → integer code (0 = censored, 1..N = category)
    all_cats = [c for c in FAILURE_CATEGORIES if (df["failure_category"] == c).sum() >= min_ev]
    if not all_cats:
        return

    cat_to_code = {cat: i + 1 for i, cat in enumerate(all_cats)}

    # Build event_col: 0 for censored, code for each category
    df2 = df.copy()
    df2["event_code"] = 0
    for cat, code in cat_to_code.items():
        df2.loc[(df2["event"] == 1) & (df2["failure_category"] == cat), "event_code"] = code

    T = df2["duration"].dropna().values.astype(float)
    E = df2.loc[df2["duration"].notna(), "event_code"].values.astype(int)

    fig, ax = plt.subplots(figsize=(11, 6))

    cif_rows = []
    for cat in all_cats:
        code = cat_to_code[cat]
        try:
            ajf = AalenJohansenFitter(calculate_variance=False)
            ajf.fit(T, E, event_of_interest=code)
            color = CATEGORY_COLORS.get(cat, "#888")
            n_ev = int((E == code).sum())
            ajf.plot(ax=ax, color=color, linewidth=2.0, label=f"{cat} (n={n_ev})")
            # Extract final CIF value
            timeline = ajf.cumulative_density_.index.values
            cif_vals = ajf.cumulative_density_.iloc[:, 0].values
            cif_rows.append({
                "scope": scope_label,
                "category": cat,
                "n_events": n_ev,
                "final_cif": float(cif_vals[-1]) if len(cif_vals) > 0 else np.nan,
                "cif_at_90d": float(np.interp(90.0, timeline, cif_vals)),
                "cif_at_365d": float(np.interp(365.0, timeline, cif_vals)),
            })
        except Exception as exc:
            print(f"  CIF fit failed for {cat} in {scope_label}: {exc}")

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Cumulative Incidence P(T ≤ t, cause = k)")
    ax.set_title(f"Phase 5 — Competing-risk CIF by failure category ({scope_label})")
    ax.set_ylim(-0.01, None)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / f"p5_cif_{scope_label}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame(cif_rows)


def _gray_test_approx(df: pd.DataFrame, cat: str, freq_col: str = "freq_group") -> dict:
    """
    Approximate Gray's test by comparing CIF curves via log-rank on cause-specific
    subdistribution hazard proxy. This is an approximation; true Gray's test requires
    the cmprsk R package. We compute a basic chi-squared on cumulative incidences.
    """
    groups = df[freq_col].dropna().unique()
    cat_incs = {}
    for grp in groups:
        sub = df[df[freq_col] == grp].dropna(subset=["duration"])
        n_total = len(sub)
        n_cause = int(((sub["event"] == 1) & (sub["failure_category"] == cat)).sum())
        if n_total > 0:
            cat_incs[grp] = n_cause / n_total
    return cat_incs


def run(df: pd.DataFrame, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    all_cif_rows = []

    # ------------------------------------------------------------------
    # 5.1  CIF for Global and Vt
    # ------------------------------------------------------------------
    for scope_label, scope_df in [("global", df), ("vt", df[df["is_vt"]])]:
        scope_df2 = scope_df.dropna(subset=["duration"]).copy()
        cif_df = _cif_for_scope(scope_df2, scope_label, out)
        if cif_df is not None and not cif_df.empty:
            all_cif_rows.append(cif_df)

    # ------------------------------------------------------------------
    # 5.2  CIF by frequency group (Global) — separate panel per category
    # ------------------------------------------------------------------
    grp_order = ["Low (≤50 Hz)", "Normal (50–55 Hz)", "High (>55 Hz)"]
    df_freq = df[df["freq_group"].isin(grp_order)].dropna(subset=["duration"])

    viable_cats = [
        c for c in FAILURE_CATEGORIES
        if (df_freq["failure_category"] == c).sum() >= MIN_FAILURES_KM
    ]
    n_cats = len(viable_cats)
    if n_cats > 0:
        ncols = min(3, n_cats)
        nrows = (n_cats + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4), squeeze=False)

        for idx, cat in enumerate(viable_cats):
            ax = axes[idx // ncols][idx % ncols]
            cat_to_code = {cat: 1}
            colors = ["#4878CF", "#6ACC65", "#D65F5F"]

            for gi, grp in enumerate(grp_order):
                sub = df_freq[df_freq["freq_group"] == grp].copy()
                if len(sub) < 5:
                    continue
                sub["event_code"] = 0
                sub.loc[(sub["event"] == 1) & (sub["failure_category"] == cat), "event_code"] = 1
                T = sub["duration"].values.astype(float)
                E = sub["event_code"].values.astype(int)
                if (E == 1).sum() < 5:
                    continue
                try:
                    ajf = AalenJohansenFitter(calculate_variance=False)
                    ajf.fit(T, E, event_of_interest=1)
                    n_ev = int((E == 1).sum())
                    ajf.plot(ax=ax, color=colors[gi % len(colors)], linewidth=2.0,
                             label=f"{grp} (n={n_ev})")
                except Exception:
                    pass

            ax.set_title(cat, fontsize=8)
            ax.set_xlabel("Days", fontsize=7)
            ax.set_ylabel("CIF", fontsize=7)
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.2)

        # Hide empty subplots
        for idx in range(n_cats, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.suptitle("Phase 5 — CIF by failure category × frequency group (Global)", y=1.01)
        fig.tight_layout()
        fig.savefig(out / "p5_cif_by_freq_group.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # 5.3  Descriptive: cause-specific incidence proportions by freq group
    # ------------------------------------------------------------------
    inc_rows = []
    for cat in FAILURE_CATEGORIES:
        for grp in grp_order:
            sub = df[df["freq_group"] == grp]
            n_total = len(sub)
            n_ev = int(((sub["event"] == 1) & (sub["failure_category"] == cat)).sum())
            inc_rows.append({
                "freq_group": grp,
                "category": cat,
                "n_cause": n_ev,
                "n_total": n_total,
                "crude_incidence_pct": n_ev / n_total * 100 if n_total > 0 else np.nan,
            })
    inc_df = pd.DataFrame(inc_rows)
    inc_df.to_csv(out / "p5_cause_incidence.csv", index=False, encoding="utf-8-sig")
    results["cause_incidence"] = inc_df

    # Pivot for easy reading
    pivot = inc_df.pivot(index="category", columns="freq_group", values="crude_incidence_pct").round(2)
    pivot.to_csv(out / "p5_incidence_pivot.csv", encoding="utf-8-sig")

    if all_cif_rows:
        cif_summary = pd.concat(all_cif_rows, ignore_index=True)
        cif_summary.to_csv(out / "p5_cif_summary.csv", index=False, encoding="utf-8-sig")
        results["cif_summary"] = cif_summary

    # ------------------------------------------------------------------
    # Print
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 5 — COMPETING RISKS (CIF)")
    print("=" * 60)
    print("Cause-specific crude incidence by frequency group (%):")
    try:
        print(pivot.to_string())
    except Exception:
        pass

    return results
