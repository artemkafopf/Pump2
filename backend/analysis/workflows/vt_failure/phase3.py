"""Phase 3 — Kaplan-Meier Survival Analysis."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

from .config import (
    CATEGORY_COLORS, FAILURE_CATEGORIES, FREQ_GROUP_COLORS,
    MAJOR_FIELDS, MIN_FAILURES_KM, VT_FIELD,
)

_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]


def _get_color(label: str, colors_dict: dict, idx: int = 0) -> str:
    return colors_dict.get(label, _PALETTE[idx % len(_PALETTE)])


def _km_stats(kmf: KaplanMeierFitter, events: np.ndarray) -> dict:
    try:
        median = float(kmf.median_survival_time_)
    except Exception:
        median = np.nan
    try:
        q25 = float(kmf.percentile(0.75))  # S(t)=0.75 → 25th percentile survival time
    except Exception:
        q25 = np.nan
    return {
        "n_total": len(events),
        "n_events": int(events.sum()),
        "median_survival": median,
        "q25_survival": q25,
    }


def _plot_km_groups(
    df: pd.DataFrame,
    group_col: str,
    groups: list,
    colors_dict: dict,
    duration_col: str,
    event_col: str,
    title: str,
    out_path: Path,
    reference_group: str | None = None,
    min_failures: int = MIN_FAILURES_KM,
) -> pd.DataFrame:
    """Plot KM curves for multiple groups. Returns a stats table."""
    fig, ax = plt.subplots(figsize=(10, 6))
    stats_rows = []
    kmfs = {}
    group_events = {}
    group_times = {}

    for i, grp in enumerate(groups):
        sub = df[df[group_col] == grp].dropna(subset=[duration_col])
        T = sub[duration_col].values.astype(float)
        E = sub[event_col].values.astype(float)
        if E.sum() < min_failures:
            continue

        kmf = KaplanMeierFitter()
        kmf.fit(T, E, label=f"{grp} (n={len(T)}, ev={int(E.sum())})")
        color = _get_color(grp, colors_dict, i)
        kmf.plot_survival_function(ax=ax, color=color, linewidth=2.0, ci_show=True, ci_alpha=0.12)

        s = _km_stats(kmf, E)
        s["group"] = grp
        stats_rows.append(s)
        kmfs[grp] = kmf
        group_events[grp] = E
        group_times[grp] = T

    # Log-rank test vs reference
    if reference_group and reference_group in kmfs:
        for grp in [g for g in groups if g != reference_group and g in kmfs]:
            try:
                lr = logrank_test(
                    group_times[reference_group], group_times[grp],
                    group_events[reference_group], group_events[grp],
                )
                p = lr.p_value
                for row in stats_rows:
                    if row["group"] == grp:
                        row["logrank_p_vs_ref"] = round(float(p), 4)
                        break
            except Exception:
                pass
    else:
        # Multivariate log-rank
        all_groups = [g for g in groups if g in kmfs]
        if len(all_groups) >= 2:
            try:
                groups_arr = np.concatenate([[g] * len(group_times[g]) for g in all_groups])
                T_all = np.concatenate([group_times[g] for g in all_groups])
                E_all = np.concatenate([group_events[g] for g in all_groups])
                lr = multivariate_logrank_test(T_all, groups_arr, E_all)
                for row in stats_rows:
                    row["logrank_p_multivariate"] = round(float(lr.p_value), 4)
            except Exception:
                pass

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival probability S(t)")
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame(stats_rows)


def run(df: pd.DataFrame, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    all_stats = []

    # ------------------------------------------------------------------
    # 3.1  All runs, by freq group — Global
    # ------------------------------------------------------------------
    grp_order = ["Low (≤50 Hz)", "Normal (50–55 Hz)", "High (>55 Hz)"]
    df_freq = df[df["freq_group"].isin(grp_order)]

    stats = _plot_km_groups(
        df_freq, "freq_group", grp_order, FREQ_GROUP_COLORS,
        "duration", "event",
        "Phase 3 — KM by frequency group (Global, all runs)",
        out / "p3_km_freq_global_all.png",
        reference_group="Normal (50–55 Hz)",
    )
    stats["scope"] = "global"; stats["subset"] = "all"
    all_stats.append(stats)

    # ------------------------------------------------------------------
    # 3.2  Mature only (TTF >= 90d), by freq group — Global
    # ------------------------------------------------------------------
    df_mature = df[(df["freq_group"].isin(grp_order)) & (
        (df["duration"] >= 90) | (df["event"] == 0)
    )]
    stats_m = _plot_km_groups(
        df_mature, "freq_group", grp_order, FREQ_GROUP_COLORS,
        "duration", "event",
        "Phase 3 — KM by frequency group (Global, mature runs ≥90d or censored)",
        out / "p3_km_freq_global_mature.png",
        reference_group="Normal (50–55 Hz)",
    )
    stats_m["scope"] = "global"; stats_m["subset"] = "mature_ge90d"
    all_stats.append(stats_m)

    # ------------------------------------------------------------------
    # 3.3  By freq group, per major field
    # ------------------------------------------------------------------
    for field in MAJOR_FIELDS:
        sub = df_freq[df_freq["field"] == field]
        if sub["event"].sum() < MIN_FAILURES_KM * 2:
            continue
        stats_f = _plot_km_groups(
            sub, "freq_group", grp_order, FREQ_GROUP_COLORS,
            "duration", "event",
            f"Phase 3 — KM by frequency group ({field})",
            out / f"p3_km_freq_{field}.png",
            reference_group="Normal (50–55 Hz)",
        )
        stats_f["scope"] = field; stats_f["subset"] = "all"
        all_stats.append(stats_f)

    # ------------------------------------------------------------------
    # 3.4  By failure category — Global and Vt
    # ------------------------------------------------------------------
    for scope, sub in [("global", df), ("vt", df[df["is_vt"]])]:
        sub_ev = sub[sub["event"] == 1]
        viable_cats = [
            c for c in FAILURE_CATEGORIES
            if (sub_ev["failure_category"] == c).sum() >= MIN_FAILURES_KM
        ]
        if not viable_cats:
            continue
        stats_c = _plot_km_groups(
            sub[sub["failure_category"].isin(viable_cats)],
            "failure_category", viable_cats, CATEGORY_COLORS,
            "duration", "event",
            f"Phase 3 — KM by failure category ({scope})",
            out / f"p3_km_category_{scope}.png",
        )
        stats_c["scope"] = scope; stats_c["subset"] = "by_category"
        all_stats.append(stats_c)

    # ------------------------------------------------------------------
    # 3.5  Vt vs Global — primary visual for executive summary
    # ------------------------------------------------------------------
    df["vt_vs_global"] = np.where(df["is_vt"], "Vt", "Global (excl. Vt)")
    fig, ax = plt.subplots(figsize=(10, 6))
    for grp, color in [("Vt", "#D65F5F"), ("Global (excl. Vt)", "#4878CF")]:
        sub = df[df["vt_vs_global"] == grp].dropna(subset=["duration"])
        T = sub["duration"].values.astype(float)
        E = sub["event"].values.astype(float)
        if E.sum() < MIN_FAILURES_KM:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(T, E, label=f"{grp} (n={len(T)}, ev={int(E.sum())})")
        kmf.plot_survival_function(ax=ax, color=color, linewidth=2.5, ci_show=True, ci_alpha=0.15)

    # Log-rank test
    t_vt = df[df["is_vt"]]["duration"].dropna().values.astype(float)
    e_vt = df[df["is_vt"]].loc[df[df["is_vt"]]["duration"].notna(), "event"].values.astype(float)
    t_gl = df[~df["is_vt"]]["duration"].dropna().values.astype(float)
    e_gl = df[~df["is_vt"]].loc[df[~df["is_vt"]]["duration"].notna(), "event"].values.astype(float)
    try:
        lr = logrank_test(t_vt, t_gl, e_vt, e_gl)
        ax.text(0.98, 0.55, f"Log-rank p = {lr.p_value:.4f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
        results["vt_vs_global_logrank_p"] = float(lr.p_value)
    except Exception:
        pass

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival probability S(t)")
    ax.set_title("Phase 3 — Vt vs Global: Kaplan-Meier survival (PRIMARY VISUAL)")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "p3_km_vt_vs_global.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save combined stats
    km_stats_df = pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()
    km_stats_df.to_csv(out / "p3_km_stats.csv", index=False, encoding="utf-8-sig")
    results["km_stats"] = km_stats_df

    # ------------------------------------------------------------------
    # Print
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 3 — KAPLAN-MEIER")
    print("=" * 60)
    if not km_stats_df.empty:
        print(km_stats_df[["scope", "subset", "group", "n_events",
                            "median_survival"]].to_string(index=False))
    vt_lr = results.get("vt_vs_global_logrank_p")
    if vt_lr is not None:
        print(f"\nVt vs Global log-rank p = {vt_lr:.4f}")

    return results
