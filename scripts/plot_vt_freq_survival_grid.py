"""
Build all combinations of:
  infant_mortality_days: 30, 60, 90
  mount_year_min: 2023, 2024, 2025  (i.e. year >= cutoff)

For each combination:
  - filter, bin by freq>55 share, compute KM curves
  - save per-combination CSV

Outputs:
  - 3x3 grid KM plot
  - summary_stats.csv  (one row per combination x freq-bin)
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from analysis.paths import results_dir, RESULTS_ROOT

_SLUG = "vt_freq_survival_grid"
OUTPUT_DIR = results_dir(_SLUG)

_LEGACY_INPUT = REPO_ROOT / "analysis_outputs" / "vt_ttf_freq55" / "vt_ttf_freq55_data.csv"


def _resolve_input_csv() -> Path:
    matches = sorted(RESULTS_ROOT.glob("vt_ttf_vs_freq55/*/vt_ttf_freq55_data.csv"), key=lambda p: p.stat().st_mtime)
    if matches:
        return matches[-1]
    if _LEGACY_INPUT.exists():
        return _LEGACY_INPUT
    raise FileNotFoundError("vt_ttf_freq55_data.csv not found; run plot_vt_ttf_vs_freq55.py first")


INPUT_CSV = _resolve_input_csv()

INFANT_THRESHOLDS = [30, 60, 90]
YEAR_CUTOFFS = [2023, 2024, 2025]

BIN_LABELS = ["y = 0", "0 < y ≤ 0.2", "0.2 < y < 0.5", "y ≥ 0.5"]
BIN_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#DD8452"]
MIN_EVENTS_FOR_KM = 5


def assign_bin(frac: float) -> str:
    if frac == 0.0:
        return BIN_LABELS[0]
    elif frac <= 0.2:
        return BIN_LABELS[1]
    elif frac < 0.5:
        return BIN_LABELS[2]
    else:
        return BIN_LABELS[3]


def kaplan_meier(durations: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(durations)
    t = durations[order]
    e = events[order]
    unique_times = np.unique(t[e == 1])
    if len(unique_times) == 0:
        return np.array([0.0]), np.array([1.0])
    times = np.concatenate([[0.0], unique_times])
    surv = np.ones(len(times))
    n = len(t)
    s = 1.0
    for i, ti in enumerate(unique_times):
        at_risk = int(np.sum(t >= ti))
        d = int(np.sum((t == ti) & (e == 1)))
        if at_risk > 0:
            s *= 1.0 - d / at_risk
        surv[i + 1] = s
    return times, surv


def km_ci(durations: np.ndarray, events: np.ndarray, t_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(durations)
    t = durations[order]
    e = events[order]
    unique_times = np.unique(t[e == 1])
    s = 1.0
    var = 0.0
    surv_vals, var_vals, t_vals = [1.0], [0.0], [0.0]
    for ti in unique_times:
        at_risk = int(np.sum(t >= ti))
        d = int(np.sum((t == ti) & (e == 1)))
        if at_risk > 0 and d > 0:
            s *= 1.0 - d / at_risk
            if at_risk > d:
                var += d / (at_risk * (at_risk - d))
        surv_vals.append(s)
        var_vals.append(var)
        t_vals.append(ti)
    s_arr = np.interp(t_grid, t_vals, surv_vals)
    v_arr = np.interp(t_grid, t_vals, var_vals)
    with np.errstate(invalid="ignore", divide="ignore"):
        log_s = np.log(np.where(s_arr > 0, s_arr, np.nan))
        log_log_s = np.log(-log_s)
        delta = 1.96 * np.sqrt(v_arr) / np.abs(log_s + 1e-30)
        lo = np.clip(np.exp(-np.exp(log_log_s + delta)), 0, 1)
        hi = np.clip(np.exp(-np.exp(log_log_s - delta)), 0, 1)
    return lo, hi


def rmst(durations: np.ndarray, events: np.ndarray, tau: float | None = None) -> float:
    """Restricted Mean Survival Time = area under the KM curve up to tau."""
    km_t, km_s = kaplan_meier(durations, events)
    if tau is None:
        tau = float(km_t[-1])
    t_clip = np.append(km_t[km_t <= tau], tau)
    s_clip = np.interp(t_clip, km_t, km_s)
    return float(np.trapezoid(s_clip, t_clip))


def build_combo(df: pd.DataFrame, year_min: int, infant_days: int) -> pd.DataFrame:
    sub = df.loc[
        (df["mount_year"] >= year_min) &
        (df["ttf_days"] >= infant_days) &
        df["frac_freq_above_55"].notna()
    ].copy()
    sub["freq_bin"] = sub["frac_freq_above_55"].apply(assign_bin)
    return sub


def combo_stats(df: pd.DataFrame, year_min: int, infant_days: int) -> list[dict]:
    # Shared tau = min of last observed event time across all freq-bin groups
    # so RMST values are comparable across groups.
    group_taus = []
    for label in BIN_LABELS:
        sub = df.loc[df["freq_bin"] == label]
        if sub["event"].eq(1).sum() >= MIN_EVENTS_FOR_KM:
            group_taus.append(float(sub.loc[sub["event"] == 1, "ttf_days"].max()))
    shared_tau = min(group_taus) if group_taus else None

    rows = []
    for label in BIN_LABELS:
        sub = df.loc[df["freq_bin"] == label]
        n_total = len(sub)
        n_fail = int(sub["event"].eq(1).sum())
        n_cens = int(sub["event"].eq(0).sum())
        med_ttf = sub["ttf_days"].median() if n_total > 0 else float("nan")
        med_fail = sub.loc[sub["event"] == 1, "ttf_days"].median() if n_fail > 0 else float("nan")
        mean_ttf = sub["ttf_days"].mean() if n_total > 0 else float("nan")
        mean_fail = sub.loc[sub["event"] == 1, "ttf_days"].mean() if n_fail > 0 else float("nan")
        rmst_val = float("nan")
        if n_fail >= MIN_EVENTS_FOR_KM and shared_tau is not None:
            rmst_val = rmst(
                sub["ttf_days"].to_numpy(dtype=float),
                sub["event"].to_numpy(dtype=float),
                tau=shared_tau,
            )
        rows.append({
            "year_min": year_min,
            "infant_excl_days": infant_days,
            "freq_bin": label,
            "n_total": n_total,
            "n_failures": n_fail,
            "n_censored": n_cens,
            "median_ttf_all": round(med_ttf, 1) if not np.isnan(med_ttf) else None,
            "median_ttf_failures": round(med_fail, 1) if not np.isnan(med_fail) else None,
            "mean_ttf_all": round(mean_ttf, 1) if not np.isnan(mean_ttf) else None,
            "mean_ttf_failures": round(mean_fail, 1) if not np.isnan(mean_fail) else None,
            "rmst_days": round(rmst_val, 1) if not np.isnan(rmst_val) else None,
            "rmst_tau_days": round(shared_tau, 0) if shared_tau is not None else None,
        })
    return rows


def draw_km_panel(ax: plt.Axes, df: pd.DataFrame, year_min: int, infant_days: int) -> None:
    t_max = max(df["ttf_days"].quantile(0.95), 100.0)
    t_grid = np.linspace(0, t_max, 400)

    any_drawn = False
    for label, color in zip(BIN_LABELS, BIN_COLORS):
        sub = df.loc[df["freq_bin"] == label]
        n_fail = int(sub["event"].eq(1).sum())
        n_total = len(sub)
        if n_fail < MIN_EVENTS_FOR_KM:
            continue
        t = sub["ttf_days"].to_numpy(dtype=float)
        e = sub["event"].to_numpy(dtype=float)
        km_t, km_s = kaplan_meier(t, e)
        s_grid = np.interp(t_grid, km_t, km_s)
        lo, hi = km_ci(t, e, t_grid)
        ax.step(t_grid, s_grid, where="post", color=color, linewidth=1.6,
                label=f"{label} (n={n_total}, {n_fail}✕)")
        ax.fill_between(t_grid, lo, hi, alpha=0.10, color=color, step="post")
        any_drawn = True

    ax.axhline(0.5, color="#bbb", linewidth=0.6, linestyle="--")
    ax.set_xlim(0, t_max)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.2, linewidth=0.4)
    n_total = len(df)
    n_fail = int(df["event"].eq(1).sum())
    ax.set_title(
        f"year ≥ {year_min},  excl. < {infant_days}d\n"
        f"n={n_total} ({n_fail} failures)",
        fontsize=8.5,
        pad=3,
    )
    if not any_drawn:
        ax.text(0.5, 0.5, "insufficient data", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="#999")


def main() -> None:
    df_all = pd.read_csv(INPUT_CSV)
    print(f"Total rows in CSV: {len(df_all)}")

    all_stats: list[dict] = []

    # --- 3x3 grid figure ---
    fig, axes = plt.subplots(
        len(YEAR_CUTOFFS), len(INFANT_THRESHOLDS),
        figsize=(14, 12),
        sharey=True,
    )

    for row_idx, year_min in enumerate(YEAR_CUTOFFS):
        for col_idx, infant_days in enumerate(INFANT_THRESHOLDS):
            combo = build_combo(df_all, year_min, infant_days)
            stats = combo_stats(combo, year_min, infant_days)
            all_stats.extend(stats)

            tag = f"year{year_min}_infant{infant_days}d"
            combo.to_csv(OUTPUT_DIR / f"{tag}.csv", index=False, encoding="utf-8-sig")

            ax = axes[row_idx][col_idx]
            draw_km_panel(ax, combo, year_min, infant_days)

            if row_idx == len(YEAR_CUTOFFS) - 1:
                ax.set_xlabel("Days", fontsize=8)
            if col_idx == 0:
                ax.set_ylabel("S(t)", fontsize=8)
            ax.tick_params(labelsize=7)

    # shared legend from first panel that has lines
    handles, labels = None, None
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break

    # build legend from bin colors (consistent across panels)
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=c, linewidth=2, label=lbl)
        for lbl, c in zip(BIN_LABELS, BIN_COLORS)
    ]
    fig.legend(
        handles=legend_handles,
        title="freq > 55 Hz share",
        loc="lower center",
        ncol=4,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.01),
        framealpha=0.9,
    )

    # column headers
    for col_idx, infant_days in enumerate(INFANT_THRESHOLDS):
        axes[0][col_idx].set_title(
            f"Infant excl. < {infant_days}d\n" + axes[0][col_idx].get_title(),
            fontsize=8.5, pad=3,
        )

    fig.suptitle(
        "Vt field — KM survival curves by freq > 55 Hz exposure\n"
        "Rows: mount year cutoff   |   Columns: infant mortality exclusion threshold",
        fontsize=11, y=1.01,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    grid_path = OUTPUT_DIR / "vt_km_grid_all_combinations.png"
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Grid plot saved: {grid_path}")

    # --- summary stats CSV ---
    stats_df = pd.DataFrame(all_stats)
    stats_path = OUTPUT_DIR / "summary_stats.csv"
    stats_df.to_csv(stats_path, index=False, encoding="utf-8-sig")
    print(f"Summary stats saved: {stats_path}")

    # print table
    print("\nSample sizes across combinations:")
    pivot = stats_df.groupby(["year_min", "infant_excl_days"])[["n_total", "n_failures"]].sum().reset_index()
    print(pivot.to_string(index=False))

    print("\nMedian TTF — failures only (days):")
    wide_med = stats_df.pivot_table(
        index=["year_min", "infant_excl_days"],
        columns="freq_bin",
        values="median_ttf_failures",
        aggfunc="first",
    )[BIN_LABELS]
    print(wide_med.to_string())

    print("\nMean TTF — failures only, arithmetic (days):")
    wide_mean = stats_df.pivot_table(
        index=["year_min", "infant_excl_days"],
        columns="freq_bin",
        values="mean_ttf_failures",
        aggfunc="first",
    )[BIN_LABELS]
    print(wide_mean.to_string())

    print("\nRMST — restricted mean survival time from KM curve (days, censoring-corrected):")
    wide_rmst = stats_df.pivot_table(
        index=["year_min", "infant_excl_days"],
        columns="freq_bin",
        values="rmst_days",
        aggfunc="first",
    )[BIN_LABELS]
    print(wide_rmst.to_string())


if __name__ == "__main__":
    main()
