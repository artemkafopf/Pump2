"""
For Vt field, mount years 2023-2026:
  - Bin runs by share of time with freq > 55 Hz into 4 groups
  - Panel 1: TTF histogram for failure events per group
  - Panel 2: Kaplan-Meier survival curves per group
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from analysis.paths import results_dir, RESULTS_ROOT

_SLUG = "vt_freq_survival"
OUTPUT_DIR = results_dir(_SLUG)

_LEGACY_INPUT = REPO_ROOT / "analysis_outputs" / "vt_ttf_freq55" / "vt_ttf_freq55_data.csv"


def _resolve_input_csv(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit)
    matches = sorted(RESULTS_ROOT.glob("vt_ttf_vs_freq55/*/vt_ttf_freq55_data.csv"), key=lambda p: p.stat().st_mtime)
    if matches:
        return matches[-1]
    if _LEGACY_INPUT.exists():
        return _LEGACY_INPUT
    raise FileNotFoundError("vt_ttf_freq55_data.csv not found; run plot_vt_ttf_vs_freq55.py first")


INPUT_CSV = _resolve_input_csv()

MOUNT_YEARS = {2023, 2024, 2025, 2026}
INFANT_MORTALITY_DAYS = 30

BIN_LABELS = ["y = 0", "0 < y ≤ 0.2", "0.2 < y < 0.5", "y ≥ 0.5"]
BIN_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#DD8452"]


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
    """Return (times, survival) step-function arrays for KM estimator."""
    order = np.argsort(durations)
    t = durations[order]
    e = events[order]

    unique_times = np.unique(t[e == 1])
    times = np.concatenate([[0.0], unique_times])
    surv = np.ones(len(times))

    n = len(t)
    s = 1.0
    for i, ti in enumerate(unique_times):
        at_risk = int(np.sum(t >= ti))
        events_at = int(np.sum((t == ti) & (e == 1)))
        if at_risk > 0:
            s *= 1.0 - events_at / at_risk
        surv[i + 1] = s

    return times, surv


def km_confidence_band(
    durations: np.ndarray,
    events: np.ndarray,
    times_grid: np.ndarray,
    alpha: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Greenwood variance → 95% log-log CI evaluated on times_grid."""
    order = np.argsort(durations)
    t = durations[order]
    e = events[order]
    unique_times = np.unique(t[e == 1])

    s = 1.0
    log_log_var = 0.0
    surv_vals: list[float] = [1.0]
    var_vals: list[float] = [0.0]
    t_vals: list[float] = [0.0]

    for ti in unique_times:
        at_risk = int(np.sum(t >= ti))
        d = int(np.sum((t == ti) & (e == 1)))
        if at_risk > 0 and d > 0:
            s *= 1.0 - d / at_risk
            if s > 0:
                log_log_var += d / (at_risk * (at_risk - d)) if at_risk > d else 0.0
        surv_vals.append(s)
        var_vals.append(log_log_var)
        t_vals.append(ti)

    surv_arr = np.array(surv_vals)
    var_arr = np.array(var_vals)
    t_arr = np.array(t_vals)

    # interpolate onto grid
    s_grid = np.interp(times_grid, t_arr, surv_arr)
    v_grid = np.interp(times_grid, t_arr, var_arr)

    z = 1.96  # 95%
    with np.errstate(invalid="ignore", divide="ignore"):
        log_s = np.log(np.where(s_grid > 0, s_grid, np.nan))
        log_log_s = np.log(-log_s)
        delta = z * np.sqrt(v_grid) / np.abs(log_s + 1e-30)
        lo = np.exp(-np.exp(log_log_s + delta))
        hi = np.exp(-np.exp(log_log_s - delta))

    lo = np.clip(lo, 0.0, 1.0)
    hi = np.clip(hi, 0.0, 1.0)
    return lo, hi


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    df = df.loc[df["mount_year"].isin(MOUNT_YEARS)].copy()
    df = df.loc[df["frac_freq_above_55"].notna()].copy()
    before = len(df)
    df = df.loc[df["ttf_days"] >= INFANT_MORTALITY_DAYS].copy()
    print(f"Dropped infant mortality (< {INFANT_MORTALITY_DAYS}d): {before - len(df)} runs")
    df["freq_bin"] = df["frac_freq_above_55"].apply(assign_bin)

    print(f"Runs 2023-2026 after infant exclusion, with freq coverage: {len(df)}")
    for label in BIN_LABELS:
        sub = df.loc[df["freq_bin"] == label]
        f = int(sub["event"].eq(1).sum())
        c = int(sub["event"].eq(0).sum())
        print(f"  {label}: {f} failures, {c} censored  (median TTF: {sub['ttf_days'].median():.0f}d)")

    fig, (ax_hist, ax_km) = plt.subplots(2, 1, figsize=(11, 12))

    # --- Panel 1: TTF histogram for failure events ---
    max_ttf = df.loc[df["event"] == 1, "ttf_days"].max()
    bins = np.arange(0, min(max_ttf + 30, 800), 30)

    for label, color in zip(BIN_LABELS, BIN_COLORS):
        sub = df.loc[(df["freq_bin"] == label) & (df["event"] == 1), "ttf_days"]
        n = len(sub)
        if n == 0:
            continue
        ax_hist.hist(
            sub,
            bins=bins,
            alpha=0.55,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            label=f"{label}  (n={n})",
            density=False,
        )

    ax_hist.set_xlabel("TTF (days)", fontsize=11)
    ax_hist.set_ylabel("Failure count", fontsize=11)
    ax_hist.set_title(
        f"Vt field (2023-2026) — TTF distribution of failure events\nby share of run time with freq > 55 Hz  (runs < {INFANT_MORTALITY_DAYS}d excluded)",
        fontsize=11,
    )
    ax_hist.legend(title="freq > 55 Hz share", fontsize=9)
    ax_hist.grid(True, alpha=0.25, axis="y")

    # --- Panel 2: KM survival curves ---
    t_max = df["ttf_days"].quantile(0.97)
    t_grid = np.linspace(0, t_max, 500)

    for label, color in zip(BIN_LABELS, BIN_COLORS):
        sub = df.loc[df["freq_bin"] == label].copy()
        if len(sub) < 5:
            continue
        t = sub["ttf_days"].to_numpy(dtype=float)
        e = sub["event"].to_numpy(dtype=float)
        n_total = len(sub)
        n_fail = int(e.sum())

        km_t, km_s = kaplan_meier(t, e)
        s_grid = np.interp(t_grid, km_t, km_s)

        lo, hi = km_confidence_band(t, e, t_grid)

        ax_km.step(t_grid, s_grid, where="post", color=color, linewidth=2,
                   label=f"{label}  (n={n_total}, {n_fail} events)")
        ax_km.fill_between(t_grid, lo, hi, alpha=0.12, color=color, step="post")

    ax_km.set_xlabel("Time (days)", fontsize=11)
    ax_km.set_ylabel("Survival probability S(t)", fontsize=11)
    ax_km.set_title(
        f"Kaplan-Meier survival curves by freq > 55 Hz exposure\n(runs < {INFANT_MORTALITY_DAYS}d excluded; shaded: 95% log-log CI)",
        fontsize=11,
    )
    ax_km.set_ylim(-0.02, 1.05)
    ax_km.set_xlim(0, t_max)
    ax_km.axhline(0.5, color="#aaa", linewidth=0.7, linestyle="--", alpha=0.6)
    ax_km.legend(title="freq > 55 Hz share", fontsize=9, loc="upper right")
    ax_km.grid(True, alpha=0.25)

    fig.tight_layout(pad=3.0)
    out_path = OUTPUT_DIR / "vt_freq55_histograms_km_2023_2026_excl_infant.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")


if __name__ == "__main__":
    main()
