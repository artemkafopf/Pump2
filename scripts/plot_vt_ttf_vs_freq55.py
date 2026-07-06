"""
Scatter plot: TTF (x) vs share of run time with freq > 55 Hz (y).
Points colored by install (mount) year. Failure events marked with X.
Vt field only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
for p in (str(REPO_ROOT), str(BACKEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from analysis.input_paths import resolve_v03_all_path
from analysis.paths import results_dir
from scripts.analyze_failure_horizon import load_runs
from scripts.data_utils import load_daily_merged

FREQ_THRESHOLD_HZ = 55.0
FIELD_CODE = "Vt"
_SLUG = "vt_ttf_vs_freq55"
OUTPUT_DIR = results_dir(_SLUG)

def compute_freq_share(runs: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["well_key"] = daily["well_id"].str.strip().str.casefold()

    records = []
    for _, run in runs.iterrows():
        well_key = str(run["Скв."]).strip().casefold()
        install = pd.Timestamp(run["Дата монтажа"])
        stop = pd.Timestamp(run["Дата остановки"])

        mask = (
            (daily["well_key"] == well_key)
            & (daily["dt"] >= install)
            & (daily["dt"] <= stop)
        )
        run_daily = daily.loc[mask].copy()

        freq_series = pd.to_numeric(run_daily["freq"], errors="coerce")
        valid = freq_series.notna()
        n_valid = int(valid.sum())
        n_above = int((freq_series > FREQ_THRESHOLD_HZ).sum()) if n_valid > 0 else 0
        frac_above = n_above / n_valid if n_valid >= 7 else float("nan")

        records.append({
            "row_id": run["row_id"],
            "well_id": str(run["Скв."]),
            "ttf_days": float(run["run_days"]),
            "event": int(run["event"]),
            "mount_year": int(install.year),
            "frac_freq_above_55": frac_above,
            "n_freq_days": n_valid,
        })

    return pd.DataFrame(records)

def main() -> None:
    runs = load_runs(resolve_v03_all_path())
    vt = runs.loc[runs["Месторождение"].astype("string").str.strip() == FIELD_CODE].copy()
    print(f"Vt runs: {len(vt)}  (failures: {int(vt['event'].eq(1).sum())}, censored: {int(vt['event'].eq(0).sum())})")

    wells = vt["Скв."].dropna().astype(str).unique().tolist()
    print(f"Loading daily data for {len(wells)} wells …")
    daily = load_daily_merged(wells)
    print(f"Daily rows loaded: {len(daily)}")

    stats = compute_freq_share(vt, daily)
    usable = stats.loc[stats["frac_freq_above_55"].notna()].copy()
    missing = len(stats) - len(usable)
    print(f"Runs with freq coverage (≥7 days): {len(usable)}  dropped (no freq data): {missing}")

    stats.to_csv(OUTPUT_DIR / "vt_ttf_freq55_data.csv", index=False, encoding="utf-8-sig")

    # --- color map by mount year ---
    years = sorted(usable["mount_year"].unique())
    year_min, year_max = min(years), max(years)
    cmap = cm.get_cmap("tab20" if len(years) <= 20 else "turbo", len(years))
    year_to_color = {yr: cmap(i) for i, yr in enumerate(years)}

    failures = usable.loc[usable["event"] == 1].copy()
    censored = usable.loc[usable["event"] == 0].copy()

    fig, ax = plt.subplots(figsize=(13, 8))

    # censored: small circles, per-year color, low alpha
    for yr in years:
        sub = censored.loc[censored["mount_year"] == yr]
        if sub.empty:
            continue
        ax.scatter(
            sub["ttf_days"],
            sub["frac_freq_above_55"],
            s=28,
            alpha=0.45,
            color=year_to_color[yr],
            marker="o",
            zorder=2,
        )

    # failures: large X markers, per-year color, high alpha, black edge
    for yr in years:
        sub = failures.loc[failures["mount_year"] == yr]
        if sub.empty:
            continue
        ax.scatter(
            sub["ttf_days"],
            sub["frac_freq_above_55"],
            s=90,
            alpha=0.9,
            color=year_to_color[yr],
            marker="X",
            edgecolors="black",
            linewidths=0.5,
            zorder=4,
            label=f"_nolegend_",
        )

    # --- legend: year colors + marker type ---
    year_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=year_to_color[yr],
                   markersize=8, label=str(yr))
        for yr in years
    ]
    type_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#888",
                   markersize=8, alpha=0.5, label=f"Censored (n={len(censored)})"),
        plt.Line2D([0], [0], marker="X", color="w", markerfacecolor="#888",
                   markersize=10, markeredgecolor="black", markeredgewidth=0.5,
                   label=f"Failure event (n={len(failures)})"),
    ]
    legend1 = ax.legend(handles=type_handles, loc="upper right", fontsize=9, title="Run outcome")
    ax.add_artist(legend1)
    ax.legend(handles=year_handles, loc="upper left", fontsize=8, title="Mount year",
              ncol=2, framealpha=0.8)

    # horizontal reference lines
    ax.axhline(0.0, color="#ccc", linewidth=0.7, linestyle="--", zorder=1)
    ax.axhline(1.0, color="#ccc", linewidth=0.7, linestyle="--", zorder=1)

    # annotate longest-running failures
    for _, row in failures.nlargest(6, "ttf_days").iterrows():
        ax.annotate(
            f"{int(row['ttf_days'])}d",
            xy=(row["ttf_days"], row["frac_freq_above_55"]),
            xytext=(6, 3),
            textcoords="offset points",
            fontsize=6.5,
            color="#333",
            alpha=0.85,
        )

    ax.set_xlabel("Run duration / TTF (days)", fontsize=12)
    ax.set_ylabel(f"Share of run time with freq > {FREQ_THRESHOLD_HZ:.0f} Hz", fontsize=12)
    ax.set_title(
        f"Vt field — TTF vs high-frequency exposure, colored by install year\n"
        f"(freq > {FREQ_THRESHOLD_HZ:.0f} Hz share over full run; ✕ = failure event)",
        fontsize=12,
    )
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.25, linewidth=0.5)

    fig.tight_layout()
    out_path = OUTPUT_DIR / "vt_ttf_vs_freq55_share.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {out_path}")

    # --- summary stats ---
    print("\n--- Summary ---")
    print(f"Failures  — median TTF: {failures['ttf_days'].median():.0f}d  |  mean freq>55 share: {failures['frac_freq_above_55'].mean():.3f}")
    print(f"Censored  — median TTF: {censored['ttf_days'].median():.0f}d  |  mean freq>55 share: {censored['frac_freq_above_55'].mean():.3f}")
    print(f"\nCorrelation (TTF vs frac_freq_above_55):")
    for label, subset in [("All", usable), ("Failures only", failures)]:
        r = subset[["ttf_days", "frac_freq_above_55"]].dropna().corr().iloc[0, 1]
        print(f"  {label}: r = {r:.3f}")
    print("\nRuns per mount year:")
    for yr, grp in usable.groupby("mount_year"):
        f = int(grp["event"].eq(1).sum())
        c = int(grp["event"].eq(0).sum())
        print(f"  {yr}: {f} failures, {c} censored")

if __name__ == "__main__":
    main()
