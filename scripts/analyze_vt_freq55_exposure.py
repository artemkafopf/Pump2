"""
Vt frequency-exposure analysis.

Builds:
1. Scatter dataset and plot for Vt runs with:
   - x = TTF
   - y = share of run time with freq > 55 Hz
   - failure events marked explicitly
   - points colored by mount year
   - source limited to mount years 2023-2026
2. For all combinations of:
   - infant mortality exclusion: 30, 60, 90 days
   - mount year cutoff: >2023, >2024, >2025
   build:
   - per-combination dataset CSV
   - failure-event histogram by freq-exposure bin
   - Kaplan-Meier survival curve by freq-exposure bin
   - summary CSV
"""
from __future__ import annotations

import json
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
BACKEND_DIR = REPO_ROOT / "backend"
for path_text in (str(REPO_ROOT), str(BACKEND_DIR)):
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from analysis.input_paths import resolve_v03_all_path
from analysis.paths import results_dir
from scripts.analyze_failure_horizon import load_runs
from scripts.data_utils import load_daily_merged


FIELD_CODE = "Vt"
FREQ_THRESHOLD_HZ = 55.0
MOUNT_YEARS = {2023, 2024, 2025, 2026}
YEAR_CUTOFFS_EXCLUSIVE = [2023, 2024, 2025]
INFANT_THRESHOLDS = [30, 60, 90]
MIN_FREQ_DAYS = 7
MIN_ROWS_PER_KM_BIN = 5

BIN_LABELS = ["y = 0", "0 < y <= 0.2", "0.2 < y < 0.5", "y > 0.5"]
BIN_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#DD8452"]

_SLUG = "vt_freq55_exposure"
OUTPUT_DIR = results_dir(_SLUG)
COMBO_DIR = OUTPUT_DIR / "combinations"
PLOT_DIR = OUTPUT_DIR / "plots"
for directory in (OUTPUT_DIR, COMBO_DIR, PLOT_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def assign_bin(frac: float) -> str:
    if pd.isna(frac):
        return "<missing>"
    if frac == 0.0:
        return BIN_LABELS[0]
    if 0.0 < frac <= 0.2:
        return BIN_LABELS[1]
    if 0.2 < frac < 0.5:
        return BIN_LABELS[2]
    return BIN_LABELS[3]


def kaplan_meier(durations: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(durations)
    durations = durations[order]
    events = events[order]
    event_times = np.unique(durations[events == 1])
    if len(event_times) == 0:
        return np.array([0.0]), np.array([1.0])

    times = np.concatenate([[0.0], event_times])
    survival = np.ones(len(times), dtype=float)
    current = 1.0
    for index, current_time in enumerate(event_times, start=1):
        at_risk = int(np.sum(durations >= current_time))
        n_events = int(np.sum((durations == current_time) & (events == 1)))
        if at_risk > 0:
            current *= 1.0 - (n_events / at_risk)
        survival[index] = current
    return times, survival


def km_confidence_band(durations: np.ndarray, events: np.ndarray, time_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(durations)
    durations = durations[order]
    events = events[order]
    event_times = np.unique(durations[events == 1])
    if len(event_times) == 0:
        one = np.ones(len(time_grid), dtype=float)
        return one, one

    current_survival = 1.0
    current_var = 0.0
    survival_values = [1.0]
    variance_values = [0.0]
    time_values = [0.0]

    for current_time in event_times:
        at_risk = int(np.sum(durations >= current_time))
        n_events = int(np.sum((durations == current_time) & (events == 1)))
        if at_risk > 0 and n_events > 0:
            current_survival *= 1.0 - (n_events / at_risk)
            if at_risk > n_events:
                current_var += n_events / (at_risk * (at_risk - n_events))
        survival_values.append(current_survival)
        variance_values.append(current_var)
        time_values.append(current_time)

    s_grid = np.interp(time_grid, time_values, survival_values)
    v_grid = np.interp(time_grid, time_values, variance_values)
    with np.errstate(invalid="ignore", divide="ignore"):
        log_s = np.log(np.where(s_grid > 0, s_grid, np.nan))
        log_log_s = np.log(-log_s)
        delta = 1.96 * np.sqrt(v_grid) / np.abs(log_s + 1e-30)
        lower = np.clip(np.exp(-np.exp(log_log_s + delta)), 0.0, 1.0)
        upper = np.clip(np.exp(-np.exp(log_log_s - delta)), 0.0, 1.0)
    return lower, upper


def compute_freq_share(runs: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    working_daily = daily.copy()
    working_daily["well_key"] = working_daily["well_id"].astype("string").str.strip().str.casefold()

    rows: list[dict[str, object]] = []
    for _, run in runs.iterrows():
        install = pd.Timestamp(run["Дата монтажа"])
        stop = pd.Timestamp(run["Дата остановки"])
        well_key = str(run["Скв."]).strip().casefold()
        run_daily = working_daily.loc[
            (working_daily["well_key"] == well_key)
            & (working_daily["dt"] >= install)
            & (working_daily["dt"] <= stop)
        ].copy()

        freq = pd.to_numeric(run_daily["freq"], errors="coerce")
        valid_mask = freq.notna()
        n_valid = int(valid_mask.sum())
        n_above = int((freq > FREQ_THRESHOLD_HZ).sum()) if n_valid > 0 else 0
        frac_above = (n_above / n_valid) if n_valid >= MIN_FREQ_DAYS else float("nan")

        rows.append(
            {
                "row_id": run["row_id"],
                "well_id": str(run["Скв."]).strip(),
                "field": run.get("Месторождение"),
                "contractor": run.get("Принадлежность"),
                "mount_date": install,
                "stop_date": stop,
                "mount_year": int(install.year),
                "ttf_days": float(run["run_days"]),
                "event": int(run["event"]),
                "frac_freq_above_55": frac_above,
                "n_freq_days": n_valid,
                "n_freq_days_above_55": n_above,
            }
        )
    result = pd.DataFrame(rows)
    result["freq_bin"] = result["frac_freq_above_55"].apply(assign_bin)
    return result


def build_scatter_plot(df: pd.DataFrame, output_path: Path) -> None:
    usable = df.loc[df["frac_freq_above_55"].notna()].copy()
    failures = usable.loc[usable["event"] == 1].copy()
    censored = usable.loc[usable["event"] == 0].copy()

    years = sorted(int(year) for year in usable["mount_year"].dropna().unique().tolist())
    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(max(len(years), 1))
    year_to_color = {year: cmap(index) for index, year in enumerate(years)}

    fig, ax = plt.subplots(figsize=(13, 8))
    for year in years:
        sub = censored.loc[censored["mount_year"] == year]
        if not sub.empty:
            ax.scatter(
                sub["ttf_days"],
                sub["frac_freq_above_55"],
                s=26,
                alpha=0.45,
                color=year_to_color[year],
                marker="o",
                zorder=2,
            )
    for year in years:
        sub = failures.loc[failures["mount_year"] == year]
        if not sub.empty:
            ax.scatter(
                sub["ttf_days"],
                sub["frac_freq_above_55"],
                s=92,
                alpha=0.9,
                color=year_to_color[year],
                marker="X",
                edgecolors="black",
                linewidths=0.5,
                zorder=4,
            )

    type_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#888", markersize=8, alpha=0.5, label=f"Censored (n={len(censored)})"),
        plt.Line2D([0], [0], marker="X", color="w", markerfacecolor="#888", markersize=10, markeredgecolor="black", markeredgewidth=0.5, label=f"Failure event (n={len(failures)})"),
    ]
    year_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=year_to_color[year], markersize=8, label=str(year))
        for year in years
    ]
    legend1 = ax.legend(handles=type_handles, loc="upper right", fontsize=9, title="Run outcome")
    ax.add_artist(legend1)
    ax.legend(handles=year_handles, loc="upper left", fontsize=8, title="Mount year", ncol=2, framealpha=0.85)

    ax.set_xlabel("Run duration / TTF (days)")
    ax.set_ylabel("Share of run time with freq > 55 Hz")
    ax.set_title("Vt field (mount years 2023-2026) — TTF vs share of time with freq > 55 Hz")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(left=0.0)
    ax.grid(True, alpha=0.25, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_combo_dataset(df: pd.DataFrame, year_cutoff_exclusive: int, infant_days: int) -> pd.DataFrame:
    result = df.loc[
        (df["mount_year"] > year_cutoff_exclusive)
        & (df["mount_year"].isin(MOUNT_YEARS))
        & (df["ttf_days"] >= infant_days)
        & (df["frac_freq_above_55"].notna())
    ].copy()
    result["freq_bin"] = result["frac_freq_above_55"].apply(assign_bin)
    return result


def combo_tag(year_cutoff_exclusive: int, infant_days: int) -> str:
    return f"year_gt_{year_cutoff_exclusive}_infant_{infant_days}d"


def build_histogram_and_km(df: pd.DataFrame, title_prefix: str, output_path: Path) -> None:
    fig, (ax_hist, ax_km) = plt.subplots(2, 1, figsize=(11, 12))

    failure_ttf = df.loc[df["event"] == 1, "ttf_days"]
    max_ttf = float(failure_ttf.max()) if not failure_ttf.empty else 180.0
    bins = np.arange(0.0, max(210.0, min(max_ttf + 30.0, 1800.0)), 30.0)

    for label, color in zip(BIN_LABELS, BIN_COLORS):
        sub = df.loc[(df["freq_bin"] == label) & (df["event"] == 1), "ttf_days"]
        if sub.empty:
            continue
        ax_hist.hist(
            sub,
            bins=bins,
            alpha=0.55,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            density=False,
            label=f"{label} (n={len(sub)})",
        )
    ax_hist.set_xlabel("TTF (days)")
    ax_hist.set_ylabel("Failure count")
    ax_hist.set_title(f"{title_prefix}\nFailure-event TTF histograms by freq>55 share bin")
    hist_handles, hist_labels = ax_hist.get_legend_handles_labels()
    if hist_handles:
        ax_hist.legend(title="Exposure bin", fontsize=9)
    ax_hist.grid(True, alpha=0.25, axis="y")

    if not df.empty:
        t_max = max(float(df["ttf_days"].quantile(0.97)), 100.0)
    else:
        t_max = 100.0
    time_grid = np.linspace(0.0, t_max, 500)
    for label, color in zip(BIN_LABELS, BIN_COLORS):
        sub = df.loc[df["freq_bin"] == label].copy()
        if len(sub) < MIN_ROWS_PER_KM_BIN:
            continue
        t = sub["ttf_days"].to_numpy(dtype=float)
        e = sub["event"].to_numpy(dtype=float)
        km_t, km_s = kaplan_meier(t, e)
        s_grid = np.interp(time_grid, km_t, km_s)
        lo, hi = km_confidence_band(t, e, time_grid)
        ax_km.step(time_grid, s_grid, where="post", color=color, linewidth=2, label=f"{label} (n={len(sub)}, events={int(e.sum())})")
        ax_km.fill_between(time_grid, lo, hi, alpha=0.12, color=color, step="post")
    ax_km.set_xlabel("Time (days)")
    ax_km.set_ylabel("Survival probability S(t)")
    ax_km.set_title(f"{title_prefix}\nKaplan-Meier survival by freq>55 share bin")
    ax_km.set_ylim(-0.02, 1.05)
    ax_km.set_xlim(0.0, t_max)
    ax_km.grid(True, alpha=0.25)
    km_handles, km_labels = ax_km.get_legend_handles_labels()
    if km_handles:
        ax_km.legend(title="Exposure bin", fontsize=9, loc="upper right")

    fig.tight_layout(pad=3.0)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    runs = load_runs(resolve_v03_all_path())
    vt_runs = runs.loc[
        runs["Месторождение"].astype("string").str.strip() == FIELD_CODE
    ].copy()
    vt_runs = vt_runs.loc[vt_runs["Дата монтажа"].dt.year.isin(MOUNT_YEARS)].copy()

    wells = vt_runs["Скв."].dropna().astype(str).unique().tolist()
    daily = load_daily_merged(wells)
    run_stats = compute_freq_share(vt_runs, daily)

    raw_csv = OUTPUT_DIR / "vt_2023_2026_freq55_run_stats.csv"
    run_stats.to_csv(raw_csv, index=False, encoding="utf-8-sig")
    build_scatter_plot(run_stats, PLOT_DIR / "vt_ttf_vs_freq55_share_2023_2026.png")

    summary_rows: list[dict[str, object]] = []
    for year_cutoff in YEAR_CUTOFFS_EXCLUSIVE:
        for infant_days in INFANT_THRESHOLDS:
            combo = build_combo_dataset(run_stats, year_cutoff, infant_days)
            tag = combo_tag(year_cutoff, infant_days)
            combo_csv = COMBO_DIR / f"{tag}.csv"
            combo.to_csv(combo_csv, index=False, encoding="utf-8-sig")

            plot_title = (
                f"Vt field | mount year > {year_cutoff} | infant exclusion < {infant_days}d\n"
                f"Runs={len(combo)}, failures={int(combo['event'].eq(1).sum()) if not combo.empty else 0}"
            )
            build_histogram_and_km(combo, plot_title, PLOT_DIR / f"{tag}_hist_km.png")

            for label in BIN_LABELS:
                sub = combo.loc[combo["freq_bin"] == label].copy()
                failure_sub = sub.loc[sub["event"] == 1].copy()
                summary_rows.append(
                    {
                        "year_cutoff_exclusive": year_cutoff,
                        "infant_exclusion_days": infant_days,
                        "freq_bin": label,
                        "n_total": int(len(sub)),
                        "n_failures": int(len(failure_sub)),
                        "n_censored": int(sub["event"].eq(0).sum()),
                        "median_ttf_all": None if sub.empty else float(sub["ttf_days"].median()),
                        "median_ttf_failures": None if failure_sub.empty else float(failure_sub["ttf_days"].median()),
                        "mean_frac_freq_above_55": None if sub.empty else float(sub["frac_freq_above_55"].mean()),
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "combination_summary.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "field": FIELD_CODE,
        "mount_years_in_source": sorted(MOUNT_YEARS),
        "year_cutoff_exclusive": YEAR_CUTOFFS_EXCLUSIVE,
        "infant_exclusion_days": INFANT_THRESHOLDS,
        "freq_threshold_hz": FREQ_THRESHOLD_HZ,
        "bin_labels": BIN_LABELS,
        "raw_csv": str(raw_csv.relative_to(REPO_ROOT)),
        "plot_dir": str(PLOT_DIR.relative_to(REPO_ROOT)),
        "combo_dir": str(COMBO_DIR.relative_to(REPO_ROOT)),
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    usable = run_stats.loc[run_stats["frac_freq_above_55"].notna()].copy()
    print(f"Vt runs in 2023-2026: {len(run_stats)}")
    print(f"Runs with frequency coverage (>= {MIN_FREQ_DAYS} days): {len(usable)}")
    print(f"Failures in usable set: {int(usable['event'].eq(1).sum())}")
    print(f"Scatter CSV: {raw_csv}")
    print(f"Combination summary: {OUTPUT_DIR / 'combination_summary.csv'}")


if __name__ == "__main__":
    main()
