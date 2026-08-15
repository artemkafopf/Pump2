"""Fleet scan: pumps that were **sped up mid-run** from 50–55 Hz to 60+ Hz.

Reads daily telemetry from the warehouse, cuts it into run windows (one physical
pump per window), and applies :mod:`analysis.features.freq_regime` to every run
with enough operating days.  Emits the qualifying runs, the near-misses with the
reason each failed, a per-field summary and trajectory figures.

Why the run window and not the well: a well's speed can differ between runs
simply because a different pump went in.  Only a change *inside* one run is a
regime change for a given pump.

Reading guards (both are written into the output tables, not just this docstring):

* daily telemetry starts **2018-01-01**.  A run installed before that has its
  early life missing; ``telemetry_left_truncated`` marks those, and their "pre"
  segment describes the telemetry window, not the pump's first months.
* only ~45 % of runs carry ≥60 operating days of valid frequency.  The scan
  reports its own denominator (``n_runs_scanned``) — the detected count is a
  share of the *scannable* fleet, never of all 2,634 runs.
* Мирнинский (Mc) obeys the standing installs-2024+ cohort rule.
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analysis.features.freq_regime import (
    FREQ_VALID_RANGE,
    HIGH_THRESHOLD,
    LOW_BAND,
    MIN_PURITY,
    MIN_SEGMENT_DAYS,
    SMOOTH_WINDOW,
    TRANSITION_DAYS,
    detect_regime_shift,
    rolling_median,
)
from analysis.paths import WAREHOUSE_DIR, results_dir
from analysis.workflows.production_risk import config as C

SLUG = "telemetry_freq_regime_shift"
WAREHOUSE_DB = WAREHOUSE_DIR / "pump2.db"

# Telemetry coverage starts here; runs installed earlier lose their early life.
TELEMETRY_START = pd.Timestamp("2018-01-01")
# Operating days either side of the changepoint for the "what did it do" windows.
NEAR_WINDOW_DAYS = 30

_RUN_COLUMNS = [
    "row_id", "well", "well_key", "field", "pad", "contractor", "event",
    "run_days", "install_date", "stop_date", "pump_type", "nominal_flow_m3d",
    "nominal_freq_hz", "motor_power_kw", "failed_node",
]


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_runs(con: sqlite3.Connection, *, mc_cohort: bool = True) -> pd.DataFrame:
    """Run registry, with the standing Мирнинский installs-2024+ cohort filter."""
    runs = pd.read_sql(f"SELECT {', '.join(_RUN_COLUMNS)} FROM raw__v03_runs", con)
    runs["install_date"] = pd.to_datetime(runs["install_date"], errors="coerce")
    runs["stop_date"] = pd.to_datetime(runs["stop_date"], errors="coerce")
    runs = runs[runs["install_date"].notna()].copy()
    if mc_cohort:
        cutoff = pd.Timestamp(C.MC_INSTALL_COHORT_START)
        drop = (runs["field"] == "Mc") & (runs["install_date"] < cutoff)
        runs = runs[~drop].copy()
    return runs


def load_daily(con: sqlite3.Connection) -> pd.DataFrame:
    """Daily telemetry restricted to valid-frequency (i.e. operating) days."""
    lo, hi = FREQ_VALID_RANGE
    daily = pd.read_sql(
        "SELECT well_key, dt, freq, qliq, watercut, load, kprod, gas_factor "
        "FROM proc__daily_merged "
        f"WHERE freq >= {lo} AND freq <= {hi} "
        "ORDER BY well_key, dt",
        con,
    )
    daily["dt"] = pd.to_datetime(daily["dt"])
    return daily


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

def _segment_means(frame: pd.DataFrame, columns: list[str], suffix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in columns:
        values = pd.to_numeric(frame[col], errors="coerce")
        out[f"{col}_{suffix}"] = float(values.mean()) if values.notna().any() else float("nan")
    return out


def scan_runs(
    runs: pd.DataFrame,
    daily: pd.DataFrame,
    *,
    low_band: tuple[float, float] = LOW_BAND,
    high_threshold: float = HIGH_THRESHOLD,
    min_segment_days: int = MIN_SEGMENT_DAYS,
    transition_days: int = TRANSITION_DAYS,
    min_purity: float = MIN_PURITY,
) -> pd.DataFrame:
    """One row per run that had enough telemetry to be scanned."""
    by_well = {key: g for key, g in daily.groupby("well_key", sort=False)}
    max_dt = daily["dt"].max() if len(daily) else pd.NaT
    context_cols = ["qliq", "watercut", "load", "kprod", "gas_factor"]

    records: list[dict] = []
    for run in runs.itertuples():
        well_daily = by_well.get(run.well_key)
        if well_daily is None:
            continue
        end = run.stop_date if pd.notna(run.stop_date) else max_dt
        window = well_daily[(well_daily["dt"] >= run.install_date) & (well_daily["dt"] <= end)]
        if len(window) < 2 * min_segment_days:
            continue

        freq = window["freq"].to_numpy(dtype=float)
        shift = detect_regime_shift(
            freq,
            low_band=low_band,
            high_threshold=high_threshold,
            min_segment_days=min_segment_days,
            transition_days=transition_days,
            min_purity=min_purity,
        )

        record: dict = {
            "row_id": run.row_id,
            "well": run.well,
            "field": run.field,
            "pad": run.pad,
            "contractor": run.contractor,
            "pump_type": run.pump_type,
            "nominal_flow_m3d": run.nominal_flow_m3d,
            "nominal_freq_hz": run.nominal_freq_hz,
            "motor_power_kw": run.motor_power_kw,
            "install_date": run.install_date.date().isoformat(),
            "stop_date": end.date().isoformat() if pd.notna(end) else None,
            "event": run.event,
            "run_days": run.run_days,
            "failed_node": run.failed_node,
            "telemetry_left_truncated": bool(run.install_date < TELEMETRY_START),
            "telemetry_first_day": window["dt"].iloc[0].date().isoformat(),
            "telemetry_last_day": window["dt"].iloc[-1].date().isoformat(),
        }
        record.update(shift.to_dict())

        k = shift.cp_index
        if k > 0:
            cp_date = window["dt"].iloc[min(k, len(window) - 1)]
            record["shift_date"] = cp_date.date().isoformat()
            record["shift_op_day"] = k
            record["shift_calendar_day"] = int((cp_date - run.install_date).days)
            record["days_shift_to_stop"] = int((end - cp_date).days) if pd.notna(end) else None
            record["shift_frac_of_run"] = (
                float((cp_date - run.install_date).days / run.run_days)
                if run.run_days and run.run_days > 0 else float("nan")
            )
            pre_end = max(k - transition_days, 1)
            post_start = min(k + transition_days, len(window) - 1)
            record.update(_segment_means(window.iloc[:pre_end], context_cols, "pre"))
            record.update(_segment_means(window.iloc[post_start:], context_cols, "post"))
            # Tight windows either side of the changepoint.  Whole-segment means
            # mix the speed change with years of natural decline; ±30 operating
            # days isolates what the speed-up itself did.
            near_lo = max(pre_end - NEAR_WINDOW_DAYS, 0)
            near_hi = min(post_start + NEAR_WINDOW_DAYS, len(window))
            record.update(_segment_means(window.iloc[near_lo:pre_end], context_cols, "pre30"))
            record.update(_segment_means(window.iloc[post_start:near_hi], context_cols, "post30"))
        records.append(record)

    return pd.DataFrame(records)


def add_rate_response(scanned: pd.DataFrame) -> pd.DataFrame:
    """Did the speed-up actually buy rate?

    Affinity says flow scales ~linearly with speed, so ``f_post/f_pre − 1`` is the
    rate gain a healthy well/pump should deliver.  ``qliq_response_pct`` is what it
    delivered over the ±30 operating-day windows.  A shift that lands far short of
    the expectation is the signature of a speed-up used to *compensate* for falling
    inflow or a degrading pump rather than to push production.
    """
    if scanned.empty or "qliq_pre30" not in scanned:
        return scanned
    out = scanned.copy()
    pre = pd.to_numeric(out["qliq_pre30"], errors="coerce")
    post = pd.to_numeric(out["qliq_post30"], errors="coerce")
    # A pre-window mean of 0 means the pump was idle into the changepoint — the
    # ratio is undefined, not infinite.
    pre = pre.where(pre > 0)
    out["qliq_response_pct"] = 100.0 * (post / pre - 1.0)
    out["qliq_affinity_expected_pct"] = 100.0 * (
        pd.to_numeric(out["post_median_hz"], errors="coerce")
        / pd.to_numeric(out["pre_median_hz"], errors="coerce") - 1.0
    )
    out["rate_shortfall_pct"] = out["qliq_response_pct"] - out["qliq_affinity_expected_pct"]
    return out


def summarise_by_field(scanned: pd.DataFrame) -> pd.DataFrame:
    """Detected count / rate per field, over the *scannable* denominator."""
    if scanned.empty:
        return pd.DataFrame(columns=["field", "n_scanned", "n_detected", "detected_pct"])
    grouped = scanned.groupby("field", dropna=False)
    out = pd.DataFrame({
        "n_scanned": grouped.size(),
        "n_detected": grouped["detected"].sum(),
        "n_failed_of_detected": grouped.apply(
            lambda g: int((g["detected"] & (g["event"] == 1)).sum()), include_groups=False
        ),
        "median_shift_hz": grouped.apply(
            lambda g: float(g.loc[g["detected"], "shift_hz"].median()) if g["detected"].any() else float("nan"),
            include_groups=False,
        ),
    }).reset_index()
    out["detected_pct"] = 100.0 * out["n_detected"] / out["n_scanned"]
    return out.sort_values("n_detected", ascending=False)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_trajectories(
    detected: pd.DataFrame,
    daily: pd.DataFrame,
    runs: pd.DataFrame,
    out_path: Path,
    *,
    max_panels: int = 24,
) -> int:
    """Small multiples of the detected runs' daily frequency, changepoint marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    picks = detected.sort_values("shift_hz", ascending=False).head(max_panels)
    if picks.empty:
        return 0
    run_index = runs.set_index("row_id")
    n = len(picks)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.4 * nrows), squeeze=False)
    max_dt = daily["dt"].max()

    for ax, row in zip(axes.ravel(), picks.itertuples()):
        run = run_index.loc[row.row_id]
        end = run["stop_date"] if pd.notna(run["stop_date"]) else max_dt
        window = daily[(daily["well_key"] == run["well_key"])
                       & (daily["dt"] >= run["install_date"]) & (daily["dt"] <= end)]
        x = np.arange(len(window))
        ax.plot(x, window["freq"].to_numpy(float), lw=0.6, color="#9aa5b1")
        ax.plot(x, rolling_median(window["freq"].to_numpy(float), SMOOTH_WINDOW),
                lw=1.6, color="#1f4e79")
        ax.axvspan(0, row.cp_index, color="#2e7d32", alpha=0.07)
        ax.axvspan(row.cp_index, len(window), color="#c62828", alpha=0.07)
        ax.axvline(row.cp_index, color="#c62828", lw=1.2, ls="--")
        ax.axhline(LOW_BAND[1], color="#888", lw=0.6, ls=":")
        ax.axhline(HIGH_THRESHOLD, color="#888", lw=0.6, ls=":")
        outcome = "отказ" if row.event == 1 else "работает/снят"
        ax.set_title(f"{row.well} · {row.field} · {row.pre_median_hz:.0f}→{row.post_median_hz:.0f} Гц · {outcome}",
                     fontsize=8)
        ax.set_ylim(40, 75)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("операционные сутки", fontsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Смена режима внутри спуска: 50–55 Гц → 60+ Гц", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return n


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    *,
    run_date: date | None = None,
    mc_cohort: bool = True,
    low_band: tuple[float, float] = LOW_BAND,
    high_threshold: float = HIGH_THRESHOLD,
    min_segment_days: int = MIN_SEGMENT_DAYS,
    min_purity: float = MIN_PURITY,
    db_path: Path | None = None,
) -> dict:
    out = results_dir(SLUG, run_date)
    con = sqlite3.connect(str(db_path or WAREHOUSE_DB))
    try:
        runs = load_runs(con, mc_cohort=mc_cohort)
        daily = load_daily(con)
    finally:
        con.close()

    scanned = scan_runs(
        runs, daily,
        low_band=low_band,
        high_threshold=high_threshold,
        min_segment_days=min_segment_days,
        min_purity=min_purity,
    )
    scanned = add_rate_response(scanned)
    detected = scanned[scanned["detected"]].copy().sort_values(["field", "shift_hz"], ascending=[True, False])
    near_miss = scanned[~scanned["detected"] & scanned["reason"].isin(
        ["pre_segment_impure", "post_segment_impure", "high_regime_not_sustained"]
    )].copy()
    near_miss = near_miss[near_miss["post_median_hz"] >= high_threshold - 2.0]

    tables = out / "tables"
    detected.to_csv(tables / "regime_shift_detected.csv", index=False, encoding="utf-8-sig")
    near_miss.sort_values("post_median_hz", ascending=False).to_csv(
        tables / "regime_shift_near_miss.csv", index=False, encoding="utf-8-sig")
    scanned.to_csv(tables / "regime_shift_all_scanned.csv", index=False, encoding="utf-8-sig")
    summarise_by_field(scanned).to_csv(tables / "regime_shift_by_field.csv", index=False, encoding="utf-8-sig")

    n_panels = plot_trajectories(detected, daily, runs, out / "figures" / "regime_shift_trajectories.png")

    reason_counts = scanned["reason"].value_counts().to_dict()
    stats = {
        "n_runs_total": int(len(runs)),
        "n_runs_scanned": int(len(scanned)),
        "n_detected": int(len(detected)),
        "n_near_miss": int(len(near_miss)),
        "n_detected_failed": int((detected["event"] == 1).sum()),
        "n_left_truncated_of_detected": int(detected["telemetry_left_truncated"].sum()),
        "median_qliq_response_pct": float(detected["qliq_response_pct"].median()),
        "median_qliq_affinity_expected_pct": float(detected["qliq_affinity_expected_pct"].median()),
        "n_reaching_affinity_expectation": int((detected["rate_shortfall_pct"] >= 0).sum()),
        "reason_counts": reason_counts,
        "n_figure_panels": n_panels,
        "out_dir": str(out),
        "params": {
            "low_band": list(low_band),
            "high_threshold": high_threshold,
            "min_segment_days": min_segment_days,
            "min_purity": min_purity,
            "transition_days": TRANSITION_DAYS,
            "smooth_window": SMOOTH_WINDOW,
            "mc_cohort_2024plus": mc_cohort,
        },
    }
    return {"stats": stats, "detected": detected, "near_miss": near_miss, "scanned": scanned}
