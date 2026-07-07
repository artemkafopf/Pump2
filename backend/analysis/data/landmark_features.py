"""Phase D — landmark frame + leakage-proof trailing-window feature builder (D0).

This is the data layer for the within-run dynamics / early-warning phase.  It turns
the 1.25M-row daily table into a **landmark frame**: one row per run × landmark
(operating ages 30, 60, 90, … while the run is still alive), carrying

* trailing-window features (14 and 30 operating-day windows) computed **past-only**
  — a feature at landmark ``L`` uses daily rows with operating-day index ≤ ``L``
  and nothing later (leakage rule 1, verified by the truncation-invariance test);
* per-horizon targets with a guard gap — a landmark at ``L`` predicts events in the
  operating-day window ``(L + g, L + g + H]`` (leakage rule 2, guard-gap test);
* age & static context (operating age, stratum, run_seq, sour flag) — the baseline
  the dynamic model must beat.

Design decisions (Phase D §0, user-confirmed 2026-07-07):

* Clock = **operating days** (qliq > 0), reconciled against ``proc__ttf_true``;
  runs whose qliq-operating-day count disagrees with ``ttf_true_best_days`` by
  > 10% are flagged (``opday_ttf_discrepancy``).
* Landmark cadence 30 operating days; horizons H ∈ {30, 60, 90}; guard gap
  g ∈ {3, 7, 14} (default 7); eligibility ≥ 30 valid telemetry days before the
  first landmark (i.e. a run needs ≥ 30 operating days to earn landmark L = 30).

Everything here is **pure-function-first** so the leakage rules are unit-testable on
synthetic daily frames without touching the warehouse: :func:`compute_window_features`
and :func:`build_landmark_targets` take arrays / frames and are exercised directly by
``backend/tests/test_landmark_features.py``.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ── Phase D §0 parameters (user-confirmed defaults) ──────────────────────────
LANDMARK_CADENCE = 30          # operating days between landmarks
HORIZONS = (30, 60, 90)        # prediction horizons (operating days); 60 primary
GUARD_GAPS = (3, 7, 14)        # guard-gap sensitivity grid; 7 default
DEFAULT_GUARD = 7
MIN_TELEMETRY_DAYS = 30        # eligibility: valid op-days before first landmark
WINDOWS = (14, 30)            # trailing-window sizes (operating days)

# Data-validity guards (mirror analysis.data.run_covariates — raw telemetry has
# physically impossible values that must be dropped before any statistic).
_FREQ_RANGE = (20.0, 75.0)
_LOAD_RANGE = (0.0, 150.0)
_WC_RANGE = (0.0, 100.0)

LOAD_EXCURSION_THR = 90.0      # load > 90 (% of rated) counts as an excursion day
FREQ_HIGH_THR = 55.0           # days above 55 Hz (over-speed exposure)
FREQ_STEP_HZ = 1.0             # |Δf| > 1 Hz day-over-day = a frequency step
OPDAY_DISCREPANCY_THR = 0.10   # |op_days - ttf_true| / ttf_true flag threshold


# ---------------------------------------------------------------------------
# Small numeric helpers (pure)
# ---------------------------------------------------------------------------

def theil_sen_slope(y: np.ndarray, x: np.ndarray | None = None) -> float:
    """Robust Theil–Sen slope of ``y`` against ``x`` (per-unit-x).

    NaNs in either series are dropped pairwise.  Fewer than 2 finite points → NaN.
    ``x`` defaults to 0..n-1 (per-step slope).
    """
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y), dtype=float) if x is None else np.asarray(x, dtype=float)
    m = np.isfinite(y) & np.isfinite(x)
    y, x = y[m], x[m]
    if len(y) < 2 or np.ptp(x) == 0:
        return float("nan")
    try:
        from scipy.stats import theilslopes
        return float(theilslopes(y, x)[0])
    except Exception:
        # median of pairwise slopes (fallback if scipy unavailable)
        slopes = []
        for i in range(len(y)):
            for j in range(i + 1, len(y)):
                if x[j] != x[i]:
                    slopes.append((y[j] - y[i]) / (x[j] - x[i]))
        return float(np.median(slopes)) if slopes else float("nan")


def _clean(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Set values outside [lo, hi] to NaN (data-validity guard)."""
    a = np.asarray(a, dtype=float)
    out = a.copy()
    out[(a < lo) | (a > hi)] = np.nan
    return out


def _mean(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if len(a) else float("nan")


def _std(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.std(a)) if len(a) >= 3 else float("nan")


# ---------------------------------------------------------------------------
# Operating-day mapping
# ---------------------------------------------------------------------------

def operating_day_index(qliq: np.ndarray) -> np.ndarray:
    """Cumulative operating-day index (qliq > 0) for a date-ordered daily series.

    Idle / missing-qliq days do **not** advance the index but keep their (integer)
    running value, so calendar rows can be sliced by operating age.  The last
    operating row's index is the run's total operating age.
    """
    q = np.nan_to_num(np.asarray(qliq, dtype=float), nan=0.0)
    return np.cumsum(q > 0).astype(int)


# ---------------------------------------------------------------------------
# Trailing-window features (pure — the leakage-proof core)
# ---------------------------------------------------------------------------

# Feature columns produced by :func:`compute_window_features` (without the window
# suffix).  Kept explicit so the builder can stamp ``_missing`` indicators and the
# tests can assert the contract.
FEATURE_STEMS = (
    "kprod_mean", "kprod_slope", "kprod_pctdrop", "qliq_slope",         # productivity
    "load_mean", "load_slope", "load_std", "load_exc",                  # electrical
    "freq_mean", "freq_std", "freq_step", "freq_days55",               # frequency
    "restarts", "days_since_restart", "longest_idle",                  # cycling
    "rintake_mean", "rintake_slope", "rzab_slope", "drawdown",         # hydraulic pressure
    "watercut_mean", "watercut_jump", "gasfactor_mean", "gasfactor_slope",  # fluid
)


def compute_window_features(daily_past: pd.DataFrame, window: int,
                            *, run_best_kprod: float | None = None) -> dict:
    """Trailing-window features from a **past-only** daily slice.

    Parameters
    ----------
    daily_past
        Daily rows of one run with operating-day index ≤ landmark, sorted by ``dt``.
        Required columns: ``dt`` (datetime), ``op_index`` (int), ``qliq``, and the
        telemetry channels ``freq``, ``load``, ``kprod``, ``rpump_intake``, ``rzab``,
        ``rpl``, ``watercut``, ``gas_factor``.  Missing channels are treated as NaN.
    window
        Trailing-window size in **operating days**.  The window is the calendar span
        covered by the last ``window`` operating days (so idle days inside it count
        for the cycling features).
    run_best_kprod
        Best (max) kprod over the whole past-only history (for the %-drop feature);
        if ``None`` it is taken from ``daily_past`` itself.

    Returns
    -------
    dict of the :data:`FEATURE_STEMS` features (NaN where unsupported by the window).
    """
    d = daily_past.sort_values("dt").reset_index(drop=True)
    q = np.nan_to_num(d["qliq"].to_numpy(dtype=float), nan=0.0)
    is_op = q > 0

    # Window = calendar span of the last ``window`` operating days.
    op_pos = np.flatnonzero(is_op)
    if len(op_pos) == 0:
        return {s: float("nan") for s in FEATURE_STEMS}
    start_pos = op_pos[-window] if len(op_pos) >= window else op_pos[0]
    win = d.iloc[start_pos:].copy()
    win_q = np.nan_to_num(win["qliq"].to_numpy(dtype=float), nan=0.0)
    win_is_op = win_q > 0

    def ch(name: str, lo: float | None = None, hi: float | None = None) -> np.ndarray:
        """Operating-day values of channel ``name`` within the window (guarded)."""
        if name not in win.columns:
            return np.array([], dtype=float)
        a = win[name].to_numpy(dtype=float)[win_is_op]
        if lo is not None:
            a = _clean(a, lo, hi)
        return a

    out: dict[str, float] = {}

    # ── Productivity decline ────────────────────────────────────────────────
    kprod = ch("kprod")
    out["kprod_mean"] = _mean(kprod)
    out["kprod_slope"] = theil_sen_slope(kprod)
    best = run_best_kprod
    if best is None and "kprod" in d.columns:
        allk = d["kprod"].to_numpy(dtype=float)[is_op]
        allk = allk[np.isfinite(allk)]
        best = float(np.max(allk)) if len(allk) else np.nan
    if best is not None and np.isfinite(best) and best > 0 and np.isfinite(out["kprod_mean"]):
        out["kprod_pctdrop"] = float((best - out["kprod_mean"]) / best)
    else:
        out["kprod_pctdrop"] = float("nan")
    out["qliq_slope"] = theil_sen_slope(win_q[win_is_op])

    # ── Electrical ──────────────────────────────────────────────────────────
    load = ch("load", *_LOAD_RANGE)
    out["load_mean"] = _mean(load)
    out["load_slope"] = theil_sen_slope(load)
    out["load_std"] = _std(load)
    out["load_exc"] = float(np.sum(load > LOAD_EXCURSION_THR)) if len(load) else float("nan")

    # ── Frequency regime ────────────────────────────────────────────────────
    freq = ch("freq", *_FREQ_RANGE)
    out["freq_mean"] = _mean(freq)
    out["freq_std"] = _std(freq)
    fv = freq[np.isfinite(freq)]
    out["freq_step"] = float(np.sum(np.abs(np.diff(fv)) > FREQ_STEP_HZ)) if len(fv) >= 2 else float("nan")
    out["freq_days55"] = float(np.sum(fv > FREQ_HIGH_THR)) if len(fv) else float("nan")

    # ── Cycling (uses the full calendar window incl. idle days) ─────────────
    on = win_is_op.astype(int)
    out["restarts"] = float(np.sum(np.diff(on) == 1)) if len(on) > 1 else 0.0
    # days since last restart: calendar days from window end back to last 0→1 edge
    restart_idx = np.flatnonzero(np.diff(on) == 1) + 1 if len(on) > 1 else np.array([], int)
    if len(restart_idx):
        last_restart_dt = win["dt"].iloc[restart_idx[-1]]
        out["days_since_restart"] = float((win["dt"].iloc[-1] - last_restart_dt).days)
    else:
        out["days_since_restart"] = float("nan")
    # longest idle spell = longest run of consecutive non-operating days in window
    out["longest_idle"] = float(_longest_run(on == 0))

    # ── Hydraulic pressure regime ───────────────────────────────────────────
    rintake = ch("rpump_intake")
    rintake = rintake[rintake > 0] if len(rintake) else rintake
    out["rintake_mean"] = _mean(rintake)
    out["rintake_slope"] = theil_sen_slope(ch("rpump_intake"))
    rzab = ch("rzab")
    out["rzab_slope"] = theil_sen_slope(rzab)
    if "rpl" in win.columns and "rzab" in win.columns:
        draw = (win["rpl"].to_numpy(dtype=float) - win["rzab"].to_numpy(dtype=float))[win_is_op]
        out["drawdown"] = _mean(draw)
    else:
        out["drawdown"] = float("nan")

    # ── Fluid ───────────────────────────────────────────────────────────────
    wc = ch("watercut", *_WC_RANGE)
    out["watercut_mean"] = _mean(wc)
    wcf = wc[np.isfinite(wc)]
    out["watercut_jump"] = float(np.max(np.diff(wcf))) if len(wcf) >= 2 else float("nan")
    gf = ch("gas_factor")
    gf = gf[gf >= 0] if len(gf) else gf
    out["gasfactor_mean"] = _mean(gf)
    out["gasfactor_slope"] = theil_sen_slope(ch("gas_factor"))

    return out


def _longest_run(mask: np.ndarray) -> int:
    """Length of the longest run of ``True`` in a boolean array."""
    best = cur = 0
    for v in np.asarray(mask, dtype=bool):
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


# ---------------------------------------------------------------------------
# Targets (pure — the guard-gap core)
# ---------------------------------------------------------------------------

def build_landmark_targets(op_age: int, terminal_op_age: int, event: int,
                           horizons=HORIZONS, guards=GUARD_GAPS) -> dict:
    """Per-(horizon × guard) target for one landmark at operating age ``op_age``.

    A landmark predicts events in the operating-day window ``(op_age+g, op_age+g+H]``.
    For each (H, g) produces:

    * ``y_H{H}_g{g}``    — 1 if the run failed inside the window, 0 if it survived
      the window, ``NaN`` if the outcome is unobservable (censored inside window, or
      the failure fell in the guard interval ``(op_age, op_age+g]``);
    * ``eligible_H{H}_g{g}`` — 1 iff this landmark contributes to the (H, g) view
      (i.e. it is observed past the guard: ``terminal_op_age > op_age + g``).

    Also always returns ``tte_land`` = ``terminal_op_age − op_age`` (operating days
    from landmark to the run's terminal age) and ``event_land`` = ``event`` — the
    survival-scoring pair used by the alarm evaluation.
    """
    d = int(terminal_op_age) - int(op_age)      # op-days from landmark to terminal
    out: dict[str, float] = {"tte_land": float(d), "event_land": int(event)}
    for H in horizons:
        for g in guards:
            key = f"H{H}_g{g}"
            observed_past_guard = d > g
            out[f"eligible_{key}"] = int(observed_past_guard)
            if not observed_past_guard:
                out[f"y_{key}"] = float("nan")          # dropped from the (H,g) view
                continue
            if event == 1 and d <= g + H:
                out[f"y_{key}"] = 1.0                    # failure inside the window
            elif d > g + H:
                out[f"y_{key}"] = 0.0                    # survived the whole window
            else:
                # event==0 and d <= g+H → censored inside window: outcome unknown
                out[f"y_{key}"] = float("nan")
    return out


# ---------------------------------------------------------------------------
# Warehouse frame builder
# ---------------------------------------------------------------------------

_DAILY_CHANNELS = ["freq", "load", "rpl", "rpump_intake", "rzab", "qliq",
                   "watercut", "gas_factor", "kprod"]


def load_daily_frame(con: sqlite3.Connection) -> pd.DataFrame:
    """One row per (run daily observation): row_id, dt + telemetry channels.

    Each daily row is assigned to the run whose ``[install_date, stop_date]`` window
    contains it (runs on a well are sequential, so the assignment is unique).
    """
    cols = ", ".join(f"dm.{c}" for c in _DAILY_CHANNELS)
    q = f"""
    SELECT m.row_id, dm.dt, {cols}
    FROM mart__weibull_input m
    JOIN proc__daily_merged dm
      ON dm.well_key = m.well_key
     AND dm.dt BETWEEN m.install_date AND m.stop_date
    ORDER BY m.row_id, dm.dt
    """
    df = pd.read_sql(q, con)
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    return df


@dataclass
class LandmarkBuildResult:
    frame: pd.DataFrame           # one row per run × landmark
    reconciliation: pd.DataFrame  # per-run op-day vs ttf_true discrepancy audit
    coverage: pd.DataFrame        # per-landmark feature coverage (measured fraction)


# Run-level context columns carried onto every landmark row.
_CONTEXT_COLS = ["well_key", "stratum_key", "field", "install_dt", "run_seq",
                 "log_run_seq", "is_sour_flagged", "has_telemetry", "event",
                 "mode_group", "event_hydraulic", "event_electro-thermal",
                 "event_protector"]


def build_landmark_frame(run_df: pd.DataFrame, daily: pd.DataFrame,
                         *, cadence: int = LANDMARK_CADENCE,
                         horizons=HORIZONS, guards=GUARD_GAPS,
                         min_telemetry_days: int = MIN_TELEMETRY_DAYS,
                         ttf_true: pd.DataFrame | None = None) -> LandmarkBuildResult:
    """Assemble the landmark frame from run-level context + the daily frame.

    ``run_df`` is one row per run (from :func:`build_competing_risks_df`); ``daily``
    is :func:`load_daily_frame`.  Returns the landmark frame plus the reconciliation
    and coverage audits.
    """
    run_df = run_df.set_index("row_id")
    ttf_map = (ttf_true.set_index("row_id")["ttf_true_best_days"].to_dict()
               if ttf_true is not None else {})

    rows: list[dict] = []
    recon: list[dict] = []
    for row_id, g in daily.groupby("row_id", sort=False):
        if row_id not in run_df.index:
            continue
        ctx = run_df.loc[row_id]
        g = g.sort_values("dt").reset_index(drop=True)
        op_index = operating_day_index(g["qliq"].to_numpy())
        g["op_index"] = op_index
        op_total = int(op_index.max()) if len(op_index) else 0

        # Reconciliation vs ttf_true.
        ttf_ref = ttf_map.get(row_id, np.nan)
        disc = (abs(op_total - ttf_ref) / ttf_ref
                if np.isfinite(ttf_ref) and ttf_ref > 0 else np.nan)
        recon.append({"row_id": row_id, "op_days_qliq": op_total,
                      "ttf_true_best_days": ttf_ref,
                      "discrepancy": round(float(disc), 3) if np.isfinite(disc) else np.nan,
                      "flag": bool(np.isfinite(disc) and disc > OPDAY_DISCREPANCY_THR)})

        # Eligibility: need ≥ min_telemetry_days operating days to earn L = cadence.
        if op_total < min_telemetry_days:
            continue

        event = int(ctx["event"])
        # run-best kprod over the whole (past-only at each landmark) history —
        # recomputed per landmark below to stay strictly past-only.
        landmark = cadence
        while landmark <= op_total:
            past = g[g["op_index"] <= landmark]
            # run-best kprod is past-only: max over op-days ≤ landmark
            allk = past.loc[past["qliq"] > 0, "kprod"].to_numpy(dtype=float)
            allk = allk[np.isfinite(allk)]
            best_kprod = float(np.max(allk)) if len(allk) else np.nan

            rec: dict = {"row_id": row_id, "op_age": landmark}
            for c in _CONTEXT_COLS:
                rec[c] = ctx[c] if c in ctx.index else np.nan
            rec["n_valid_op_days"] = int((past["qliq"] > 0).sum())

            for w in WINDOWS:
                feats = compute_window_features(past, w, run_best_kprod=best_kprod)
                for stem, val in feats.items():
                    rec[f"{stem}_w{w}"] = val

            rec.update(build_landmark_targets(landmark, op_total, event,
                                              horizons=horizons, guards=guards))
            rows.append(rec)
            landmark += cadence

    frame = pd.DataFrame(rows)
    recon_df = pd.DataFrame(recon)

    # ── _missing indicators + coverage audit ────────────────────────────────
    feat_cols = [f"{s}_w{w}" for w in WINDOWS for s in FEATURE_STEMS]
    cov_rows = []
    if not frame.empty:
        for c in feat_cols:
            if c not in frame.columns:
                frame[c] = np.nan
            frame[f"{c}_missing"] = frame[c].isna().astype(np.int8)
            n = len(frame)
            n_meas = int(frame[c].notna().sum())
            cov_rows.append({"feature": c, "n_landmarks": n, "n_measured": n_meas,
                             "pct_measured": round(100.0 * n_meas / n, 1) if n else 0.0})
    coverage = pd.DataFrame(cov_rows)

    return LandmarkBuildResult(frame=frame, reconciliation=recon_df, coverage=coverage)


def build_from_warehouse(warehouse_db, *, cadence: int = LANDMARK_CADENCE,
                         horizons=HORIZONS, guards=GUARD_GAPS,
                         min_telemetry_days: int = MIN_TELEMETRY_DAYS,
                         tte_col: str = "ttf_mix") -> LandmarkBuildResult:
    """End-to-end warehouse build (thin orchestration for the D0 script)."""
    from analysis.data.competing_risks_loader import build_competing_risks_df

    run_df = build_competing_risks_df(tte_col=tte_col)
    con = sqlite3.connect(warehouse_db)
    try:
        daily = load_daily_frame(con)
        ttf_true = pd.read_sql(
            "SELECT row_id, ttf_true_best_days FROM proc__ttf_true", con)
    finally:
        con.close()
    return build_landmark_frame(run_df, daily, cadence=cadence, horizons=horizons,
                                guards=guards, min_telemetry_days=min_telemetry_days,
                                ttf_true=ttf_true)


def load_latest_landmark_frame() -> pd.DataFrame:
    """Load the most recent materialised landmark frame from ``results/``.

    Downstream Phase D scripts (D1–D5) read the frame D0 wrote rather than rebuild
    it.  Raises ``FileNotFoundError`` if D0 has not been run.
    """
    from analysis.paths import RESULTS_ROOT
    base = RESULTS_ROOT / "phase_d_landmarks"
    if not base.exists():
        raise FileNotFoundError("no phase_d_landmarks results — run phase_d_landmarks.py first")
    for d in sorted((p for p in base.iterdir() if p.is_dir()), reverse=True):
        for fn, reader in (("landmark_frame.parquet", pd.read_parquet),
                           ("landmark_frame.csv", pd.read_csv)):
            p = d / "tables" / fn
            if p.exists():
                df = reader(p)
                df["install_dt"] = pd.to_datetime(df["install_dt"], errors="coerce")
                return df
    raise FileNotFoundError("no landmark_frame under results/phase_d_landmarks/*/tables/")


__all__ = [
    "LANDMARK_CADENCE", "HORIZONS", "GUARD_GAPS", "DEFAULT_GUARD",
    "MIN_TELEMETRY_DAYS", "WINDOWS", "FEATURE_STEMS",
    "theil_sen_slope", "operating_day_index", "compute_window_features",
    "build_landmark_targets", "load_daily_frame", "build_landmark_frame",
    "build_from_warehouse", "LandmarkBuildResult", "load_latest_landmark_frame",
]
