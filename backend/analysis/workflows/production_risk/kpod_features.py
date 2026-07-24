"""Kpod (коэффициент подачи) feature family for the Vt TTF-covariate analysis.

Kpod = actual liquid rate / pump design capacity.  Two variants, both canonical
(see :mod:`analysis.features.operational`):

* ``kpod``       = ``qliq / nominal_flow_m3d``                — off-design at 50 Hz;
* ``kpod_freq``  = ``qliq / (nominal_flow_m3d · f/50)``       — off-design scaled to
  the *running* frequency (affinity law).

Their disagreement localises where frequency trim matters and is itself reported
(a run held off nominal purely by a frequency cut reads as off-design on ``kpod``
but on-design on ``kpod_freq``).

**Window family (workflow plan §2).**  Every telemetry covariate is computed three
ways so the analysis can compare which member correlates better rather than
pre-committing:

* ``*_t0``   — mean over the first ``t0_op_days`` operating days (the causal anchor,
  immune to the failure read backwards);
* ``*_run``  — mean over the whole run **excluding the last ``guard_days`` calendar
  days before stop** (chronic loading, with the tail guard that stops a failing
  pump's collapsing qliq from leaking the outcome into the feature);
* ``frac_days_*`` — share of guarded op-days in a Kpod band (chronic dose of
  excursion, the alternative to the mean).

The tail guard is non-negotiable and **differential**: event runs carry a death
tail, censored runs do not, so an unguarded run mean "wins" a correlation contest
partly mechanically.  Runs with too few guarded op-days have no honest run-long
value and fall back to ``*_t0`` with ``kpod_run_source = 't0_fallback'`` — never
silently; short runs are exactly the infant failures.

**Qnom / freq resolution (plan §1 imputation rules).**  Each run gets a design flow
and a nominal frequency from the best available source, tagged in
``kpod_qnom_source`` ∈ {mart, big, type_parse, missing} so the U-shape claim can be
shown robust to dropping the weakest (``type_parse``) tier:

* ``nominal_flow_m3d``: mart ``mart__weibull_input`` → Big «Насос / Ном. Произв» →
  parse the pump type name (ЭЦН5-80 → 80 м³/сут);
* ``nominal_freq_hz``:  mart → Big «Номинальная частота» → **assume 50 Hz**.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.data.equipment_big import load_equipment_big
from analysis.data.pump_type_parser import parse_pump_type
from analysis.features.operational import KPOD_UNDERLOAD, compute_kpod_features
from analysis.paths import WAREHOUSE_DIR
from analysis.workflows.production_risk import crosswalk

# Telemetry range guard mirrors analysis.data.run_covariates._FREQ_RANGE.
FREQ_RANGE = (20.0, 75.0)

DEFAULT_T0_OP_DAYS = 30
DEFAULT_GUARD_DAYS = 30
#: A guarded run window needs at least this many op-days to carry an honest mean;
#: fewer and the run falls back to its t0 value (flagged).
MIN_RUN_OP_DAYS = 5
#: Default exposure-share bands (plan §3.5; refined after the spline binning).
DEFAULT_BAND_LO = 0.7
DEFAULT_BAND_HI = 1.1
#: Over-speed threshold (Hz): share of op-days above this is the frequency-dose feature
#: (mirrors ``run_covariates`` ``freq_above_55hz``; 50 Hz is the design point).
FREQ_OVERSPEED_HZ = 55.0


# ---------------------------------------------------------------------------
# Qnom / nominal-frequency resolution
# ---------------------------------------------------------------------------

def _mart_nominals(db_path: Path | None = None) -> pd.DataFrame:
    con = sqlite3.connect(str(db_path or (WAREHOUSE_DIR / "pump2.db")))
    try:
        m = pd.read_sql(
            "SELECT well_key, install_date, nominal_flow_m3d, nominal_freq_hz, pump_type "
            "FROM mart__weibull_input",
            con,
        )
    finally:
        con.close()
    m["code"] = m["well_key"].map(crosswalk.norm_well)
    m["install"] = pd.to_datetime(m["install_date"], errors="coerce")
    m["q_mart"] = pd.to_numeric(m["nominal_flow_m3d"], errors="coerce")
    m["f_mart"] = pd.to_numeric(m["nominal_freq_hz"], errors="coerce")
    return m.dropna(subset=["code", "install"])[["code", "install", "q_mart", "f_mart", "pump_type"]]


def _big_nominals(equipment_big_path: Path | None = None) -> pd.DataFrame:
    big = load_equipment_big(equipment_big_path) if equipment_big_path else load_equipment_big()
    big = big[big["is_esp_strict"] == True].copy()  # noqa: E712 — pandas mask
    big["code"] = big["well_key"].map(crosswalk.norm_well)
    big["install"] = pd.to_datetime(big["install_date"], errors="coerce")
    big["q_big"] = pd.to_numeric(big["q_nom_m3d"], errors="coerce")
    big["f_big"] = pd.to_numeric(big["freq_nom_hz"], errors="coerce")
    return big.dropna(subset=["code", "install"])[["code", "install", "q_big", "f_big", "gno_type"]]


def _asof_nearest(left: pd.DataFrame, right: pd.DataFrame, tol_days: int) -> pd.DataFrame:
    """Nearest per-code install match within ``tol_days`` (mirrors attach_equipment)."""
    left = left.copy()
    left["_row"] = np.arange(len(left))
    merged = pd.merge_asof(
        left.sort_values("install"),
        right.sort_values("install"),
        on="install",
        by="code",
        tolerance=pd.Timedelta(days=tol_days),
        direction="nearest",
    )
    return merged.sort_values("_row").drop(columns="_row").reset_index(drop=True)


def _parse_qnom(pump_type: object, gno_type: object) -> float:
    """Design flow from the pump-type string (last-resort tier)."""
    for txt in (pump_type, gno_type):
        if txt is None or (isinstance(txt, float) and np.isnan(txt)):
            continue
        q = parse_pump_type(txt).q_design_m3d
        if q is not None and np.isfinite(q) and q > 0:
            return float(q)
    return np.nan


def resolve_qnom(
    pop: pd.DataFrame,
    *,
    tol_days: int = 30,
    db_path: Path | None = None,
    equipment_big_path: Path | None = None,
) -> pd.DataFrame:
    """Attach ``nominal_flow_m3d``, ``nominal_freq_hz``, ``kpod_qnom_source`` to ``pop``.

    Priority mart → big → type_parse (plan §1).  Missing frequency assumes 50 Hz.
    Returns ``pop`` with the three columns added (index preserved).
    """
    out = pop.copy()
    base = out[["code", "install"]].copy()
    base = _asof_nearest(base, _mart_nominals(db_path), tol_days)
    base = _asof_nearest(base, _big_nominals(equipment_big_path), tol_days)

    q_mart = base["q_mart"]
    q_big = base["q_big"]
    q_parse = np.array([_parse_qnom(pt, gt) for pt, gt in zip(base["pump_type"], base["gno_type"])])

    qnom = np.where(q_mart.notna() & (q_mart > 0), q_mart, np.nan)
    src = np.where(q_mart.notna() & (q_mart > 0), "mart", "")
    use_big = np.isnan(qnom) & q_big.notna().to_numpy() & (q_big.fillna(0).to_numpy() > 0)
    qnom = np.where(use_big, q_big, qnom)
    src = np.where(use_big, "big", src)
    use_parse = np.isnan(qnom) & np.isfinite(q_parse) & (q_parse > 0)
    qnom = np.where(use_parse, q_parse, qnom)
    src = np.where(use_parse, "type_parse", src)
    src = np.where(src == "", "missing", src)

    # nominal frequency: mart → big → 50 Hz (plan §1).
    fnom = np.where(base["f_mart"].notna() & (base["f_mart"] > 0), base["f_mart"], np.nan)
    fnom = np.where(np.isnan(fnom) & base["f_big"].notna().to_numpy(), base["f_big"], fnom)
    fnom = np.where(np.isnan(fnom) | (fnom <= 0), 50.0, fnom)

    out["nominal_flow_m3d"] = qnom
    out["nominal_freq_hz"] = fnom
    out["kpod_qnom_source"] = src
    return out


# ---------------------------------------------------------------------------
# Daily telemetry → per-run window features
# ---------------------------------------------------------------------------

def load_kpod_dailies(well_keys: list[str], *, db_path: Path | None = None) -> pd.DataFrame:
    """qliq / freq daily rows for ``well_keys`` (casefolded well_key space)."""
    if not well_keys:
        return pd.DataFrame(columns=["well_key", "dt", "qliq", "freq"])
    con = sqlite3.connect(str(db_path or (WAREHOUSE_DIR / "pump2.db")))
    try:
        placeholders = ",".join("?" * len(well_keys))
        df = pd.read_sql(
            "SELECT well_key, dt, qliq, freq FROM proc__daily_merged "
            f"WHERE well_key IN ({placeholders})",
            con, params=list(well_keys),
        )
    finally:
        con.close()
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    lo, hi = FREQ_RANGE
    f = pd.to_numeric(df["freq"], errors="coerce")
    df["freq"] = f.where((f >= lo) & (f <= hi))  # out-of-range → NaN, like the SQL guard
    df["qliq"] = pd.to_numeric(df["qliq"], errors="coerce")
    return df.dropna(subset=["dt"])


def run_kpod_windows(
    dts: np.ndarray,
    qliq: np.ndarray,
    freq: np.ndarray,
    install: pd.Timestamp,
    stop: pd.Timestamp,
    qnom: float,
    *,
    t0_op_days: int = DEFAULT_T0_OP_DAYS,
    guard_days: int = DEFAULT_GUARD_DAYS,
    band_lo: float = DEFAULT_BAND_LO,
    band_hi: float = DEFAULT_BAND_HI,
    min_run_op_days: int = MIN_RUN_OP_DAYS,
) -> dict:
    """Compute the Kpod window family for one run from its daily arrays.

    Pure (no DB): ``dts`` datetimes, ``qliq``/``freq`` aligned daily arrays over
    ``[install, stop]``.  Operating day := ``qliq > 0``.  Returns t0 means, guarded
    run means, guarded frac-days bands (plain Kpod), the t0→run drift, and coverage
    counts + ``kpod_run_source`` ∈ {run, t0_fallback, none}.
    """
    dts = pd.to_datetime(pd.Series(dts)).to_numpy()
    q = np.asarray(qliq, dtype=float)
    f = np.asarray(freq, dtype=float)
    order = np.argsort(dts)
    dts, q, f = dts[order], q[order], f[order]

    op = np.isfinite(q) & (q > 0)
    dts_o, q_o, f_o = dts[op], q[op], f[op]

    out: dict = {
        "kpod_t0": np.nan, "kpod_freq_t0": np.nan, "kpod_t0_n_days": 0,
        "kpod_run": np.nan, "kpod_freq_run": np.nan, "kpod_run_n_days": 0,
        "frac_days_kpod_below": np.nan, "frac_days_kpod_above": np.nan,
        "kpod_run_source": "none",
        # Running frequency (Hz) is its own covariate — the operating speed the pump
        # is driven at — and does NOT need Qnom, so it is populated even for runs with
        # no design flow.
        "freq_t0": np.nan, "freq_run": np.nan, "frac_days_freq_over": np.nan,
        "freq_run_source": "none",
    }
    if len(q_o) == 0:
        return out

    # -- running-frequency window family (Qnom-independent) -------------------
    def _fmean(arr: np.ndarray) -> float:
        v = arr[np.isfinite(arr)]
        return float(v.mean()) if len(v) else np.nan

    guard_cut = pd.Timestamp(stop) - pd.Timedelta(days=guard_days)
    keep = dts_o <= np.datetime64(guard_cut)
    f_t0, f_run = f_o[:t0_op_days], f_o[keep]
    out["freq_t0"] = _fmean(f_t0)
    if np.isfinite(f_run).sum() >= min_run_op_days:
        out["freq_run"] = _fmean(f_run)
        fv = f_run[np.isfinite(f_run)]
        out["frac_days_freq_over"] = float(np.mean(fv > FREQ_OVERSPEED_HZ)) if len(fv) else np.nan
        out["freq_run_source"] = "run"
    elif np.isfinite(out["freq_t0"]):
        out["freq_run"] = out["freq_t0"]
        out["freq_run_source"] = "t0_fallback"

    if not np.isfinite(qnom) or qnom <= 0:
        return out

    # -- t0 window: first t0_op_days operating days ---------------------------
    t0 = compute_kpod_features(q_o[:t0_op_days], qnom, f_o[:t0_op_days])
    out["kpod_t0"] = t0["kpod_mean"]
    out["kpod_freq_t0"] = t0["kpod_freq_mean"]
    out["kpod_t0_n_days"] = int(t0["n_op_days"])

    # -- guarded run window: op-days up to stop − guard_days -------------------
    q_r, f_r = q_o[keep], f_o[keep]
    if len(q_r) >= min_run_op_days:
        run = compute_kpod_features(q_r, qnom, f_r)
        kp = q_r / qnom
        out["kpod_run"] = run["kpod_mean"]
        out["kpod_freq_run"] = run["kpod_freq_mean"]
        out["kpod_run_n_days"] = int(run["n_op_days"])
        out["frac_days_kpod_below"] = float(np.mean(kp < band_lo))
        out["frac_days_kpod_above"] = float(np.mean(kp > band_hi))
        out["kpod_run_source"] = "run"
    elif out["kpod_t0_n_days"] > 0:
        # Too few guarded op-days for an honest run mean — fall back to t0, flagged.
        out["kpod_run"] = out["kpod_t0"]
        out["kpod_freq_run"] = out["kpod_freq_t0"]
        out["kpod_run_n_days"] = out["kpod_t0_n_days"]
        out["kpod_run_source"] = "t0_fallback"
    return out


@dataclass
class KpodCoverage:
    n_runs: int
    n_with_qnom: int
    n_with_t0: int
    n_with_run: int          # honest guarded run window
    n_run_fallback: int      # fell back to t0

    def as_frame(self) -> pd.DataFrame:
        d = self.__dict__
        row = dict(d)
        row["share_run"] = self.n_with_run / self.n_runs if self.n_runs else np.nan
        return pd.DataFrame([row])


def build_kpod_family(
    pop: pd.DataFrame,
    *,
    as_of: pd.Timestamp | None = None,
    t0_op_days: int = DEFAULT_T0_OP_DAYS,
    guard_days: int = DEFAULT_GUARD_DAYS,
    band_lo: float = DEFAULT_BAND_LO,
    band_hi: float = DEFAULT_BAND_HI,
    suffix: str = "",
    db_path: Path | None = None,
    equipment_big_path: Path | None = None,
) -> tuple[pd.DataFrame, KpodCoverage]:
    """Attach the Kpod window family to ``pop`` (an ``esp_population`` selection with
    Qnom already resolved by :func:`resolve_qnom`).

    ``suffix`` lets a second guard length be materialised side by side (e.g.
    ``suffix='_g60'`` for the sensitivity contest) without clobbering the primary
    columns.  Open (running) runs are windowed to ``as_of`` when ``end`` is NaT.
    """
    if "nominal_flow_m3d" not in pop.columns:
        raise KeyError("build_kpod_family needs resolve_qnom() first")
    out = pop.copy()
    out["well_key"] = out["code"].astype(str).str.casefold()
    as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.max

    dailies = load_kpod_dailies(sorted(out["well_key"].unique().tolist()), db_path=db_path)
    by_well = {k: g for k, g in dailies.groupby("well_key", sort=False)}

    rows: list[dict] = []
    for _, run in out.iterrows():
        rec: dict = {"_idx": run.name}
        g = by_well.get(run["well_key"])
        if g is None or pd.isna(run["install"]) or not np.isfinite(run["nominal_flow_m3d"]):
            rows.append(rec)
            continue
        install = pd.Timestamp(run["install"])
        stop = pd.Timestamp(run["end"]) if pd.notna(run["end"]) else as_of
        win = g[(g["dt"] >= install) & (g["dt"] <= stop)]
        feats = run_kpod_windows(
            win["dt"].to_numpy(), win["qliq"].to_numpy(), win["freq"].to_numpy(),
            install, stop, float(run["nominal_flow_m3d"]),
            t0_op_days=t0_op_days, guard_days=guard_days,
            band_lo=band_lo, band_hi=band_hi,
        )
        rec.update(feats)
        rows.append(rec)

    feat = pd.DataFrame(rows).set_index("_idx")
    feat["kpod_run_minus_t0"] = feat["kpod_run"] - feat["kpod_t0"]
    feat["freq_run_minus_t0"] = feat["freq_run"] - feat["freq_t0"]
    if suffix:
        feat = feat.rename(columns={c: f"{c}{suffix}" for c in feat.columns})
    out = out.join(feat)

    run_src_col = f"kpod_run_source{suffix}"
    cov = KpodCoverage(
        n_runs=int(len(out)),
        n_with_qnom=int((out["kpod_qnom_source"] != "missing").sum()),
        n_with_t0=int(out[f"kpod_t0{suffix}"].notna().sum()),
        n_with_run=int((out[run_src_col] == "run").sum()),
        n_run_fallback=int((out[run_src_col] == "t0_fallback").sum()),
    )
    return out, cov


__all__ = [
    "resolve_qnom",
    "build_kpod_family",
    "run_kpod_windows",
    "load_kpod_dailies",
    "KpodCoverage",
    "KPOD_UNDERLOAD",
    "FREQ_RANGE",
    "DEFAULT_T0_OP_DAYS",
    "DEFAULT_GUARD_DAYS",
    "DEFAULT_BAND_LO",
    "DEFAULT_BAND_HI",
    "FREQ_OVERSPEED_HZ",
]
