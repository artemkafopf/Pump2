"""Operational (Block 2) daily-telemetry feature derivations — pure functions.

Phase C C0 factors the Block-2 covariate maths out of the SQL builder into small
array-level functions so the two derivations the spec singles out — **kpod**
(коэффициент подачи = Ql/Qnominal) and **restart counting** — are unit-testable on
synthetic daily frames, independent of the warehouse.

Definitions (canonical — the SQL early-window path in
``analysis.data.run_covariates`` mirrors these exactly):

* ``kpod = qliq / nominal_flow_m3d``  (fraction of BEP delivery at design freq)
* ``kpod_freq = qliq / (nominal_flow_m3d · f/50)``  (BEP scaled to the running
  frequency — the affinity-law flow at *this* speed)
* ``frac_kpod_below_0p7`` — share of operating days with kpod < 0.7 (underload/gas)
* a **restart** is a 0→>0 transition in daily qliq (pump comes back on after an
  idle day); counted over the early *calendar* window and expressed per-100-days
  because restart shock needs the idle structure (idle days are absent from the
  operating-day axis).
* a **frequency step** is a day-over-day |Δf| > 1 Hz among valid-frequency days.

All functions ignore NaNs the way the SQL ``AVG``/guards do (invalid telemetry is
dropped before averaging), so a synthetic frame with a couple of out-of-range or
missing values exercises the same code path the warehouse hits.
"""
from __future__ import annotations

import numpy as np

# Day-over-day |Δf| above this (Hz) counts as a frequency step (thermal cycling).
FREQ_STEP_HZ = 1.0
# kpod below this is "underload" (gas interference / off-BEP low side).
KPOD_UNDERLOAD = 0.7
# Minimum valid days before a within-window std is meaningful.
MIN_STD_DAYS = 3


def compute_kpod_features(
    qliq: np.ndarray,
    nominal_flow_m3d: float,
    freq: np.ndarray | None = None,
) -> dict:
    """kpod mean, frequency-scaled kpod mean, and frac-below-0.7 over a daily window.

    Parameters
    ----------
    qliq
        Daily liquid rate (m³/d) over the window's operating days (qliq > 0).
    nominal_flow_m3d
        Pump nominal (design) flow at 50 Hz.  A non-positive / NaN nominal makes
        every kpod undefined → NaNs returned.
    freq
        Optional daily running frequency (Hz); required for ``kpod_freq_mean``.

    Returns
    -------
    dict with ``kpod_mean``, ``kpod_freq_mean``, ``frac_kpod_below_0p7``,
    ``n_op_days``.
    """
    q = np.asarray(qliq, dtype=float)
    nan = float("nan")
    if not np.isfinite(nominal_flow_m3d) or nominal_flow_m3d <= 0:
        return {"kpod_mean": nan, "kpod_freq_mean": nan,
                "frac_kpod_below_0p7": nan, "n_op_days": 0}

    kpod = q / nominal_flow_m3d
    valid = np.isfinite(kpod)
    n = int(valid.sum())
    if n == 0:
        return {"kpod_mean": nan, "kpod_freq_mean": nan,
                "frac_kpod_below_0p7": nan, "n_op_days": 0}

    kpod_mean = float(np.mean(kpod[valid]))
    frac_below = float(np.mean(kpod[valid] < KPOD_UNDERLOAD))

    kpod_freq_mean = nan
    if freq is not None:
        f = np.asarray(freq, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            denom = nominal_flow_m3d * f / 50.0
            kpod_freq = np.where((f > 0) & np.isfinite(f), q / denom, np.nan)
        vf = np.isfinite(kpod_freq)
        if vf.any():
            kpod_freq_mean = float(np.mean(kpod_freq[vf]))

    return {"kpod_mean": kpod_mean, "kpod_freq_mean": kpod_freq_mean,
            "frac_kpod_below_0p7": frac_below, "n_op_days": n}


def compute_cycling_features(
    qliq: np.ndarray,
    freq: np.ndarray | None = None,
    *,
    span_days: int | None = None,
) -> dict:
    """Restart count, frequency-step count (both per-100d) and early-window stds.

    A *restart* is a 0→>0 transition in the daily qliq sequence; a *freq step* is
    a day-over-day |Δf| > ``FREQ_STEP_HZ`` among consecutive valid-frequency days.
    Both are normalised to per-100-days using ``span_days`` (defaults to the length
    of the qliq sequence — i.e. the calendar span of the window).

    ``qliq`` NaNs are treated as idle (0), matching the SQL ``COALESCE(qliq,0)``
    behaviour used for restart detection.  ``freq`` NaNs (out-of-range telemetry)
    are dropped for both the step count and the std.
    """
    q = np.asarray(qliq, dtype=float)
    span = int(span_days) if span_days else max(len(q), 1)

    on = (np.nan_to_num(q, nan=0.0) > 0).astype(int)
    n_restarts = int(np.sum(np.diff(on) == 1)) if len(on) > 1 else 0

    freq_std = float("nan")
    n_steps = 0
    if freq is not None:
        f = np.asarray(freq, dtype=float)
        dfreq = np.abs(np.diff(f))
        n_steps = int(np.nansum(dfreq > FREQ_STEP_HZ))
        fv = f[np.isfinite(f)]
        if len(fv) >= MIN_STD_DAYS:
            freq_std = float(np.std(fv))

    return {
        "n_restarts_per_100d": round(100.0 * n_restarts / span, 3),
        "n_freq_steps_per_100d": round(100.0 * n_steps / span, 3),
        "freq_std": round(freq_std, 3) if np.isfinite(freq_std) else float("nan"),
        "n_restarts_raw": n_restarts,
        "n_steps_raw": n_steps,
        "span_days": span,
    }


def std_from_moments(mean: float, sq_mean: float, n: int) -> float:
    """Population std from AVG(x) and AVG(x²) (the SQL sum-of-squares path).

    Guards the tiny-negative that floating-point AVG(x²)−AVG(x)² can produce and
    returns NaN when fewer than :data:`MIN_STD_DAYS` days back the moments.
    """
    if n is None or n < MIN_STD_DAYS or not (np.isfinite(mean) and np.isfinite(sq_mean)):
        return float("nan")
    var = max(sq_mean - mean * mean, 0.0)
    return float(np.sqrt(var))


__all__ = [
    "compute_kpod_features",
    "compute_cycling_features",
    "std_from_moments",
    "FREQ_STEP_HZ",
    "KPOD_UNDERLOAD",
    "MIN_STD_DAYS",
]
