"""Within-run frequency **regime-shift** detection on daily telemetry.

Answers: *which pumps ran at 50–55 Hz and were later moved to 60+ Hz, without
being pulled?*  The shift has to happen **inside one run** (one physical pump,
one installation) — a speed difference between two runs of the same well is a
new pump, not a regime change, so the run window is the unit of analysis.

Detection (pure functions here, warehouse wiring in
``analysis.workflows.production_risk.freq_regime_shift``):

1. Keep operating days only — a valid daily frequency in ``FREQ_VALID_RANGE``.
   Idle days simply drop out of the axis, so the clock is *operating days*.
2. Smooth with a centred rolling **median** (``SMOOTH_WINDOW`` days).  A median
   kills single-day VFD spikes and restart transients that a mean would smear
   into the segments on both sides.
3. Scan every admissible split ``k`` and score it by band **purity**:
   ``frac(pre days in [50,55]) + frac(post days ≥ 60)``.  The best ``k`` is the
   changepoint.  A ``transition_days`` collar around ``k`` is excluded from both
   purities so a multi-week ramp is not charged against either segment.
4. Qualify the run: both purities above threshold, both segment medians inside
   their bands, and a **sustained** high streak (``min_segment_days``) — a
   two-week excursion to 60 Hz is not a regime change.

The scan is O(n) per run after two cumulative sums, so the whole fleet
(~1,200 runs with enough telemetry) scans in a couple of seconds.

Guards worth knowing before reading any output:

* daily telemetry starts **2018-01-01** — runs installed earlier are truncated
  and their "pre" segment may be missing entirely (``telemetry_left_truncated``).
* ``freq > 65`` is above the range any of the fleet's drives is rated for;
  it is kept but flagged (``freq_max``) rather than silently clipped.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

# Daily frequency outside this range is instrumentation noise, not a setpoint.
FREQ_VALID_RANGE = (20.0, 75.0)

# The "before" band — nameplate 50 Hz through the mild over-speed the fleet
# treats as normal.  Half-Hz slack on each side absorbs telemetry rounding.
LOW_BAND = (49.5, 55.5)
# The "after" threshold — 60 Hz and above (0.5 Hz slack for the same reason).
HIGH_THRESHOLD = 59.5

# Centred rolling-median window (operating days).
SMOOTH_WINDOW = 7
# Each segment must hold at least this many operating days to be a regime.
MIN_SEGMENT_DAYS = 30
# Operating days around the changepoint excluded from both purity counts.
TRANSITION_DAYS = 7
# Minimum share of a segment's days inside its band.
MIN_PURITY = 0.70


def rolling_median(values: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    """Centred rolling median over the operating-day axis (NaN-tolerant edges)."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n == 0:
        return v
    half = window // 2
    out = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        w = v[lo:hi]
        w = w[~np.isnan(w)]
        out[i] = np.median(w) if len(w) else np.nan
    return out


@dataclass
class RegimeShift:
    """Best 50–55 Hz → 60+ Hz split found in one run's operating-day series."""

    n_op_days: int
    detected: bool
    reason: str
    # index into the operating-day series: first day of the post segment
    cp_index: int = -1
    pre_days: int = 0
    post_days: int = 0
    pre_purity: float = float("nan")
    post_purity: float = float("nan")
    pre_median_hz: float = float("nan")
    post_median_hz: float = float("nan")
    shift_hz: float = float("nan")
    high_streak_days: int = 0
    freq_max: float = float("nan")
    reverted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _longest_true_streak(mask: np.ndarray) -> int:
    best = cur = 0
    for flag in mask:
        cur = cur + 1 if flag else 0
        best = max(best, cur)
    return best


def detect_regime_shift(
    freq: np.ndarray,
    *,
    low_band: tuple[float, float] = LOW_BAND,
    high_threshold: float = HIGH_THRESHOLD,
    min_segment_days: int = MIN_SEGMENT_DAYS,
    transition_days: int = TRANSITION_DAYS,
    min_purity: float = MIN_PURITY,
    smooth_window: int = SMOOTH_WINDOW,
) -> RegimeShift:
    """Find the best low→high split in one run's **operating-day** frequency series.

    ``freq`` must already be filtered to valid operating days and ordered by date.
    Returns a :class:`RegimeShift` whose ``detected`` flag says whether the run
    qualifies; the metrics are populated either way so near-misses are auditable.
    """
    f = np.asarray(freq, dtype=float)
    n = len(f)
    if n < 2 * min_segment_days:
        return RegimeShift(n_op_days=n, detected=False, reason="too_few_op_days")

    smooth = rolling_median(f, smooth_window)
    lo, hi = low_band
    is_low = (smooth >= lo) & (smooth <= hi)
    is_high = smooth >= high_threshold

    # Cumulative counts so every candidate split costs O(1).
    c_low = np.concatenate([[0], np.cumsum(is_low)])
    c_high = np.concatenate([[0], np.cumsum(is_high)])

    ks = np.arange(min_segment_days, n - min_segment_days + 1)
    # Collar around k excluded from both segments (ramp days belong to neither).
    pre_end = np.maximum(ks - transition_days, 1)
    post_start = np.minimum(ks + transition_days, n - 1)
    pre_purity = c_low[pre_end] / pre_end
    post_purity = (c_high[n] - c_high[post_start]) / (n - post_start)
    score = pre_purity + post_purity
    best = int(np.argmax(score))
    k = int(ks[best])

    pre = smooth[: pre_end[best]]
    post = smooth[post_start[best] :]
    pre_median = float(np.nanmedian(pre)) if len(pre) else float("nan")
    post_median = float(np.nanmedian(post)) if len(post) else float("nan")
    streak = _longest_true_streak(is_high[post_start[best] :])
    tail = smooth[-min_segment_days:]
    reverted = bool(np.nanmedian(tail) < high_threshold)

    result = RegimeShift(
        n_op_days=n,
        detected=False,
        reason="",
        cp_index=k,
        pre_days=int(pre_end[best]),
        post_days=int(n - post_start[best]),
        pre_purity=float(pre_purity[best]),
        post_purity=float(post_purity[best]),
        pre_median_hz=pre_median,
        post_median_hz=post_median,
        shift_hz=post_median - pre_median,
        high_streak_days=int(streak),
        freq_max=float(np.nanmax(f)),
        reverted=reverted,
    )

    if not (lo <= pre_median <= hi):
        result.reason = "pre_median_outside_low_band"
    elif post_median < high_threshold:
        result.reason = "post_median_below_high_threshold"
    elif result.pre_purity < min_purity:
        result.reason = "pre_segment_impure"
    elif result.post_purity < min_purity:
        result.reason = "post_segment_impure"
    elif streak < min_segment_days:
        result.reason = "high_regime_not_sustained"
    else:
        result.detected = True
        result.reason = "ok"
    return result
