"""Tests for the within-run 50-55 Hz -> 60+ Hz regime-shift detector."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from analysis.features.freq_regime import (
    HIGH_THRESHOLD,
    LOW_BAND,
    detect_regime_shift,
    rolling_median,
)
from analysis.workflows.production_risk.freq_regime_shift import (
    add_rate_response,
    scan_runs,
    summarise_by_field,
)


def _series(*blocks: tuple[float, int]) -> np.ndarray:
    return np.concatenate([np.full(n, hz, dtype=float) for hz, n in blocks])


# ---------------------------------------------------------------------------
# rolling_median
# ---------------------------------------------------------------------------

def test_rolling_median_kills_single_day_spike():
    f = np.full(21, 52.0)
    f[10] = 70.0
    smooth = rolling_median(f, window=7)
    assert smooth[10] == pytest.approx(52.0)


def test_rolling_median_tracks_a_real_step():
    f = _series((52.0, 20), (62.0, 20))
    smooth = rolling_median(f, window=7)
    assert smooth[5] == pytest.approx(52.0)
    assert smooth[34] == pytest.approx(62.0)


# ---------------------------------------------------------------------------
# detect_regime_shift — positives
# ---------------------------------------------------------------------------

def test_clean_step_is_detected():
    result = detect_regime_shift(_series((52.0, 100), (62.0, 100)))
    assert result.detected
    assert result.reason == "ok"
    assert result.pre_median_hz == pytest.approx(52.0)
    assert result.post_median_hz == pytest.approx(62.0)
    assert result.shift_hz == pytest.approx(10.0)
    assert abs(result.cp_index - 100) <= 10


def test_ramped_step_is_detected_despite_the_transition():
    """A three-week ramp belongs to neither regime — the collar absorbs it."""
    ramp = np.linspace(55.0, 60.0, 21)
    result = detect_regime_shift(np.concatenate([_series((52.0, 90)), ramp, _series((61.0, 90))]))
    assert result.detected, result.reason
    assert result.pre_median_hz <= LOW_BAND[1]
    assert result.post_median_hz >= HIGH_THRESHOLD


def test_shift_that_later_reverts_is_flagged_but_still_detected():
    f = np.concatenate([_series((52.0, 80)), _series((62.0, 200)), _series((52.0, 40))])
    result = detect_regime_shift(f)
    assert result.detected, result.reason
    assert result.reverted


# ---------------------------------------------------------------------------
# detect_regime_shift — negatives, each with its own reason
# ---------------------------------------------------------------------------

def test_flat_50hz_run_is_not_a_shift():
    result = detect_regime_shift(_series((52.0, 200)))
    assert not result.detected
    assert result.reason == "post_median_below_high_threshold"


def test_run_that_starts_high_is_not_a_shift():
    """Always-60 Hz is a setpoint, not a change of regime."""
    result = detect_regime_shift(_series((60.0, 200)))
    assert not result.detected
    assert result.reason == "pre_median_outside_low_band"


def test_step_that_stops_short_of_60hz_is_not_a_shift():
    result = detect_regime_shift(_series((50.0, 100), (57.0, 100)))
    assert not result.detected
    assert result.reason == "post_median_below_high_threshold"


def test_brief_60hz_excursion_is_not_a_regime():
    f = np.concatenate([_series((52.0, 120)), _series((62.0, 12)), _series((52.0, 120))])
    result = detect_regime_shift(f)
    assert not result.detected


def test_downward_shift_is_not_detected():
    result = detect_regime_shift(_series((62.0, 100), (52.0, 100)))
    assert not result.detected


def test_too_short_series_is_rejected_before_scanning():
    result = detect_regime_shift(_series((52.0, 20), (62.0, 20)))
    assert not result.detected
    assert result.reason == "too_few_op_days"
    assert result.cp_index == -1


def test_high_regime_must_be_sustained_not_intermittent():
    """Cycling between 62 and 50 Hz has the median and the purity but no streak."""
    blocks = [_series((52.0, 60))]
    for _ in range(6):
        blocks.append(_series((62.0, 20)))
        blocks.append(_series((50.0, 14)))
    result = detect_regime_shift(np.concatenate(blocks), min_purity=0.5)
    assert not result.detected
    assert result.reason == "high_regime_not_sustained"
    assert result.post_median_hz == pytest.approx(62.0)  # median alone would pass
    assert result.high_streak_days < 30


# ---------------------------------------------------------------------------
# scan_runs — warehouse wiring on a synthetic frame
# ---------------------------------------------------------------------------

def _daily_frame(well_key: str, start: str, freqs: np.ndarray) -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(freqs), freq="D")
    return pd.DataFrame({
        "well_key": well_key,
        "dt": dates,
        "freq": freqs,
        "qliq": 100.0,
        "watercut": 50.0,
        "load": 60.0,
        "kprod": 1.0,
        "gas_factor": 30.0,
    })


def test_scan_runs_splits_by_run_window_not_by_well():
    """Two runs of one well, 52 Hz then 62 Hz, must NOT read as one regime shift.

    A different pump went in — the speed difference is between installations.
    """
    daily = pd.concat([
        _daily_frame("W1", "2022-01-01", _series((52.0, 200))),
        _daily_frame("W1", "2022-08-01", _series((62.0, 200))),
    ], ignore_index=True)
    runs = pd.DataFrame([
        {"row_id": 1, "well": "W1", "well_key": "W1", "field": "Ya", "pad": "p",
         "contractor": "brt", "event": 1, "run_days": 200.0,
         "install_date": pd.Timestamp("2022-01-01"), "stop_date": pd.Timestamp("2022-07-19"),
         "pump_type": "x", "nominal_flow_m3d": 80.0, "nominal_freq_hz": 50.0,
         "motor_power_kw": 32.0, "failed_node": "ЭЦН"},
        {"row_id": 2, "well": "W1", "well_key": "W1", "field": "Ya", "pad": "p",
         "contractor": "brt", "event": 0, "run_days": 200.0,
         "install_date": pd.Timestamp("2022-08-01"), "stop_date": pd.Timestamp("2023-02-16"),
         "pump_type": "x", "nominal_flow_m3d": 80.0, "nominal_freq_hz": 50.0,
         "motor_power_kw": 32.0, "failed_node": "нет"},
    ])
    scanned = scan_runs(runs, daily)
    assert len(scanned) == 2
    assert not scanned["detected"].any()


def test_scan_runs_detects_a_within_run_shift_and_reports_context():
    daily = _daily_frame("W2", "2022-01-01", _series((52.0, 120), (62.0, 120)))
    runs = pd.DataFrame([{
        "row_id": 7, "well": "W2", "well_key": "W2", "field": "Vt", "pad": "p",
        "contractor": "slb", "event": 1, "run_days": 240.0,
        "install_date": pd.Timestamp("2022-01-01"), "stop_date": pd.Timestamp("2022-08-28"),
        "pump_type": "x", "nominal_flow_m3d": 80.0, "nominal_freq_hz": 50.0,
        "motor_power_kw": 32.0, "failed_node": "ПЭД",
    }])
    scanned = scan_runs(runs, daily)
    row = scanned.iloc[0]
    assert bool(row["detected"])
    assert row["shift_op_day"] == pytest.approx(120, abs=10)
    assert row["days_shift_to_stop"] > 0
    assert row["shift_calendar_day"] > 0
    assert not row["telemetry_left_truncated"]
    # context means are carried for both segments
    assert row["qliq_pre"] == pytest.approx(100.0)
    assert row["qliq_post"] == pytest.approx(100.0)

    summary = summarise_by_field(scanned)
    assert summary.loc[summary["field"] == "Vt", "n_detected"].iloc[0] == 1
    assert summary.loc[summary["field"] == "Vt", "detected_pct"].iloc[0] == pytest.approx(100.0)


def test_rate_response_is_measured_against_the_affinity_expectation():
    scanned = pd.DataFrame({
        "qliq_pre30": [100.0, 100.0, 0.0],
        "qliq_post30": [120.0, 100.0, 50.0],
        "pre_median_hz": [50.0, 50.0, 50.0],
        "post_median_hz": [60.0, 60.0, 60.0],
    })
    out = add_rate_response(scanned)
    # +20 % delivered against +20 % expected — the shift paid off exactly
    assert out["qliq_response_pct"].iloc[0] == pytest.approx(20.0)
    assert out["qliq_affinity_expected_pct"].iloc[0] == pytest.approx(20.0)
    assert out["rate_shortfall_pct"].iloc[0] == pytest.approx(0.0)
    # flat rate on a +20 % speed-up is a 20-point shortfall
    assert out["rate_shortfall_pct"].iloc[1] == pytest.approx(-20.0)
    # idle into the changepoint — undefined, never infinite
    assert np.isnan(out["qliq_response_pct"].iloc[2])


def test_scan_runs_skips_runs_without_enough_telemetry():
    daily = _daily_frame("W3", "2022-01-01", _series((52.0, 20), (62.0, 20)))
    runs = pd.DataFrame([{
        "row_id": 9, "well": "W3", "well_key": "W3", "field": "Ya", "pad": "p",
        "contractor": "brt", "event": 1, "run_days": 40.0,
        "install_date": pd.Timestamp("2022-01-01"), "stop_date": pd.Timestamp("2022-02-09"),
        "pump_type": "x", "nominal_flow_m3d": 80.0, "nominal_freq_hz": 50.0,
        "motor_power_kw": 32.0, "failed_node": "ЭЦН",
    }])
    assert scan_runs(runs, daily).empty
