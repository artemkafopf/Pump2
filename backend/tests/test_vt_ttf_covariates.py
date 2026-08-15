"""Unit tests for the pure Kpod feature-family logic (Vt TTF-covariate workflow).

Covers the two properties the workflow leans on and that are hard to eyeball once
the DB is in the loop: the **differential tail guard** (§2) and the **Qnom source
ladder** (§1).  DB-driven builders are exercised end-to-end by the run script, not
here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.workflows.production_risk import kpod_features as K


def _daily(install: str, n_days: int, qliq, freq=50.0):
    dts = pd.date_range(install, periods=n_days, freq="D")
    q = np.asarray(qliq, dtype=float)
    f = np.full(n_days, freq, dtype=float) if np.isscalar(freq) else np.asarray(freq, dtype=float)
    return dts.to_numpy(), q, f


def test_tail_guard_excludes_the_death_tail():
    # 100 op-days at Kpod 1.0 (qliq==Qnom) then a 30-day collapse before stop.
    qnom = 100.0
    body = [100.0] * 100
    tail = [10.0] * 30            # failing pump's qliq collapses before the pull
    dts, q, f = _daily("2022-01-01", 130, body + tail)
    install, stop = pd.Timestamp("2022-01-01"), pd.Timestamp("2022-01-01") + pd.Timedelta(days=129)
    r = K.run_kpod_windows(dts, q, f, install, stop, qnom, guard_days=30)
    # The guarded run mean must see only the healthy body (Kpod≈1.0), not the tail.
    assert r["kpod_run_source"] == "run"
    assert abs(r["kpod_run"] - 1.0) < 1e-6
    # Without the guard the last 30 days would drag the mean well below 1.0.
    assert r["kpod_run"] > 0.99


def test_short_run_falls_back_to_t0_flagged():
    # A run whose entire body is inside the guard window has no honest run mean.
    qnom = 80.0
    dts, q, f = _daily("2022-03-01", 20, [80.0] * 20)   # 20 op-days, all in last 30d
    install, stop = pd.Timestamp("2022-03-01"), pd.Timestamp("2022-03-01") + pd.Timedelta(days=19)
    r = K.run_kpod_windows(dts, q, f, install, stop, qnom, guard_days=30)
    assert r["kpod_run_source"] == "t0_fallback"
    assert r["kpod_run"] == r["kpod_t0"]
    assert r["kpod_t0_n_days"] == 20


def test_frac_days_bands_are_guarded():
    qnom = 100.0
    # 60 days below 0.7 (qliq 50), 40 days above 1.1 (qliq 130), then collapse tail.
    body = [50.0] * 60 + [130.0] * 40
    dts, q, f = _daily("2021-01-01", 130, body + [5.0] * 30)
    install, stop = pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-01") + pd.Timedelta(days=129)
    r = K.run_kpod_windows(dts, q, f, install, stop, qnom, guard_days=30,
                           band_lo=0.7, band_hi=1.1)
    assert abs(r["frac_days_kpod_below"] - 0.6) < 1e-6
    assert abs(r["frac_days_kpod_above"] - 0.4) < 1e-6


def test_kpod_freq_scales_with_running_frequency():
    # At 40 Hz the affinity-law capacity is Qnom*40/50, so a fixed qliq reads a
    # higher freq-adjusted Kpod than the plain one.
    qnom = 100.0
    dts, q, f = _daily("2022-01-01", 40, [80.0] * 40, freq=40.0)
    install, stop = pd.Timestamp("2022-01-01"), pd.Timestamp("2022-01-01") + pd.Timedelta(days=39)
    r = K.run_kpod_windows(dts, q, f, install, stop, qnom, guard_days=0)
    assert abs(r["kpod_run"] - 0.8) < 1e-6                 # 80/100
    assert abs(r["kpod_freq_run"] - (80.0 / (100.0 * 40 / 50))) < 1e-6  # 1.0


def test_no_qnom_returns_nan_family():
    dts, q, f = _daily("2022-01-01", 40, [80.0] * 40)
    install, stop = pd.Timestamp("2022-01-01"), pd.Timestamp("2022-01-01") + pd.Timedelta(days=39)
    r = K.run_kpod_windows(dts, q, f, install, stop, np.nan)
    assert np.isnan(r["kpod_run"]) and r["kpod_run_source"] == "none"


def test_frequency_family_is_qnom_independent_and_tail_guarded():
    # Running frequency is a covariate in its own right; it must be populated even
    # when Qnom is missing (kpod is NaN there) and must respect the same tail guard.
    body = [52.0] * 100          # steady 52 Hz body
    tail = [58.0] * 30           # a late over-speed episode inside the guard window
    dts, q, f = _daily("2022-01-01", 130, [80.0] * 130, freq=np.array(body + tail))
    install, stop = pd.Timestamp("2022-01-01"), pd.Timestamp("2022-01-01") + pd.Timedelta(days=129)
    r = K.run_kpod_windows(dts, q, f, install, stop, np.nan, guard_days=30)
    assert np.isnan(r["kpod_run"])                       # no Qnom → kpod undefined
    assert abs(r["freq_run"] - 52.0) < 1e-6              # guard excludes the 58 Hz tail
    assert r["freq_run_source"] == "run"
    assert r["frac_days_freq_over"] == 0.0               # no >55 Hz day survives the guard


def test_frequency_overspeed_share():
    freqs = [58.0] * 40 + [50.0] * 60                    # 40% of op-days over 55 Hz
    dts, q, f = _daily("2021-01-01", 100, [70.0] * 100, freq=np.array(freqs))
    install, stop = pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-01") + pd.Timedelta(days=99)
    r = K.run_kpod_windows(dts, q, f, install, stop, np.nan, guard_days=0)
    assert abs(r["frac_days_freq_over"] - 0.4) < 1e-6


def test_resolve_qnom_source_ladder():
    # mart wins when present; big fills a mart gap; type_parse (off the mart pump_type
    # string when its flow is null) is the last resort; missing when all are silent.
    pop = pd.DataFrame({
        "code": ["VT_1", "VT_2", "VT_3", "VT_4"],
        "install": pd.to_datetime(["2022-01-01"] * 4),
    })
    inst = pd.to_datetime(["2022-01-01"])

    def fake_mart(_):
        return pd.DataFrame({
            "code": ["VT_1", "VT_3"],
            "install": pd.to_datetime(["2022-01-01", "2022-01-01"]),
            "q_mart": [120.0, np.nan],           # VT_3 has a type string but no flow
            "f_mart": [50.0, np.nan],
            "pump_type": [None, "ЭЦН5-80-1600"],
        })

    def fake_big(_):
        return pd.DataFrame({"code": ["VT_2"], "install": inst,
                             "q_big": [200.0], "f_big": [np.nan], "gno_type": [None]})

    import analysis.workflows.production_risk.kpod_features as mod
    orig_m, orig_b = mod._mart_nominals, mod._big_nominals
    mod._mart_nominals, mod._big_nominals = fake_mart, fake_big
    try:
        res = mod.resolve_qnom(pop)
    finally:
        mod._mart_nominals, mod._big_nominals = orig_m, orig_b
    src = dict(zip(res["code"], res["kpod_qnom_source"]))
    qn = dict(zip(res["code"], res["nominal_flow_m3d"]))
    assert src["VT_1"] == "mart"
    assert src["VT_2"] == "big"
    assert src["VT_3"] == "type_parse" and abs(qn["VT_3"] - 80.0) < 1e-6
    assert src["VT_4"] == "missing"
    # nominal frequency defaults to 50 where every source is silent.
    assert float(res.loc[res["code"] == "VT_2", "nominal_freq_hz"].iloc[0]) == 50.0
