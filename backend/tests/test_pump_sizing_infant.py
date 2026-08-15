"""Tests for the infant-band pump-sizing analysis.

The load-bearing machinery here is the counting-process split that lets one Cox model
carry a different ``log_qnom`` coefficient per follow-up interval.  It is easy to get
subtly wrong in ways that return plausible numbers, so the split is pinned from several
directions, and the ridge-penalty trap that flattened the first run of this analysis has
its own regression test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.workflows.production_risk import pump_sizing_infant as PSI

CUTS = (30.0, 90.0, 365.0)


def _frame(t, ev, q=None, well=None):
    n = len(t)
    return pd.DataFrame({
        PSI.CLOCK: np.asarray(t, dtype=float),
        PSI.EVENT_COL: np.asarray(ev, dtype=int),
        "log_qnom": np.log(np.asarray(q if q is not None else np.full(n, 250.0), dtype=float)),
        "nominal_flow_m3d": np.asarray(q if q is not None else np.full(n, 250.0), dtype=float),
        "well_key": well if well is not None else [f"w{i}" for i in range(n)],
    })


# ---------------------------------------------------------------------------
# Counting-process split
# ---------------------------------------------------------------------------

def test_split_produces_one_row_per_interval_entered():
    long = PSI.to_counting_process(_frame([500.0], [1]), cuts=CUTS)
    assert len(long) == 4
    assert long["start"].tolist() == [0.0, 30.0, 90.0, 365.0]
    assert long["stop"].tolist() == [30.0, 90.0, 365.0, 500.0]


def test_event_flag_lands_only_on_the_final_interval():
    long = PSI.to_counting_process(_frame([500.0], [1]), cuts=CUTS)
    assert long[PSI.EVENT_COL].tolist() == [0, 0, 0, 1]


def test_censored_run_carries_no_event_anywhere():
    long = PSI.to_counting_process(_frame([500.0], [0]), cuts=CUTS)
    assert long[PSI.EVENT_COL].sum() == 0


def test_short_run_occupies_only_the_first_interval():
    long = PSI.to_counting_process(_frame([12.0], [1]), cuts=CUTS)
    assert len(long) == 1
    assert long["stop"].iloc[0] == 12.0
    assert long[PSI.EVENT_COL].iloc[0] == 1


def test_run_ending_exactly_on_a_cut_does_not_create_an_empty_interval():
    long = PSI.to_counting_process(_frame([30.0], [1]), cuts=CUTS)
    assert len(long) == 1
    assert long["stop"].iloc[0] == 30.0
    assert (long["stop"] > long["start"]).all()


def test_every_interval_has_positive_width():
    rng = np.random.default_rng(0)
    t = rng.uniform(0.5, 900.0, 300)
    long = PSI.to_counting_process(_frame(t, rng.integers(0, 2, 300)), cuts=CUTS)
    assert (long["stop"] > long["start"]).all()


def test_total_events_are_conserved_by_the_split():
    rng = np.random.default_rng(1)
    d = _frame(rng.uniform(0.5, 900.0, 500), rng.integers(0, 2, 500))
    long = PSI.to_counting_process(d, cuts=CUTS)
    assert long[PSI.EVENT_COL].sum() == d[PSI.EVENT_COL].sum()


def test_total_exposure_is_conserved_by_the_split():
    rng = np.random.default_rng(2)
    d = _frame(rng.uniform(0.5, 900.0, 400), rng.integers(0, 2, 400))
    long = PSI.to_counting_process(d, cuts=CUTS)
    assert (long["stop"] - long["start"]).sum() == pytest.approx(d[PSI.CLOCK].sum(), rel=1e-10)


def test_band_columns_are_zero_outside_their_own_interval():
    long = PSI.to_counting_process(_frame([500.0], [1], q=[400.0]), cuts=CUTS)
    cols = [f"log_qnom__b{i}" for i in range(4)]
    m = long[cols].to_numpy()
    assert np.count_nonzero(m) == 4                      # one non-zero per row
    assert np.allclose(np.diag(m), np.log(400.0))        # on the diagonal


# ---------------------------------------------------------------------------
# The validity gate and the penalty trap
# ---------------------------------------------------------------------------

def _ph_data(n=1200, coef=0.6, seed=0, with_wells=False):
    """Exponential PH data with a known log_qnom coefficient and real censoring."""
    rng = np.random.default_rng(seed)
    q = np.exp(rng.uniform(np.log(50), np.log(1500), n))
    lam = np.exp(coef * (np.log(q) - np.log(250))) / 600.0
    t = rng.exponential(1.0 / lam)
    c = rng.uniform(30, 1500, n)
    wells = [f"w{i % (n // 4)}" for i in range(n)] if with_wells else None
    return _frame(np.minimum(t, c), (t <= c).astype(int), q=q, well=wells)


def test_counting_process_reproduces_plain_cox():
    """THE gate: collapsing the band columns must give back a plain Cox exactly."""
    g = PSI._check_reproduces_plain_cox(_ph_data())
    assert g["flag"] == "OK"
    assert g["abs_diff"] < 1e-3


def test_counting_process_reproduces_plain_cox_with_strata():
    g = PSI._check_reproduces_plain_cox(_ph_data(with_wells=True), strata="well_key")
    assert g["flag"] == "OK"
    assert g["abs_diff"] < 1e-3


def test_piecewise_recovers_a_constant_effect_as_constant():
    """Data generated with ONE coefficient must not show a spurious band gradient."""
    pw = PSI.piecewise_size_effect(_ph_data(n=3000, coef=0.6, seed=5), cuts=CUTS)
    assert len(pw) >= 3
    for _, r in pw.iterrows():
        assert r["HR_lo"] <= np.exp(0.6) <= r["HR_hi"], f"band {r['band']} missed the truth"
    assert PSI.band_homogeneity_test(pw)["p"] > 0.05


def test_ridge_penalty_flattens_the_bands_toward_the_null():
    """Regression test for the bug that produced the first (wrong) run of this analysis.

    A penalizer of 0.01 collapsed the real gradient 1.13/1.38/1.66/1.87 to
    1.02/1.04/1.12/1.15.  The default must stay 0, and a penalised fit must announce
    itself so a shrunk coefficient is never read as a null finding.
    """
    d = _ph_data(n=2000, coef=0.8, seed=9)
    free = PSI.piecewise_size_effect(d, cuts=CUTS, penalizer=0.0)
    pen = PSI.piecewise_size_effect(d, cuts=CUTS, penalizer=5.0)
    assert (pen["HR_per_e_fold"] < free["HR_per_e_fold"]).all()
    # ...and every penalised band is closer to HR = 1 than its unpenalised counterpart.
    assert (np.abs(np.log(pen["HR_per_e_fold"])) < np.abs(np.log(free["HR_per_e_fold"]))).all()


def test_unstable_bands_are_flagged_not_reported():
    """HR 14.8 [0.39, 557] is not a number and must be marked."""
    pw = pd.DataFrame({"HR_per_e_fold": [1.5, 14.8], "HR_lo": [1.2, 0.39],
                       "HR_hi": [1.9, 557.0], "note": ["", ""]})
    pw["unstable"] = pw["HR_hi"] / pw["HR_lo"] > 20.0
    assert pw["unstable"].tolist() == [False, True]
    # and an unstable band must not drive the homogeneity test
    assert np.isnan(PSI.band_homogeneity_test(pw[pw["unstable"]])["p"])


def test_homogeneity_test_detects_a_real_gradient():
    pw = pd.DataFrame({
        "HR_per_e_fold": [1.05, 1.40, 1.70, 1.90],
        "HR_lo": [0.95, 1.25, 1.55, 1.70], "HR_hi": [1.16, 1.57, 1.86, 2.12],
    })
    assert PSI.band_homogeneity_test(pw)["p"] < 0.01


def test_homogeneity_test_accepts_a_flat_profile():
    pw = pd.DataFrame({
        "HR_per_e_fold": [1.50, 1.52, 1.48, 1.51],
        "HR_lo": [1.35, 1.37, 1.33, 1.36], "HR_hi": [1.67, 1.69, 1.65, 1.68],
    })
    assert PSI.band_homogeneity_test(pw)["p"] > 0.5


# ---------------------------------------------------------------------------
# Mechanism + descriptive helpers
# ---------------------------------------------------------------------------

def test_failure_node_detects_a_planted_mode_shift():
    """Continuous size range with the node probability tied to size."""
    rng = np.random.default_rng(4)
    n = 900
    q = np.exp(rng.uniform(np.log(60), np.log(1200), n))
    p_motor = (np.log(q) - np.log(60)) / (np.log(1200) - np.log(60))   # 0 -> 1 with size
    node = np.where(rng.uniform(size=n) < p_motor, "motor", "seal")
    d = _frame(rng.uniform(1, 80, n), np.ones(n), q=q)
    d["failed_node"] = node
    _, test = PSI.failure_node_by_size(d)
    assert test["p"] < 0.01
    assert "differs" in test["note"]


def test_failure_node_degenerate_size_distribution_is_reported_not_crashed():
    """Two distinct pump sizes collapse every tercile edge — must degrade, not raise."""
    rng = np.random.default_rng(4)
    n = 300
    d = _frame(rng.uniform(1, 80, 2 * n), np.ones(2 * n),
               q=np.concatenate([np.full(n, 80.0), np.full(n, 900.0)]))
    d["failed_node"] = np.array(["seal"] * n + ["motor"] * n)
    _, test = PSI.failure_node_by_size(d)
    assert "p" not in test
    assert test["note"] == "degenerate table"


def test_failure_node_quiet_when_modes_are_size_independent():
    rng = np.random.default_rng(6)
    n = 600
    d = _frame(rng.uniform(1, 80, n), np.ones(n),
               q=np.exp(rng.uniform(np.log(60), np.log(1200), n)))
    d["failed_node"] = rng.choice(["seal", "motor", "cable"], n)
    _, test = PSI.failure_node_by_size(d)
    assert test["p"] > 0.05


def test_band_labels_partition_the_timeline():
    t = pd.Series([0.5, 29.9, 30.0, 89.0, 90.0, 364.0, 365.0, 5000.0])
    lab = PSI._band_labels(t)
    assert lab.notna().all()
    assert lab.nunique() == 4


def test_hazard_by_tercile_orders_by_pump_size():
    rng = np.random.default_rng(8)
    d = _ph_data(n=900, coef=0.9, seed=8)
    out = PSI.hazard_by_size_tercile(d)
    assert list(out["size_tercile"]) == ["small", "mid", "large"]
    assert out["qnom_median"].is_monotonic_increasing
    # a positive PH coefficient means bigger pumps must have lower RMST
    assert out["rmst"].iloc[0] > out["rmst"].iloc[-1]
