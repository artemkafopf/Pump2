"""Tests for the CatBoost failure-rate line and the injected-p_fail_fn refactor.

These are deliberately CatBoost-free (no model fitting): the bridge math, the injection
parity, the covariate-join tiers, and the well-clustered fold assignment are all tested
without touching ``catboost``.  The end-to-end fit is exercised by the comparison run.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk import failure_rate as fr
from analysis.workflows.production_risk import failure_rate_catboost as cb
from analysis.workflows.production_risk.survival import StrataModel, current_pump_p_fail


# --------------------------------------------------------------------------- #
# Bridge math                                                                  #
# --------------------------------------------------------------------------- #
def test_survival_anchors_and_monotone():
    surv, b90 = cb.build_cb_survival(100.0, 300.0, 600.0)
    assert b90 == 600.0
    assert surv(0.0) == 1.0
    assert surv(100.0) == pytest.approx(0.90, abs=1e-9)   # F(b10)=0.10
    assert surv(300.0) == pytest.approx(0.50, abs=1e-9)   # F(b50)=0.50
    assert surv(600.0) == pytest.approx(0.10, abs=1e-9)   # F(b90)=0.90
    prev = 1.0
    for t in range(0, 3000, 5):
        s = surv(float(t))
        assert s <= prev + 1e-12
        assert 0.0 < s <= 1.0
        prev = s


def test_exponential_tail_hazard_continuity():
    surv, _ = cb.build_cb_survival(100.0, 300.0, 600.0)
    # implied hazard of the b50->b90 segment == tail hazard just past b90
    import math
    lam = math.log(0.5 / 0.1) / (600.0 - 300.0)
    assert surv(700.0) == pytest.approx(0.10 * math.exp(-lam * 100.0), rel=1e-6)


def test_degenerate_anchors_no_div0():
    for anchors in [(0.0, 0.0, 0.0), (-5.0, -5.0, -5.0), (50.0, 50.0, 50.0)]:
        surv, b90 = cb.build_cb_survival(*anchors)
        assert b90 > 0
        # valid steep-but-finite CDF: strictly between 0 and 1, monotone, no exception
        assert surv(0.0) == 1.0
        assert 0.0 < surv(b90) <= 1.0
        # p_fail well-defined and finite everywhere
        assert 0.0 <= cb.catboost_p_fail(surv, 10.0, 30.0) <= 1.0


def test_quantile_crossing_guard_still_monotone():
    # raw predict_ttf is only row-sorted, but feed crossed/tied anchors anyway
    surv, _ = cb.build_cb_survival(400.0, 200.0, 200.0)
    prev = 1.0
    for t in range(0, 1500, 5):
        s = surv(float(t))
        assert s <= prev + 1e-12
        prev = s


def test_bridge_reproduces_weibull_conditional():
    """Fed a Weibull-shaped S, catboost_p_fail must equal current_pump_p_fail exactly."""
    model = StrataModel(bundle_date=C.BUNDLE_DATE)
    params, _ = model.resolve("Vt", "nonsour", "Pooled")
    weibull_surv = lambda t: float(model.S(float(t), params))
    for age in (0, 50, 200, 900):
        for op in (10.0, 30.0, 90.0):
            got = cb.catboost_p_fail(weibull_surv, float(age), op)
            want = current_pump_p_fail({age: 1.0}, params, op, model)
            assert got == pytest.approx(want, abs=1e-12)


def test_pmf_wrapper_matches_current_pump_p_fail():
    model = StrataModel(bundle_date=C.BUNDLE_DATE)
    params, _ = model.resolve("Ya", "nonsour", "Pooled")
    weibull_surv = lambda t: float(model.S(float(t), params))
    pmf = {30: 0.4, 400: 0.6}
    got = cb.catboost_p_fail_pmf(weibull_surv, pmf, 45.0)
    want = current_pump_p_fail(pmf, params, 45.0, model)
    assert got == pytest.approx(want, abs=1e-12)


# --------------------------------------------------------------------------- #
# Injection parity: default callable leaves the Weibull path byte-identical     #
# --------------------------------------------------------------------------- #
class _StubPlan:
    months: list[str] = []          # no plan months -> uptime-branch replay
    op_days = pd.DataFrame()
    op_days_raw = pd.DataFrame()
    cal_days = pd.Series(dtype=float)


def test_injected_default_equals_weibull_closure():
    model = StrataModel(bundle_date=C.BUNDLE_DATE)
    params, _ = model.resolve("Mc", "nonsour", "Pooled")
    months = fr._month_range("2020-01", "2021-12")
    active = pd.DataFrame()
    common = dict(
        plan=_StubPlan(), active=active, model=model, field="УН-Тест", code="Mc_777",
        months=months, start=datetime(2020, 3, 1), end=datetime(2021, 6, 1),
        params=params, total_op=None, global_pooled=False,
    )
    rows_default: list[dict] = []
    fr._append_interval_predictions(rows_default, **common)  # p_fail_fn=None -> Weibull
    rows_explicit: list[dict] = []
    explicit = lambda age_pmf, op_days: current_pump_p_fail(age_pmf, params, op_days, model)
    fr._append_interval_predictions(rows_explicit, **common, p_fail_fn=explicit)
    assert rows_default == rows_explicit
    assert rows_default  # non-empty: the interval actually produced predictions


# --------------------------------------------------------------------------- #
# Well-clustered cross-fit fold assignment                                     #
# --------------------------------------------------------------------------- #
def test_fold_membership_is_well_clustered():
    # 6 wells, several runs each; folds must never split a well across folds
    join_keys = []
    for w in ("ya_1", "ya_2", "vt_3", "vt_4", "mc_5", "mc_6"):
        join_keys += [w] * 3
    fold = cb.CatBoostFailureModel._fold_membership(join_keys, n_folds=3)
    df = pd.DataFrame({"well": join_keys, "fold": fold})
    per_well_folds = df.groupby("well")["fold"].nunique()
    assert (per_well_folds == 1).all()
    assert set(fold.tolist()) == {0, 1, 2}


# --------------------------------------------------------------------------- #
# Covariate join tiers + diagnostics (no model fitting)                        #
# --------------------------------------------------------------------------- #
def _prewired_model() -> cb.CatBoostFailureModel:
    """A CatBoostFailureModel with quantiles injected directly (skip the fit)."""
    frame = pd.DataFrame({"row_id": [], "well_key": [], "install_date": [], "field": []})
    m = cb.CatBoostFailureModel(frame, well_field={"Ya_100": "УН-А"})
    # matched run: well ya_100 installed 2021-01-10
    m._quant["insample"]["r1"] = (100.0, 300.0, 600.0)
    m._quant["xfit"]["r1"] = (120.0, 320.0, 640.0)
    m._by_well["ya_100"] = [(pd.Timestamp("2021-01-10"), "r1")]
    # same-well categorical-only fallback
    m._quant_well["ya_100"] = (80.0, 260.0, 520.0)
    # absent-well field grid
    m._quant_grid[("Ya", "2020-2022")] = (90.0, 280.0, 560.0)
    return m


def test_resolve_curve_match_tier():
    m = _prewired_model()
    surv, b90, matched = m._resolve_curve("ya_100", datetime(2021, 1, 12), "xfit")  # within ±7d
    assert matched is True
    assert b90 == 640.0  # xfit quantile used


def test_resolve_curve_same_well_partial_tier():
    m = _prewired_model()
    # same well but install date far from the only run -> categorical-only (partial)
    surv, b90, matched = m._resolve_curve("ya_100", datetime(2023, 6, 1), "xfit")
    assert matched is False
    assert b90 == 520.0


def test_resolve_curve_absent_well_grid_tier(monkeypatch):
    m = _prewired_model()
    monkeypatch.setattr(cb.crosswalk, "map_model_field_from_well", lambda code: "Ya")
    surv, b90, matched = m._resolve_curve("ya_999", datetime(2021, 5, 1), "xfit")
    assert matched is False
    assert b90 == 560.0  # ("Ya","2020-2022") grid entry


def test_p_fail_fn_accumulates_covariate_and_tail_diagnostics():
    m = _prewired_model()
    diag = cb._ReplayDiag()
    fn = m.p_fail_fn("ya_100", datetime(2021, 1, 12), "УН-А", "xfit", diag)  # matched, b90=640
    fn({100: 1.0}, 30.0)            # age below b90 -> not a tail eval
    fn({700: 1.0}, 30.0)            # age above b90 -> tail eval
    assert diag.n_eval["УН-А"] == pytest.approx(2.0)
    assert diag.n_eval_past_b90["УН-А"] == pytest.approx(1.0)
    assert diag.mass_full["УН-А"] > 0.0
    assert diag.mass_partial.get("УН-А", 0.0) == 0.0
    share = diag.covariate_share_by_field()
    assert share["УН-А"] == pytest.approx(1.0)      # all mass from the matched (full) run
    assert share["__global__"] == pytest.approx(1.0)
    tail = diag.past_b90_share_by_field()
    assert tail["УН-А"] == pytest.approx(0.5)
