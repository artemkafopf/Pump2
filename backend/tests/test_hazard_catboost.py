"""Tests for the discrete-time hazard CatBoost line."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.workflows.production_risk import hazard_catboost as hc


def _pop(rows: list[dict]) -> pd.DataFrame:
    base = {c: 1.0 for c in hc.COVARIATES}
    return pd.DataFrame([{**base, "code": f"w{i}", "entry": 0.0, **r} for i, r in enumerate(rows)])


class TestPersonPeriod:
    def test_exposure_sums_to_tte(self):
        pp = hc.person_period(_pop([{"tte": 200.0, "event": 1}]))
        assert pp["exposure"].sum() == pytest.approx(200.0)

    def test_event_lands_in_exactly_one_bin(self):
        pp = hc.person_period(_pop([{"tte": 200.0, "event": 1}]))
        assert pp["y"].sum() == 1
        hit = pp[pp["y"] == 1].iloc[0]
        assert hit["age_lo"] < 200.0 <= hit["age_hi"]

    def test_censored_run_contributes_exposure_but_no_event(self):
        pp = hc.person_period(_pop([{"tte": 200.0, "event": 0}]))
        assert pp["y"].sum() == 0
        assert pp["exposure"].sum() == pytest.approx(200.0)

    def test_day_zero_failure_kept_in_first_bin(self):
        # esp_population clips day-0 startup failures to 0.5 d rather than dropping them.
        pp = hc.person_period(_pop([{"tte": 0.5, "event": 1}]))
        assert pp["y"].sum() == 1
        assert pp.iloc[0]["age_lo"] == 0.0
        assert pp["exposure"].sum() == pytest.approx(0.5)

    def test_left_truncation_drops_pre_entry_exposure(self):
        pp = hc.person_period(_pop([{"tte": 200.0, "event": 1, "entry": 150.0}]))
        assert pp["exposure"].sum() == pytest.approx(50.0)
        assert (pp["age_hi"] > 150.0).all()
        assert pp["y"].sum() == 1

    def test_run_ended_before_entry_contributes_nothing(self):
        pp = hc.person_period(_pop([{"tte": 100.0, "event": 1, "entry": 150.0}]))
        assert pp.empty

    def test_bins_are_disjoint_and_ordered(self):
        pp = hc.person_period(_pop([{"tte": 900.0, "event": 0}]))
        assert (pp["age_lo"].to_numpy()[1:] == pp["age_hi"].to_numpy()[:-1]).all()


class TestEmpiricalHazard:
    def test_constant_hazard_population_recovers_its_rate(self):
        # Exponential lifetimes at a known rate; the binned estimate must land near it.
        rng = np.random.default_rng(0)
        rate = 0.001
        t = rng.exponential(1.0 / rate, size=4000)
        pp = hc.person_period(_pop([{"tte": float(x), "event": 1} for x in t]))
        emp = hc.empirical_hazard(pp)
        mid = emp[(emp["age_lo"] >= 60.0) & (emp["age_lo"] <= 650.0)]
        got = 30.4 * mid["events"].sum() / mid["pump_days"].sum()
        assert got == pytest.approx(30.4 * rate, rel=0.15)


def _flag(pop: pd.DataFrame) -> pd.DataFrame:
    out = pop.copy()
    out["covariates_present"] = out[hc.COVARIATES].notna().all(axis=1)
    return out


class TestCoverageReport:
    def test_splits_hazard_by_covariate_availability(self):
        # The diagnostic that matters: hot runs missing covariates, cold runs having them.
        hot = [{"tte": 50.0, "event": 1, **{c: np.nan for c in hc.COVARIATES}} for _ in range(10)]
        cold = [{"tte": 500.0, "event": 0} for _ in range(10)]
        rep = hc.coverage_report(_flag(_pop(hot + cold)))
        present = rep[rep["covariates_present"]].iloc[0]
        missing = rep[~rep["covariates_present"]].iloc[0]
        assert missing["n_events"] == 10 and present["n_events"] == 0
        assert missing["fail_per_month"] > present["fail_per_month"]


class TestSupportOverlap:
    def test_disjoint_covariate_is_not_transferable(self):
        # The pbubble_atm case: train 230..259, test 0..158 -> the model would extrapolate.
        train = _pop([{"tte": 100.0, "event": 1, "pbubble_atm": v} for v in (230.0, 250.0, 259.0)])
        test = _pop([{"tte": 100.0, "event": 1, "pbubble_atm": v} for v in (0.0, 80.0, 158.0)])
        row = hc.support_overlap(train, test, covariates=["pbubble_atm"]).iloc[0]
        assert row["share_in_train_range"] == 0.0
        assert not row["transferable"]

    def test_constant_test_covariate_is_not_transferable(self):
        # The nominal_freq_hz case: inside the train range, but no variance to exploit.
        train = _pop([{"tte": 100.0, "event": 1, "nominal_freq_hz": v} for v in (0.0, 50.0, 220.0)])
        test = _pop([{"tte": 100.0, "event": 1, "nominal_freq_hz": 50.0} for _ in range(3)])
        row = hc.support_overlap(train, test, covariates=["nominal_freq_hz"]).iloc[0]
        assert row["share_in_train_range"] == 1.0  # in range...
        assert row["test_is_constant"]             # ...but useless
        assert not row["transferable"]

    def test_overlapping_varying_covariate_is_transferable(self):
        train = _pop([{"tte": 100.0, "event": 1, "freq_std_early": v} for v in (0.0, 5.0, 14.0)])
        test = _pop([{"tte": 100.0, "event": 1, "freq_std_early": v} for v in (1.0, 6.0, 9.0)])
        row = hc.support_overlap(train, test, covariates=["freq_std_early"]).iloc[0]
        assert row["transferable"]


class TestDiscreteHazardModel:
    @staticmethod
    def _fit_constant(rate: float, n: int = 3000, seed: int = 0) -> hc.DiscreteHazardModel:
        rng = np.random.default_rng(seed)
        t = rng.exponential(1.0 / rate, size=n)
        pop = _pop([{"tte": float(x), "event": 1} for x in t])
        return hc.DiscreteHazardModel(random_seed=1).fit(hc.person_period(pop))

    def test_recovers_a_constant_hazard(self):
        rate = 0.002
        m = self._fit_constant(rate)
        cov = {c: 1.0 for c in hc.COVARIATES}
        got = m.rate([100.0, 300.0, 600.0], cov)
        assert np.allclose(got, rate, rtol=0.3)

    def test_p_fail_matches_the_exponential_closed_form(self):
        rate = 0.002
        m = self._fit_constant(rate)
        cov = {c: 1.0 for c in hc.COVARIATES}
        got = m.p_fail(100.0, 30.4, cov)
        assert got == pytest.approx(1.0 - np.exp(-rate * 30.4), rel=0.3)

    def test_p_fail_is_memoryless_under_constant_hazard(self):
        m = self._fit_constant(0.002)
        cov = {c: 1.0 for c in hc.COVARIATES}
        assert m.p_fail(100.0, 30.4, cov) == pytest.approx(m.p_fail(800.0, 30.4, cov), rel=0.35)

    def test_recovers_a_spike_then_flat_hazard(self):
        # The shape the shipped Weibull cannot express: 10x hazard for the first 30 d,
        # flat afterwards. This is the whole reason for modelling the hazard directly.
        rng = np.random.default_rng(3)
        spike, plateau = 0.02, 0.002
        t = np.where(
            rng.random(6000) < 1 - np.exp(-spike * 30),
            rng.uniform(0.5, 30.0, 6000),
            30.0 + rng.exponential(1.0 / plateau, 6000),
        )
        pop = _pop([{"tte": float(x), "event": 1} for x in t])
        m = hc.DiscreteHazardModel(random_seed=1).fit(hc.person_period(pop))
        cov = {c: 1.0 for c in hc.COVARIATES}
        early = float(m.rate([10.0], cov)[0])
        late = float(m.rate([600.0], cov)[0])
        assert early / late > 4.0, f"infant spike not recovered: {early:.5f} vs {late:.5f}"

    def test_learns_a_covariate_effect(self):
        rng = np.random.default_rng(5)
        rows = []
        for _ in range(3000):
            hot = rng.random() < 0.5
            rate = 0.006 if hot else 0.001
            rows.append({
                "tte": float(rng.exponential(1.0 / rate)),
                "event": 1,
                "nominal_freq_hz": 60.0 if hot else 50.0,
                "pbubble_atm": 1.0,
                "freq_std_early": 1.0,
                "log_glf_mean_opdays": 1.0,
            })
        m = hc.DiscreteHazardModel(random_seed=1).fit(hc.person_period(_pop(rows)))
        base = {"pbubble_atm": 1.0, "freq_std_early": 1.0, "log_glf_mean_opdays": 1.0}
        hot = float(m.rate([200.0], {**base, "nominal_freq_hz": 60.0})[0])
        cold = float(m.rate([200.0], {**base, "nominal_freq_hz": 50.0})[0])
        assert hot > 2.0 * cold

    def test_survival_is_monotone_decreasing(self):
        m = self._fit_constant(0.002)
        cov = {c: 1.0 for c in hc.COVARIATES}
        s = [m.survival(t, cov) for t in (0.0, 50.0, 200.0, 800.0)]
        assert s[0] == pytest.approx(1.0)
        assert all(a >= b for a, b in zip(s, s[1:]))

    def test_fit_refuses_too_few_events(self):
        pop = _pop([{"tte": 100.0, "event": 0} for _ in range(20)])
        with pytest.raises(ValueError, match="at least 3 events"):
            hc.DiscreteHazardModel().fit(hc.person_period(pop))
