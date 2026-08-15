"""Tests for the Mc failure/ГТМ competing-risks accounting.

Three families the handoff requires:
  * at-risk accounting sanity (open runs present, single clock, cohort filter on);
  * decomposition identity (failure + ГТМ + running partition the fleet / CIF);
  * a synthetic two-hazard simulation recovering known cause-specific parameters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.models.survival import cif as CIF
from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk import mc_competing_risks as M


# --------------------------------------------------------------------------- #
# Fixtures                                                                      #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def pop() -> pd.DataFrame:
    return M.build_population(C.SVOD_OPEN_ASOF)


# --------------------------------------------------------------------------- #
# At-risk accounting sanity                                                     #
# --------------------------------------------------------------------------- #
def test_open_runs_present_and_on_single_calendar_clock(pop: pd.DataFrame):
    asof = pd.Timestamp(C.SVOD_OPEN_ASOF)
    open_runs = pop[pop["is_open"]]
    assert len(open_runs) > 30, "running Mc fleet vanished — a silent open-run loss"
    # Every open run is timed on the calendar clock (t_cal == as_of - install),
    # NOT the mixed-clock tte.  This is the mixed-clock trap guard.
    cal_age = (asof - open_runs["install"]).dt.days.clip(lower=M.P.DAY_ZERO_TTE).astype(float)
    assert np.allclose(open_runs["t_cal"].to_numpy(float), cal_age.to_numpy(float))
    # open runs carry a positive current age = t_cal
    assert (open_runs["current_age"] > 0).all()
    assert np.allclose(open_runs["current_age"], open_runs["t_cal"])


def test_mc_cohort_filter_is_on(pop: pd.DataFrame):
    # No Мирнинский run installed before the 2024 cohort start may survive.
    assert (pop["install"] >= pd.Timestamp(C.MC_INSTALL_COHORT_START)).all()
    assert set(pop["field"].unique()).issubset(set(C.MC_COHORT_FIELDS))


def test_cause_codes_partition_the_population(pop: pd.DataFrame):
    codes = set(pop["cause_code"].unique())
    assert codes.issubset({M.CENSORED, M.FAILURE, M.GTM})
    n = len(pop)
    n_run = int((pop["cause_code"] == M.CENSORED).sum())
    n_fail = int((pop["cause_code"] == M.FAILURE).sum())
    n_gtm = int((pop["cause_code"] == M.GTM).sum())
    assert n_run + n_fail + n_gtm == n
    # every open run is censored and every censored-yet-closed row is admin-censored
    assert (pop.loc[pop["is_open"], "cause_code"] == M.CENSORED).all()


# --------------------------------------------------------------------------- #
# Decomposition / CIF identity                                                  #
# --------------------------------------------------------------------------- #
def test_cif_partitions_all_cause_incidence(pop: pd.DataFrame):
    t = pop["t_cal"].to_numpy(float)
    ec = pop["cause_code"].to_numpy(int)
    cif_f = CIF.aalen_johansen_cif(t, ec, M.FAILURE)
    cif_g = CIF.aalen_johansen_cif(t, ec, M.GTM)
    km_all = CIF.all_cause_km(t, (ec > 0).astype(int))  # 1 - S_all
    for tt in (90.0, 180.0, 365.0, 720.0):
        total = cif_f.at(tt) + cif_g.at(tt)
        # failure + ГТМ incidence == all-cause incidence (running = 1 - total ≥ 0)
        assert total == pytest.approx(km_all.at(tt), abs=1e-9)
        running = 1.0 - total
        assert -1e-9 <= running <= 1.0 + 1e-9


def test_renewal_conserves_fleet_mass():
    # A single well seeded at age 0; competing hazards + renewal keep exactly one
    # pump in the well at all times, so cumulative removals equal cumulative
    # renewals and mass is conserved.  We verify total event flow stays bounded by
    # the exposure and that both flows are non-negative.
    haz = M.BandHazards(
        fail=np.full(len(M.HAZARD_BANDS), 2e-3),
        gtm=np.full(len(M.HAZARD_BANDS), 1e-3),
    )
    fail, gtm = M.simulate_renewal([(0, 0, 1.0)], haz, n_days=365)
    assert (fail >= 0).all() and (gtm >= 0).all()
    # With constant total hazard 3e-3/day and one well, expected events over a year
    # ~ 365*3e-3 ≈ 1.1; renewal keeps it near that (mass never grows).
    assert 0.8 < (fail.sum() + gtm.sum()) < 1.4
    # cause split follows the hazard ratio (2:1)
    assert fail.sum() / gtm.sum() == pytest.approx(2.0, rel=0.05)


def test_decomposition_columns_are_consistent(pop: pd.DataFrame):
    asof = pd.Timestamp(C.SVOD_OPEN_ASOF)
    haz = M.band_hazards_from_pop(pop)
    obs = M.observed_window_prediction(pop, haz)
    dec = M.combined_decomposition(pop, haz, obs, as_of=asof, horizon_months=12)
    fc = dec[dec["segment"] == "forecast"]
    assert len(fc) == 12
    # total pulls == failures + ГТМ (up to 4-decimal CSV rounding)
    assert np.allclose(fc["m3_total_pulls"], fc["m1_obs_failures"] + fc["m2_gtm"], atol=2e-4)
    # observed-window predicted totals reproduce the actual counts (band hazards are
    # the empirical rates ⇒ bookkeeping is internally consistent)
    # (integer-day replay grid vs continuous person-days -> ~1e-4 discretization gap)
    assert obs["pred_failures"].sum() == pytest.approx(obs["actual_failures"].sum(), rel=1e-3)
    assert obs["pred_gtm"].sum() == pytest.approx(obs["actual_gtm"].sum(), rel=1e-3)


# --------------------------------------------------------------------------- #
# Synthetic two-hazard recovery                                                 #
# --------------------------------------------------------------------------- #
def _weibull_rvs(rng, beta, eta, n):
    return eta * (-np.log(rng.uniform(size=n))) ** (1.0 / beta)


def test_synthetic_competing_risks_recovers_parameters():
    rng = np.random.default_rng(7)
    n = 6000
    beta_f, eta_f = 0.9, 400.0
    beta_g, eta_g = 1.3, 300.0
    tf = _weibull_rvs(rng, beta_f, eta_f, n)
    tg = _weibull_rvs(rng, beta_g, eta_g, n)
    cens = rng.uniform(100, 900, size=n)  # administrative censoring
    obs = np.minimum(np.minimum(tf, tg), cens)
    cause = np.where(
        (tf <= tg) & (tf <= cens), M.FAILURE,
        np.where((tg < tf) & (tg <= cens), M.GTM, M.CENSORED),
    )
    df = pd.DataFrame(
        {
            "t_cal": np.clip(obs, 0.5, None),
            "cause_code": cause,
            "well_key": np.arange(n),  # one run per well -> no clustering shrinkage
        }
    )
    fit_f = M.fit_cause_specific(df, M.FAILURE, n_boot=0)
    fit_g = M.fit_cause_specific(df, M.GTM, n_boot=0)
    # cause-specific Weibull (other cause treated as censoring) recovers the truth
    assert fit_f.beta == pytest.approx(beta_f, rel=0.12)
    assert fit_f.eta == pytest.approx(eta_f, rel=0.15)
    assert fit_g.beta == pytest.approx(beta_g, rel=0.12)
    assert fit_g.eta == pytest.approx(eta_g, rel=0.15)


def test_piecewise_hazard_totals_match_event_counts(pop: pd.DataFrame):
    # hazard * person-days summed over bands == number of events (definitional).
    for code in (M.FAILURE, M.GTM):
        h = M.piecewise_hazard(pop, code)
        recovered = float((h["hazard_per_day"] * h["person_days"]).sum())
        assert recovered == pytest.approx(float(h["n_events"].sum()), rel=1e-9)
