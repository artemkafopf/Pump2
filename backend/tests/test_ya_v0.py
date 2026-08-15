"""Tests for the Ya v0 base competing-risks model.

Everything here runs on **synthetic populations** with known parameters — no warehouse, no
Свод — so the suite stays fast and tests the estimator rather than today's data.  The three
things worth breaking on are:

* the competing-risks composition identity ``S_all + Σ CIF_k ≡ 1`` (a wrong CIF construction
  is silent: the curves still look plausible, they just do not add up);
* recovery of known cause-specific parameters from censored competing-risks data, including
  the case where a *third* of the sample is administratively censored;
* the clock verdict actually firing on a differentially-available clock — the whole point of
  the audit is that it refuses the honest-looking-but-mixed column.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from analysis.models.survival import mixture_baseline as MB  # noqa: E402
from analysis.workflows.production_risk import ya_v0 as M  # noqa: E402
from analysis.workflows.production_risk.competing_risks_core import (  # noqa: E402
    CENSORED, FAILURE, GTM,
)


# ---------------------------------------------------------------------------
# Synthetic populations
# ---------------------------------------------------------------------------
def _weibull(rng, n, beta, eta):
    return eta * rng.weibull(beta, size=n)


def make_competing_pop(
    n: int = 2500,
    *,
    fail=(1.2, 500.0),
    gtm=(1.6, 900.0),
    admin_censor: float | None = 1200.0,
    seed: int = 7,
    n_wells: int = 200,
    well_frailty: float = 0.0,
) -> pd.DataFrame:
    """Latent failure times for two independent causes; the earlier one is observed.

    This is exactly the data-generating process v0 assumes, so a fit that cannot recover it
    here has an implementation bug rather than a modelling disagreement.

    ``well_frailty`` (log-normal σ on each well's η) makes runs of one well genuinely
    dependent.  At the default 0 the rows are iid *despite* sharing well codes — which is
    why a cluster bootstrap is NOT expected to be wider there, and why the test that checks
    it must ask for frailty explicitly.
    """
    rng = np.random.default_rng(seed)
    well_idx = np.arange(n) % n_wells
    if well_frailty > 0:
        frailty = np.exp(rng.normal(0.0, well_frailty, size=n_wells))[well_idx]
    else:
        frailty = np.ones(n)
    t_fail = _weibull(rng, n, fail[0], 1.0) * fail[1] * frailty
    t_gtm = _weibull(rng, n, gtm[0], 1.0) * gtm[1] * frailty
    t = np.minimum(t_fail, t_gtm)
    code = np.where(t_fail <= t_gtm, FAILURE, GTM)
    if admin_censor is not None:
        cens = rng.uniform(0.3 * admin_censor, admin_censor, size=n)
        censored = cens < t
        t = np.where(censored, cens, t)
        code = np.where(censored, CENSORED, code)
    t = np.clip(t, 0.5, None)

    install = pd.Timestamp("2015-01-01") + pd.to_timedelta(
        rng.integers(0, 3800, size=n), unit="D")
    return pd.DataFrame({
        "code": [f"YA_{w:04d}" for w in well_idx],
        "well_key": [f"YA_{w:04d}" for w in well_idx],
        "install": install,
        "end": install + pd.to_timedelta(t, unit="D"),
        "t_cal": t,
        "t_mix": t * 0.9,
        "t_nno": t * 0.98,
        "t_nno_reported": np.where(code == CENSORED, np.nan, t * 0.98),
        "t_nno_substituted": code == CENSORED,
        "t_nno_valid": True,
        "t_op": np.where(rng.random(n) < 0.5, t * 0.9, np.nan),
        "t_mix_source": np.where(rng.random(n) < 0.5, "measured", "imputed_field_kexp"),
        "cause_code": code,
        "cause": pd.Series(code).map({CENSORED: "running", FAILURE: "failure", GTM: "gtm"}),
        "is_open": code == CENSORED,
        "current_age": t,
        "pull_reason": np.where(code == GTM, "ГТМ", np.where(code == FAILURE, "клин", "")),
        "has_failed_unit": code == FAILURE,
        "contractor_group": "brt",
        "h2s_class": "nonsour",
    })


@pytest.fixture(scope="module")
def pop() -> pd.DataFrame:
    return make_competing_pop()


@pytest.fixture(scope="module")
def model(pop) -> M.YaV0Fit:
    # 8 starts keeps the suite quick; the recovery assertions below are loose enough that a
    # slightly worse local optimum still passes, and tight enough to catch a real regression.
    return M.fit_v0(pop, "t_cal", "full", num_starts=8)


# ---------------------------------------------------------------------------
# Composition: the identity that a wrong CIF construction breaks silently
# ---------------------------------------------------------------------------
def test_cif_identity_holds(model):
    """S_all(t) + Σ_k CIF_k(t) == 1 at every grid point."""
    f = model.curve_frame(max_time=2000.0, n_grid=4000)
    assert np.abs(f["identity_error"]).max() < 5e-3


def test_s_all_is_the_product_of_cause_curves(model):
    u = np.linspace(0.0, 1500.0, 200)
    expected = model.S_cause("failure", u) * model.S_cause("gtm", u)
    np.testing.assert_allclose(model.S_all(u), expected, rtol=1e-10)


def test_cif_is_monotone_and_bounded(model):
    f = model.curve_frame(max_time=2000.0)
    for cause in M.CAUSES:
        c = f[f"cif_{cause}"].to_numpy(float)
        assert np.all(np.diff(c) >= -1e-12), f"CIF for {cause} decreases"
        assert 0.0 <= c.min() and c.max() <= 1.0


def test_cif_beats_naive_km_by_construction(model):
    """The CIF must sit BELOW the net (1−S) curve: the competing cause removes pumps first.

    Exact in the continuum (``S_all ≤ S_k`` inside the same integrand), so the only slack
    allowed is quadrature error, which falls off as 1/n_grid — hence a tolerance far tighter
    than the effect being tested (the gap reaches ~0.14 by day 730 on real Ya).
    """
    u = np.linspace(1.0, 1500.0, 300)
    f = model.curve_frame(max_time=1500.0)
    for cause in M.CAUSES:
        cif = np.interp(u, f["time"], f[f"cif_{cause}"])
        net = 1.0 - np.asarray(model.S_cause(cause, u), dtype=float)
        assert np.all(cif <= net + 1e-4)


def test_window_probability_is_consistent(model):
    p = model.window_probability(180.0, 365.0)
    assert 0.0 <= p["any_pull"] <= 1.0
    # Causes partition the all-cause pull probability (up to grid interpolation error).
    assert abs((p["failure"] + p["gtm"]) - p["any_pull"]) < 5e-3
    # Conditioning on survival must not produce a certainty.
    assert p["failure"] > 0 and p["gtm"] > 0


def test_window_probability_grows_with_horizon(model):
    short = model.window_probability(90.0, 90.0)["any_pull"]
    long = model.window_probability(90.0, 720.0)["any_pull"]
    assert long > short


# ---------------------------------------------------------------------------
# Recovery of known parameters
# ---------------------------------------------------------------------------
def test_recovers_known_cause_specific_weibulls():
    """Cause-specific fits recover the generating β/η despite the competing cause AND
    administrative censoring — the property that makes 'censor the other cause' legitimate."""
    p = make_competing_pop(n=4000, fail=(1.3, 600.0), gtm=(1.8, 1000.0),
                           admin_censor=1500.0, seed=11)
    for cause, (beta, eta) in (("failure", (1.3, 600.0)), ("gtm", (1.8, 1000.0))):
        f = M.fit_cause(p, "t_cal", cause, num_starts=8)
        # k1 is the truth here; if AIC picks k2 the mixture must still imply the same curve,
        # so compare on RMST rather than on the raw parameters.
        rmst_fit = M.MB.life_from_S(lambda t, f=f: MB.mixture_S(
            t, f.w1, f.beta1, f.eta1, f.beta2, f.eta2))["rmst"]
        rmst_true = MB.life_from_S(lambda t: np.exp(-((t / eta) ** beta)))["rmst"]
        assert abs(rmst_fit - rmst_true) / rmst_true < 0.06, (
            f"{cause}: RMST {rmst_fit} vs true {rmst_true}")


def test_parametric_cif_tracks_aalen_johansen(pop, model):
    """The independence assumption, checked the way the workflow checks it."""
    chk = M.cif_check(pop, "t_cal", model, times=(180.0, 365.0, 730.0))
    assert np.abs(chk["param_minus_aj"]).max() < 0.03
    assert np.abs(chk["s_all_model_minus_km"]).max() < 0.03


def test_naive_km_overstates_incidence(pop, model):
    """Censoring the competing cause as if it were random inflates the incidence."""
    chk = M.cif_check(pop, "t_cal", model, times=(365.0, 730.0))
    assert (chk["naive_overstatement_abs"] > 0).all()


def test_km_control_matches_the_fit(pop, model):
    for cause in M.CAUSES:
        km = M.km_cause(pop, "t_cal", cause)
        fit_rmst = model.life(cause)["rmst"]
        assert abs(fit_rmst - km["rmst"]) / km["rmst"] < 0.06


# ---------------------------------------------------------------------------
# Degenerate-spike guard
# ---------------------------------------------------------------------------
def test_k2_degenerate_flags_a_spike_component():
    spike = MB.BaselineFit(
        model_kind="k2", w1=0.5, beta1=1.0, eta1=1.6, beta2=1.2, eta2=800.0,
        n=100, events=50, loglik_k1=0.0, aic_k1=0.0, loglik_k2=0.0, aic_k2=0.0,
        delta_aic=-10.0, k2_short_life=1.6, k2_long_life=800.0)
    assert M.k2_degenerate(spike)


def test_k2_degenerate_flags_a_vanishing_weight():
    thin = MB.BaselineFit(
        model_kind="k2", w1=0.001, beta1=0.8, eta1=200.0, beta2=1.2, eta2=800.0,
        n=100, events=50, loglik_k1=0.0, aic_k1=0.0, loglik_k2=0.0, aic_k2=0.0,
        delta_aic=-10.0, k2_short_life=200.0, k2_long_life=800.0)
    assert M.k2_degenerate(thin)


def test_k2_degenerate_passes_a_healthy_mixture_and_ignores_k1():
    healthy = MB.BaselineFit(
        model_kind="k2", w1=0.3, beta1=0.9, eta1=150.0, beta2=1.4, eta2=900.0,
        n=100, events=50, loglik_k1=0.0, aic_k1=0.0, loglik_k2=0.0, aic_k2=0.0,
        delta_aic=-10.0, k2_short_life=150.0, k2_long_life=900.0)
    assert not M.k2_degenerate(healthy)
    k1 = MB.BaselineFit(
        model_kind="k1", w1=0.0, beta1=0.8, eta1=1.0, beta2=0.8, eta2=1.0,
        n=100, events=50, loglik_k1=0.0, aic_k1=0.0, loglik_k2=0.0, aic_k2=0.0,
        delta_aic=5.0, k2_short_life=1.0, k2_long_life=1.0)
    assert not M.k2_degenerate(k1), "the guard must not fire on a k1 selection"


# ---------------------------------------------------------------------------
# Clock audit / verdict
# ---------------------------------------------------------------------------
def test_clock_audit_covers_every_clock_and_outcome(pop):
    audit = M.clock_audit(pop)
    assert set(audit["scope"]) == {"all", "failure", "gtm", "running"}
    assert set(audit["cohort"]) == set(M.COHORTS)
    for clock in M.AUDIT_CLOCKS:
        assert (audit["clock"] == clock).any(), f"{clock} missing from the audit"


def test_verdict_disqualifies_the_nno_clock_and_passes_t_cal(pop):
    """``t_nno_reported`` exists for events but not for running pumps — the mixed-clock trap."""
    v = M.clock_verdict(M.clock_audit(pop)).set_index(["cohort", "clock"])
    assert v.loc[("full", "t_cal"), "eligible_as_fit_clock"]
    assert v.loc[("full", "t_cal"), "severity"] == "ok"
    assert not v.loc[("full", "t_nno_reported"), "eligible_as_fit_clock"]
    assert v.loc[("full", "t_nno_reported"), "severity"] == "disqualified_structural"
    assert v.loc[("full", "t_nno_reported"), "availability_spread"] > 0.9


def test_verdict_grades_differential_imputation(pop):
    """A clock that is always *available* but only sometimes *measured* is graded, not passed."""
    p = pop.copy()
    # Make the imputation strongly differential: failures imputed, censorings measured.
    p["t_mix_source"] = np.where(p["cause_code"] == CENSORED, "measured", "imputed_field_kexp")
    v = M.clock_verdict(M.clock_audit(p)).set_index(["cohort", "clock"])
    assert v.loc[("full", "t_mix"), "availability_spread"] == 0.0
    assert not v.loc[("full", "t_mix"), "eligible_as_fit_clock"]
    assert v.loc[("full", "t_mix"), "severity"].startswith("disqualified_differential")


# ---------------------------------------------------------------------------
# Cohort handling
# ---------------------------------------------------------------------------
def test_cohort_modern_is_an_install_date_filter(pop):
    modern = M.cohort(pop, "modern")
    assert (modern["install"] >= pd.Timestamp(M.MODERN_INSTALL_START)).all()
    assert len(modern) < len(pop)
    # A cohort filter, NOT left truncation: every kept run keeps its FULL exposure from age 0.
    # (Compared row-wise by original index — synthetic wells reuse (code, install) pairs, so
    # a merge on those keys would fan out and compare unrelated runs.)
    np.testing.assert_allclose(modern["t_cal"], pop.loc[modern.index, "t_cal"])
    assert modern["current_age"].equals(pop.loc[modern.index, "current_age"])


def test_cohort_rejects_an_unknown_name(pop):
    with pytest.raises(ValueError):
        M.cohort(pop, "ancient")


def test_population_table_accounts_for_every_run(pop):
    t = M.population_table(pop).set_index("cohort")
    row = t.loc["full"]
    assert row["n_failure"] + row["n_gtm"] + row["n_running"] == row["n_runs"] == len(pop)


def test_cohort_scan_reports_censoring_alongside_life(pop):
    scan = M.cohort_scan(pop, "t_cal", cuts=(date(2018, 1, 1), date(2022, 1, 1)),
                         num_starts=2)
    assert {"censored_share", "max_t_cal", "failure_rmst730", "failure_km_rmst730"} <= set(scan)
    # Later cutoff ⇒ strictly fewer runs and heavier administrative censoring.
    assert scan.iloc[1]["n_runs"] < scan.iloc[0]["n_runs"]


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def test_bootstrap_brackets_the_point_estimate(pop):
    f = M.fit_cause(pop, "t_cal", "failure", num_starts=8)
    b = M.bootstrap_cause(pop, "t_cal", "failure", f, n_boot=30, seed=3)
    assert b["n_boot_ok"] >= 20
    point = M.MB.life_from_S(lambda t: MB.mixture_S(
        t, f.w1, f.beta1, f.eta1, f.beta2, f.eta2))["rmst"]
    assert b["rmst_lo"] <= point <= b["rmst_hi"]


def test_bootstrap_resamples_wells_not_runs():
    """On genuinely clustered data the well bootstrap is WIDER — the reason it is the default.

    Needs real intra-well dependence to have anything to detect, so this builds a population
    with well-level frailty; on iid rows the two intervals coincide, which is correct and not
    what this test is about.
    """
    clustered_pop = make_competing_pop(n=2000, n_wells=40, well_frailty=0.6, seed=17)
    f = M.fit_cause(clustered_pop, "t_cal", "failure", num_starts=8)
    clustered = M.bootstrap_cause(clustered_pop, "t_cal", "failure", f, n_boot=60, seed=5)
    solo = clustered_pop.copy()
    solo[M.CLUSTER] = [f"w{i}" for i in range(len(solo))]   # every run its own cluster
    unclustered = M.bootstrap_cause(solo, "t_cal", "failure", f, n_boot=60, seed=5)
    width_clustered = clustered["rmst_hi"] - clustered["rmst_lo"]
    width_solo = unclustered["rmst_hi"] - unclustered["rmst_lo"]
    assert width_clustered > width_solo, (
        f"cluster bootstrap {width_clustered:.1f} d not wider than run bootstrap "
        f"{width_solo:.1f} d — the clustering is not being applied")


# ---------------------------------------------------------------------------
# Cause assignment
# ---------------------------------------------------------------------------
def test_cause_assignment_audit_covers_all_closed_runs(pop):
    audit = M.cause_assignment_audit(pop)
    assert audit["n_runs"].sum() == int((~pop["is_open"]).sum())
    assert abs(audit["share_of_closed"].sum() - 1.0) < 1e-6


def test_cause_assignment_separates_workover_from_failure(pop):
    audit = M.cause_assignment_audit(pop)
    wo = audit[audit["branch"] == "reason_workover"]
    assert set(wo["cause"]) == {"gtm"}, "a ГТМ reason must never land on the failure cause"
