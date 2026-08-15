"""Tests for the growing-ESP-field simulation workflow.

Covers the KM math on a hand-worked example, simulation invariants, the
competing-risks workover logic, statistical recovery of the true Weibull by the
censoring-aware fit (and the failures-only bias), and the save/load round trip.
"""
from __future__ import annotations

from dataclasses import replace as dataclasses_replace

import numpy as np
import pandas as pd
import pytest

from analysis.workflows.field_sim import (
    SimConfig,
    aggregate_bins,
    aggregate_by_snapshot,
    commission_days,
    mode_columns,
    observe_at,
    population_at,
    run_experiment,
    simulate_runs,
)
from analysis.workflows.field_sim.config import DAYS_PER_YEAR, TTF_REFERENCES
from analysis.workflows.field_sim.fit import (
    MIN_FAIL_FOR_FIT,
    _weibull_mle,
    fit_snapshot,
    km_curve_failures_only,
)
from analysis.workflows.field_sim.hazard import (
    freq_theta,
    layers_from_config,
    make_freq_layer,
    make_ql_layer,
    make_well_layer,
    split_layers,
    theta_mixture,
    theta_spread,
)
from analysis.workflows.field_sim.km import kaplan_meier, rmst, weibull_survival
from analysis.workflows.field_sim.metrics import (
    base_cumhaz,
    eta_components,
    eta_equivalent_days,
    eta_from_life,
    expected_life,
    expected_rates,
    life_given_theta,
    mixture_survival,
    snapshot_metrics,
    weibull_life,
)
from analysis.workflows.field_sim.store import load_experiment, save_experiment


# ── KM math ──────────────────────────────────────────────────────────────────
def test_km_hand_worked_scenario_a():
    """Pool of 4: fail@1, workover@2, fail@3, running@4 → S ends at 0.375.

    Reproduces scenario (a) from the KM tutorial: only failures step the curve;
    the workover (event=0) just thins the risk set.
    """
    dur = np.array([1.0, 2.0, 3.0, 4.0])
    ev = np.array([1, 0, 1, 0])
    t, s, _ = kaplan_meier(dur, ev)
    assert t.tolist() == [0.0, 1.0, 3.0]
    assert s[1] == pytest.approx(0.75)
    assert s[-1] == pytest.approx(0.375)


def test_km_censoring_before_failure_protects_curve():
    """Scenario (c): workover@1, fail@2, workover@3, running@4 → S ends at 2/3."""
    dur = np.array([1.0, 2.0, 3.0, 4.0])
    ev = np.array([0, 1, 0, 0])
    _, s, _ = kaplan_meier(dur, ev)
    assert s[-1] == pytest.approx(2.0 / 3.0)


def test_rmst_matches_weibull_area_for_exponential():
    """RMST(0, tau) of a dense exact-Weibull KM ≈ analytic area for β=1."""
    eta = 365.0
    t = np.linspace(1, 3000, 4000)
    surv = weibull_survival(t, 1.0, eta)
    # emulate an (essentially) uncensored KM by using the true curve directly
    tau = 700.0
    analytic = eta * (1 - np.exp(-tau / eta))  # ∫0^tau exp(-t/eta) dt
    approx = rmst(t, surv, tau)
    assert approx == pytest.approx(analytic, rel=0.02)


# ── simulation invariants ────────────────────────────────────────────────────
def test_commission_and_population_ramp_then_plateau():
    cfg = SimConfig(plateau_wells=100, ramp_years=10, total_years=20)
    c = commission_days(cfg)
    assert len(c) == 100
    assert c[0] == 0.0
    assert c[-1] == pytest.approx(10 * DAYS_PER_YEAR, abs=1.0)
    assert np.all(np.diff(c) >= 0)
    # linear-ish during ramp, flat after
    assert population_at(cfg, 5 * DAYS_PER_YEAR) == pytest.approx(50, abs=3)
    assert population_at(cfg, 10 * DAYS_PER_YEAR) == 100
    assert population_at(cfg, 20 * DAYS_PER_YEAR) == 100


def test_simulation_is_reproducible():
    cfg = SimConfig(plateau_wells=20, n_seeds=1)
    a = simulate_runs(cfg, 42)
    b = simulate_runs(cfg, 42)
    pd.testing.assert_frame_equal(a, b)
    assert not simulate_runs(cfg, 43).equals(a)


def test_runs_are_chained_with_cause_dependent_downtime():
    cfg = SimConfig(plateau_wells=5, downtime_fail_days=7, downtime_workover_days=3,
                    workover_mode="statistical", beta_wo=1.3, wo_eta_fraction=0.8)
    runs = simulate_runs(cfg, 7)
    for _, g in runs.groupby("slot"):
        g = g.sort_values("run_idx")
        starts = g["start"].to_numpy()
        ends = g["end"].to_numpy()
        gaps = np.where(g["cause"].to_numpy()[:-1] == "workover", 3.0, 7.0)
        assert np.allclose(starts[1:], ends[:-1] + gaps)
        # downtime_after column records the cause-specific gap
        assert np.allclose(g["downtime_after"].to_numpy(),
                           np.where(g["cause"].to_numpy() == "workover", 3.0, 7.0))
    assert set(runs["cause"]) <= {"fail", "workover"}


def test_observe_at_is_consistent_across_snapshots():
    cfg = SimConfig(plateau_wells=30)
    runs = simulate_runs(cfg, 3)
    n_earlier = len(observe_at(runs, 3 * DAYS_PER_YEAR))
    n_later = len(observe_at(runs, 8 * DAYS_PER_YEAR))
    assert n_later >= n_earlier  # more runs have started
    obs = observe_at(runs, 6 * DAYS_PER_YEAR)
    assert (obs["duration"] > 0).all() or (obs["duration"] >= 0).all()
    # running rows are censored; failed rows are events
    assert set(obs.loc[obs["status"] == "running", "event"]) <= {0}
    assert set(obs.loc[obs["status"] == "fail", "event"]) == {1}


# ── competing-risks workover logic ───────────────────────────────────────────
def test_deterministic_pm_caps_run_length_and_hides_late_failures():
    cfg = SimConfig(plateau_wells=40, workover_mode="deterministic", pm_fraction=0.8)
    pm_age = cfg.pm_age_days()
    runs = simulate_runs(cfg, 11)
    assert runs["run_len"].max() <= pm_age + 1e-6
    # every completed run beyond pm_age would have to be a workover — none are failures
    assert not ((runs["cause"] == "fail") & (runs["run_len"] > pm_age + 1e-6)).any()


def test_ttf_reference_prices_both_workover_fractions_off_the_same_number():
    """The two fractions are fractions OF this, so the choice has to move both."""
    for ref in TTF_REFERENCES:
        cfg = SimConfig(ttf_reference=ref, pm_fraction=0.8, wo_eta_fraction=0.5)
        assert cfg.pm_age_days() == pytest.approx(0.8 * cfg.ttf_ref_days())
        assert cfg.wo_eta_days() == pytest.approx(0.5 * cfg.ttf_ref_days())


def test_rmst_reference_is_the_mean_truncated_at_tau():
    # β < 1 puts a long tail past τ, so restricting the mean there bites hardest
    cfg = SimConfig(beta_fail=0.8, eta_fail_days=365.0, rmst_tau_days=730.0)
    mean_ref = dataclasses_replace(cfg, ttf_reference="mean").ttf_ref_days()
    rmst_ref = dataclasses_replace(cfg, ttf_reference="rmst").ttf_ref_days()
    assert rmst_ref == pytest.approx(expected_life(cfg)["rmst"])
    assert rmst_ref < mean_ref
    # …and τ is its ceiling: pushing τ out recovers the unrestricted mean
    far = dataclasses_replace(cfg, ttf_reference="rmst", rmst_tau_days=200_000.0)
    assert far.ttf_ref_days() == pytest.approx(mean_ref, rel=1e-3)


def test_rmst_reference_survives_competing_modes():
    """No closed form there, so it is a quadrature — it still has to bracket sanely."""
    cfg = SimConfig(modes_on=True, ttf_reference="rmst", rmst_tau_days=730.0)
    assert 0.0 < cfg.ttf_ref_days() < min(e for _, _, e in cfg.modes())
    assert cfg.ttf_ref_days() == pytest.approx(life_given_theta(cfg, 1.0)["rmst"])


def test_statistical_workover_produces_both_causes():
    cfg = SimConfig(plateau_wells=60, workover_mode="statistical", beta_wo=1.3, wo_eta_fraction=0.8)
    runs = simulate_runs(cfg, 5)
    counts = runs["cause"].value_counts()
    assert counts.get("fail", 0) > 0
    assert counts.get("workover", 0) > 0


# ── fitting behaviour ────────────────────────────────────────────────────────
def test_fit_snapshot_returns_nan_when_too_few_failures():
    obs = pd.DataFrame(
        {
            "duration": [10.0, 20.0, 30.0, 40.0],
            "event": [1, 0, 0, 0],  # only one failure < MIN_FAIL_FOR_FIT
            "status": ["fail", "running", "running", "running"],
        }
    )
    assert MIN_FAIL_FOR_FIT >= 2
    out = fit_snapshot(obs)
    assert np.isnan(out["beta_cens"]) and np.isnan(out["eta_cens"])


def test_censored_fit_recovers_truth_and_failures_only_is_biased_low():
    """The headline experiment, in miniature (kept small so it runs in seconds)."""
    cfg = SimConfig(
        plateau_wells=80, n_seeds=12, n_fit_snapshots=6,
        km_snapshot_years=[20], beta_fail=1.0, eta_fail_days=365.0,
    )
    res = run_experiment(cfg)
    agg = aggregate_by_snapshot(res.fit_table).sort_values("snap_year")
    last = agg.iloc[-1]
    # censoring-aware fit lands on the truth
    assert last["beta_cens_median"] == pytest.approx(1.0, abs=0.12)
    assert last["eta_cens_median"] == pytest.approx(365.0, rel=0.08)
    # failures-only systematically underestimates eta (drops the long survivors)
    assert last["eta_fo_median"] < last["eta_cens_median"]


def test_planned_workover_censoring_still_recovered_by_censored_fit():
    """Even pulling every pump at 0.8×TTF, the censored fit recovers η; FO collapses."""
    cfg = SimConfig(
        plateau_wells=80, n_seeds=12, n_fit_snapshots=6, km_snapshot_years=[20],
        workover_mode="deterministic", pm_fraction=0.8,
    )
    res = run_experiment(cfg)
    last = aggregate_by_snapshot(res.fit_table).sort_values("snap_year").iloc[-1]
    assert last["eta_cens_median"] == pytest.approx(365.0, rel=0.10)
    # failures-only badly low because all runs are truncated at 0.8*mean
    assert last["eta_fo_median"] < 0.7 * cfg.eta_fail_days


def test_all_pulls_fit_is_shorter_lived_and_not_stalled():
    """All-pulls (WO counted as failure) has η below the true failure law, and the
    multi-start keeps it from stalling at the failure-scale init (365)."""
    cfg = SimConfig(plateau_wells=80, n_seeds=10, n_fit_snapshots=6, km_snapshot_years=[20],
                    workover_mode="deterministic", pm_fraction=0.8)
    last = aggregate_by_snapshot(run_experiment(cfg).fit_table).sort_values("snap_year").iloc[-1]
    assert last["eta_all_median"] < last["eta_cens_median"]        # all-cause pull-time is shorter
    assert abs(last["eta_all_median"] - 365.0) > 30                # not stuck at the 365 init


# ── operational metrics: downtime, oil loss, workover programme effect ────────
def test_snapshot_metrics_downtime_and_oil_loss_are_consistent():
    cfg = SimConfig(plateau_wells=40, workover_mode="statistical", beta_wo=1.3,
                    wo_eta_fraction=0.8, downtime_fail_days=7, downtime_workover_days=3)
    runs = simulate_runs(cfg, 3)
    m = snapshot_metrics(runs, cfg, 20 * DAYS_PER_YEAR)
    # exposure + downtime ≈ calendar well-days (small edge slack near the horizon)
    assert m["exposure_days"] + m["downtime_days"] == pytest.approx(m["well_days"], rel=0.02)
    assert 0.0 < m["oil_loss_pct"] < 100.0
    assert m["fail_rate_well_yr"] > 0.0
    assert m["pull_rate_well_yr"] >= m["fail_rate_well_yr"]  # pulls include workovers


def test_workover_programme_under_memoryless_failures_adds_loss_without_cutting_failures():
    """β=1: PM cannot lower the failure rate; it only adds pulls and oil loss."""
    cfg = SimConfig(plateau_wells=80, n_seeds=12, n_fit_snapshots=5, km_snapshot_years=[20],
                    workover_mode="deterministic", pm_fraction=0.8, beta_fail=1.0)
    res = run_experiment(cfg, counterfactual=True)
    assert res.fit_table_nowo is not None
    withp = aggregate_by_snapshot(res.fit_table).sort_values("snap_year").iloc[-1]
    nowo = aggregate_by_snapshot(res.fit_table_nowo).sort_values("snap_year").iloc[-1]
    # failure rate essentially unchanged (memoryless), pulls and oil loss up
    assert withp["fail_rate_well_yr_median"] == pytest.approx(nowo["fail_rate_well_yr_median"], rel=0.08)
    assert withp["pull_rate_well_yr_median"] > 1.3 * nowo["pull_rate_well_yr_median"]
    assert withp["oil_loss_pct_median"] > nowo["oil_loss_pct_median"]


# ── analytic estimates from parameters (renewal-reward) ──────────────────────
def test_expected_rates_none_closed_form():
    cfg = SimConfig(beta_fail=1.0, eta_fail_days=365.0, downtime_fail_days=7.0)
    e = expected_rates(cfg)
    assert e["workover_per_well_year"] == 0.0
    assert e["fail_per_well_year"] == pytest.approx(365.0 / (365.0 + 7.0), rel=1e-6)
    assert e["oil_loss_pct"] == pytest.approx(100.0 * 7.0 / 372.0, rel=1e-6)


@pytest.mark.parametrize("kw", [
    {},
    {"workover_mode": "deterministic", "pm_fraction": 0.8},
    {"workover_mode": "statistical", "beta_wo": 1.3, "wo_eta_fraction": 0.8},
])
def test_expected_rates_match_simulation_plateau(kw):
    cfg = SimConfig(plateau_wells=80, n_seeds=15, n_fit_snapshots=5, km_snapshot_years=[20], **kw)
    e = expected_rates(cfg)
    last = aggregate_by_snapshot(run_experiment(cfg).fit_table).sort_values("snap_year").iloc[-1]
    assert last["fail_rate_well_yr_median"] == pytest.approx(e["fail_per_well_year"], rel=0.06)
    assert last["pull_rate_well_yr_median"] == pytest.approx(e["pull_per_well_year"], rel=0.06)
    assert last["oil_loss_pct_median"] == pytest.approx(e["oil_loss_pct"], rel=0.08)


def test_expected_life_closed_form_for_exponential():
    cfg = SimConfig(beta_fail=1.0, eta_fail_days=365.0, rmst_tau_days=730.0)
    L = expected_life(cfg)
    assert L["mean"] == pytest.approx(365.0)
    assert L["mrl0"] == L["mean"]
    assert L["median"] == pytest.approx(365.0 * np.log(2.0))          # ≈ 253
    assert L["rmst"] == pytest.approx(365.0 * (1 - np.exp(-2.0)))     # ≈ 315.6


def test_life_summaries_recover_truth_naive_mean_biased_low():
    """KM RMST/MRL/median → truth; naive Σt/N mean stays below."""
    cfg = SimConfig(plateau_wells=80, n_seeds=12, n_fit_snapshots=5, km_snapshot_years=[20],
                    beta_fail=1.0, eta_fail_days=365.0, rmst_tau_days=730.0)
    L = expected_life(cfg)
    last = aggregate_by_snapshot(run_experiment(cfg).fit_table).sort_values("snap_year").iloc[-1]
    assert last["life_median_median"] == pytest.approx(L["median"], rel=0.10)
    assert last["life_rmst_median"] == pytest.approx(L["rmst"], rel=0.10)
    # naive observed mean underestimates the KM mean (drops long survivors)
    assert last["obs_ttf_fail_median"] < last["life_mrl0_median"]


# ── covariate hazard layers ──────────────────────────────────────────────────
@pytest.mark.parametrize("shape", ["log", "linear"])
def test_ql_layer_is_monotone_with_the_requested_bin_ratio(shape):
    """theta rises across the bins and the top/bottom bin ratio is exactly 2x."""
    spec = make_ql_layer(50.0, 250.0, 5, ratio=2.0, shape=shape)
    assert spec.n_bins == 5
    assert np.all(np.diff(spec.theta) > 0)
    assert spec.theta[-1] / spec.theta[0] == pytest.approx(2.0, rel=1e-9)
    # centering leaves the population mean at 1 (probabilities are uniform)
    assert float(np.sum(spec.probs * spec.theta)) == pytest.approx(1.0)


def test_freq_layer_is_a_u_shape_anchored_on_the_endpoints():
    """theta = 1 + k(1 - f/50)^2 reaches `edge` at 40/60 Hz; bin centers sit inside."""
    spec = make_freq_layer(40.0, 60.0, 5, edge=2.0, center=False)
    assert spec.theta[2] == pytest.approx(1.0)              # the 50 Hz vertex
    assert spec.theta[0] == pytest.approx(spec.theta[-1])   # symmetric about 50
    assert spec.theta[0] > spec.theta[1] > spec.theta[2]    # U, not monotone
    # the underlying shape hits `edge` exactly at 40/60 Hz ...
    assert freq_theta(np.array([40.0, 50.0, 60.0]), 40.0, 60.0, 2.0) == pytest.approx([2.0, 1.0, 2.0])
    # ... so the extreme *bins*, centered at 42/58 Hz, fall short of it
    assert 1.5 < spec.theta[0] < 2.0


def test_theta_mixture_is_the_cross_product_of_the_layers():
    cfg = SimConfig(ql_layer_on=True, freq_layer_on=True, ql_bins=5, freq_bins=5)
    probs, theta = theta_mixture(layers_from_config(cfg))
    assert probs.size == theta.size == 25
    assert float(probs.sum()) == pytest.approx(1.0)
    assert float(np.sum(probs * theta)) == pytest.approx(1.0)  # both layers centered


def test_layers_off_leave_the_simulation_bit_identical():
    """No enabled layer must not touch the generator (theta is 1, bins are -1)."""
    base = SimConfig(plateau_wells=15)
    with_off = dataclasses_replace(base, ql_layer_on=False, freq_layer_on=False)
    a, b = simulate_runs(base, 5), simulate_runs(with_off, 5)
    pd.testing.assert_frame_equal(a, b)
    assert (a["theta"] == 1.0).all()
    assert (a["ql_bin"] == -1).all() and (a["freq_bin"] == -1).all()


def test_theta_shortens_life_in_the_high_hazard_bins():
    """The realised per-bin mean run length follows eta * theta**(-1/beta)."""
    cfg = SimConfig(plateau_wells=120, beta_fail=1.0, ql_layer_on=True, ql_theta_ratio=2.0)
    spec = layers_from_config(cfg)[0]
    runs = simulate_runs(cfg, 17)
    means = runs.groupby("ql_bin")["run_len"].mean().sort_index().to_numpy()
    expected = cfg.eta_fail_days * spec.theta ** (-1.0 / cfg.beta_fail)
    assert np.all(np.diff(means) < 0)                       # higher Ql -> shorter life
    assert means == pytest.approx(expected, rel=0.10)


def test_expected_life_uses_the_mixture_not_the_baseline_weibull():
    """With a layer on, the truth is a Weibull mixture: E[T] rises by Jensen."""
    off = SimConfig(beta_fail=1.0, eta_fail_days=365.0)
    on = dataclasses_replace(off, ql_layer_on=True, ql_theta_ratio=3.0)
    assert eta_components(off)[0].size == 1
    assert eta_components(on)[0].size == on.ql_bins
    # mean theta is 1, but E[eta] = eta0 * E[theta^-1] > eta0 (convexity)
    assert expected_life(on)["mean"] > expected_life(off)["mean"]
    # the mixture survival is a proper survival curve
    s = mixture_survival(np.array([0.0, 100.0, 1e5]), on)
    assert s[0] == pytest.approx(1.0)
    assert s[1] < 1.0 and s[-1] < 1e-6
    # and the median solves S(t) = 0.5 on that curve
    med = expected_life(on)["median"]
    assert float(mixture_survival(np.array([med]), on)[0]) == pytest.approx(0.5, abs=1e-6)


def test_bin_table_recovers_the_theta_ordering_that_produced_it():
    cfg = SimConfig(plateau_wells=60, n_seeds=4, n_fit_snapshots=4, km_snapshot_years=[20],
                    freq_layer_on=True, freq_theta_edge=2.0)
    res = run_experiment(cfg)
    assert res.bin_table is not None
    agg = aggregate_bins(res.bin_table, "freq", 20.0)
    assert list(agg["center"]) == [42.0, 46.0, 50.0, 54.0, 58.0]
    # uniform assignment -> roughly equal bin populations
    assert agg["n_runs"].max() < 1.4 * agg["n_runs"].min()
    # the U-shape shows up in the observed per-bin life: the 50 Hz bin lives longest
    life = agg["km_rmst_median"].to_numpy()
    assert life.argmax() == 2
    assert life[2] > life[0] and life[2] > life[-1]


def test_aggregate_bins_can_isolate_one_seed():
    """Pooled sums the seeds; a single seed keeps its own counts and closes the band."""
    cfg = SimConfig(plateau_wells=40, n_seeds=4, n_fit_snapshots=4, km_snapshot_years=[20],
                    ql_layer_on=True)
    res = run_experiment(cfg)
    pooled = aggregate_bins(res.bin_table, "ql", 20.0)
    seeds = sorted(int(s) for s in res.bin_table["seed"].unique())
    per_seed = [aggregate_bins(res.bin_table, "ql", 20.0, s) for s in seeds]

    assert (pooled["n_seeds"] == cfg.n_seeds).all()
    for one in per_seed:
        assert (one["n_seeds"] == 1).all()
        assert list(one["bin"]) == list(pooled["bin"])
        # one realization has nothing to spread over, so the band collapses
        assert (one["km_rmst_p90"] - one["km_rmst_p10"]).abs().max() == pytest.approx(0.0)
    # pooled counts are exactly the seeds added up
    assert pooled["n_fail"].to_numpy() == pytest.approx(
        sum(o["n_fail"].to_numpy() for o in per_seed))
    # ... while the pooled band is genuinely wide
    assert (pooled["km_rmst_p90"] - pooled["km_rmst_p10"]).max() > 0

    assert aggregate_bins(res.bin_table, "ql", 20.0, seed=-1).empty


def test_no_bin_table_when_every_layer_is_off():
    cfg = SimConfig(plateau_wells=20, n_seeds=2, n_fit_snapshots=3, km_snapshot_years=[10])
    assert run_experiment(cfg).bin_table is None


# ── the naive failures-only KM stored next to the censoring-aware one ────────
def test_failures_only_curve_is_stored_and_reads_shorter_lived():
    """Dropping censored runs biases the curve low, and the bias shrinks with age.

    Early on most runs are still alive, so the survivors are missing from the
    sample entirely and the curve collapses; by the horizon almost everything has
    failed and the two curves converge on the truth.
    """
    cfg = SimConfig(plateau_wells=80, n_seeds=6, n_fit_snapshots=5,
                    km_snapshot_years=[3, 10, 20], beta_fail=1.0, eta_fail_days=365.0)
    res = run_experiment(cfg)
    truth = expected_life(cfg)["rmst"]

    bias = []
    for year in ("3", "10", "20"):
        c = res.km_curves[year]
        assert {"time_fo", "surv_fo", "n_runs_fo"} <= set(c)
        assert c["n_runs_fo"] == c["n_fail"]           # only the failures survive the drop
        aware = rmst(np.array(c["time"]), np.array(c["surv"]), 730.0)
        naive = rmst(np.array(c["time_fo"]), np.array(c["surv_fo"]), 730.0)
        assert naive < aware
        bias.append((truth - naive) / truth)
    assert bias[0] > bias[1] > bias[2]                 # monotonically less biased
    assert bias[0] > 0.20 and bias[-1] < 0.10


def test_failures_only_curve_is_one_minus_the_ecdf_of_the_failures():
    """With nothing censored the KM has nothing to correct and collapses to 1-ECDF."""
    dur = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    ev = np.array([1, 0, 1, 0, 1])
    c = km_curve_failures_only(dur, ev)
    assert c["n_runs"] == c["n_fail"] == 3
    assert c["time"] == [0.0, 10.0, 30.0, 50.0]
    assert c["surv"] == pytest.approx([1.0, 2 / 3, 1 / 3, 0.0])


# ── individual-well frailty layer ────────────────────────────────────────────
def test_well_layer_spans_the_requested_theta_ratio_and_stays_centered():
    spec = make_well_layer(5, ratio=4.0)
    assert spec.per_well
    assert np.all(np.diff(spec.theta) > 0)
    assert spec.theta[-1] / spec.theta[0] == pytest.approx(4.0, rel=1e-9)
    assert float(np.sum(spec.probs * spec.theta)) == pytest.approx(1.0)
    # geometric, not linear: equal steps in log theta
    assert np.diff(np.log(spec.theta)) == pytest.approx(
        np.full(4, np.log(4.0) / 4), rel=1e-9)


def test_well_theta_is_drawn_once_per_slot_and_never_redrawn():
    """The whole point of a frailty: a well's runs share one multiplier."""
    cfg = SimConfig(plateau_wells=40, total_years=25, well_layer_on=True, well_theta_ratio=6.0)
    runs = simulate_runs(cfg, 4)
    per_slot = runs.groupby("slot")[["theta", "well_bin"]].nunique()
    assert (per_slot["theta"] == 1).all()
    assert (per_slot["well_bin"] == 1).all()
    assert runs.groupby("slot")["run_idx"].max().max() > 1     # slots really do renew
    assert set(runs["well_bin"]) == set(range(cfg.well_bins))


def test_well_and_run_layers_compose_and_split_by_schedule():
    cfg = SimConfig(well_layer_on=True, ql_layer_on=True, well_bins=3, ql_bins=4)
    well, run = split_layers(layers_from_config(cfg))
    assert [s.key for s in well] == ["well"] and [s.key for s in run] == ["ql"]
    probs, theta = theta_mixture(layers_from_config(cfg))
    assert probs.size == theta.size == 12                      # the cross product
    runs = simulate_runs(cfg, 9)
    # each run's theta is the product of its two bins
    w = {s.key: s for s in layers_from_config(cfg)}
    expected = w["well"].theta[runs["well_bin"]] * w["ql"].theta[runs["ql_bin"]]
    assert runs["theta"].to_numpy() == pytest.approx(expected)


def test_well_frailty_drags_the_pooled_shape_below_the_true_one():
    """Every pump wears out (β = 1.3) yet the pooled fit returns β < 1.

    This is the heterogeneity mechanism the β-shape study rests on, reproduced
    on its own numbers: the marginal law of a mixture is not the law of its
    members, and the wider the spread of log θ the further the fitted shape
    falls. How far it has to fall to cross 1 depends on the true β — at 1.4 a
    ×16 spread only reaches ≈1.04 — so the crossing is asserted where the study
    claims it, not as a property of any β.
    """
    base = SimConfig(plateau_wells=150, total_years=20, beta_fail=1.3, eta_fail_days=730.0)
    fits = []
    for ratio in (None, 4.0, 16.0):
        cfg = base if ratio is None else dataclasses_replace(
            base, well_layer_on=True, well_theta_ratio=ratio, well_bins=5)
        obs = observe_at(simulate_runs(cfg, 101), cfg.horizon_days())
        fits.append(fit_snapshot(obs)["beta_cens"])
    assert fits[0] == pytest.approx(1.3, abs=0.12)     # control: no spread -> the truth
    assert fits[0] > fits[1] > fits[2]                 # monotone in the spread
    assert fits[2] < 1.0                               # and it crosses below 1


def test_expected_rates_average_per_well_ratios_not_ratios_of_averages():
    """Each well runs its own renewal process, so the fleet rate is E[p/E cycle].

    Averaging numerator and denominator first would understate the rate badly:
    a frail well contributes more pulls per year *because* its cycles are short.
    """
    cfg = SimConfig(plateau_wells=300, total_years=60, ramp_years=2,
                    well_layer_on=True, well_theta_ratio=6.0, well_bins=5)
    e = expected_rates(cfg)
    sim = snapshot_metrics(simulate_runs(cfg, 5), cfg, cfg.horizon_days())
    assert e["fail_per_well_year"] == pytest.approx(sim["fail_rate_well_yr"], rel=0.06)
    assert e["oil_loss_pct"] == pytest.approx(sim["oil_loss_pct"], rel=0.08)
    # the naive collapse would land far below — check we are not accidentally it
    probs, theta = theta_mixture(layers_from_config(cfg))
    mean_run = float(np.sum(probs * cfg.eta_fail_days * theta ** (-1.0 / cfg.beta_fail)))
    naive = DAYS_PER_YEAR / (mean_run + cfg.downtime_fail_days)
    assert e["fail_per_well_year"] > 1.3 * naive


def test_theta_spread_reports_ratio_and_sd_of_log_theta():
    cfg = SimConfig(well_layer_on=True, well_theta_ratio=9.0, well_bins=3)
    span, sd = theta_spread(layers_from_config(cfg))
    assert span == pytest.approx(9.0)
    assert sd == pytest.approx(float(np.std(np.log([1.0, 3.0, 9.0]))), rel=1e-9)
    assert theta_spread([]) == (1.0, 0.0)


# ── competing failure modes ──────────────────────────────────────────────────
MODES = [{"name": "pump", "beta": 1.6, "eta_days": 520.0},
         {"name": "cable", "beta": 0.8, "eta_days": 900.0}]


def test_modes_off_is_the_single_baseline_weibull():
    cfg = SimConfig(beta_fail=1.2, eta_fail_days=400.0)
    assert cfg.modes() == [("failure", 1.2, 400.0)]
    assert cfg.n_modes() == 1
    # and the simulation is untouched by the mode machinery
    runs = simulate_runs(cfg, 5)
    assert (runs["fail_mode"] == 0).all()


def test_competing_modes_take_the_minimum_and_record_the_winner():
    cfg = SimConfig(plateau_wells=120, total_years=20, modes_on=True, fail_modes=MODES)
    runs = simulate_runs(cfg, 7)
    fails = runs[runs["cause"] == "fail"]
    assert set(fails["fail_mode"]) == {0, 1}
    # the mode with the shorter life wins more often, and its wins are shorter
    assert (fails["fail_mode"] == 0).mean() > 0.5
    # the combined law is shorter-lived than either mode alone
    solo = [SimConfig(beta_fail=m["beta"], eta_fail_days=m["eta_days"]).true_mean_ttf_days()
            for m in MODES]
    assert cfg.true_mean_ttf_days() < min(solo)
    assert fails["run_len"].mean() == pytest.approx(cfg.true_mean_ttf_days(), rel=0.06)


def test_combined_survival_is_the_product_of_the_modes():
    cfg = SimConfig(modes_on=True, fail_modes=MODES)
    t = np.array([0.0, 100.0, 500.0, 2000.0])
    prod = np.prod([np.exp(-((t / m["eta_days"]) ** m["beta"])) for m in MODES], axis=0)
    assert mixture_survival(t, cfg) == pytest.approx(prod)
    assert base_cumhaz(t, cfg) == pytest.approx(-np.log(prod))
    # theta multiplies the *total* cumulative hazard, so it factors straight out
    assert mixture_survival(t, dataclasses_replace(cfg, well_layer_on=True, well_bins=1)) \
        == pytest.approx(prod)


def test_pooled_fit_tracks_the_combined_law_not_either_mode():
    """The pooled η lands *below every mode's* η — arithmetic, not a broken fit.

    Competing hazards add, so the combined law reaches 1/e sooner than the
    earliest single mode does. Anchoring the pooled fit against the best Weibull
    approximation of the true combined law (fitted on a large uncensored sample)
    is what distinguishes "correctly shorter" from "wrongly shorter".
    """
    cfg = SimConfig(plateau_wells=200, total_years=25, modes_on=True, fail_modes=MODES)
    eq = eta_equivalent_days(cfg)
    assert eq < min(m["eta_days"] for m in MODES)

    rng = np.random.default_rng(0)
    draws = np.min([m["eta_days"] * rng.weibull(m["beta"], 200_000) for m in MODES], axis=0)
    ref = _weibull_mle(draws, np.ones(draws.size, dtype=int))

    fitted = fit_snapshot(observe_at(simulate_runs(cfg, 101), cfg.horizon_days()))
    assert fitted["eta_cens"] == pytest.approx(ref["eta"], rel=0.05)
    assert fitted["beta_cens"] == pytest.approx(ref["beta"], rel=0.05)
    # ... and that reference is itself close to (just under) the 1/e crossing
    assert 0.85 * eq < ref["eta"] < eq


def test_eta_equivalent_is_the_age_where_the_combined_law_hits_1_over_e():
    cfg = SimConfig(modes_on=True, fail_modes=MODES)
    e = eta_equivalent_days(cfg)
    assert float(mixture_survival(np.array([e]), cfg)[0]) == pytest.approx(np.exp(-1.0))
    assert e < min(m["eta_days"] for m in MODES)
    # one mode: it is exactly the scale, no solving involved
    assert eta_equivalent_days(SimConfig(eta_fail_days=411.0)) == 411.0


def test_life_given_theta_matches_the_closed_form_when_there_is_one_mode():
    cfg = SimConfig(beta_fail=1.3, eta_fail_days=600.0, rmst_tau_days=730.0)
    for theta in (0.5, 1.0, 2.5):
        got = life_given_theta(cfg, theta)
        want = weibull_life(1.3, 600.0 * theta ** (-1.0 / 1.3), 730.0)
        assert {k: got[k] for k in want} == pytest.approx(want)


def test_cause_specific_fit_recovers_each_mode_the_pooled_fit_recovers_neither():
    """The headline of the competing-modes feature, in miniature."""
    cfg = SimConfig(plateau_wells=120, total_years=25, n_seeds=5, n_fit_snapshots=4,
                    km_snapshot_years=[25], modes_on=True, fail_modes=MODES)
    res = run_experiment(cfg)
    assert mode_columns(res.fit_table) == [0, 1]
    last = aggregate_by_snapshot(res.fit_table).sort_values("snap_year").iloc[-1]
    for m, spec in enumerate(MODES):
        assert last[f"beta_m{m}_median"] == pytest.approx(spec["beta"], rel=0.10)
        assert last[f"eta_m{m}_median"] == pytest.approx(spec["eta_days"], rel=0.10)
        assert last[f"n_fail_m{m}"] > 0
    # the pooled fit lands between the two shapes — i.e. on neither
    pooled = last["beta_cens_median"]
    assert min(m["beta"] for m in MODES) < pooled < max(m["beta"] for m in MODES)
    assert last["eta_cens_median"] < min(m["eta_days"] for m in MODES)


def test_mode_columns_absent_when_a_single_law_is_simulated():
    cfg = SimConfig(plateau_wells=20, n_seeds=2, n_fit_snapshots=3, km_snapshot_years=[10])
    res = run_experiment(cfg)
    assert mode_columns(res.fit_table) == []
    assert "beta_m0" not in res.fit_table.columns


def test_modes_and_layers_compose_in_the_simulated_life():
    """theta scales every mode, so the per-bin mean life follows the mixture."""
    cfg = SimConfig(plateau_wells=150, total_years=25, modes_on=True, fail_modes=MODES,
                    well_layer_on=True, well_theta_ratio=4.0, well_bins=4)
    spec = layers_from_config(cfg)[0]
    runs = simulate_runs(cfg, 21)
    means = runs.groupby("well_bin")["run_len"].mean().sort_index().to_numpy()
    expected = np.array([life_given_theta(cfg, float(t))["mean"] for t in spec.theta])
    assert np.all(np.diff(means) < 0)
    assert means == pytest.approx(expected, rel=0.10)


# ── eta calculator (life summary -> scale) ───────────────────────────────────
@pytest.mark.parametrize("beta", [0.8, 1.0, 1.6, 2.5])
@pytest.mark.parametrize("kind", ["mean", "median", "rmst"])
def test_eta_from_life_round_trips_through_weibull_life(beta, kind):
    tau, target = 730.0, 400.0
    eta = eta_from_life(target, kind, beta, tau)
    assert weibull_life(beta, eta, tau)[kind] == pytest.approx(target, rel=1e-6)


def test_eta_from_life_rejects_an_rmst_beyond_tau():
    """RMST(0, tau) < tau always — asking for more is not a solvable eta."""
    with pytest.raises(ValueError, match="not reachable"):
        eta_from_life(800.0, "rmst", 1.0, 730.0)


# ── persistence ──────────────────────────────────────────────────────────────
def test_save_load_round_trip(tmp_path, monkeypatch):
    import analysis.workflows.field_sim.store as store

    monkeypatch.setattr(store, "FIELD_SIM_ROOT", tmp_path / "field_sim")
    cfg = SimConfig(plateau_wells=20, n_seeds=3, n_fit_snapshots=4, km_snapshot_years=[10],
                    workover_mode="deterministic", pm_fraction=0.8,
                    ql_layer_on=True, freq_layer_on=True)
    res = run_experiment(cfg, label="roundtrip", counterfactual=True)
    save_experiment(res)
    loaded = load_experiment(res.experiment_id)
    assert loaded.label == "roundtrip"
    assert loaded.config.plateau_wells == 20
    assert set(loaded.km_curves) == set(res.km_curves)
    pd.testing.assert_frame_equal(
        loaded.fit_table.reset_index(drop=True), res.fit_table.reset_index(drop=True)
    )
    # counterfactual twin and per-bin table survive the round trip
    assert loaded.fit_table_nowo is not None
    pd.testing.assert_frame_equal(
        loaded.fit_table_nowo.reset_index(drop=True), res.fit_table_nowo.reset_index(drop=True)
    )
    assert loaded.bin_table is not None
    pd.testing.assert_frame_equal(
        loaded.bin_table.reset_index(drop=True), res.bin_table.reset_index(drop=True)
    )
    assert loaded.config.ql_layer_on and loaded.config.freq_layer_on


def test_app_defaults_persist_and_never_raise(tmp_path, monkeypatch):
    """The explorer's saved input defaults outlive the process that set them."""
    import analysis.workflows.field_sim.store as store

    path = tmp_path / "field_sim" / "app_defaults.json"
    monkeypatch.setattr(store, "FIELD_SIM_ROOT", tmp_path / "field_sim")
    monkeypatch.setattr(store, "APP_DEFAULTS_PATH", path)

    assert store.load_app_defaults() == {}          # nothing saved yet
    values = {"plateau_wells": 37, "total_years": 13.0, "ql_shape": "linear"}
    assert store.save_app_defaults(values) == path
    assert store.load_app_defaults() == values      # survives a fresh read

    path.write_text("{ not json", encoding="utf-8")
    assert store.load_app_defaults() == {}          # corrupt file is not fatal
    path.write_text('["a list"]', encoding="utf-8")
    assert store.load_app_defaults() == {}          # neither is the wrong shape

    store.save_app_defaults(values)
    assert store.save_app_defaults({}) is None      # empty clears it
    assert not path.exists()
    assert store.load_app_defaults() == {}


def test_app_defaults_file_does_not_confuse_the_experiment_listing(tmp_path, monkeypatch):
    """The preferences file sits in the same directory the listing walks."""
    import analysis.workflows.field_sim.store as store

    root = tmp_path / "field_sim"
    monkeypatch.setattr(store, "FIELD_SIM_ROOT", root)
    monkeypatch.setattr(store, "APP_DEFAULTS_PATH", root / "app_defaults.json")
    cfg = SimConfig(plateau_wells=10, n_seeds=2, n_fit_snapshots=3, km_snapshot_years=[5])
    store.save_experiment(run_experiment(cfg, label="one"))
    store.save_app_defaults({"plateau_wells": 5})

    listed = store.list_experiments()
    assert [r["label"] for r in listed] == ["one"]


def test_config_dict_round_trip_and_validation():
    cfg = SimConfig(workover_mode="statistical", beta_wo=1.3, wo_eta_fraction=1.2)
    assert SimConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()
    with pytest.raises(ValueError):
        SimConfig(workover_mode="bogus").validate()
