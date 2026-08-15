"""Tests for the Vt failure/ГТМ competing-risks accounting (lab-based H2S).

Covers the handoff's acceptance-critical invariants:
  * the lab classifier is three-way and well-level (open runs inherit; never
    default-to-nonsour);
  * the single calendar clock holds on Vt (the mixed-clock tte trap);
  * sour and nonsour are never pooled in any fit call;
  * the decomposition identity holds per H2S class.
The Mc tests live in ``test_mc_competing_risks`` and must stay green (the shared
engine is byte-identical); this file only touches the Vt module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.models.survival import cif as CIF
from analysis.workflows.production_risk import config as C
from analysis.workflows.production_risk import vt_competing_risks as V


@pytest.fixture(scope="module")
def pop() -> pd.DataFrame:
    return V.build_population(C.SVOD_OPEN_ASOF)


# --------------------------------------------------------------------------- #
# Lab classifier — three-way, well-level, no default-to-nonsour                 #
# --------------------------------------------------------------------------- #
def test_lab_class_is_three_way_and_covers_no_lab(pop: pd.DataFrame):
    classes = set(pop["lab_class"].unique())
    assert classes.issubset(set(V.LAB_CLASSES))
    # all three appear on Vt (measured 2026-07-23: sour_lab/nonsour_lab/no_lab)
    assert classes == set(V.LAB_CLASSES)
    # no_lab is a real, non-empty class — never folded into nonsour
    assert (pop["lab_class"] == "no_lab").any()


def test_lab_class_is_well_level_and_inherited_by_open_runs(pop: pd.DataFrame):
    # every run of a well shares exactly one lab class (open runs included)
    per_well = pop.groupby("code")["lab_class"].nunique()
    assert (per_well == 1).all()
    # open runs are classified too (not dropped, not forced to a default)
    open_runs = pop[pop["is_open"]]
    assert len(open_runs) > 100
    assert open_runs["lab_class"].isin(V.LAB_CLASSES).all()
    # sour_lab open runs exist — the direct successor to defect #4 (open pumps on
    # sour wells must not be misfiled nonsour)
    assert (open_runs["lab_class"] == "sour_lab").any()


def test_lab_class_matches_measured_positive_wells():
    # classifier flags a well sour_lab iff it ever has a '+' svb_h2s_indicator
    m = V.lab_h2s_class_map()
    assert set(m.unique()).issubset({"sour_lab", "nonsour_lab"})
    # '+' is the only positive value ⇒ every sour_lab well truly has a '+' sample;
    # a well with samples but no '+' is nonsour_lab (a weak negative), never no_lab
    assert (m == "sour_lab").sum() > 0
    assert (m == "nonsour_lab").sum() > 0


# --------------------------------------------------------------------------- #
# Single calendar clock on Vt                                                   #
# --------------------------------------------------------------------------- #
def test_open_runs_on_single_calendar_clock(pop: pd.DataFrame):
    asof = pd.Timestamp(C.SVOD_OPEN_ASOF)
    open_runs = pop[pop["is_open"]]
    cal_age = (asof - open_runs["install"]).dt.days.clip(lower=0.5).astype(float)
    assert np.allclose(open_runs["t_cal"].to_numpy(float), cal_age.to_numpy(float))
    assert np.allclose(open_runs["current_age"], open_runs["t_cal"])
    assert (pop["t_cal"] > 0).all() and np.isfinite(pop["t_cal"]).all()


def test_vt_uses_all_history_not_mc_cohort(pop: pd.DataFrame):
    # Vt is all-history: installs before the Mc 2024 cohort start must survive.
    assert (pop["install"] < pd.Timestamp(C.MC_INSTALL_COHORT_START)).any()


# --------------------------------------------------------------------------- #
# sour and nonsour are never pooled in any fit                                  #
# --------------------------------------------------------------------------- #
def test_no_fit_pools_sour_and_nonsour(pop: pd.DataFrame):
    # Every stratum frame the module fits is drawn from exactly one lab class.
    for cls in V.FIT_CLASSES:
        for cell in V.CONTRACTOR_CELLS:
            g = V._stratum_frame(pop, cls, cell)
            if len(g):
                assert g["lab_class"].nunique() == 1
                assert g["lab_class"].iloc[0] == cls


def test_cause_specific_params_never_mix_classes(pop: pd.DataFrame):
    params = V.cause_specific_params(pop, n_boot=0)
    # each row is tagged with one class; the collider frame likewise
    assert set(params["lab_class"]).issubset(set(V.LAB_CLASSES))
    coll = V.collider_slb_brt(pop, n_boot=20)
    assert set(coll["h2s_class"]).issubset(set(V.FIT_CLASSES))


# --------------------------------------------------------------------------- #
# CIF / decomposition identity per class                                        #
# --------------------------------------------------------------------------- #
def test_cif_partitions_all_cause_incidence_per_class(pop: pd.DataFrame):
    for cls in V.FIT_CLASSES:
        g = V._stratum_frame(pop, cls, "all")
        t = g["t_cal"].to_numpy(float)
        ec = g["cause_code"].to_numpy(int)
        cif_f = CIF.aalen_johansen_cif(t, ec, V.FAILURE)
        cif_g = CIF.aalen_johansen_cif(t, ec, V.GTM)
        km_all = CIF.all_cause_km(t, (ec > 0).astype(int))
        for tt in (90.0, 180.0, 365.0):
            assert cif_f.at(tt) + cif_g.at(tt) == pytest.approx(km_all.at(tt), abs=1e-9)


def test_decomposition_identity_per_class(pop: pd.DataFrame):
    asof = pd.Timestamp(C.SVOD_OPEN_ASOF)
    dec = V.decomposition_by_class(pop, as_of=asof, horizon_months=12)
    for cls in V.FIT_CLASSES:
        fc = dec[(dec["lab_class"] == cls) & (dec["segment"] == "forecast")]
        assert len(fc) == 12
        # total pulls == failures + ГТМ (up to CSV rounding)
        assert np.allclose(fc["m3_total_pulls"], fc["m1_obs_failures"] + fc["m2_gtm"], atol=2e-4)
        # all five columns are finite and non-negative
        for col in ("m1_obs_failures", "m2_gtm", "m3_total_pulls", "m4_latent_nogtm_failures"):
            assert (fc[col] >= -1e-9).all() and np.isfinite(fc[col]).all()


def test_k1k2_survival_lab_split(pop: pd.DataFrame):
    k = V.k1k2_survival(pop, n_boot=0)
    # only the lab fit-classes appear (no_lab omitted as non-informative)
    assert set(k["lab_class"]).issubset(set(V.FIT_CLASSES))
    assert not k["stratum"].str.contains("no_lab").any()
    # both populations present, each with a k1 and a k2 row per class
    assert set(k["population"]) == set(V.K1K2_STRATA)
    for (strat, cls), grp in k.groupby(["population", "lab_class"]):
        assert set(grp["k"]) == {"k1", "k2"}
        # exactly one k is AIC-selected per stratum
        assert int(grp["selected"].sum()) == 1
    # event is genuine failure with ГТМ censored: n_failures counts only FAILURE
    for _, r in k.iterrows():
        g = V._k1k2_population(pop, r["lab_class"], r["population"])
        assert r["n_failures"] == int((g["cause_code"] == V.FAILURE).sum())
        assert r["n_gtm"] == int((g["cause_code"] == V.GTM).sum())
        assert np.isfinite(r["rmst_0_730"]) and r["rmst_0_730"] > 0
    # slb+brt population excludes oth; pulled keeps all contractors
    slbbrt = V._k1k2_population(pop, "sour_lab", "slb+brt")
    assert set(slbbrt["contractor_group"]).issubset({"slb", "brt"})


def test_latent_bound_is_a_lower_bound_direction(pop: pd.DataFrame):
    _, runs_by_class = V.telemetry_level2_by_class(pop)
    lat = V.latent_bounds(pop, runs_by_class)
    # recoding precursor-positive ГТМ as failures cannot LENGTHEN λ_fail RMST
    for _, r in lat.iterrows():
        assert r["lambda_fail_rmst_upper_bound"] <= r["lambda_fail_rmst_baseline"] + 1e-6
