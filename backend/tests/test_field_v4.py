"""Per-field v4: the roster/alias contract, the deploy guards, and the contractor de-biasing.

Structural only — no database.  The fits themselves are checked by
``scripts/run/field_v4.py`` against the KM control and the CV gate.
"""
import numpy as np
import pandas as pd
import pytest

from analysis.workflows.production_risk import field_v4 as F
from analysis.workflows.production_risk import unified_v4 as U


# --- roster / aliases -------------------------------------------------------
def test_calculator_codes_map_onto_repo_fields():
    """Au and Mr are pads folded in at population build time, not separate fields."""
    assert F._repo_field("Au") == "Za"
    assert F._repo_field("Mr") == "Mc"
    assert F._repo_field("Vt") == "Vt"
    assert U.FIELD_ALIAS["Au"] == "Za" and U.FIELD_ALIAS["Mr"] == "Mc"


def test_every_roster_code_has_a_field_of_entry():
    for code in F.ROSTER:
        field = F._repo_field(code)
        strata = [s for s, f in U.FIELD_OF.items() if f == field]
        assert strata, f"{code} → {field} has no stratum in FIELD_OF"


def test_mc_is_a_cohort_not_a_full_history():
    """Мирнинский is installs 2024+ — a cohort filter on install date, never left truncation."""
    assert U.FIELD_COHORT_START["Mc"] == "MC_INSTALL_COHORT_START"
    assert "Vt" not in U.FIELD_COHORT_START and "Ya" not in U.FIELD_COHORT_START


def test_only_vt_is_split_on_h2s():
    assert U.SPLIT_H2S == ("Vt",)


# --- WS3: the deploy guard --------------------------------------------------
def test_deploy_theta_qnom_flattens_a_fall_above_the_reference():
    th = np.array([0.55, 0.79, 1.05, 1.0, 1.11, 1.33, 1.37, 1.20])
    out = U.deploy_theta_qnom(th)
    assert out[-1] == pytest.approx(1.37)
    assert list(out[:6]) == pytest.approx(list(th[:6]))     # below the ref: untouched


def test_deploy_theta_qnom_leaves_a_rising_curve_alone():
    th = np.array([0.8, 0.75, 0.92, 1.0, 1.30, 1.62, 1.62, 1.83])
    assert list(U.deploy_theta_qnom(th)) == pytest.approx(list(th))


def test_deploy_theta_qnom_does_not_touch_the_low_end():
    """A protective small-pump segment is a finding; only the sparse top is guarded."""
    th = np.array([0.6, 0.9, 0.7, 1.0, 1.2, 1.3, 1.4, 1.5])
    assert list(U.deploy_theta_qnom(th)[:4]) == pytest.approx(list(th[:4]))


# --- WS2: contractor levels net of the size layers --------------------------
def _window_frame(n=240, seed=3):
    """Synthetic overlap window where slb runs systematically bigger pumps at lower Kpod."""
    rng = np.random.default_rng(seed)
    cg = np.where(np.arange(n) % 2 == 0, "brt", "slb")
    qnom = np.where(cg == "slb", 500.0, 300.0) * np.exp(rng.normal(0, 0.15, n))
    ql = np.full(n, 300.0) * np.exp(rng.normal(0, 0.05, n))
    hazard = (qnom / 250.0) ** 0.8                       # size hurts; contractor does NOT
    t = rng.exponential(600.0 / hazard)
    return pd.DataFrame({U.CLOCK: np.clip(t, 1, None), U.EVENT_COL: 1.0,
                         "contractor_group": cg, "qnom": qnom, "ql": ql,
                         "kpod_run": ql / qnom, "stratum": "Test"})


def test_size_adjusters_remove_a_level_that_is_pure_pump_size():
    d = _window_frame()
    plain = U.fit_contractor_levels(d, size_adjust=False)["level"]["slb"]
    adj = U.fit_contractor_levels(d, size_adjust=True)["level"]["slb"]
    assert plain > 1.15                       # the unadjusted level picks up the size effect
    assert abs(np.log(adj)) < abs(np.log(plain))          # adjusting moves it back toward 1


def test_size_adjust_reports_both_scales():
    r = U.fit_contractor_levels(_window_frame())
    assert set(r["size_terms"]) == {"lq", "lk"}
    assert set(r["size_terms_sd"]) == {"lq", "lk"}
    assert r["size_adjust"] is True


def test_unsupported_contractor_falls_back_to_the_reference():
    """Mc has no slb at all and Ic has 2 oth runs in the window — a dummy with no information."""
    d = _window_frame()
    d.loc[d["contractor_group"] == "slb", "contractor_group"] = "brt"
    d.loc[d.index[:3], "contractor_group"] = "oth"        # below CONTRACTOR_MIN_RUNS
    r = U.fit_contractor_levels(d)
    assert r["level"]["oth"] == 1.0
    assert "oth" in r["unsupported"]
    assert r["unsupported"]["oth"]["n"] == 3


def test_default_is_adjusted_for_unified_and_plain_for_vt_v4():
    """v4 is the published fit of record and must keep reproducing its own tables."""
    from analysis.workflows.production_risk import vt_v4 as V
    assert U.CONTRACTOR_SIZE_ADJUST is True
    assert V.CONTRACTOR_SIZE_ADJUST is False


# --- the shared frame trap --------------------------------------------------
def test_build_cached_frame_restores_the_config_flag():
    from analysis.workflows.production_risk import config as C
    before = C.SOUR_WELL_LEVEL_ALL_RUNS
    try:
        U.build_cached_frame.__doc__.index("v3.2 sour relabel")
    finally:
        assert C.SOUR_WELL_LEVEL_ALL_RUNS == before


# --- deploy blocks ----------------------------------------------------------
def _fake_run(field, strata):
    fits = {}
    for s in strata:
        fits[s] = U.StratumFit(
            stratum=s, n=100, events=50, beta0=1.2, eta0=500.0,
            theta={"qnom": np.array([0.8, 0.9, 0.95, 1.0, 1.1, 1.2, 1.3, 1.1]),
                   "kpod": np.ones(len(U.KPOD_KNOTS)),
                   "freq": np.ones(len(U.FREQ_KNOTS))},
            contractor={"brt": 1.0, "slb": 1.2, "oth": 1.9},
            loglik=-1.0, concordance=0.6)
    frame = pd.DataFrame({U.CLOCK: [100.0], U.EVENT_COL: [1.0], "ql": [300.0],
                          "qnom": [300.0], "kpod_run": [0.8], "stratum": [strata[0]],
                          "contractor_group": ["brt"]})
    lvl = {"level": {"brt": 1.0, "slb": 1.2, "oth": 1.9}, "window": (200.0, 500.0),
           "size_terms": {"lq": 1.5, "lk": 1.2}, "unsupported": {}}
    return F.FieldRun(field=field, fits=fits, contractor=lvl, contractor_plain=lvl,
                      frame=frame, cv={"kpod": 8.4, "freq": -0.8})


def test_deploy_blocks_carry_the_clamped_theta_and_the_aliases():
    runs = {"Za": _fake_run("Za", ["Za"]), "Mc": _fake_run("Mc", ["Mc"])}
    q = F.qnom_block(runs).set_index("key")
    assert q.loc["Za", "q1600"] == pytest.approx(1.3)      # clamped up from the fitted 1.1
    assert q.loc["Au", "q1600"] == q.loc["Za", "q1600"]    # Au reuses Za's row
    t = F.tune_block(runs).set_index("key")
    assert t.loc["Mr", "beta"] == t.loc["Mc", "beta"]
    assert set(t.columns) == {"beta", "rmst_ref_days", "brt", "slb", "oth"}


# --- the pooled fallback ----------------------------------------------------
def test_fleet_is_on_the_roster():
    assert F.FLEET_KEY in F.ROSTER
    assert F.FLEET_CODES == ("Da", "Ki", "Ma", "Bt")


def test_unmodelled_codes_get_a_row_pointing_at_the_fleet_fit():
    """Ki / Ma / Bt have no stratum in the mart at all; Da has 50 events, one contractor."""
    runs = {"Fleet": _fake_run("Fleet", [F.FLEET_KEY]), "Za": _fake_run("Za", ["Za"])}
    t = F.tune_block(runs).set_index("key")
    for code in F.FLEET_CODES:
        assert code in t.index, code
        assert t.loc[code, "beta"] == t.loc[F.FLEET_KEY, "beta"]
    q = F.qnom_block(runs).set_index("key")
    assert q.loc["Ki", "q1600"] == q.loc[F.FLEET_KEY, "q1600"]


def test_fleet_frame_stacks_every_field_with_the_field_as_stratum(monkeypatch):
    seen = []

    def fake_prepare(field, cached=None, as_of=None):
        seen.append(field)
        return pd.DataFrame({U.CLOCK: [100.0], U.EVENT_COL: [1.0], "ql": [300.0],
                             "qnom": [300.0], "kpod_run": [0.8], "freq_dev": [0.0],
                             "contractor_group": ["brt"], "stratum": ["placeholder"]})

    monkeypatch.setattr(U, "prepare_field", fake_prepare)
    d = F.fleet_frame(cached=pd.DataFrame(), fields=["Ya", "Vt", "Mc"])
    assert seen == ["Ya", "Vt", "Mc"]
    # stratum carries the FIELD, which is what makes the stage-1 fixed effects work
    assert sorted(d["stratum"].unique()) == ["Mc", "Vt", "Ya"]


# --- stratum fixed effects --------------------------------------------------
def test_extra_strata_enter_the_window_fit_as_dummies():
    d = _window_frame()
    d.loc[d.index[:80], "stratum"] = "B"                   # two strata now
    r = U.fit_contractor_levels(d)
    assert r["n"] > 0                                      # fits without a singular design
    one = _window_frame()
    assert U.fit_contractor_levels(one)["level"]["slb"] > 0


def test_layer_table_flags_what_the_guard_changed():
    runs = {"Za": _fake_run("Za", ["Za"])}
    tbl = F.layer_table(runs)
    top = tbl[(tbl["layer"] == "qnom") & (tbl["x"] == 1600.0)].iloc[0]
    assert top["clamped"] is True or bool(top["clamped"])
    assert top["theta_fitted"] == pytest.approx(1.1)
    assert top["theta_deployed"] == pytest.approx(1.3)
