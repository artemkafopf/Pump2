"""Tests for the Рприем reconstruction (:mod:`analysis.features.intake_pressure`,
:mod:`analysis.models.ml.intake_pressure_ml`, :mod:`analysis.workflows.intake_pressure`).

Two kinds of claim are pinned here.  The first is ordinary physics: the head relation
inverts, Vogel is monotone and bounded, the linear form reads back the constants it was
built from.  The second is the set of **traps** the measurements turned up, which are the
reason this package is shaped the way it is — Рзаб must never reach a feature set, the
target must never reach one either, and the honest baseline must stay in the candidate
list.  Those are regression tests against a future refactor quietly reintroducing the
leak.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.features import intake_pressure as IP
from analysis.models.ml import intake_pressure_ml as ML
from analysis.workflows.intake_pressure import gas_at_intake as GI


# ---------------------------------------------------------------------------
# Physical primitives
# ---------------------------------------------------------------------------

def test_column_atm_matches_hand_calculation():
    """1000 kg/m³ over 10.33 m is one atmosphere."""
    assert IP.column_atm(1000.0, 10.3323) == pytest.approx(1.0, rel=1e-3)


def test_liquid_density_interpolates_between_oil_and_water():
    assert IP.liquid_density(0.0) == pytest.approx(IP.RHO_OIL)
    assert IP.liquid_density(100.0) == pytest.approx(IP.RHO_WATER)
    assert IP.liquid_density(50.0) == pytest.approx((IP.RHO_OIL + IP.RHO_WATER) / 2)


def test_vogel_bounded_and_monotone_decreasing_in_rate():
    p = IP.vogel_pwf(200.0, np.array([0.0, 100.0, 300.0, 600.0, 900.0]), 1000.0)
    assert np.all(np.diff(p) < 0), "more rate must mean lower bottomhole pressure"
    assert p[0] == pytest.approx(200.0), "zero rate returns reservoir pressure"
    assert np.all((p >= 0) & (p <= 200.0))


def test_vogel_at_absolute_open_flow_is_zero_not_imaginary():
    """q >= q_max clamps rather than going imaginary — the guard that keeps a noisy
    daily rate from producing NaN for a whole well."""
    assert IP.vogel_pwf(200.0, 1500.0, 1000.0) == pytest.approx(0.0)


def test_fit_annulus_h_recovers_a_planted_column():
    rng = np.random.default_rng(0)
    wc = rng.uniform(0, 100, 500)
    pi = rng.uniform(20, 80, 500)
    h_true = 137.0
    rzab = pi + IP.column_atm(IP.liquid_density(wc), h_true)
    assert IP.fit_annulus_h(rzab, pi, wc) == pytest.approx(h_true, rel=1e-6)


def test_fit_annulus_h_is_median_so_spikes_do_not_move_it():
    """The daily inversion spans -2556 … 6495 m on real telemetry; the reduction has to
    be a median or a single spike carries the well."""
    wc = np.full(200, 50.0)
    pi = np.full(200, 40.0)
    rzab = pi + IP.column_atm(IP.liquid_density(wc), 100.0)
    rzab[:20] = 590.0  # spikes inside the pressure guard
    assert IP.fit_annulus_h(rzab, pi, wc) == pytest.approx(100.0, rel=1e-6)


# ---------------------------------------------------------------------------
# The linear form round-trips its own constants
# ---------------------------------------------------------------------------

def _synthetic_panel(n_wells=6, n_days=200, seed=0, a=90.0, b=0.05, h=120.0):
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(n_wells):
        q = rng.uniform(50, 400, n_days)
        wc = rng.uniform(5, 95, n_days)
        aw = a + 10 * w
        pi = IP.predict_intake_linear(q, wc, aw, b, h) + rng.normal(0, 0.5, n_days)
        rows.append(pd.DataFrame({
            "well_key": f"w{w}", "field": "Ya",
            "dt": pd.date_range("2022-01-01", periods=n_days),
            "qliq": q, "watercut": wc, "rpump_intake": pi,
            "rpl": rng.uniform(150, 200, n_days), "rzab": pi + 10,
            "gas_factor": rng.uniform(80, 250, n_days), "freq": rng.uniform(45, 55, n_days),
            "load": rng.uniform(40, 90, n_days), "kprod": rng.uniform(1, 10, n_days),
            "vg_m": 2400.0, "pump_depth_m": 2700.0, "q_nom_m3d": 250.0,
            "head_nom_m": 2100.0, "stages": 300.0, "pump_od_mm": 103.0,
            "pbubble_atm": 240.0, "op_day": np.arange(n_days), "month": 6,
            "qoil": q * (1 - wc / 100), "qwater": q * wc / 100,
            "rho_mix_kgm3": IP.liquid_density(wc), "kpod": q / 250.0,
        }))
    df = pd.concat(rows, ignore_index=True)
    df["is_observed"] = True
    return df


def test_linear_fit_recovers_planted_constants():
    df = _synthetic_panel()
    m = IP.PhysicalIntakeModel(mode="linear").fit(df)
    ct = m.calibration_table()
    assert len(ct) == 6
    assert ct["b"].median() == pytest.approx(0.05, abs=0.01)
    assert ct["h_eff_m"].median() == pytest.approx(120.0, abs=15.0)
    assert ct["resid_mad"].max() < 2.0


def test_linear_predict_is_accurate_on_its_own_generator():
    df = _synthetic_panel()
    m = IP.PhysicalIntakeModel(mode="linear").fit(df)
    err = np.abs(m.predict(df) - df["rpump_intake"].to_numpy())
    assert np.median(err) < 1.0


def test_unseen_well_falls_back_to_field_tier_not_nan():
    """The cold-well case: no own history, so the field pool must answer.  This is the
    tier the validation reports separately because it is far weaker — but it must exist."""
    df = _synthetic_panel()
    m = IP.PhysicalIntakeModel(mode="linear").fit(df)
    cold = df[df["well_key"] == "w0"].copy()
    cold["well_key"] = "brand_new_well"
    assert m.predict_source(cold)[0] == "field"
    assert np.isfinite(m.predict(cold)).all()


def test_predictions_are_clipped_to_the_physical_guard():
    df = _synthetic_panel()
    m = IP.PhysicalIntakeModel(mode="linear").fit(df)
    wild = df.copy()
    wild["qliq"] = 1e6  # drives the linear form far negative
    p = m.predict(wild)
    assert np.all((p >= 0.5) & (p <= 400.0))


# ---------------------------------------------------------------------------
# The baseline that everything must beat
# ---------------------------------------------------------------------------

def test_well_median_baseline_is_per_well_and_falls_back_for_unseen():
    df = _synthetic_panel()
    b = IP.WellMedianBaseline().fit(df)
    p = b.predict(df)
    for w, g in df.groupby("well_key"):
        assert np.allclose(p[df["well_key"] == w], np.median(g["rpump_intake"]))
    cold = df.head(5).assign(well_key="unseen")
    assert np.isfinite(b.predict(cold)).all()


# ---------------------------------------------------------------------------
# The leakage traps — regression tests, not style checks
# ---------------------------------------------------------------------------

def test_rzab_is_never_a_feature():
    """Рзаб is «Расчетное забойное давление» — calculated FROM the target, correlated
    0.933 with it, and present on only 10.3 % of the gap.  A model given it scores
    beautifully in CV and fails in production."""
    for variant in ("plain", "physics", "residual"):
        feats = ML.IntakeMLModel(variant=variant).feature_names()
        assert "rzab" not in feats
    assert "rzab" in ML.FORBIDDEN


def test_target_and_mask_are_never_features():
    for variant in ("plain", "physics", "residual"):
        feats = ML.IntakeMLModel(variant=variant).feature_names()
        assert "rpump_intake" not in feats
        assert "is_observed" not in feats


def test_forbidden_feature_is_rejected_loudly():
    m = ML.IntakeMLModel(variant="plain")
    ML.DAILY_FEATURES.append("rzab")
    try:
        with pytest.raises(ValueError, match="forbidden"):
            m.feature_names()
    finally:
        ML.DAILY_FEATURES.remove("rzab")


def test_physics_variant_adds_physics_features_plain_does_not():
    plain = set(ML.IntakeMLModel(variant="plain").feature_names())
    phys = set(ML.IntakeMLModel(variant="physics").feature_names())
    assert set(ML.PHYS_FEATURES).isdisjoint(plain)
    assert set(ML.PHYS_FEATURES) <= phys


def test_history_features_are_absent_when_disabled():
    """The cold regime must not be able to reach a per-well aggregate of the target."""
    feats = set(ML.IntakeMLModel(variant="plain", use_history=False).feature_names())
    assert set(ML.HIST_FEATURES).isdisjoint(feats)


# ---------------------------------------------------------------------------
# ML realizations run and stay physical
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", ["plain", "physics", "residual"])
def test_ml_variants_fit_and_predict_in_range(variant):
    df = _synthetic_panel(n_wells=5, n_days=150)
    m = ML.IntakeMLModel(variant=variant, params=dict(ML.DEFAULT_PARAMS, iterations=40))
    m.fit(df)
    p = m.predict(df)
    assert len(p) == len(df)
    assert np.all((p >= 0.5) & (p <= 400.0))
    assert np.isfinite(p).all()


def test_residual_variant_adds_the_physical_baseline_back():
    """The property that makes this 'physics-informed' rather than physics-flavoured: the
    booster learns a residual and the physical prediction is added back verbatim, so the
    physical model owns the level.  Checked as an identity, not a correlation."""
    from catboost import Pool

    df = _synthetic_panel(n_wells=4, n_days=120)
    m = ML.IntakeMLModel(variant="residual", params=dict(ML.DEFAULT_PARAMS, iterations=30))
    m.fit(df)
    raw = np.asarray(
        m.model.predict(Pool(m._design(df)[m._feats], cat_features=ML.CAT_FEATURES)), float
    )
    base = m.phys.predict(df)
    expected = np.clip(raw + base, 0.5, 400.0)
    assert np.allclose(m.predict(df), expected)


def test_add_physics_features_static_column_is_physical():
    df = _synthetic_panel(n_wells=2, n_days=50)
    out = ML.add_physics_features(df, None)
    # 2400 m of ~930 kg/m3 liquid is ~215 atm
    assert out["p_static_atm"].between(180, 240).all()
    assert out["phys_pred_atm"].isna().all(), "no model supplied -> no prediction"


# ---------------------------------------------------------------------------
# β propagation
# ---------------------------------------------------------------------------

def test_gor_unit_conversion_scales_by_oil_density():
    """«Газовый фактор, м3/т» is per tonne; Standing wants per m³ of stock-tank oil."""
    assert GI.gor_m3_per_m3(100.0) == pytest.approx(100.0 * GI.DEFAULT_PVT.oil_sg)
    assert GI.gor_m3_per_m3(100.0) < 100.0


def test_attach_beta_produces_a_fraction():
    df = _synthetic_panel(n_wells=3, n_days=80)
    out = GI.attach_beta(df, "rpump_intake")
    b = out["beta"].dropna()
    assert len(b) > 0
    assert ((b >= 0) & (b <= 1)).all()


def test_beta_falls_as_intake_pressure_rises():
    """The monotonicity the whole propagation rests on: more pressure keeps more gas in
    solution, so β must fall."""
    df = _synthetic_panel(n_wells=1, n_days=60)
    lo = GI.attach_beta(df.assign(rpump_intake=30.0), "rpump_intake")["beta"]
    hi = GI.attach_beta(df.assign(rpump_intake=150.0), "rpump_intake")["beta"]
    assert np.nanmean(hi) < np.nanmean(lo)


def test_beta_error_profile_reports_bands_and_sensitivity():
    df = _synthetic_panel(n_wells=4, n_days=120)
    df["p_intake_hat"] = df["rpump_intake"] * 1.15
    prof = GI.beta_error_profile(df)
    assert len(prof) > 0
    assert {"beta_mae", "dbeta_per_atm", "p_mae_atm"} <= set(prof.columns)
    assert (prof["beta_mae"] >= 0).all()


def test_range_compression_biases_beta_toward_its_middle():
    """The trap that makes imputed β unusable for exposure shares: a reconstruction that
    is compressed toward the mean *understates* β where pressure is low (gassiest wells)
    and *overstates* it where pressure is high.  Measured on the fleet the bias runs
    −0.091 → +0.050 across the pressure bands; here the mechanism is reproduced exactly."""
    df = _synthetic_panel(n_wells=6, n_days=150)
    p = df["rpump_intake"].to_numpy()
    # Shrink 20 % toward the mean — the same direction a boosted tree errs in.
    df["p_intake_hat"] = p.mean() + 0.8 * (p - p.mean())
    prof = GI.beta_error_profile(df).sort_values("band")
    lo, hi = prof.iloc[0], prof.iloc[-1]
    assert lo["beta_bias"] < 0 < hi["beta_bias"], (
        "compression must under-state beta at low pressure and over-state it at high"
    )


def test_beta_rank_stability_is_perfect_for_an_exact_reconstruction():
    df = _synthetic_panel(n_wells=8, n_days=100)
    df["p_intake_hat"] = df["rpump_intake"]
    r = GI.beta_rank_stability(df)
    assert r["spearman"] == pytest.approx(1.0)
