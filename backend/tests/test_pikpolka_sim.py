"""The Python port of the calculator: layer maths, the ladder fix, and the sweep contract.

Workbook-parity checks live in ``pikpolka_sim.validate_against_workbook`` — they need the
operator's files, which are outside the repo, so they are not asserted here.  What is asserted
is everything that can go wrong without Excel: the layer forms, the two divergences from the
sheet that are deliberate, and the day-grid bookkeeping.
"""
import numpy as np
import pytest

from analysis.workflows.production_risk import pikpolka_sim as S


# --- layers -----------------------------------------------------------------
def test_theta_at_reference_is_one():
    assert S.theta_qnom("Vt_nonsour", 250.0) == pytest.approx(1.0)
    assert S.theta_qnom("Vt_sour", 250.0) == pytest.approx(1.0)
    assert S.theta_kpod(0.8) == pytest.approx(1.0)          # inside the plateau


def test_theta_freq_50_is_not_one():
    """The shipped anchors put θ(50 Hz) = 0.9 — 'all θ = 1' is NOT the life at 50 Hz."""
    assert S.theta_freq(50.0) == pytest.approx(0.9)
    assert S.theta_freq(40.0) == pytest.approx(1.8)
    assert S.theta_freq(60.0) == pytest.approx(1.3)
    assert S.theta_freq(70.0) == pytest.approx(3.75, rel=1e-6)


def test_theta_freq_clamped_outside_35_70():
    assert S.theta_freq(20.0) == pytest.approx(S.theta_freq(35.0))
    assert S.theta_freq(90.0) == pytest.approx(S.theta_freq(70.0))


def test_null_ctrl_switches_both_operator_layers_off():
    """Scenario 0: θ_freq ≡ 1 and θ_Kpod ≡ 1 across the whole clamped range, exactly."""
    for f in (35, 40, 45, 50, 55, 60, 65, 70, 20, 90):
        assert S.theta_freq(float(f), S.NULL_CTRL) == pytest.approx(1.0)
    for k in (0.05, 0.15, 0.2, 0.5, 0.8, 1.0, 1.2, 1.6, 1.8, 3.0):
        assert S.theta_kpod(float(k), S.NULL_CTRL) == pytest.approx(1.0)


def test_null_ctrl_leaves_only_the_fitted_layers():
    """With the priors off, life at the reference IS the fitted RMST_ref."""
    lay = S.Layers(key="Vt_nonsour", ctrl=dict(S.NULL_CTRL))
    assert lay.life_at_ref() == pytest.approx(S.rmst(lay.eta0, lay.beta), rel=1e-9)
    # …and Qnom still moves it, because that layer is fitted, not assumed
    assert lay.life_days(contractor="brt", qnom=1600, freq=50, kpod=0.8) < lay.life_at_ref()


def test_theta_qnom_flat_outside_the_knots():
    assert S.theta_qnom("Vt_nonsour", 10.0) == pytest.approx(S.theta_qnom("Vt_nonsour", 60.0))
    assert S.theta_qnom("Vt_nonsour", 5000.0) == pytest.approx(
        S.theta_qnom("Vt_nonsour", 1600.0))


def test_sour_qnom_is_clamped_flat_above_1000():
    """WS3: free-fitted the sour arm turns down above 1000 on 20 runs / 14 events."""
    assert S.QNOM["Vt_sour"][-1] == pytest.approx(S.QNOM["Vt_sour"][-2])
    assert S.QNOM_PRECLAMP["Vt_sour"][-1] < S.QNOM_PRECLAMP["Vt_sour"][-2]
    assert S.theta_qnom("Vt_sour", 1600.0) == pytest.approx(S.theta_qnom("Vt_sour", 1000.0))


def test_contractor_matcher_handles_both_alphabets():
    b, s, o = 1.0, 1.2, 1.9
    for name in ("Борец", "БОРЕЦ", "Borets", "борец-РЕДА", ""):
        assert S.contractor_mult(name, b, s, o) == b
    for name in ("Шлюмберже", "ШЛЮМБЕРЖЕ", "SLB", "Schlumberger", "Слайб"):
        assert S.contractor_mult(name, b, s, o) == s
    for name in ("Новые_Технологии", "Новомет", "ССК"):
        assert S.contractor_mult(name, b, s, o) == o


def test_eta_from_rmst_round_trips():
    for beta, target in ((1.25, 407.284), (1.34, 183.8), (0.9, 300.0)):
        eta = S.eta_from_rmst(target, beta)
        assert S.rmst(eta, beta) == pytest.approx(target, rel=1e-6)


def test_life_falls_as_theta_rises():
    lay = S.Layers(key="Vt_nonsour")
    small = lay.life_days(contractor="brt", qnom=100, freq=50, kpod=0.8)
    big = lay.life_days(contractor="brt", qnom=1600, freq=50, kpod=0.8)
    assert big < small


def test_life_at_ref_differs_from_bare_baseline():
    """θ_freq(50) = 0.9, so the reference life is NOT RMST(η₀, β₀)."""
    lay = S.Layers(key="Vt_nonsour")
    assert lay.life_at_ref() > S.rmst(lay.eta0, lay.beta)


# --- the ladder fix (WS1) ---------------------------------------------------
LADDER = (30, 35, 60, 80, 100, 125, 160, 200, 250, 320, 400, 500, 700, 800, 1000, 1250, 1500)


def test_lookup_nominal_picks_next_larger():
    assert S.lookup_nominal(800, LADDER) == 800
    assert S.lookup_nominal(801, LADDER) == 1000
    assert S.lookup_nominal(1, LADDER) == 30


def test_lookup_nominal_above_the_ladder_falls_back_to_the_top():
    """The workbook's 999999 default drove Kpod to ~0 and cost ~40 % of the modelled life."""
    assert S.lookup_nominal(1600, LADDER) == 1500
    assert S.lookup_nominal(5000, LADDER) == 1500


def test_above_ladder_life_is_continuous_and_monotone():
    lay = S.Layers(key="Vt_nonsour")

    def life(ql):
        nom = S.lookup_nominal(ql, LADDER)
        return lay.life_days(contractor="brt", qnom=nom, freq=50, kpod=ql / nom)

    lives = [life(q) for q in (1400, 1500, 1550, 1600, 1800)]
    assert all(np.isfinite(lives))
    assert lives == sorted(lives, reverse=True)          # no cliff at the top of the ladder
    assert min(lives) > 0.5 * max(lives)                 # and no collapse


# --- decline & water cut ----------------------------------------------------
def _cfg(**kw):
    base = dict(niz=100_000.0, tiz=0.0, ladder=LADDER,
                wcut_curve=((0.0, 0.0), (0.5, 0.5), (1.0, 1.0)),
                netback=10_000.0, downtime_days=12, days=400,
                branches=(S.Branch("base", 300.0, 50.0, 320.0, "Борец"),))
    base.update(kw)
    return S.SheetConfig(**base)


def test_decline_switches_form_at_day_181():
    """Phase 1 is curved to day 180, then the sheet switches to a linear phase."""
    cfg = _cfg(decline_6m=0.1, decline_after=0.9, bend=0.5)
    assert S.decline_ql(cfg, 300.0, 1) == pytest.approx(300.0)
    d180, d181, d182 = (S.decline_ql(cfg, 300.0, d) for d in (180, 181, 182))
    assert d181 == pytest.approx(d180)                  # the linear phase starts at factor 1
    assert d182 < d181
    # extrapolating phase 1 instead would have gone negative well before the grid ends
    assert S.decline_ql(cfg, 300.0, 1000) > 0


def test_wcut_takes_the_smallest_point_at_or_above():
    curve = ((0.9, 0.9), (1.00085, 0.99), (1.00227, 0.995), (1.0, 1.0))
    assert S._wcut(curve, 0.99903) == 1.0               # next-larger, not first-encountered
    assert S._wcut(curve, 5.0) == 0.0                   # off the end → XLOOKUP's if_not_found


def test_dead_well_stays_dead():
    """Once water cut hits 1 the cumulative freezes, so the well cannot come back to life."""
    # tiz == niz ⇒ the recovery share starts at 0 and grows with cumulative oil
    cfg = _cfg(days=300, niz=1000.0, tiz=1000.0, wcut0=0.0)
    g = S.simulate_branch(cfg, cfg.branches[0], S.Layers(key="Vt_nonsour"))
    zero_from = np.flatnonzero(g["oil"].to_numpy() <= 0)
    assert len(zero_from)
    assert (g["oil"].to_numpy()[zero_from[0]:] <= 0).all()


# --- day grid ---------------------------------------------------------------
def test_failures_renew_and_carry_the_remainder():
    cfg = _cfg(days=1400)          # reference life is ~400 d, so 1400 gives three cycles
    g = S.simulate_branch(cfg, cfg.branches[0], S.Layers(key="Vt_nonsour"))
    days = g.loc[g["fail"] == 0, "day"].to_numpy()
    assert len(days) >= 2
    gaps = np.diff(days)
    assert (gaps > 1).all()


def test_downtime_zeroes_revenue_for_the_window():
    cfg = _cfg(days=600, downtime_days=12)
    g = S.simulate_branch(cfg, cfg.branches[0], S.Layers(key="Vt_nonsour"))
    first = int(g.loc[g["fail"] == 0, "day"].iloc[0])
    window = g[(g["day"] >= first) & (g["day"] < first + 12)]
    assert (window["up"] == 0).all()
    assert (window["revenue"] == 0).all()
    assert int(g.loc[g["day"] == first + 12, "up"].iloc[0]) == 1


def test_sweep_rejects_unknown_axes():
    cfg = _cfg()
    with pytest.raises(KeyError):
        S.sweep(cfg, nonsense=[1, 2])


def test_sweep_shape_and_monotonicity():
    cfg = _cfg(days=500)
    g = S.sweep(cfg, **{"branch0:freq": [50.0, 60.0, 65.0]})
    assert len(g) == 3
    nno = g.sort_values("branch0:freq")["nno_days"].to_numpy()
    assert nno[0] > nno[1] > nno[2]         # the shipped freq prior penalises over-speed
