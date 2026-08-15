"""Contract tests for the unified v4 model and the FREE polyline shape it rests on.

Structural, so they stay fast: what is pinned is the *contract* — that no arm imposes a
shape, that the composition rule is θ-multiply-then-convert-once, that the thin sour stratum
borrows the shapes it cannot estimate, and that the two findings which overturned earlier
conclusions (Kpod is real; the tent is what hid it) cannot be silently reverted.
"""
import numpy as np
import pytest

from analysis.models.survival import polyline_ph as P
from analysis.workflows.production_risk import unified_v4 as U


# ---------------------------------------------------------------------------
# The FREE shape
# ---------------------------------------------------------------------------
def _free_arm(ridge: float = 0.0) -> P.ArmSpec:
    return P.ArmSpec("k", "k", (0.2, 0.5, 0.8, 1.2), 0.8, P.FREE, ridge)


def test_free_arm_admits_negative_increments():
    """The whole point: a free arm may go *down*, which mono and tent structurally cannot."""
    a = _free_arm()
    lt = P.log_theta_from_increments(a, np.array([0.3, -0.4, 0.2]))
    assert np.any(np.diff(lt) < 0)
    assert a.increment_bounds[0] < 0


@pytest.mark.parametrize("shape", [P.MONO, P.TENT])
def test_one_sided_shapes_still_clip(shape):
    """Regression: adding FREE must not relax the sign constraint on the other two."""
    a = P.ArmSpec("k", "k", (0.2, 0.5, 0.8, 1.2), 0.8, shape, 0.0)
    lt = P.log_theta_from_increments(a, np.array([0.3, -0.4, 0.2]))
    assert a.increment_bounds == (0.0, 4.0)
    if shape == P.MONO:
        assert np.all(np.diff(lt) >= -1e-12)
    else:
        assert np.all(lt >= -1e-12)          # tent: θ ≥ 1 everywhere


def test_free_arm_is_pinned_at_the_reference():
    a = _free_arm()
    lt = P.log_theta_from_increments(a, np.array([0.3, -0.4, 0.2]))
    assert lt[a.pin_index] == pytest.approx(0.0)


def test_tent_cannot_represent_a_monotone_effect():
    """The measurement error behind every previous 'Kpod is null' verdict.

    A tent is θ ≥ 1 on both arms, so the closest it can get to a rising line is flat below
    the pin — it must discard the entire protective half of a monotone effect.
    """
    tent = P.ArmSpec("k", "k", (0.2, 0.5, 0.8, 1.2), 0.8, P.TENT, 0.0)
    best = np.inf
    truth = np.log([0.6, 0.8, 1.0, 1.25])
    for _ in range(400):
        inc = np.abs(np.random.default_rng(_).normal(0, 0.4, tent.n_increments))
        best = min(best, float(np.sum((P.log_theta_from_increments(tent, inc) - truth) ** 2)))
    free = P.log_theta_from_increments(_free_arm(), np.diff(truth))
    assert float(np.sum((free - truth) ** 2)) == pytest.approx(0.0, abs=1e-12)
    assert best > 0.1, "if a tent could fit a monotone curve, the Kpod null was not an artifact"


def test_offset_enters_with_coefficient_fixed_at_one():
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 400
    d = pd.DataFrame({
        "t": rng.gamma(2.0, 150.0, n) + 5.0,
        "e": rng.integers(0, 2, n).astype(float),
        "k": rng.uniform(0.2, 1.2, n),
        "off": np.zeros(n),
    })
    a = P.ArmSpec("k", "k", (0.2, 0.5, 0.8, 1.2), 0.8, P.FREE, 4.0)
    plain = P.fit_polyline_ph(d, clock="t", event_col="e", arms=(a,))
    zero = P.fit_polyline_ph(d, clock="t", event_col="e", arms=(a,), offset="off")
    assert zero.loglik == pytest.approx(plain.loglik, rel=1e-6)

    d["off"] = np.log(2.0)                    # a known doubling must be absorbed by η₀
    doubled = P.fit_polyline_ph(d, clock="t", event_col="e", arms=(a,), offset="off")
    assert doubled.eta0 > plain.eta0
    assert doubled.loglik == pytest.approx(plain.loglik, rel=1e-3)


def test_all_null_offset_raises_instead_of_fitting_nothing():
    import pandas as pd

    d = pd.DataFrame({"t": [10.0, 20.0, 30.0], "e": [1.0, 0.0, 1.0],
                      "k": [0.3, 0.8, 1.1], "off": [np.nan] * 3})
    a = P.ArmSpec("k", "k", (0.2, 0.5, 0.8, 1.2), 0.8, P.FREE, 4.0)
    with pytest.raises(ValueError, match="no complete-case rows"):
        P.fit_polyline_ph(d, clock="t", event_col="e", arms=(a,), offset="off")


# ---------------------------------------------------------------------------
# Model contract
# ---------------------------------------------------------------------------
def _fit(stratum: str = "Vt_nonsour") -> U.StratumFit:
    """Hand-set parameters close to the shipped fit — exercises the contract without refitting."""
    return U.StratumFit(
        stratum=stratum, n=249, events=127, beta0=1.2511, eta0=517.8,
        theta={"qnom": np.array([0.82, 0.75, 0.92, 1.00, 1.30, 1.64, 1.64, 1.83]),
               "kpod": np.array([0.58, 0.67, 0.80, 1.00, 1.20, 1.23, 1.27]),
               "freq": np.array([0.94, 0.95, 1.02, 1.07, 1.00, 0.95, 1.00])},
        contractor={"brt": 1.0, "slb": 1.1999, "oth": 1.9098},
        loglik=-1.0, concordance=0.689,
        shared_shape=("kpod", "freq"))


def test_every_layer_is_a_free_polyline():
    """The owner's requirement — no layer may assume its shape up front."""
    assert set(U.SHAPES.values()) == {P.FREE}
    assert all(a.shape == P.FREE for a in U.arms())


def test_thetas_multiply_but_rmst_multipliers_do_not():
    """The standing trap in this workflow: composing RMST multipliers overstates by ~30 %."""
    f = _fit()
    both = f.compose(contractor="oth", qnom=1000, kpod=1.6)
    only_c = f.compose(contractor="oth")["rmst_mult"]
    only_q = f.compose(qnom=1000)["rmst_mult"]
    only_k = f.compose(kpod=1.6)["rmst_mult"]
    assert both["theta_total"] == pytest.approx(
        f.contractor["oth"] * float(f.theta_at("qnom", 1000)[0]) * float(f.theta_at("kpod", 1.6)[0]))
    assert both["rmst_mult"] < only_c * only_q * only_k
    assert both["eta_eff"] == pytest.approx(f.eta0 * both["theta_total"] ** (-1 / f.beta0))


def test_reference_point_is_neutral():
    f = _fit()
    r = f.compose()
    assert r["theta_total"] == pytest.approx(1.0)
    assert r["rmst_mult"] == pytest.approx(1.0)


def test_layers_are_clamped_outside_the_knots():
    """Flat extrapolation is the one assumption left; it must not become a trend."""
    f = _fit()
    for name, knots in U.KNOTS.items():
        assert float(f.theta_at(name, knots[0] - 1e6)[0]) == pytest.approx(f.theta[name][0])
        assert float(f.theta_at(name, knots[-1] + 1e6)[0]) == pytest.approx(f.theta[name][-1])


def test_rmst_is_the_integral_not_the_weibull_mean():
    """Standing rule: report RMST(0, 730).  With β < 1 the mean is a tail nobody observes."""
    from scipy.special import gamma

    eta, beta = 900.0, 0.8
    assert U.rmst(eta, beta) < eta * gamma(1 + 1 / beta)
    assert U.rmst(eta, beta, horizon=1e-6) == pytest.approx(0.0, abs=1e-5)


def test_sour_borrows_the_shapes_it_cannot_estimate():
    """81 events cannot carry ~20 free polyline parameters; the pooling is explicit."""
    assert _fit("Vt_sour").shared_shape == ("kpod", "freq")
    assert "qnom" not in _fit("Vt_sour").shared_shape       # the rate slope stays per-stratum


def test_ya_is_a_single_stratum():
    """All 2145 Ya runs carry one H₂S class — a Ya sour split is not available, not declined."""
    assert U.FIELD_OF["Ya"] == "Ya"
    assert [s for s in U.STRATA if U.FIELD_OF[s] == "Ya"] == ["Ya"]


# ---------------------------------------------------------------------------
# The two findings that overturned earlier conclusions
# ---------------------------------------------------------------------------
def test_kpod_is_kept_and_its_evidence_travels_with_it():
    """Kpod was dropped from Ya v3 and pinned to θ≡1 in Vt v4.  Both were wrong."""
    assert U.LAYER_CV_DELTA[("Vt", "kpod")] > 5.0
    assert U.LAYER_CV_DELTA[("Ya", "kpod")] > 0.0
    assert "kpod" in {a.name for a in U.arms()}


def test_frequency_is_carried_as_a_declared_null():
    """It is fitted and plotted, but it does not earn its place out of sample in either field.

    Pinned so that a future run cannot quietly promote it to a measured effect.
    """
    assert U.LAYER_CV_DELTA[("Vt", "freq")] < 0.0
    assert U.LAYER_CV_DELTA[("Ya", "freq")] < U.LAYER_CV_DELTA[("Ya", "kpod")]


def test_contractor_window_resolves_at_call_time():
    """A default bound at def time silently ignores sensitivity runs — that bug has happened."""
    import inspect

    sig = inspect.signature(U.fit_contractor_levels)
    assert sig.parameters["window"].default is None


def test_ql_is_not_a_layer():
    """log Ql = log Qnom + log Kpod — with both present, Ql is their product, not evidence."""
    assert "ql" not in {a.name for a in U.arms()}
    assert "ql" not in U.KNOTS


# ---------------------------------------------------------------------------
# Operator rotation of the nameplate curve
# ---------------------------------------------------------------------------
class TestQnomRotation:
    TH = np.array([0.82, 0.75, 0.92, 1.00, 1.30, 1.64, 1.64, 1.83])

    def test_fold_per_decade_reads_the_average_slope(self):
        decades = np.log10(U.QNOM_KNOTS[-1] / U.QNOM_KNOTS[0])
        assert U.qnom_fold_per_decade(self.TH) == pytest.approx(
            (self.TH[-1] / self.TH[0]) ** (1.0 / decades))
        assert U.qnom_fold_per_decade(np.ones_like(self.TH)) == pytest.approx(1.0)

    def test_rotation_hits_the_requested_slope(self):
        for target in (1.1, 1.6, 2.4):
            out = U.rotate_qnom(self.TH, target)
            assert U.qnom_fold_per_decade(out) == pytest.approx(target)

    def test_the_pivot_is_held_at_one_so_the_baseline_is_never_recalibrated(self):
        """θ(250) = 1 must survive any rotation — the reference RMST depends on it."""
        pin = list(U.QNOM_KNOTS).index(U.QNOM_REF)
        for target in (0.8, 1.0, 1.5, 3.0):
            assert U.rotate_qnom(self.TH, target)[pin] == pytest.approx(1.0)

    def test_identity_at_the_fitted_slope(self):
        assert U.rotate_qnom(self.TH, U.qnom_fold_per_decade(self.TH)) == pytest.approx(self.TH)

    def test_rotation_is_two_sided_about_the_pivot(self):
        """Steepening lifts the curve above the pivot and lowers it below — that IS a rotation."""
        out = U.rotate_qnom(self.TH, U.qnom_fold_per_decade(self.TH) * 1.5)
        k = np.asarray(U.QNOM_KNOTS, float)
        assert np.all(out[k > U.QNOM_REF] > self.TH[k > U.QNOM_REF])
        assert np.all(out[k < U.QNOM_REF] < self.TH[k < U.QNOM_REF])

    def test_local_shape_is_preserved_only_the_trend_moves(self):
        """A rotation is a constant slope in log-log, so knot-to-knot residuals are unchanged."""
        out = U.rotate_qnom(self.TH, 2.0)
        lk = np.log(np.asarray(U.QNOM_KNOTS, float))
        d0 = np.diff(np.log(self.TH)) / np.diff(lk)
        d1 = np.diff(np.log(out)) / np.diff(lk)
        assert d1 - d0 == pytest.approx(np.full_like(d0, (d1 - d0)[0]))

    def test_flattening_works_as_well_as_steepening(self):
        out = U.rotate_qnom(self.TH, 1.0)
        assert U.qnom_fold_per_decade(out) == pytest.approx(1.0)
        assert out[-1] < self.TH[-1]

    def test_does_not_mutate_its_input_and_stays_positive(self):
        before = self.TH.copy()
        assert np.all(U.rotate_qnom(self.TH, 4.0) > 0)
        assert self.TH == pytest.approx(before)
        assert U.rotate_qnom(self.TH, 0.0) == pytest.approx(self.TH)   # ignored, not NaN
