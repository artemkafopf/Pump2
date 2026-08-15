"""Contract tests for the Ya frequency layer forms.

Covers the deployable three-anchor polynomial: what each parameter is allowed to move, and
why the polynomial is taken in log θ rather than in θ.
"""
import numpy as np
import pytest

from analysis.workflows.production_risk import ya_freq_empirical as E


# ---------------------------------------------------------------------------
# Three-anchor polynomial frequency form (n = 3 / n = 4)
# ---------------------------------------------------------------------------
class TestFreqPoly:
    """The deployable frequency form: θ(40), θ(50), θ(60) [+ one optional cubic]."""

    ANCH = (1.45, 1.00, 1.20)

    @pytest.mark.parametrize("cubic", [0.0, -0.0008, 0.0015])
    def test_anchors_are_hit_exactly_for_any_cubic(self, cubic):
        """The three sliders mean exactly what they say — that is the point of the form."""
        got = E.theta_freq_poly(np.array([40.0, 50.0, 60.0]), *self.ANCH, cubic=cubic)
        assert got == pytest.approx(np.array(self.ANCH), rel=1e-12)

    def test_n3_is_nested_in_n4_at_cubic_zero(self):
        g = np.linspace(35, 70, 200)
        assert E.theta_freq_poly(g, *self.ANCH, cubic=0.0) == pytest.approx(
            E.theta_freq_quadratic(g, *self.ANCH))

    def test_only_the_linear_coefficient_pays_for_the_cubic(self):
        """Algebraic claim in the docstring: a and c are independent of d; b absorbs it."""
        a = E.freq_poly_coeffs(*self.ANCH, cubic=0.0)
        b = E.freq_poly_coeffs(*self.ANCH, cubic=0.0015)
        assert a["c"] == pytest.approx(b["c"])
        assert a["a"] == pytest.approx(b["a"])
        assert a["b"] != pytest.approx(b["b"])          # b is what pays for d

    def test_cubic_moves_the_curve_BETWEEN_the_anchors_too(self):
        """Guards a retracted claim: pinned coefficients ≠ a pinned curve.

        ``a`` and ``c`` being independent of ``d`` does NOT confine the cubic to the tails —
        ``u³`` is non-zero at u = ±5 as well.  Anything that documents this form as
        "tails only" is wrong.
        """
        mid = np.array([45.0, 55.0])
        d0 = E.theta_freq_poly(mid, *self.ANCH, cubic=0.0)
        d1 = E.theta_freq_poly(mid, *self.ANCH, cubic=0.0015)
        assert np.any(np.abs(d1 / d0 - 1.0) > 0.4)

    def test_raw_cubic_is_violently_nonlinear_outside_the_anchors(self):
        """Why the app exposes a fourth ANCHOR and never the coefficient itself."""
        assert float(E.theta_freq_poly(70.0, *self.ANCH, cubic=0.0015)[0]) > 1e3

    def test_fourth_anchor_is_hit_while_the_first_three_hold(self):
        for t70 in (1.4, 3.5, 5.0):
            d = E.cubic_from_fourth_anchor(*self.ANCH, t70)
            got = E.theta_freq_poly(np.array([40.0, 50.0, 60.0, 70.0]), *self.ANCH, cubic=d)
            assert got == pytest.approx(np.array([*self.ANCH, t70]), rel=1e-9)

    def test_fourth_anchor_at_the_parabolas_own_value_is_exactly_n3(self):
        t70 = float(E.theta_freq_poly(70.0, *self.ANCH)[0])
        assert E.cubic_from_fourth_anchor(*self.ANCH, t70) == pytest.approx(0.0, abs=1e-15)

    @pytest.mark.parametrize("f4", [40.0, 50.0, 60.0])
    def test_fourth_anchor_refuses_to_sit_on_an_existing_one(self, f4):
        with pytest.raises(ValueError, match="not identified"):
            E.cubic_from_fourth_anchor(*self.ANCH, 1.5, f4=f4)

    def test_theta_is_positive_everywhere_even_for_a_deep_valley(self):
        """Why the polynomial is in log θ: a plain polynomial in θ can cross zero here."""
        g = np.linspace(20, 90, 500)
        assert np.all(E.theta_freq_poly(g, 2.8, 0.55, 2.8) > 0.0)

    def test_clamped_outside_the_band(self):
        lo, hi = E.FREQ_CLAMP
        for x, edge in ((lo - 40.0, lo), (hi + 40.0, hi)):
            assert float(E.theta_freq_poly(x, *self.ANCH)[0]) == pytest.approx(
                float(E.theta_freq_poly(edge, *self.ANCH)[0]))

    def test_flat_anchors_give_a_flat_layer(self):
        g = np.linspace(35, 70, 100)
        assert E.theta_freq_poly(g, 1.0, 1.0, 1.0) == pytest.approx(np.ones_like(g))

    def test_valley_and_ridge_are_distinguished(self):
        assert E.freq_poly_edges(1.45, 1.00, 1.20)["kind"] == "valley"
        assert E.freq_poly_edges(0.80, 1.00, 0.90)["kind"] == "ridge"

    def test_optimum_is_not_assumed_to_sit_at_50(self):
        """θ(50) = 1 does not make 50 Hz the optimum — asymmetric anchors move the vertex."""
        e = E.freq_poly_edges(1.05, 1.00, 1.60)
        assert e["opt_hz"] < 50.0

    def test_round_trip_through_the_least_squares_fit(self):
        g = np.linspace(35, 70, 120)
        th = E.theta_freq_poly(g, *self.ANCH, cubic=0.0009)
        r = E.fit_freq_poly(g, th, degree=3)
        assert (r["t40"], r["t50"], r["t60"]) == pytest.approx(self.ANCH, rel=1e-6)
        assert r["cubic"] == pytest.approx(0.0009, rel=1e-6)
        assert r["rmse_log"] == pytest.approx(0.0, abs=1e-9)

    def test_fit_degree_two_forces_the_cubic_to_zero(self):
        g = np.linspace(35, 70, 120)
        th = E.theta_freq_poly(g, *self.ANCH, cubic=0.0009)
        assert E.fit_freq_poly(g, th, degree=2)["cubic"] == 0.0
        assert E.fit_freq_quadratic(g, th)["cubic"] == 0.0

    def test_extra_parameter_never_fits_worse(self):
        g = np.linspace(35, 70, 60)
        th = E.theta_freq_poly(g, 1.3, 1.02, 1.4) * (1 + 0.02 * np.sin(g))
        assert (E.fit_freq_poly(g, th, degree=3)["rmse_log"]
                <= E.fit_freq_poly(g, th, degree=2)["rmse_log"] + 1e-12)
