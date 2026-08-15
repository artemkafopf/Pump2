"""Tests for the free-gas-at-intake feature (:mod:`analysis.features.free_gas`).

The physics claims that the sizing analysis rests on are pinned here: β is zero above
bubble point, monotone in the things it must be monotone in, water cut is protective, and
— the property the whole "PVT-free" argument depends on — anchoring makes β's *ordering*
invariant to the assumed API / temperature / gas gravity.
"""
from __future__ import annotations

import numpy as np
import pytest

from analysis.features import free_gas as FG


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------

def test_solution_gor_monotone_in_pressure():
    p = np.array([10.0, 50.0, 100.0, 200.0, 300.0])
    rs = FG.solution_gor(p)
    assert np.all(np.diff(rs) > 0)
    assert np.all(rs > 0)


def test_solution_gor_nonpositive_pressure_is_nan():
    assert np.isnan(FG.solution_gor(0.0))
    assert np.isnan(FG.solution_gor(-5.0))


def test_gas_fvf_inverse_in_pressure():
    """Bg ∝ 1/P — gas at 100 atm occupies a tenth of its volume at 10 atm."""
    assert FG.gas_fvf(10.0) / FG.gas_fvf(100.0) == pytest.approx(10.0, rel=1e-9)


def test_oil_fvf_above_one_and_increasing_with_rs():
    bo = FG.oil_fvf(np.array([0.0, 50.0, 150.0, 250.0]))
    assert np.all(bo > 1.0)
    assert np.all(np.diff(bo) > 0)


# ---------------------------------------------------------------------------
# Anchoring — the property the PVT-free claim rests on
# ---------------------------------------------------------------------------

def test_anchored_rs_hits_gor_exactly_at_bubble_point():
    gor, pb = 222.0, 250.0
    assert FG.solution_gor_anchored(pb, pb, gor) == pytest.approx(gor, rel=1e-12)


def test_anchored_rs_is_capped_above_bubble_point():
    gor, pb = 222.0, 250.0
    assert FG.solution_gor_anchored(400.0, pb, gor) == pytest.approx(gor, rel=1e-12)


def test_anchored_rs_independent_of_pvt_at_the_anchor():
    """Whatever the PVT set, the anchored curve passes through (P_b, GOR)."""
    gor, pb = 180.0, 240.0
    vals = [float(FG.solution_gor_anchored(pb, pb, gor, p)) for p in FG.pvt_sensitivity_grid()]
    assert np.allclose(vals, gor, rtol=1e-10)


def test_anchoring_makes_beta_ranking_pvt_invariant():
    """The load-bearing claim: β's ORDER does not depend on the PVT assumptions.

    β's level is an upper bound with an unknown separation offset, so only its ranking is
    used downstream.  If this ever fails, every β-based conclusion needs re-deriving.
    """
    rng = np.random.default_rng(0)
    n = 400
    p = rng.uniform(20.0, 150.0, n)
    gor = rng.uniform(50.0, 600.0, n)
    wct = rng.uniform(0.0, 95.0, n)
    pb = rng.uniform(180.0, 300.0, n)

    ref = FG.free_gas_fraction(p, gor, wct, p_bubble_atm=pb)
    for pvt in FG.pvt_sensitivity_grid():
        b = FG.free_gas_fraction(p, gor, wct, p_bubble_atm=pb, pvt=pvt)
        ok = np.isfinite(ref) & np.isfinite(b)
        rho = np.corrcoef(np.argsort(np.argsort(ref[ok])), np.argsort(np.argsort(b[ok])))[0, 1]
        assert rho > 0.999, f"anchored beta ranking moved under {pvt}: rho={rho}"


def test_unanchored_beta_ranking_is_less_stable_than_anchored():
    """Sanity that the anchoring is doing work, not decorating a tautology."""
    rng = np.random.default_rng(1)
    n = 400
    p, gor = rng.uniform(20.0, 150.0, n), rng.uniform(50.0, 600.0, n)
    wct, pb = rng.uniform(0.0, 95.0, n), rng.uniform(180.0, 300.0, n)
    extreme = FG.PVT(api_gravity=36.0, temp_c=60.0, gas_gravity=0.9)

    a0 = FG.free_gas_fraction(p, gor, wct, p_bubble_atm=pb)
    a1 = FG.free_gas_fraction(p, gor, wct, p_bubble_atm=pb, pvt=extreme)
    u0 = FG.free_gas_fraction(p, gor, wct)
    u1 = FG.free_gas_fraction(p, gor, wct, pvt=extreme)
    # Anchored levels move less than unanchored levels under the same PVT swing.
    assert np.nanmedian(np.abs(a1 - a0)) < np.nanmedian(np.abs(u1 - u0))


# ---------------------------------------------------------------------------
# β behaviour
# ---------------------------------------------------------------------------

def test_beta_zero_at_and_above_bubble_point():
    """No gas has broken out of solution at or above P_b, so β is exactly 0."""
    assert FG.free_gas_fraction(250.0, 222.0, 30.0, p_bubble_atm=250.0) == pytest.approx(0.0)
    assert FG.free_gas_fraction(300.0, 222.0, 30.0, p_bubble_atm=250.0) == pytest.approx(0.0)


def test_beta_increases_as_intake_pressure_falls():
    p = np.array([200.0, 150.0, 100.0, 50.0, 20.0])
    b = FG.free_gas_fraction(p, 222.0, 30.0, p_bubble_atm=250.0)
    assert np.all(np.diff(b) > 0), b


def test_beta_increases_with_gor():
    b = FG.free_gas_fraction(50.0, np.array([50.0, 150.0, 300.0, 600.0]), 30.0,
                             p_bubble_atm=250.0)
    assert np.all(np.diff(b) > 0), b


def test_watercut_is_protective():
    """Gas rides the oil phase: at fixed GOR more water means less free gas.

    This is the mechanism behind the counter-intuitive positive watercut-Kpod correlation
    in the fleet, so it is pinned rather than left as a comment.
    """
    b = FG.free_gas_fraction(50.0, 222.0, np.array([0.0, 30.0, 60.0, 90.0, 99.0]),
                             p_bubble_atm=250.0)
    assert np.all(np.diff(b) < 0), b
    assert b[-1] < 0.2 * b[0]


def test_beta_bounded_in_unit_interval():
    rng = np.random.default_rng(7)
    b = FG.free_gas_fraction(rng.uniform(1, 300, 2000), rng.uniform(0, 2000, 2000),
                             rng.uniform(0, 100, 2000), p_bubble_atm=rng.uniform(50, 400, 2000))
    v = b[np.isfinite(b)]
    assert len(v) > 1000
    assert v.min() >= 0.0 and v.max() <= 1.0


def test_separation_reduces_beta_monotonically():
    kw = dict(p_bubble_atm=250.0)
    full = FG.free_gas_fraction(50.0, 222.0, 30.0, **kw)
    half = FG.free_gas_fraction(50.0, 222.0, 30.0, separation=0.5, **kw)
    none = FG.free_gas_fraction(50.0, 222.0, 30.0, separation=1.0, **kw)
    assert full > half > none
    assert none == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Guards — the warehouse really does contain these values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p,gor,wct", [
    (-2.0, 222.0, 30.0),        # rpump_intake min in the warehouse
    (8990.0, 222.0, 30.0),      # rpump_intake max
    (50.0, 4_896_117.0, 30.0),  # gas_factor max
    (50.0, 222.0, -58.7),       # watercut min
    (50.0, 222.0, 172_858.0),   # watercut max
])
def test_impossible_telemetry_becomes_nan(p, gor, wct):
    assert np.isnan(FG.free_gas_fraction(p, gor, wct, p_bubble_atm=250.0))


def test_guards_can_be_disabled_for_synthetic_input():
    v = FG.free_gas_fraction(50.0, 222.0, 30.0, p_bubble_atm=250.0, apply_guards=False)
    assert np.isfinite(v)


# ---------------------------------------------------------------------------
# Window summary + gates
# ---------------------------------------------------------------------------

def test_free_gas_window_shares_are_consistent_with_bands():
    rng = np.random.default_rng(3)
    n = 500
    out = FG.free_gas_window(rng.uniform(20, 120, n), rng.uniform(100, 400, n),
                             rng.uniform(0, 90, n), p_bubble_atm=250.0)
    assert out["beta_n_days"] == n
    assert 0.0 <= out["beta_mean"] <= 1.0
    assert out["beta_p90"] <= out["beta_max"]
    shares = [out[f"frac_beta_above_{str(b).replace('.', 'p')}"] for b in FG.BETA_BANDS]
    assert all(np.diff(shares) <= 1e-12), "exposure share must fall as the band rises"


def test_free_gas_window_all_nan_input():
    out = FG.free_gas_window(np.full(5, np.nan), np.full(5, np.nan), np.full(5, np.nan))
    assert out["beta_n_days"] == 0
    assert np.isnan(out["beta_mean"])


def test_validate_pvt_ratio_is_one_when_gor_is_generated_from_standing():
    """Self-consistency: if GOR is *defined* as Rs(P_b), the gate must return 1.0."""
    pb = np.array([200.0, 250.0, 300.0])
    gor = FG.solution_gor(pb)
    g = FG.validate_pvt_against_bubble_point(pb, gor)
    assert g["ratio_median"] == pytest.approx(1.0, rel=1e-9)
    assert g["share_within_2x"] == 1.0


def test_pvt_sensitivity_grid_shape():
    grid = FG.pvt_sensitivity_grid()
    assert len(grid) == 27
    assert len({(p.api_gravity, p.temp_c, p.gas_gravity) for p in grid}) == 27
