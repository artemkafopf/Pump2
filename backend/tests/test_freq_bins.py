"""Tests for the frequency-binning estimator layer.

Synthetic data throughout: the edge algebra and the statistics are pure functions of
``(x, y, edges)`` and testing them against the warehouse would be testing the warehouse.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.workflows.production_risk import freq_bins as FB


# ── edge algebra ─────────────────────────────────────────────────────────────

def test_sanitize_sorts_dedups_and_enforces_min_width():
    assert FB.sanitize_edges([50, 30, 40]) == [30.0, 40.0, 50.0]
    # 40.2 is inside min width of 40 → collapsed away, not raised on
    assert FB.sanitize_edges([40, 40.2, 45], min_width=0.5) == [40.0, 45.0]


def test_uniform_edges_is_a_whole_grid():
    e = FB.uniform_edges(30, 70, 2)
    assert e[0] == 30 and e[-1] == 70 and len(e) == 21
    assert np.allclose(np.diff(e), 2.0)


def test_merge_split_move_are_inverse_where_they_should_be():
    e = [30.0, 40.0, 50.0, 60.0]
    assert FB.merge_bin(e, 1) == [30.0, 40.0, 60.0]          # bins 1+2 fuse
    assert FB.split_bin(e, 0) == [30.0, 35.0, 40.0, 50.0, 60.0]
    assert FB.split_bin(FB.merge_bin(e, 1), 1, at=50.0) == e  # merge then re-cut
    assert FB.move_edge(e, 1, 45.0) == [30.0, 45.0, 50.0, 60.0]
    assert e == [30.0, 40.0, 50.0, 60.0]                     # inputs never mutated


def test_move_edge_clamps_instead_of_crossing():
    e = [30.0, 40.0, 50.0]
    # dragged past its right neighbour: parks one min-width short, keeps bin identity
    assert FB.move_edge(e, 1, 99.0, min_width=0.5) == [30.0, 49.5, 50.0]
    assert FB.move_edge(e, 1, -99.0, min_width=0.5) == [30.0, 30.5, 50.0]


def test_merge_and_split_reject_out_of_range_bins():
    e = [30.0, 40.0, 50.0]
    with pytest.raises(IndexError):
        FB.merge_bin(e, 1)          # last bin has no successor
    with pytest.raises(ValueError):
        FB.split_bin(e, 0, at=45.0)  # cut outside bin 0


def test_bin_index_closes_the_top_edge():
    e = [30.0, 40.0, 50.0]
    idx = FB.bin_index([29.9, 30.0, 39.9, 40.0, 50.0, 50.1], e)
    assert list(idx) == [-1, 0, 0, 1, 1, -1]


# ── the adaptive builder ─────────────────────────────────────────────────────

def _clustered(n_mid=400, n_tail=30, seed=0):
    """Dense 48–56 Hz, sparse tails — the shape the real panel has."""
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.uniform(48, 56, n_mid), rng.uniform(30, 48, n_tail),
                           rng.uniform(56, 70, n_tail)])


def test_adaptive_edges_lands_on_whole_hz_inside_the_bounds():
    e = FB.adaptive_edges(_clustered(), min_n=40)
    assert all(abs(v - round(v)) < 1e-9 for v in e)
    assert e[0] >= FB.FREQ_BOUNDS[0] and e[-1] <= FB.FREQ_BOUNDS[1]


def test_adaptive_edges_never_leaves_an_empty_bin_at_an_end():
    """The cap can strand an empty sliver at a bound; it is trimmed, not drawn."""
    x = _clustered()
    e = FB.adaptive_edges(x, min_n=40, max_width=5.0)
    n = FB.bin_counts(x, e)
    assert n[0] > 0 and n[-1] > 0


def test_adaptive_seed_stays_the_anchor_when_the_anchor_is_populated():
    rng = np.random.default_rng(11)
    x = np.concatenate([rng.uniform(50, 51, 60), rng.uniform(30, 70, 600)])
    e = FB.adaptive_edges(x, min_n=40)
    assert 50.0 in e and 51.0 in e          # the seed bin survives as its own bin


def test_adaptive_seed_widens_when_the_anchor_is_thin():
    """A seed short of the target grows — alternating sides, so it stays on the anchor."""
    rng = np.random.default_rng(12)
    x = np.concatenate([rng.uniform(50, 51, 5),          # a hole exactly on the anchor
                        rng.uniform(44, 50, 300), rng.uniform(51, 58, 300)])
    e = FB.adaptive_edges(x, min_n=40)
    k = int(FB.bin_index([50.5], e)[0])
    assert e[k + 1] - e[k] > 1.0
    assert e[k] <= 50.0 and e[k + 1] >= 51.0


def test_adaptive_edges_meets_the_target_wherever_it_is_free_to():
    """A bin short of ``min_n`` must have a reason: the width cap, or the end of the axis.

    Those are the only two things that stop a bin growing, so a thin bin anywhere else
    would mean the sweep gave up early.
    """
    x = _clustered()
    e = np.asarray(FB.adaptive_edges(x, min_n=40, max_width=5.0))
    n, w = FB.bin_counts(x, e), np.diff(e)
    at_bound = np.zeros(len(w), bool)
    at_bound[0] = at_bound[-1] = True           # outermost bins ran out of axis
    assert ((n >= 40) | (w >= 5.0 - 1e-9) | at_bound).all()


def test_adaptive_edges_never_exceeds_the_width_cap():
    x = _clustered()
    for cap in (2.0, 3.0, 5.0, 8.0):
        w = np.diff(FB.adaptive_edges(x, min_n=40, max_width=cap))
        assert w.max() <= cap + 1e-9


def test_the_width_cap_outranks_the_target():
    """With a tight cap the sparse tails come back thin — the documented trade."""
    x = _clustered()
    assert FB.bin_counts(x, FB.adaptive_edges(x, min_n=40, max_width=2.0)).min() < 40


def test_adaptive_edges_widens_only_where_it_must():
    x = _clustered()
    w = np.diff(FB.adaptive_edges(x, min_n=40))
    # the dense band gets the 1 Hz quantum; the sparse tails run out to the cap
    assert w.min() == 1.0
    assert w.max() > 2.0


def test_adaptive_edges_keeps_the_short_tail_when_asked():
    x = _clustered()
    kept = FB.adaptive_edges(x, min_n=40, max_width=20.0, merge_tails=False)
    merged = FB.adaptive_edges(x, min_n=40, max_width=20.0)
    assert len(kept) >= len(merged)
    assert FB.bin_counts(x, kept).min() <= FB.bin_counts(x, merged).min()


def test_adaptive_edges_survives_a_target_no_bin_can_reach():
    x = _clustered()
    e = FB.adaptive_edges(x, min_n=10_000, max_width=1000.0)
    assert len(e) == 2                          # one bin, not an infinite loop
    e5 = FB.adaptive_edges(x, min_n=10_000)     # capped: a uniform 5 Hz grid, still finite
    assert np.diff(e5).max() <= FB.MAX_BIN_WIDTH_HZ + 1e-9


def test_merge_thin_bins_reaches_the_target():
    x = _clustered()
    e = FB.merge_thin_bins(FB.uniform_edges(30, 70, 2), x, min_n=40)
    assert FB.bin_counts(x, e).min() >= 40


def test_equal_count_edges_is_more_even_than_a_fixed_grid():
    """Rounding edges to a readable 0.5 Hz costs some balance; the claim is *relative*."""
    x = _clustered(seed=3)
    cv = lambda n: float(np.std(n) / np.mean(n))          # noqa: E731
    q = FB.bin_counts(x, FB.equal_count_edges(x, 8, bounds=FB.FREQ_BOUNDS))
    grid = FB.bin_counts(x, FB.uniform_edges(30, 70, 5))  # same bin count, fixed width
    assert (q > 0).all()
    assert cv(q) < 0.5 * cv(grid)


# ── statistics ───────────────────────────────────────────────────────────────

def test_bin_stats_recovers_the_centres_it_claims():
    x = np.repeat([35.0, 45.0], 200)
    rng = np.random.default_rng(1)
    # lognormal: median == geometric mean in expectation, mean is exp(s^2/2) higher
    y = np.exp(rng.normal(np.log(np.repeat([200.0, 400.0], 200)), 0.6))
    t = FB.bin_stats(x, y, [30.0, 40.0, 50.0], n_boot=0)
    assert list(t["n"]) == [200, 200]
    assert t.loc[0, "median"] == pytest.approx(200, rel=0.15)
    assert t.loc[1, "geomean"] == pytest.approx(400, rel=0.15)
    assert (t["mean"] > t["median"]).all()
    assert t["sigma_log"].between(0.5, 0.7).all()


def test_bin_stats_flags_the_mixture_the_median_hides():
    """Half the runs die at ~30 d, half at ~600 d — same median as a tight 600 d bin."""
    rng = np.random.default_rng(2)
    early = np.exp(rng.normal(np.log(30), 0.2, 100))
    late = np.exp(rng.normal(np.log(900), 0.2, 110))
    x = np.concatenate([np.full(210, 35.0), np.full(210, 45.0)])
    y = np.concatenate([np.concatenate([early, late]),
                        np.exp(rng.normal(np.log(500), 0.2, 210))])
    t = FB.bin_stats(x, y, [30.0, 40.0, 50.0], n_boot=0)
    assert t.loc[0, "share_under_90d"] == pytest.approx(0.476, abs=0.05)
    assert t.loc[1, "share_under_90d"] == 0.0
    assert t.loc[0, "sigma_log"] > 3 * t.loc[1, "sigma_log"]


def test_bin_stats_bootstrap_brackets_the_point_estimate():
    rng = np.random.default_rng(4)
    x = np.full(300, 45.0)
    y = np.exp(rng.normal(np.log(300), 0.7, 300))
    t = FB.bin_stats(x, y, [40.0, 50.0], n_boot=200)
    for k in FB.STATS:
        assert t.loc[0, f"{k}_lo"] <= t.loc[0, k] <= t.loc[0, f"{k}_hi"]


def test_geomean_is_the_more_efficient_estimate_of_the_same_centre():
    """The efficiency argument for reporting the geometric mean, made numerically.

    On lognormal data the two estimate the same centre, but the median's sampling sd is
    ``sqrt(pi/2)`` ≈ 1.25× the geometric mean's — so it is the noisier curve to read a
    shape off.  Measured across repeated samples, not from one bootstrap: a single draw's
    bootstrap width is itself noisy enough to come out either way.
    """
    rng = np.random.default_rng(9)
    med, geo = [], []
    for _ in range(300):
        y = np.exp(rng.normal(np.log(300), 0.7, 200))
        med.append(float(np.median(y)))
        geo.append(float(np.exp(np.mean(np.log(y)))))
    assert np.mean(med) == pytest.approx(np.mean(geo), rel=0.05)
    assert np.std(med) / np.std(geo) == pytest.approx(np.sqrt(np.pi / 2), rel=0.15)


def test_trimmed_mean_ignores_the_tail_the_mean_chases():
    y = np.concatenate([np.full(98, 100.0), [10_000.0, 20_000.0]])
    x = np.full(100, 45.0)
    t = FB.bin_stats(x, y, [40.0, 50.0], trim=0.1, n_boot=0)
    assert t.loc[0, "mean"] > 300
    assert t.loc[0, "trimmed"] == pytest.approx(100.0)


def test_empty_bins_are_reported_not_dropped():
    t = FB.bin_stats([45.0] * 10, [100.0] * 10, [30.0, 40.0, 50.0], n_boot=0)
    assert len(t) == 2 and t.loc[0, "n"] == 0 and t.loc[1, "n"] == 10


def test_bin_stats_counts_what_falls_outside_the_grid():
    t = FB.bin_stats([35.0, 45.0, 95.0], [100.0] * 3, [30.0, 50.0], n_boot=0)
    assert t.attrs["n_outside"] == 1


# ── censoring-aware ──────────────────────────────────────────────────────────

def test_km_bin_stats_beats_the_naive_summary_under_censoring():
    """Random censoring: the naive median of failures is biased low, KM recovers the truth.

    This is the whole reason the censoring tab exists — the fact panel's median is the
    ``naive_median`` column, and the gap below is what conditioning on "already failed" costs.
    """
    rng = np.random.default_rng(5)
    true = rng.weibull(1.5, 4000) * 600.0
    cens = rng.uniform(0.0, 1200.0, 4000)
    t = np.minimum(true, cens)
    ev = (true <= cens).astype(float)
    out = FB.km_bin_stats(np.full(4000, 45.0), t, ev, [40.0, 50.0], n_boot=0)
    true_median = float(np.median(true))
    assert out.loc[0, "km_median"] == pytest.approx(true_median, rel=0.10)
    assert out.loc[0, "naive_median"] < 0.8 * true_median
    assert out.loc[0, "events"] + out.loc[0, "censored"] == 4000


def test_km_median_is_undefined_under_heavy_administrative_censoring():
    """Everyone censored at 300 d while the true median is ~450: KM cannot reach 0.5.

    NaN rather than a number is the point — a bin whose curve never crosses half has no
    median, and inventing one from the failures alone is exactly the naive answer.
    """
    rng = np.random.default_rng(15)
    true = rng.weibull(1.5, 2000) * 600.0
    t = np.minimum(true, 300.0)
    ev = (true <= 300.0).astype(float)
    out = FB.km_bin_stats(np.full(2000, 45.0), t, ev, [40.0, 50.0], n_boot=0)
    assert np.isnan(out.loc[0, "km_median"])
    assert out.loc[0, "rmst"] > 0                    # RMST is still defined


def test_km_median_is_nan_when_the_curve_never_falls_to_half():
    t = np.full(50, 100.0)
    ev = np.zeros(50)                      # everyone still running
    out = FB.km_bin_stats(np.full(50, 45.0), t, ev, [40.0, 50.0], n_boot=0)
    assert np.isnan(out.loc[0, "km_median"])
    assert out.loc[0, "surv_at_horizon"] == 1.0


def test_km_rmst_is_bounded_by_the_horizon():
    rng = np.random.default_rng(6)
    t = rng.weibull(1.2, 500) * 400
    ev = np.ones(500)
    out = FB.km_bin_stats(np.full(500, 45.0), t, ev, [40.0, 50.0], horizon=730.0, n_boot=0)
    assert 0 < out.loc[0, "rmst"] <= 730.0


# ── discrete (catalog) axes ──────────────────────────────────────────────────

CATALOG = [50.0, 80.0, 125.0, 160.0, 200.0, 250.0, 320.0, 400.0, 500.0]


def _catalog_x(counts=(3, 40, 25, 44, 39, 33, 38, 23, 22)):
    return np.repeat(CATALOG, counts).astype(float)


def test_catalog_edges_give_one_bin_per_size_and_lose_nobody():
    x = _catalog_x()
    e = FB.catalog_edges(x, bounds=(20.0, 2000.0), scale="log")
    assert len(e) - 1 == len(CATALOG)
    counts = FB.bin_counts(x, e)
    assert list(counts) == [3, 40, 25, 44, 39, 33, 38, 23, 22]
    assert (FB.bin_index(x, e) >= 0).all()               # nothing falls outside the grid
    # every cut sits strictly between two neighbouring sizes
    for lo, v, hi in zip(e, CATALOG, e[1:]):
        assert lo < v < hi


def test_catalog_edges_do_not_grow_a_bin_to_reach_a_target():
    """The whole point: a size installed 3 times keeps its own bin."""
    x = _catalog_x()
    cat = FB.catalog_edges(x, bounds=(20.0, 2000.0), scale="log")
    rule = FB.adaptive_edges(x, anchor=(200.0, 224.0), min_n=40, bounds=(20.0, 2000.0),
                             step=0.05, max_width=0.30, scale="log", min_width=0.0)
    assert FB.bin_counts(x, cat).min() == 3
    assert len(cat) > len(rule)                          # the rule merges sizes, this does not


def test_catalog_groups_values_the_grid_cannot_separate():
    """243/244 and 250/252 are one pump recorded twice — a 2 % grid cannot split them."""
    x = np.array([243.0] * 2 + [244.0] * 9 + [250.0] * 33 + [252.0] + [320.0] * 10)
    g = FB.catalog_values(x, scale="log", tol=1.02)
    assert [sorted(set(v)) for v in g] == [[243.0, 244.0], [250.0, 252.0], [320.0]]
    e = FB.catalog_edges(x, scale="log")
    assert list(FB.bin_counts(x, e)) == [11, 34, 10]
    # ...and a real neighbour 3 % away is NOT folded in
    assert len(FB.catalog_values([464.0, 479.0], scale="log", tol=1.02)) == 2


def test_catalog_edges_survive_a_single_value_and_a_linear_axis():
    e = FB.catalog_edges([200.0] * 5, scale="log")
    assert e[0] < 200.0 < e[-1] and len(e) == 2
    lin = FB.catalog_edges(np.repeat([50.0, 60.0, 70.0], 4), scale="linear", tol=0.5)
    assert list(FB.bin_counts(np.repeat([50.0, 60.0, 70.0], 4), lin)) == [4, 4, 4]


def test_observed_centres_put_the_marker_on_the_value():
    x = _catalog_x()
    y = np.full(x.size, 300.0)
    e = FB.catalog_edges(x, bounds=(20.0, 2000.0), scale="log")
    t = FB.bin_stats(x, y, e, n_boot=0)
    assert not np.allclose(t["f_mid"], CATALOG)          # bracket middles, not the sizes
    c = FB.observed_centres(t, x, e)
    assert list(c["f_mid"]) == pytest.approx(CATALOG)
    assert list(c["f_mid_grid"]) == pytest.approx(list(t["f_mid"]))
    assert list(c["f_lo"]) == pytest.approx(list(t["f_lo"]))   # the bracket is untouched


def test_observed_centres_keep_the_grid_midpoint_where_a_bin_is_empty():
    x = np.array([50.0] * 5 + [500.0] * 5)
    y = np.full(10, 300.0)
    e = [40.0, 60.0, 100.0, 600.0]                       # middle bin holds nothing
    t = FB.bin_stats(x, y, e, n_boot=0)
    c = FB.observed_centres(t, x, e)
    assert c.loc[0, "f_mid"] == 50.0 and c.loc[2, "f_mid"] == 500.0
    assert c.loc[1, "f_mid"] == t.loc[1, "f_mid"] == 80.0


def test_only_qnom_is_declared_discrete():
    """A flag on the wrong axis silently changes that axis's default grid."""
    assert [k for k, p in FB.PROPERTIES.items() if p.discrete] == ["qnom"]


# ── normalisation to a reference bin ─────────────────────────────────────────

def test_normalize_puts_one_at_the_reference_and_shares_elsewhere():
    x = np.repeat([35.0, 45.0, 55.0], 100)
    y = np.repeat([100.0, 200.0, 400.0], 100)
    t = FB.bin_stats(x, y, [30.0, 40.0, 50.0, 60.0], n_boot=0)
    n = FB.normalize_to_bin(t, ["mean", "median"], 45.0)          # middle bin is the ref
    assert n.loc[1, "median"] == pytest.approx(1.0)
    assert list(n["median"]) == pytest.approx([0.5, 1.0, 2.0])
    assert n.attrs["ref_bin"] == 1 and (n.attrs["ref_lo"], n.attrs["ref_hi"]) == (40.0, 50.0)
    assert n.attrs["ref_values"]["median"] == pytest.approx(200.0)
    assert list(t["median"]) == pytest.approx([100.0, 200.0, 400.0])   # input untouched


def test_normalize_divides_each_statistic_by_its_own_reference():
    """A shared divisor would fold the mean/median level gap into the shape."""
    rng = np.random.default_rng(11)
    x = np.repeat([35.0, 45.0], 400)
    y = np.exp(rng.normal(np.log(np.repeat([150.0, 300.0], 400)), 0.8))
    t = FB.bin_stats(x, y, [30.0, 40.0, 50.0], n_boot=0)
    assert t.loc[1, "mean"] > t.loc[1, "median"] * 1.1            # the level gap is real
    n = FB.normalize_to_bin(t, ["mean", "median"], 45.0)
    assert n.loc[1, "mean"] == pytest.approx(1.0)
    assert n.loc[1, "median"] == pytest.approx(1.0)
    # ...and the shapes stay the ratios they were, not the ratios plus that gap
    assert n.loc[0, "mean"] == pytest.approx(t.loc[0, "mean"] / t.loc[1, "mean"])


def test_normalize_scales_the_bootstrap_band_with_its_curve():
    rng = np.random.default_rng(12)
    x = np.repeat([35.0, 45.0], 300)
    y = np.exp(rng.normal(np.log(np.repeat([200.0, 300.0], 300)), 0.5))
    t = FB.bin_stats(x, y, [30.0, 40.0, 50.0], n_boot=100)
    n = FB.normalize_to_bin(t, ["median"], 45.0)
    assert n.loc[0, "median_lo"] == pytest.approx(t.loc[0, "median_lo"] / t.loc[1, "median"])
    assert n.loc[0, "median_lo"] < n.loc[0, "median"] < n.loc[0, "median_hi"]


def test_normalize_blanks_a_statistic_the_reference_bin_cannot_supply():
    """An undefined KM median in the reference bin blanks that curve — never rescales it."""
    rng = np.random.default_rng(13)
    t = np.concatenate([rng.weibull(1.3, 200) * 300, rng.weibull(1.3, 200) * 300])
    x = np.repeat([35.0, 45.0], 200)
    ev = np.concatenate([np.ones(200), np.zeros(200)])   # right bin: nothing ever fails
    kt = FB.km_bin_stats(x, t, ev, [30.0, 40.0, 50.0], n_boot=0)
    assert np.isnan(kt.loc[1, "km_median"])
    n = FB.normalize_to_bin(kt, ["km_median", "rmst"], 45.0)
    assert n.attrs["unnormalised"] == ["km_median"]
    assert n["km_median"].isna().all()
    assert n.loc[1, "rmst"] == pytest.approx(1.0)         # rmst is always defined, so it works


def test_normalize_refuses_a_reference_outside_the_grid():
    t = FB.bin_stats(np.full(50, 45.0), np.full(50, 300.0), [40.0, 50.0], n_boot=0)
    with pytest.raises(ValueError):
        FB.normalize_to_bin(t, ["median"], 65.0)
    with pytest.raises(ValueError):                      # inside the grid, outside the axis
        FB.normalize_to_bin(t, ["median"], 45.0, bounds=(46.0, 70.0))


# ── populations: fields, the fleet, and the Vt sour split ────────────────────

def _vt_panel():
    return pd.DataFrame({
        "field": ["Vt"] * 6 + ["Ya"] * 4,
        "h2s_class": ["sour"] * 2 + ["nonsour"] * 4 + ["nonsour"] * 4,
    })


def test_population_mask_splits_vt_and_passes_a_plain_field_through():
    p = _vt_panel()
    assert FB.resolve_population("Vt_sour") == ("Vt", "sour")
    assert FB.resolve_population("Ya") == ("Ya", None)
    assert int(FB.population_mask(p, "Vt").sum()) == 6
    assert int(FB.population_mask(p, "Vt_sour").sum()) == 2
    assert int(FB.population_mask(p, "Vt_nonsour").sum()) == 4
    assert int(FB.population_mask(p, FB.FLEET).sum()) == 10
    # the halves partition the field: no run is in both, none is in neither
    assert (FB.population_mask(p, "Vt_sour") ^ FB.population_mask(p, "Vt_nonsour")).sum() == 6


def test_selectable_populations_offers_the_split_under_its_field(monkeypatch):
    counts = pd.Series({"Ya": 696, "Vt": 279, "Az": 134,
                        "Vt_sour": 103, "Vt_nonsour": 176})
    monkeypatch.setattr(FB, "field_failure_counts",
                        lambda *a, **k: counts[["Ya", "Vt", "Az"]])
    monkeypatch.setattr(FB, "population_counts", lambda *a, **k: counts)
    assert FB.selectable_populations(min_n=100) == [
        FB.FLEET, "Ya", "Vt", "Vt_sour", "Vt_nonsour", "Az"]
    # a half too thin for its own panel drops out; the field keeps its panel
    assert FB.selectable_populations(min_n=150) == [FB.FLEET, "Ya", "Vt", "Vt_nonsour"]
    # ...and when the parent field is gone, its halves go with it
    assert FB.selectable_populations(min_n=400) == [FB.FLEET, "Ya"]


def test_the_sour_split_inherits_the_vt_layer_models():
    """The Vt fits carry the stratum inside them, so either half uses the same models."""
    assert FB.models_for_field("Vt_sour") == FB.models_for_field("Vt")
    assert FB.models_for_field("Vt_nonsour") == FB.models_for_field("Vt")
    assert FB.models_for_field("Vt")                     # and that list is not empty


# ── properties and the log axis ──────────────────────────────────────────────

def test_every_property_is_self_consistent():
    """The registry is data; a typo in it would only show up as a silent bad grid."""
    for key, p in FB.PROPERTIES.items():
        assert p.key == key
        assert p.bounds[0] < p.bounds[1]
        assert p.bounds[0] <= p.anchor[0] < p.anchor[1] <= p.bounds[1]
        assert p.scale in ("linear", "log")
        assert p.step > 0 and p.max_width >= p.step and p.min_width > 0 and p.ui_step > 0
        assert p.source == "svod" or p.source in FB.ATTACHERS
        if p.scale == "log":
            assert p.bounds[0] > 0, f"{key}: a log axis cannot reach 0"


def test_every_property_source_has_a_loader():
    """A property whose ``source`` has no attacher would fail only at click time."""
    for key, p in FB.PROPERTIES.items():
        assert p.source == "svod" or p.source in FB.ATTACHERS, key
        if p.source != "svod":
            cols, _ = FB.ATTACHERS[p.source]
            assert p.column in cols, f"{key}: {p.column} is not brought by {p.source}"


def test_attach_source_is_a_noop_when_nothing_is_needed():
    df = pd.DataFrame({"code": ["a"], "install": [pd.Timestamp("2020-01-01")]})
    assert FB.attach_source(df, "svod") is df
    # already carrying the columns → no join, no warehouse read
    have = df.assign(**{c: 1.0 for c in FB.PRESSURE_COLS})
    assert FB.attach_source(have, "pressure") is have


def test_prepare_refuses_an_axis_the_panel_was_not_loaded_for():
    """Better a named error than a silently empty panel."""
    df = pd.DataFrame({"code": ["a"], "install": [pd.Timestamp("2020-01-01")],
                       "ttf": [100.0]})
    with pytest.raises(KeyError, match="with_warehouse|pzab|source"):
        FB.prepare(df, FB.PROPERTIES["pzab"])


def test_composition_shares_sum_to_one_per_bin():
    x = np.repeat([35.0, 45.0, 55.0], 40)
    lab = np.array(["Ya"] * 60 + ["Vt"] * 30 + ["Az"] * 30)
    mix = FB.composition(x, lab, [30.0, 40.0, 50.0, 60.0])
    tot = mix.groupby("bin")["share"].sum()
    assert np.allclose(tot, 1.0)
    assert set(mix["bin"]) == {0, 1, 2}


def test_composition_folds_the_long_tail_into_other():
    x = np.full(70, 45.0)
    lab = np.array(["a"] * 30 + ["b"] * 20 + ["c"] * 10 + ["d"] * 5 + ["e"] * 3 + ["f"] * 2)
    mix = FB.composition(x, lab, [40.0, 50.0], top=3)
    assert set(mix["label"]) == {"a", "b", "c", "прочие"}
    assert mix.loc[mix["label"] == "прочие", "n"].iloc[0] == 10       # d + e + f


def test_composition_skips_empty_bins():
    mix = FB.composition(np.full(10, 45.0), ["Ya"] * 10, [30.0, 40.0, 50.0])
    assert set(mix["bin"]) == {1}


def test_selectable_fields_gates_panels_not_fleet_membership(monkeypatch):
    """A field too thin for its own panel still belongs to the fleet it is part of."""
    counts = pd.Series({"Ya": 696, "Vt": 279, "Az": 134, "Mc": 31, "Da": 28})
    monkeypatch.setattr(FB, "field_failure_counts", lambda *a, **k: counts)
    assert FB.selectable_fields(min_n=100) == [FB.FLEET, "Ya", "Vt", "Az"]
    assert FB.selectable_fields(min_n=30) == [FB.FLEET, "Ya", "Vt", "Az", "Mc"]
    assert FB.selectable_fields(min_n=10_000) == [FB.FLEET]      # fleet is always offered


def test_only_a_shared_pin_model_is_offered_on_the_fleet():
    """A per-field model is pinned to its own η₀; pooling those on one panel mixes zeros.

    ``field_v5`` is the exception and is offered: it carries a θ_Qnom per field but they are
    all pinned at the same Qnom 250, so a mixed panel is rescaled toward one reference pump.
    """
    offered = FB.models_for_field(FB.FLEET)
    assert offered == ["field_v5"]
    assert all(FB.LAYER_MODEL_INFO[k]["field"] == "*" for k in offered)


def _lognormal_x(seed=30, n=3000, lo=5.0, hi=2000.0):
    rng = np.random.default_rng(seed)
    v = np.exp(rng.normal(np.log(150.0), 1.1, n))
    return v[(v >= lo) & (v <= hi)]


def test_log_axis_bins_are_equal_ratio_not_equal_width():
    x = _lognormal_x()
    p = FB.PROPERTIES["ql"]
    e = np.asarray(FB.adaptive_edges(x, anchor=p.anchor, bounds=p.bounds, step=p.step,
                                     max_width=p.max_width, scale="log",
                                     min_width=p.min_width, sig=p.sig))
    ratio, width = e[1:] / e[:-1], np.diff(e)
    assert ratio.max() <= 10 ** p.max_width * 1.02          # the cap, in ratio terms
    assert ratio.min() > 1.0
    # widths span orders of magnitude while ratios stay in a narrow band — the whole point
    assert width.max() / width.min() > 20
    assert ratio.max() / ratio.min() < 3


def test_log_axis_meets_the_target_where_the_ratio_cap_allows():
    """Same contract as the linear axis: short only when capped or out of axis."""
    x = _lognormal_x()
    p = FB.PROPERTIES["ql"]
    e = np.asarray(FB.adaptive_edges(x, anchor=p.anchor, bounds=p.bounds, step=p.step,
                                     max_width=p.max_width, scale="log",
                                     min_width=p.min_width, sig=p.sig, min_n=40))
    n = FB.bin_counts(x, e)
    at_cap = (e[1:] / e[:-1]) >= 10 ** p.max_width * 0.98
    at_bound = np.zeros(len(n), bool)
    at_bound[0] = at_bound[-1] = True
    assert ((n >= 40) | at_cap | at_bound).all()
    assert (n[~at_cap & ~at_bound] >= 40).all()


def test_log_split_halves_the_ratio_not_the_width():
    e = FB.split_bin([100.0, 400.0], 0, scale="log", min_width=1.0)
    assert e[1] == pytest.approx(200.0)
    assert FB.split_bin([100.0, 400.0], 0, scale="linear", min_width=1.0)[1] == 250.0


def test_uniform_edges_on_a_log_axis_is_a_ratio_ladder():
    e = np.asarray(FB.uniform_edges(10.0, 1000.0, 2.0, scale="log", min_width=0.5))
    assert np.allclose(e[1:] / e[:-1], 2.0, rtol=0.02)


def test_sanitize_does_not_collapse_a_fine_grid():
    """min_width is in the caller's units — a Hz-sized default would eat a Kpod grid."""
    kpod = [0.05, 0.10, 0.15, 0.20]
    assert FB.sanitize_edges(kpod, min_width=0.0) == kpod
    assert FB.sanitize_edges(kpod, min_width=0.02) == kpod
    assert len(FB.sanitize_edges(kpod, min_width=0.5)) == 2       # would have, at 0.5


def test_bin_stats_never_silently_merges_a_fine_grid():
    x = np.repeat([0.07, 0.12, 0.17], 50)
    t = FB.bin_stats(x, np.full(150, 300.0), [0.05, 0.10, 0.15, 0.20], n_boot=0)
    assert list(t["n"]) == [50, 50, 50]


# ── the trend ────────────────────────────────────────────────────────────────

def _parabolic_panel(seed=20, n_per=80, noise=0.05):
    """Bins on a known log-parabola peaking at 50 Hz, all of them well populated."""
    rng = np.random.default_rng(seed)
    f = np.repeat(np.arange(42.0, 59.0), n_per) + rng.uniform(0, 1, 17 * n_per)
    peak, curve = np.log(500.0), -0.004
    y = np.exp(peak + curve * (f - 50.0) ** 2 + rng.normal(0, noise, f.size))
    return f, y, FB.uniform_edges(42, 59, 1)


def test_fit_trend_recovers_a_known_optimum():
    f, y, e = _parabolic_panel()
    tab = FB.bin_stats(f, y, e, n_boot=0)
    fit = FB.fit_trend(tab, stat="median", form="poly2", min_n=40)
    assert fit.opt_hz == pytest.approx(50.0, abs=0.6)
    assert fit.opt_value == pytest.approx(500.0, rel=0.06)
    assert fit.weighted_r2 > 0.9


def test_fit_trend_uses_only_the_reliable_bins():
    """A wild 3-run bin at 60 Hz must not move the fit at all."""
    f, y, e = _parabolic_panel()
    f2 = np.concatenate([f, [60.2, 60.4, 60.6]])
    y2 = np.concatenate([y, [5.0, 4000.0, 3.0]])
    e2 = e + [61.0]
    tab = FB.bin_stats(f2, y2, e2, n_boot=0)
    clean = FB.fit_trend(FB.bin_stats(f, y, e, n_boot=0), stat="median", min_n=40)
    dirty = FB.fit_trend(tab, stat="median", min_n=40)
    assert dirty.n_bins == clean.n_bins
    assert dirty.opt_hz == pytest.approx(clean.opt_hz, abs=1e-6)
    assert dirty.support[1] <= 59.0            # the thin bin is outside the support


def test_fit_trend_is_drawn_only_where_it_was_fitted():
    f, y, e = _parabolic_panel()
    fit = FB.fit_trend(FB.bin_stats(f, y, e, n_boot=0), stat="mean", min_n=40)
    assert fit.support == (42.0, 59.0)


def test_fit_trend_returns_none_rather_than_an_exact_fit():
    """Three reliable bins cannot support a 3-parameter parabola honestly."""
    rng = np.random.default_rng(21)
    f = np.repeat([45.0, 50.0, 55.0], 60) + rng.uniform(0, 1, 180)
    y = np.exp(rng.normal(np.log(400), 0.3, 180))
    tab = FB.bin_stats(f, y, [44.0, 49.0, 54.0, 59.0], n_boot=0)
    assert FB.fit_trend(tab, stat="median", form="poly2", min_n=40) is None
    assert FB.fit_trend(tab, stat="median", form="poly1", min_n=40) is not None


def test_every_trend_form_predicts_finite_positive_life():
    f, y, e = _parabolic_panel()
    tab = FB.bin_stats(f, y, e, n_boot=0)
    grid = np.linspace(42, 59, 50)
    for form in FB.TREND_FORMS:
        fit = FB.fit_trend(tab, stat="median", form=form, min_n=40)
        v = fit.predict(grid)
        assert np.all(np.isfinite(v)) and np.all(v > 0), form


def test_trend_weights_discount_a_dispersed_bin():
    """Two bins of equal n: the one with triple the log-spread gets ~1/9 the weight."""
    d = pd.DataFrame({"n": [100, 100], "sigma_log": [0.5, 1.5]})
    w = FB._trend_weights(d)
    assert w[0] / w[1] == pytest.approx(9.0, rel=1e-6)


def test_fit_trend_on_a_log_axis_uses_the_log_coordinate():
    """A power law in x is a straight line in log-log; the raw-unit fit cannot see it.

    This is the failure the ``scale`` argument exists to prevent: a parabola in raw m³/d over
    a 5–2000 axis is a straight line with a rounding error, and it parks its vertex wherever
    the last decade happens to fall.
    """
    rng = np.random.default_rng(31)
    edges = list(np.geomspace(10, 1000, 12))
    centres = np.sqrt(np.asarray(edges[:-1]) * np.asarray(edges[1:]))
    x = np.repeat(centres, 60) * rng.uniform(0.98, 1.02, 11 * 60)
    y = 5000.0 * x ** -0.5 * np.exp(rng.normal(0, 0.05, x.size))   # exact power law
    tab = FB.bin_stats(x, y, edges, n_boot=0)
    log_fit = FB.fit_trend(tab, stat="median", form="poly1", min_n=40,
                           scale="log", ref=100.0)
    raw_fit = FB.fit_trend(tab, stat="median", form="poly1", min_n=40,
                           scale="linear", ref=100.0)
    assert log_fit.weighted_r2 > 0.99                 # log-log line: essentially exact
    assert raw_fit.weighted_r2 < 0.85                 # raw units: cannot represent it
    assert log_fit.params[1] == pytest.approx(-0.5 * np.log(10), rel=0.05)  # the exponent


def test_fit_trend_flags_an_edge_maximum():
    """On a monotone curve there is no interior optimum and the app must not claim one."""
    rng = np.random.default_rng(32)
    edges = list(np.geomspace(10, 1000, 12))
    centres = np.sqrt(np.asarray(edges[:-1]) * np.asarray(edges[1:]))
    x = np.repeat(centres, 60) * rng.uniform(0.98, 1.02, 11 * 60)
    y = 5000.0 * x ** -0.5 * np.exp(rng.normal(0, 0.05, x.size))
    fit = FB.fit_trend(FB.bin_stats(x, y, edges, n_boot=0), stat="median", form="poly1",
                       min_n=40, scale="log", ref=100.0)
    assert fit.opt_at_edge
    assert fit.opt_hz == pytest.approx(fit.support[0], rel=0.05)

    f, yv, e2 = _parabolic_panel()
    interior = FB.fit_trend(FB.bin_stats(f, yv, e2, n_boot=0), stat="median", min_n=40)
    assert not interior.opt_at_edge


def test_fit_trend_can_key_reliability_on_another_column():
    """The KM table's reliability is its event count, not its run count."""
    f, y, e = _parabolic_panel()
    tab = FB.bin_stats(f, y, e, n_boot=0)
    tab["events"] = (tab["n"] // 4).astype(int)          # every bin now "thin" on events
    assert FB.fit_trend(tab, stat="median", min_n=40, count_col="events") is None
    assert FB.fit_trend(tab, stat="median", min_n=15, count_col="events") is not None


# ── removing fitted layers ───────────────────────────────────────────────────

def _toy_model(theta_map, beta=0.8):
    """A LayerModel whose θ is a lookup, so the arithmetic can be checked exactly."""
    def theta_of(panel, layer):
        x = pd.to_numeric(panel[FB.LAYER_COLUMN[layer]], errors="coerce").to_numpy(float)
        return np.where(np.isfinite(x), theta_map[layer](x), 1.0)

    return FB.LayerModel(key="toy", label="toy", field="Ya",
                         layers=tuple(theta_map), beta_of=lambda p: np.full(len(p), beta),
                         theta_of=theta_of)


def _toy_panel(n=6):
    return pd.DataFrame({"ql": np.linspace(50, 500, n), "kpod": np.linspace(0.3, 1.2, n),
                         "ttf": np.linspace(100, 600, n)})


def test_remove_layers_is_the_aft_offset_it_claims():
    p = _toy_panel()
    m = _toy_model({"ql": lambda x: x / 250.0})
    adj, meta = FB.remove_layers(p, "ttf", m, ["ql"])
    expect = p["ttf"] * (p["ql"] / 250.0) ** (1 / 0.8)
    assert np.allclose(adj, expect)
    assert meta["layers"] == ("ql",) and meta["n_unadjusted"] == 0


def test_removing_layers_multiplies_their_thetas():
    p = _toy_panel()
    m = _toy_model({"ql": lambda x: x / 250.0, "kpod": lambda k: 1.0 + k})
    one = FB.remove_layers(p, "ttf", m, ["ql"])[0]
    both = FB.remove_layers(p, "ttf", m, ["ql", "kpod"])[0]
    assert np.allclose(both, p["ttf"] * ((p["ql"] / 250.0) * (1 + p["kpod"])) ** (1 / 0.8))
    assert not np.allclose(one, both)


def test_a_flat_layer_is_a_no_op():
    """Ya v2.1's Kpod arm is a measured null; removing it must change nothing."""
    p = _toy_panel()
    m = _toy_model({"kpod": lambda k: np.ones_like(k)})
    adj, _ = FB.remove_layers(p, "ttf", m, ["kpod"])
    assert np.allclose(adj, p["ttf"])


def test_the_offset_is_monotone_so_order_statistics_survive():
    """Strictly increasing in t ⇒ the median of the adjusted panel is a real median.

    This is why the whole binned panel can be adjusted and still read the same way; a
    subtracted residual would not have the property.
    """
    p = _toy_panel(50)
    p["ttf"] = np.sort(np.random.default_rng(7).uniform(50, 900, 50))
    m = _toy_model({"ql": lambda x: np.full_like(x, 1.4)})
    adj, _ = FB.remove_layers(p, "ttf", m, ["ql"])
    assert np.all(np.diff(adj.to_numpy()) > 0)
    assert adj.iloc[np.argsort(p["ttf"].to_numpy())[len(p) // 2]] == adj.sort_values().iloc[len(p) // 2]


def test_runs_without_the_covariate_pass_through_and_are_counted():
    p = _toy_panel()
    p.loc[[1, 4], "ql"] = np.nan
    m = _toy_model({"ql": lambda x: x / 250.0})
    adj, meta = FB.remove_layers(p, "ttf", m, ["ql"])
    assert meta["n_unadjusted"] == 2
    assert np.allclose(adj.iloc[[1, 4]], p["ttf"].iloc[[1, 4]])      # untouched
    assert not np.allclose(adj.iloc[[0, 2]], p["ttf"].iloc[[0, 2]])  # the rest adjusted


def test_removing_nothing_returns_the_raw_times():
    p = _toy_panel()
    adj, meta = FB.remove_layers(p, "ttf", _toy_model({"ql": lambda x: x}), [])
    assert np.allclose(adj, p["ttf"]) and meta["layers"] == ()


def test_remove_layers_rejects_a_layer_the_model_does_not_have():
    with pytest.raises(KeyError, match="freq"):
        FB.remove_layers(_toy_panel(), "ttf", _toy_model({"ql": lambda x: x}), ["freq"])


def _aft_sample(n=4000, c=-0.5, shape=1.3, seed=13, group_shift=None, ref=250.0):
    rng = np.random.default_rng(seed)
    g = None if group_shift is None else rng.integers(0, len(group_shift), n)
    x = np.exp(rng.normal(np.log(ref), 1.0, n))
    if g is not None:
        # each group sits at its own life level AND its own rate level — the configuration
        # that makes a pooled slope wrong
        x = x * np.exp(np.asarray([0.0, 1.2])[g] if len(group_shift) == 2 else 0.0)
        eta = np.exp(np.log(400) + np.asarray(group_shift)[g] + c * np.log(x / ref))
    else:
        eta = np.exp(np.log(400) + c * np.log(x / ref))
    T = eta * rng.weibull(shape, n)
    C = rng.uniform(0, 2500, n)
    return x, np.minimum(T, C), (T <= C).astype(float), g, T


def test_empirical_layer_recovers_a_known_slope_under_censoring():
    x, t, ev, _, _ = _aft_sample(c=-0.5, shape=1.3)
    f = FB.fit_empirical_layer(x, t, ev, ref=250.0)
    assert f.c == pytest.approx(-0.5, abs=0.06)
    assert f.shape == pytest.approx(1.3, rel=0.1)
    assert f.events < f.n                       # censoring really was present and used
    assert 0 < f.se < 0.1 and abs(f.z) > 5


def test_group_intercepts_rescue_a_slope_a_pooled_fit_gets_wrong():
    """The fleet case: fields differ in BOTH life level and rate, so pooling confounds."""
    x, t, ev, g, _ = _aft_sample(c=-0.5, group_shift=[0.0, 1.1], seed=17)
    grouped = FB.fit_empirical_layer(x, t, ev, ref=250.0, groups=g)
    pooled = FB.fit_empirical_layer(x, t, ev, ref=250.0)
    assert grouped.c == pytest.approx(-0.5, abs=0.08)
    assert abs(pooled.c + 0.5) > abs(grouped.c + 0.5) + 0.1     # pooling is worse
    assert grouped.n_groups == 2 and pooled.n_groups == 1


def test_empirical_layer_refuses_to_fit_too_few_events():
    x, t, ev, _, _ = _aft_sample(n=200)
    with pytest.raises(ValueError, match="событий"):
        FB.fit_empirical_layer(x, t, np.zeros_like(ev), min_events=30)


def test_time_multiplier_inverts_the_fitted_slope_and_clamps():
    f = FB.EmpiricalFit(column="ql", ref=250.0, c=-0.5, se=0.01, shape=1.2, n=100,
                        events=100, support=(50.0, 500.0))
    assert f.time_multiplier(1000.0)[0] == pytest.approx((500 / 250) ** 0.5)  # clamped
    assert f.time_multiplier(250.0)[0] == pytest.approx(1.0)
    assert f.per_doubling == pytest.approx(1 - 2 ** -0.5)


def test_removing_the_empirical_layer_flattens_what_it_fitted():
    """The end-to-end contract: fit on these runs, take it off, the dependence is gone.

    The slope is fitted on the CENSORED observations and then applied to the true lifetimes,
    so what is checked is whether the offset removes the generating dependence — not whether
    censoring happens to leave the observed correlation at zero, which it does not.
    """
    x, t, ev, _, true_t = _aft_sample(c=-0.5, seed=19)
    f = FB.fit_empirical_layer(x, t, ev, ref=250.0)
    panel = pd.DataFrame({"ql": x, "ttf": true_t})
    adj, meta = FB.remove_layers(panel, "ttf", FB.empirical_layer_model(f), ["ql"])
    before = np.corrcoef(np.log(x), np.log(true_t))[0, 1]
    after = np.corrcoef(np.log(x), np.log(adj))[0, 1]
    assert before < -0.25
    assert abs(after) < 0.03
    assert meta["beta_median"] == 1.0        # beta == 1: theta IS the time-scale multiplier


def test_layer_registry_is_consistent():
    assert set(FB.LAYER_MODEL_INFO) == set(FB.LAYER_MODELS)
    for key, info in FB.LAYER_MODEL_INFO.items():
        assert info["field"] in ("Ya", "Vt", "Mc", "*")
        assert key in FB.models_for_field("Ya" if info["field"] == "*" else info["field"])
        for lay in info["layers"]:
            assert lay in FB.LAYER_AXIS and lay in FB.LAYER_COLUMN
            # every layer names a real axis, so the app can spot a self-cancelling choice
            assert FB.LAYER_AXIS[lay][1] in FB.PROPERTIES


def test_only_the_pooled_model_is_wired_where_a_field_has_no_fit_of_its_own():
    """Mc has no single-field layer model — it gets the per-field v5 one and nothing else."""
    assert FB.models_for_field("Mc") == ["field_v5"]
    assert "ya_v21" in FB.models_for_field("Ya")
    assert "ya_v21" not in FB.models_for_field("Vt")


# ── the flatness test ────────────────────────────────────────────────────────

def test_permutation_flatness_finds_a_real_step():
    rng = np.random.default_rng(7)
    x = np.repeat([35.0, 45.0], 300)
    y = np.exp(rng.normal(np.log(np.repeat([150.0, 600.0], 300)), 0.5))
    r = FB.permutation_flatness(x, y, [30.0, 40.0, 50.0], stat="median", n_perm=200)
    assert r.p_value < 0.02 and r.n_bins == 2


def test_permutation_flatness_does_not_find_a_shape_that_is_not_there():
    rng = np.random.default_rng(8)
    x = rng.uniform(40, 60, 900)
    y = np.exp(rng.normal(np.log(400), 0.7, 900))
    r = FB.permutation_flatness(x, y, FB.uniform_edges(40, 60, 5), stat="median",
                                n_perm=300, min_n=40)
    assert r.p_value > 0.05


# ── round trip ───────────────────────────────────────────────────────────────

def test_parse_edges_round_trips_both_separators():
    e = [30.0, 45.5, 70.0]
    assert FB.parse_edges("|".join(f"{v:g}" for v in e)) == e
    assert FB.parse_edges("30, 45.5, 70") == e


# --- the per-field layer model (v5) -----------------------------------------
class TestFieldV5LayerModel:
    """One θ_Qnom per field, read from the blocks the calculators are wired from."""

    def _panel(self):
        return pd.DataFrame({
            "field": ["Ya", "Vt", "Vt", "Az", "Ki", "Mc"],
            "h2s_class": ["nonsour", "sour", "nonsour", "nonsour", "nonsour", "nonsour"],
            "qnom": [250.0, 250.0, 800.0, 800.0, 800.0, 800.0],
            "kpod": [0.8] * 6, "freq": [50.0] * 6, "ttf": [400.0] * 6,
        })

    def test_offered_on_every_population_including_the_fleet(self):
        for pop in ("Ya", "Vt", "Vt_sour", "Az", "Ic", FB.FLEET):
            assert "field_v5" in FB.models_for_field(pop), pop

    def test_vt_splits_on_h2s_and_an_unmodelled_field_falls_back(self):
        p = self._panel()
        keys = list(FB.field_v5_key(p["field"], p["h2s_class"]))
        assert keys[:4] == ["Ya", "Vt_sour", "Vt_nonsour", "Az"]
        # Ki has no fit of its own; whatever key it lands on must carry the pooled parameters
        betas, blocks = FB._field_v5_blocks()
        assert betas[keys[4]] == pytest.approx(betas["Fleet"])
        assert blocks[keys[4]][1].tolist() == blocks["Fleet"][1].tolist()

    def test_theta_is_one_at_the_shared_reference_pump(self):
        """Every field pins θ_Qnom at Qnom 250 — that is what makes a mixed panel legitimate."""
        m = FB.layer_model("field_v5")
        p = self._panel().assign(qnom=250.0)
        assert np.allclose(m.theta(p, "qnom"), 1.0)

    def test_beta_is_per_field(self):
        m = FB.layer_model("field_v5")
        b = m.beta(self._panel())
        assert len(set(np.round(b, 4))) > 1
        assert (b > 0.5).all() and (b < 3.0).all()

    def test_removal_lengthens_a_run_on_a_heavy_pump(self):
        m = FB.layer_model("field_v5")
        p = self._panel()
        adj, _ = FB.remove_layers(p, "ttf", m, ["qnom"])
        adj = adj.to_numpy(float)
        assert adj[0] == pytest.approx(400.0)          # at the reference pump: untouched
        assert adj[2] > 400.0                          # Qnom 800 carries θ > 1 -> scaled up

    def test_operator_layers_come_from_the_base_scenario(self):
        from analysis.workflows.production_risk import pikpolka_sim as PS
        m = FB.layer_model("field_v5")
        p = self._panel()
        assert float(m.theta(p.assign(freq=60.0), "freq")[0]) == pytest.approx(
            PS.theta_freq(60.0, PS.CTRL_BASE))
        assert float(m.theta(p.assign(kpod=0.3), "kpod")[0]) == pytest.approx(
            PS.theta_kpod(0.3, PS.CTRL_BASE))


# ── selection funnel: what the panel count is a count OF ─────────────────────

class TestPopulationFunnel:
    """The app shows this table under «Отказов в панели», so it has one job: reconcile.

    Its top row must be the sheet the operator remembers and its bottom row the number
    on the metric, with every filter in between named and counted.
    """

    def test_fleet_and_field_slices_reconcile_with_the_panel(self, svod_workbook):
        wb = svod_workbook
        fleet = FB.population_funnel(FB.FLEET, workbook=wb.path)
        assert int(fleet["kept"].iloc[0]) == wb.n_rows
        assert int(fleet["kept"].iloc[-1]) == len(wb.survivors)
        ya = FB.population_funnel("Ya", workbook=wb.path)
        assert int(ya["kept"].iloc[-1]) == 2
        # a field never sees the unmapped rows, so its funnel loses nothing to that stage
        assert int(ya.loc[ya["stage"].str.contains("месторождение"), "dropped"].iloc[0]) == 0

    def test_dropped_is_the_step_to_step_difference(self, svod_workbook):
        f = FB.population_funnel(FB.FLEET, workbook=svod_workbook.path)
        assert list(f["dropped"]) == [0] + [1] * (len(f) - 1)

    def test_sour_half_is_sliced_not_pooled(self, svod_workbook):
        wb = svod_workbook
        sour = FB.population_funnel("Vt_sour", workbook=wb.path)
        nonsour = FB.population_funnel("Vt_nonsour", workbook=wb.path)
        assert int(sour["kept"].iloc[-1]) == 1
        assert int(nonsour["kept"].iloc[-1]) == 0
        # the halves partition Vt at every stage, which is what makes them comparable
        vt = FB.population_funnel("Vt", workbook=wb.path)
        assert list(sour["kept"] + nonsour["kept"]) == list(vt["kept"])

    def test_axis_losses_are_appended_to_the_sheet_stages(self, svod_workbook):
        wb = svod_workbook
        prop = FB.PROPERTIES["freq"]
        panel = FB.load_fact("Ya", workbook=wb.path)
        prepared = FB.prepare(panel, prop)
        full = FB.funnel_table("Ya", prop, prepared, workbook=wb.path)
        sheet = FB.population_funnel("Ya", workbook=wb.path)
        assert len(full) == len(sheet) + 2
        assert int(full["kept"].iloc[-1]) == len(prepared)
        assert prop.label in full["stage"].iloc[-2]
        assert list(full["step"]) == list(range(len(full)))
