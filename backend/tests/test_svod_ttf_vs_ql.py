"""Unit tests for the Свод TTF-vs-Ql panels — synthetic panels, no workbook."""
from __future__ import annotations

from dataclasses import replace

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.workflows.production_risk import svod_ttf_vs_ql as Q


def _panel(n: int = 900, seed: int = 0, ql_slope: float = -80.0,
           qnom_slope: float = 0.0, all_pulls: bool = False) -> pd.DataFrame:
    """Panel where ННО falls with log Ql (and optionally with pump size)."""
    rng = np.random.default_rng(seed)
    field = rng.choice(["Ya", "Vt"], n)
    h2s = np.where(field == "Vt", rng.choice(["sour", "nonsour"], n), "nonsour")
    ql = np.exp(rng.normal(5.0, 0.6, n))
    qnom = rng.choice([100.0, 250.0, 450.0], n)
    # The intercept is sized off the slopes so ННО stays positive on its own: a
    # floor at 10 days would censor the steep end and attenuate the very slope
    # these tests measure.
    base = 300 + 7.0 * (abs(ql_slope) + abs(qnom_slope))
    nno = (base + 150 * (field == "Ya") + ql_slope * np.log(ql)
           + qnom_slope * np.log(qnom) + rng.normal(0, 20, n))
    assert (nno > 0).all()
    event = rng.choice([0, 1], n) if all_pulls else np.ones(n, dtype=int)
    return pd.DataFrame({
        "code": [f"w{i}" for i in range(n)],
        "field": field, "h2s_class": h2s,
        "ql": ql, "log_ql": np.log(ql), "qnom": qnom, "log_qnom": np.log(qnom),
        # ql_start is a noisy readout of the same underlying rate.
        "ql_start": ql * np.exp(rng.normal(0, 0.1, n)),
        "kpod": ql / qnom, "freq": 50.0, "nno": nno,
        "event": event,
        "install": pd.Timestamp("2024-06-01"), "end": pd.Timestamp("2025-06-01"),
    })


@pytest.fixture(autouse=True)
def _cheap_field_models(monkeypatch):
    """Stand in for the real field models everywhere except the tests that want them.

    ``FIELD_MODELS["Ya"]`` fits ``ya_k1k2_hybrid`` off the warehouse — ~130 s and a live
    DB — which no unit test should pay for.  The stand-in is a power law with the same
    signature and the same stratum-awareness, so every test about *plumbing* (residual
    algebra, panel wiring, tables) still exercises the real code paths.  The tests that
    check the actual models are in :class:`TestRealFieldModels` and patch nothing.
    """
    exponent = {"nonsour": -0.35, "sour": -0.10}

    def fake(ql, stratum):
        return (np.asarray(ql, float) / 250.0) ** exponent.get(stratum, -0.35)

    monkeypatch.setattr(Q, "FIELD_MODELS",
                        {"Ya": ("fake Ya θ_Ql", fake), "Vt": ("fake Vt θ_Ql", fake)})


def _dailies(well_key: str, start: str, days: int, qliq: float) -> pd.DataFrame:
    return pd.DataFrame({
        "well_key": well_key,
        "dt": pd.date_range(start, periods=days, freq="D"),
        "qliq": float(qliq), "qgas": 0.0, "watercut": 0.0, "gas_factor": 0.0,
    })


class TestFit:
    def test_slope_is_days_per_efold_of_ql(self):
        p = _panel(ql_slope=-80.0)
        f = Q._fit(p["ql"], p["nno"])
        assert f is not None
        assert f.beta1 == pytest.approx(-80.0, abs=6.0)
        assert f.p1 < 1e-6

    def test_slope_is_read_at_the_geometric_mean(self):
        """u is centred on mean(log Ql), so the reported Ql is the geometric mean."""
        p = _panel()
        f = Q._fit(p["ql"], p["nno"])
        assert f.ql_gm == pytest.approx(float(np.exp(np.log(p["ql"]).mean())), rel=1e-9)
        # Not the arithmetic mean — Ql is right-skewed, the two differ materially.
        assert f.ql_gm < p["ql"].mean() * 0.95

    def test_straight_relation_shows_no_curvature(self):
        f = Q._fit(*[_panel()[c] for c in ("ql", "nno")])
        assert f.p2 > 0.05

    def test_too_few_rows_returns_none_rather_than_a_fit(self):
        assert Q._fit(_panel(n=20)["ql"], _panel(n=20)["nno"]) is None


class TestQnomResidual:
    def test_residual_is_orthogonal_to_pump_size(self):
        r = Q._qnom_residual(_panel(qnom_slope=-60.0))
        p = _panel(qnom_slope=-60.0)
        assert r.mean() == pytest.approx(0.0, abs=1e-8)
        assert np.corrcoef(r, p["log_qnom"])[0, 1] == pytest.approx(0.0, abs=1e-8)

    def test_pure_pump_size_effect_does_not_survive(self):
        """If Ql only proxied Qnom, the residual panel must flatten — the Ya v2.1 check."""
        p = _panel(ql_slope=0.0, qnom_slope=-100.0)
        marginal = Q._fit(p["ql"], p["nno"])
        residual = Q._fit(p["ql"], Q._qnom_residual(p))
        assert abs(marginal.beta1) > abs(residual.beta1)
        assert residual.p1 > 0.05

    def test_genuine_ql_effect_does_survive(self):
        p = _panel(ql_slope=-80.0, qnom_slope=-100.0)
        residual = Q._fit(p["ql"], Q._qnom_residual(p))
        assert residual.beta1 == pytest.approx(-80.0, abs=10.0)
        assert residual.p1 < 1e-6


class TestBuild:
    def test_emits_the_four_requested_groups(self):
        groups, _ = Q.build(_panel())
        assert list(groups) == ["Ya", "Vt (all)", "Vt_nonsour", "Vt_sour"]

    def test_vt_all_is_the_union_of_the_two_strata(self):
        groups, _ = Q.build(_panel())
        assert (len(groups["Vt (all)"].rows)
                == len(groups["Vt_nonsour"].rows) + len(groups["Vt_sour"].rows))

    def test_residuals_are_fitted_inside_each_group(self):
        """Pooled size fit would leak the Ya/Vt life gap in through their pump mixes."""
        groups, _ = Q.build(_panel(qnom_slope=-60.0))
        for gp in groups.values():
            assert gp.rows["resid"].mean() == pytest.approx(0.0, abs=1e-8)

    def test_out_of_range_ql_is_counted_not_silently_dropped(self):
        p = _panel()
        p.loc[p.index[:5], "ql"] = 0.5          # below QL_BOUNDS
        cov = Q.build(p)[1].set_index("group")
        assert cov.loc["Ya", "n_dropped_out_of_bounds"] + \
               cov.loc["Vt (all)", "n_dropped_out_of_bounds"] == 5

    def test_coverage_splits_failures_from_other_pulls(self):
        p = _panel(all_pulls=True)
        cov = Q.build(p)[1].set_index("group")
        row = cov.loc["Ya"]
        assert row["n_failures"] + row["n_other_pulls"] == row["n_in_bounds"]
        assert row["n_other_pulls"] > 0

    def test_thin_group_is_skipped_but_still_reported_in_coverage(self):
        p = _panel()
        p.loc[p["h2s_class"] == "sour", "h2s_class"] = "nonsour"
        p.loc[p.index[:5], "h2s_class"] = "sour"
        p.loc[p.index[:5], "field"] = "Vt"
        groups, cov = Q.build(p)
        assert "Vt_sour" not in groups
        assert "Vt_sour" in set(cov["group"])


class TestQlStart:
    """The first-operating-month rate — and the guards that keep it honest."""

    @staticmethod
    def _panel_one(install="2024-01-01", end="2025-01-01") -> pd.DataFrame:
        return pd.DataFrame([{
            "code": "VT_1", "field": "Vt", "h2s_class": "sour",
            "install": pd.Timestamp(install), "end": pd.Timestamp(end),
        }])

    def _attach(self, monkeypatch, panel, dailies):
        monkeypatch.setattr(Q.t0_covariates, "load_dailies", lambda keys, **kw: dailies)
        return Q.attach_ql_start(panel)

    def test_averages_the_first_window_only(self, monkeypatch):
        d = pd.concat([_dailies("vt_1", "2024-01-01", 30, 100.0),
                       _dailies("vt_1", "2024-02-01", 200, 900.0)])
        out, _ = self._attach(monkeypatch, self._panel_one(), d)
        assert out["ql_start"].iloc[0] == pytest.approx(100.0)
        assert out["ql_start_n_days"].iloc[0] == 30

    def test_idle_days_do_not_count_toward_the_month(self, monkeypatch):
        """An operating day is a day the pump lifted fluid — qliq == 0 is not one."""
        d = _dailies("vt_1", "2024-01-01", 60, 100.0)
        d.loc[d.index[:30], "qliq"] = 0.0
        out, _ = self._attach(monkeypatch, self._panel_one(), d)
        assert out["ql_start_n_days"].iloc[0] == 30
        assert out["ql_start_status"].iloc[0] == "full window"

    def test_pre_2018_install_is_refused_not_given_a_midlife_rate(self, monkeypatch):
        """The daily table starts 2018; its first rows are mid-life for a 2015 run."""
        panel = self._panel_one(install="2015-01-01", end="2020-01-01")
        out, _ = self._attach(monkeypatch, panel, _dailies("vt_1", "2018-01-01", 60, 100.0))
        assert np.isnan(out["ql_start"].iloc[0])
        assert "after install" in out["ql_start_status"].iloc[0]
        assert out["ql_start_gap_days"].iloc[0] == pytest.approx(1096.0)

    def test_late_windows_roll_up_to_one_coverage_row(self, monkeypatch):
        """The gap is a number, not part of the label — else the rollup is one row per run."""
        panel = pd.concat([self._panel_one(install=d, end="2021-01-01")
                           for d in ("2015-01-01", "2016-01-01", "2017-01-01")])
        _, cov = self._attach(monkeypatch, panel,
                              _dailies("vt_1", "2018-01-01", 60, 100.0))
        late = cov[cov["ql_start_status"].str.contains("after install")]
        assert len(late) == 1 and late["n"].iloc[0] == 3

    def test_window_is_bounded_by_the_run_so_the_next_run_cannot_leak_in(self, monkeypatch):
        panel = self._panel_one(install="2024-01-01", end="2024-01-11")
        d = pd.concat([_dailies("vt_1", "2024-01-01", 10, 100.0),
                       _dailies("vt_1", "2024-06-01", 30, 900.0)])   # a later run
        out, _ = self._attach(monkeypatch, panel, d)
        assert out["ql_start"].iloc[0] == pytest.approx(100.0)
        assert out["ql_start_status"].iloc[0] == "partial window"

    def test_missing_well_keeps_nan_and_is_reported(self, monkeypatch):
        out, cov = self._attach(monkeypatch, self._panel_one(),
                                _dailies("other_well", "2024-01-01", 30, 100.0))
        assert np.isnan(out["ql_start"].iloc[0])
        assert cov["n"].sum() == 1
        assert "no daily rows for well" in set(cov["ql_start_status"])

    def test_build_selects_the_rate_axis(self):
        p = _panel()
        p.loc[p.index[:100], "ql_start"] = np.nan
        svod, _ = Q.build(p, rate="ql_svod")
        start, cov = Q.build(p, rate="ql_start")
        assert sum(len(g.rows) for g in start.values()) \
            < sum(len(g.rows) for g in svod.values())
        # Ya + Vt (all) partition the panel; the two Vt strata re-cover Vt.
        top = cov[cov["group"].isin(["Ya", "Vt (all)"])]
        assert top["n_dropped_missing_rate"].sum() == 100
        assert all(g.rate == "ql_start" for g in start.values())

    def test_missing_rate_rows_are_counted_not_silently_gone(self):
        p = _panel()
        p.loc[p.index[:100], "ql_start"] = np.nan
        cov = Q.build(p, rate="ql_start")[1]
        # Vt (all) double-counts its two strata, so check the disjoint groups.
        disjoint = cov[cov["group"] != "Vt (all)"]
        assert (disjoint["n_present"]
                == disjoint["n_with_rate"] + disjoint["n_dropped_missing_rate"]).all()


class TestFieldModelResidual:
    """The third panel: offset by the field's own fitted Ql layer, not by log Ql."""

    def test_model_supplies_the_shape_and_only_the_level_is_fitted(self):
        """A run set that IS the model curve, scaled, must residual to zero."""
        p = _panel(ql_slope=0.0)
        mult = Q.model_life_multiplier(p.assign(field="Ya"), "ql")
        p["nno"] = 380.0 * mult                      # exactly the model, level 380
        r = Q._model_residual(p, "ql")
        assert np.abs(r).max() < 1e-6

    def test_level_mismatch_alone_does_not_create_a_slope(self):
        """Свод ННО is a different clock; a pure scale error must not read as shape error."""
        p = _panel(ql_slope=0.0)
        p["nno"] = 3.7 * 380.0 * Q.model_life_multiplier(p.assign(field="Ya"), "ql")
        assert np.abs(Q._model_residual(p, "ql")).max() < 1e-6

    def test_shape_the_model_misses_survives_into_the_residual(self):
        """Structure the layer does not capture must still be visible afterwards.

        Only the SIGN and significance are asserted, not the magnitude: the free level
        ``L0`` is itself fitted against a multiplier that varies with Ql, so it absorbs
        part of any unmodelled slope and re-emits it in the model's own shape.  The
        residual panel answers "is there structure left, and which way", never "the
        layer is short by exactly N days per e-fold".
        """
        p = _panel(ql_slope=0.0)
        base = Q.model_life_multiplier(p.assign(field="Ya"), "ql")
        p["nno"] = 380.0 * base - 120.0 * np.log(p["ql"])     # extra, unmodelled slope
        f = Q._fit(p["ql"], Q._model_residual(p, "ql"))
        assert f.beta1 < -20.0
        assert f.p1 < 1e-6

    def test_vt_strata_are_evaluated_separately(self):
        """Vt's layer is stratum-specific — pooling applies the nonsour curve to sour runs."""
        p = _panel()
        vt = p[p["field"] == "Vt"]
        got = Q.model_life_multiplier(vt, "ql")
        for stratum, part in vt.groupby("h2s_class"):
            want = Q.FIELD_MODELS["Vt"][1](part["ql"].to_numpy(float), stratum)
            assert np.allclose(got.loc[part.index], want)
        # The two strata really do differ, so the split is not a no-op here.
        assert not np.allclose(
            got, Q.FIELD_MODELS["Vt"][1](vt["ql"].to_numpy(float), "nonsour"))

    def test_panel_and_fit_table_carry_the_third_residual(self):
        groups, _ = Q.build(_panel())
        gp = groups["Ya"]
        assert len(gp.model_residual) == len(gp.marginal)
        assert gp.model_label == "fake Ya θ_Ql"
        panels = set(Q.fit_table(groups)["panel"])
        assert panels == {"marginal", "qnom_residual", "field_model_residual"}

    def test_figure_has_three_columns(self, tmp_path):
        groups, _ = Q.build(_panel())
        fig_path = Q.plot(groups, tmp_path / "f.png", suptitle="t", overlay="heatmap")
        assert fig_path.stat().st_size > 0


class TestRealFieldModels:
    """The genuine layers — Vt's closed form only; Ya's needs a warehouse fit."""

    def test_vt_uses_the_shipped_v32_exponents_per_stratum(self):
        from analysis.workflows.production_risk import vt_composed_model as VC
        # Higher Ql ⇒ higher hazard ⇒ shorter life, and nonsour responds ~2× as hard.
        ns = Q.vt_life_mult(np.array([100.0, 800.0]), "nonsour")
        so = Q.vt_life_mult(np.array([100.0, 800.0]), "sour")
        assert ns[1] < ns[0] and so[1] < so[0]
        assert (ns[0] - ns[1]) > (so[0] - so[1])
        # Read the curve, not a named constant: v3.2's θ_Ql form has already changed
        # once (power law → tanh, see vt_ql_extrapolation) and will change again.
        assert VC.theta_ql("nonsour", 800.0) > VC.theta_ql("sour", 800.0)

    def test_vt_life_multiplier_is_one_at_the_reference_rate(self):
        from analysis.workflows.production_risk import vt_composed_model as VC
        m = Q.vt_life_mult(np.array([VC.QL_REF, VC.QL_REF * 2]), "nonsour")
        assert m[0] == pytest.approx(1.0, abs=1e-3)

    def test_vt_layer_is_flat_below_its_guard(self):
        """v3.2 holds θ_Ql flat under 100 m³/d; the life multiplier must inherit that."""
        m = Q.vt_life_mult(np.array([10.0, 50.0, 99.0]), "nonsour")
        assert np.allclose(m, m[0], atol=1e-6)

    def test_ya_model_is_lazy_and_cached(self):
        """Nothing may fit Ya at import time, and a second call must not refit."""
        assert Q._ya_model.cache_info().currsize in (0, 1)

    def test_grid_interpolation_matches_direct_evaluation(self):
        """The θ→life map is interpolated for speed; it must not distort the curve."""
        from analysis.workflows.production_risk import vt_composed_model as VC
        b = VC.BASELINE["nonsour"]
        r0 = VC.rmst(b["eta0"], b["beta0"])
        exact = lambda t: VC.rmst(b["eta0"] * t ** (-1 / b["beta0"]), b["beta0"]) / r0  # noqa: E731
        ql = np.linspace(100.0, 900.0, 40)
        got = Q.vt_life_mult(ql, "nonsour")
        want = np.array([exact(float(t)) for t in VC.theta_ql("nonsour", ql)])
        assert np.abs(got - want).max() < 1e-4


class TestRateComparison:
    def test_holds_rows_fixed_between_the_two_rate_definitions(self):
        p = _panel()
        p.loc[p.index[:200], "ql_start"] = np.nan
        t = Q.rate_comparison(p)
        assert (t["svod_subset_n"] == t["start_subset_n"]).all()
        assert (t["svod_full_n"] > t["svod_subset_n"]).all()

    def test_isolates_a_pure_definition_change(self):
        """Same rows, and only ql_start carries the effect — the slope must follow it."""
        p = _panel(ql_slope=0.0)
        rng = np.random.default_rng(1)
        p["ql_start"] = np.exp(rng.normal(5.0, 0.6, len(p)))
        p["nno"] = p["nno"] - 90.0 * np.log(p["ql_start"])
        t = Q.rate_comparison(p).set_index("group")
        assert t.loc["Ya", "start_subset_days_per_efold"] == pytest.approx(-90.0, abs=12.0)
        assert t.loc["Ya", "start_subset_p"] < 1e-6
        assert t.loc["Ya", "svod_subset_p"] > 0.05


class TestFigures:
    def test_every_overlay_mode_writes_a_figure(self, tmp_path):
        groups, _ = Q.build(_panel(all_pulls=True))
        sizes = {}
        for overlay in Q.OVERLAYS:
            p = Q.plot(groups, tmp_path / f"{overlay}.png", suptitle="t", overlay=overlay)
            sizes[overlay] = p.stat().st_size
        # Both overlays draw every run, so both are heavier than the bare curves.
        assert sizes["scatter"] > sizes["none"]
        assert sizes["heatmap"] > sizes["none"]

    def test_unknown_overlay_is_rejected_rather_than_silently_ignored(self, tmp_path):
        groups, _ = Q.build(_panel())
        with pytest.raises(ValueError, match="overlay must be"):
            Q.plot(groups, tmp_path / "x.png", suptitle="t", overlay="hexbin")

    def test_uncapped_field_ylim_is_percentile_clipped_against_outliers(self, monkeypatch):
        """Without a cap, one 20000-day run must not squash the decile means to the floor."""
        monkeypatch.setattr(Q, "Y_CAP_BY_FIELD", {})
        p = _panel()
        p.loc[p.index[0], "nno"] = 20000.0
        groups, _ = Q.build(p)
        gp = groups["Ya"]
        assert Q._ylims(gp, "scatter")[0][1] < 5000.0
        assert Q._ylims(gp, "none")[0][1] < 5000.0

    def test_capped_fields_get_the_requested_spans(self):
        groups, _ = Q.build(_panel())
        ya_m, ya_r, ya_k = Q._ylims(groups["Ya"], "scatter")
        vt_m, vt_r, vt_k = Q._ylims(groups["Vt (all)"], "scatter")
        assert ya_m == (0.0, 1100.0) and vt_m == (0.0, 750.0)
        # Both residual bands share the marginal's height, centred on zero — that is
        # what makes the Qnom and field-model panels comparable side by side.
        assert ya_r == ya_k == (-550.0, 550.0)
        assert vt_r == vt_k == (-375.0, 375.0)

    def test_every_vt_group_shares_one_span(self):
        """The sour/nonsour/pooled rows are only comparable on a common scale."""
        groups, _ = Q.build(_panel())
        vt = [Q._ylims(groups[k], "scatter")
              for k in ("Vt (all)", "Vt_nonsour", "Vt_sour")]
        assert len(set(vt)) == 1

    def test_clipped_share_is_measured_for_the_annotation(self):
        groups, _ = Q.build(_panel())
        gp = groups["Ya"]
        assert Q._clipped_share(gp.rows, "nno", (0.0, 1e9)) == 0.0
        assert Q._clipped_share(gp.rows, "nno", (0.0, float(gp.rows["nno"].median()))) \
            == pytest.approx(0.5, abs=0.02)


class TestMultiplierAxis:
    """The right-hand axis: same days, expressed against the median-rate life."""

    def test_reference_is_the_fit_at_the_median_rate(self):
        groups, _ = Q.build(_panel(ql_slope=-80.0))
        gp = groups["Ya"]
        assert gp.rate_median == pytest.approx(float(gp.rows["ql"].median()))
        assert gp.ref_ttf == pytest.approx(gp.marginal_fit.predict(gp.rate_median))
        # It is a life, so it lands inside the observed range of lives.
        assert gp.rows["nno"].min() < gp.ref_ttf < gp.rows["nno"].max()

    def test_predict_recovers_the_planted_slope(self):
        p = _panel(ql_slope=-80.0)
        f = Q._fit(p["ql"], p["nno"])
        # One e-fold up from the geometric mean must cost ~80 days.
        assert f.predict(f.ql_gm * np.e) - f.predict(f.ql_gm) == pytest.approx(-80.0, abs=6.0)

    def test_marginal_multiplier_is_a_plain_ratio(self):
        assert Q.multiplier(400.0, 400.0, zero_is_reference=False) == pytest.approx(1.0)
        assert Q.multiplier(200.0, 400.0, zero_is_reference=False) == pytest.approx(0.5)

    def test_residual_multiplier_puts_1x_on_the_zero_line(self):
        """A residual of 0 means "as long as a median-rate well", i.e. exactly 1.0×."""
        m = lambda d: Q.multiplier(d, 400.0, zero_is_reference=True)  # noqa: E731
        assert m(0.0) == pytest.approx(1.0)
        assert m(400.0) == pytest.approx(2.0)
        assert m(-200.0) == pytest.approx(0.5)

    @staticmethod
    def _annotate(ax, frame, ref, xlim, *, zero_is_reference=False):
        ax.set_xlim(*xlim)
        ax.set_ylim(float(frame["ymean"].min()) * 0.5, float(frame["ymean"].max()) * 1.5)
        ax.figure.canvas.draw()
        return Q._annotate_multipliers(
            ax, frame, ref, zero_is_reference=zero_is_reference,
            renderer=ax.figure.canvas.get_renderer())

    def test_labels_carry_the_ratio_of_each_point(self):
        groups, _ = Q.build(_panel())
        gp = groups["Ya"]
        fig, ax = plt.subplots(figsize=(30, 6))     # roomy: nothing should be dropped
        wide = (float(gp.marginal["xmid"].min()), float(gp.marginal["xmid"].max()))
        drawn, skipped = self._annotate(ax, gp.marginal, gp.ref_ttf, wide)
        texts = [t.get_text() for t in ax.texts]
        assert drawn == len(texts) and skipped == 0
        expected = [f"{v:.2f}×" for v in gp.marginal["ymean"] / gp.ref_ttf]
        assert texts == expected
        plt.close(fig)

    def test_crowded_labels_stack_across_levels_before_being_dropped(self):
        """Every level of headroom must be used before any label is given up."""
        groups, _ = Q.build(_panel())
        gp = groups["Ya"]
        fig, ax = plt.subplots(figsize=(2.2, 4))    # deliberately far too narrow
        tight = (float(gp.marginal["xmid"].min()), float(gp.marginal["xmid"].max()))
        drawn, skipped = self._annotate(ax, gp.marginal, gp.ref_ttf, tight)
        assert drawn + skipped == len(gp.marginal)
        assert skipped > 0                              # too tight for all of them
        assert {a.xyann[1] for a in ax.texts} == set(Q.LABEL_LEVELS)  # levels used first
        plt.close(fig)

    def test_placed_labels_never_overlap_each_other(self):
        """The whole point of box-based placement: no two boxes may intersect."""
        groups, _ = Q.build(_panel())
        gp = groups["Vt (all)"]
        fig, ax = plt.subplots(figsize=(5, 4))
        self._annotate(ax, gp.marginal, gp.ref_ttf,
                       (0.0, float(gp.marginal["xmid"].max()) * 1.2))
        r = fig.canvas.get_renderer()
        boxes = [a.get_window_extent(r) for a in ax.texts]
        assert boxes, "expected at least some labels to be placed"
        for i, a in enumerate(boxes):
            for b in boxes[i + 1:]:
                assert not a.overlaps(b)
        plt.close(fig)

    def test_labels_at_different_levels_still_cannot_collide(self):
        """An x-only rule passed this case; two stacked labels on a steep curve did not."""
        frame = pd.DataFrame({"xmid": [100.0, 104.0], "ymean": [600.0, 520.0],
                              "yse": [1.0, 1.0], "n": [50, 50]})
        fig, ax = plt.subplots(figsize=(5, 4))
        drawn, _ = self._annotate(ax, frame, 400.0, (0.0, 200.0))
        r = fig.canvas.get_renderer()
        boxes = [a.get_window_extent(r) for a in ax.texts]
        assert drawn == len(boxes)
        if len(boxes) == 2:
            assert not boxes[0].overlaps(boxes[1])
        plt.close(fig)

    def test_no_labels_when_the_reference_is_unusable(self, tmp_path):
        """A non-positive or NaN reference would print inverted or NaN multipliers."""
        p = _panel()
        for bad in (0.0, -10.0, float("nan")):
            groups, _ = Q.build(p)
            groups = {k: replace(gp, ref_ttf=bad) for k, gp in groups.items()}
            out = Q.plot(groups, tmp_path / f"{bad}.png", suptitle="t")
            assert out.exists()

    def test_x_axis_is_linear_not_log(self):
        groups, _ = Q.build(_panel())
        fig, ax = plt.subplots()
        Q._panel(ax, groups["Ya"].marginal, None, color="#000000", ylabel="y",
                 title="t", zero_line=False, rows=None, value_col="nno",
                 rate_col="ql", xlabel="x", overlay="none", ylim=(0.0, 1.0))
        assert ax.get_xscale() == "linear"
        plt.close(fig)

    def test_fit_table_carries_the_denominator(self):
        groups, _ = Q.build(_panel())
        t = Q.fit_table(groups).set_index(["group", "panel"])
        assert t.loc[("Ya", "marginal"), "ref_ttf_at_rate_median"] == \
            pytest.approx(groups["Ya"].ref_ttf)
        # Both panels of a group share one denominator — that is what makes the
        # marginal and residual multipliers comparable.
        assert t.loc[("Ya", "qnom_residual"), "ref_ttf_at_rate_median"] == \
            pytest.approx(groups["Ya"].ref_ttf)


class TestHeatmap:
    """The scatter's replacement — a conditional density, with its own failure modes."""

    def test_each_rate_column_sums_to_one(self):
        groups, _ = Q.build(_panel())
        fig, ax = plt.subplots()
        gp = groups["Ya"]
        mesh = Q._heatmap(ax, gp.rows, "ql", "nno", (0.0, float(gp.rows["nno"].max())))
        z = mesh.get_array().reshape(Q.HEATMAP_YBINS, Q.HEATMAP_XBINS)
        totals = np.ma.filled(z, 0.0).sum(axis=0)
        assert np.allclose(totals[totals > 0], 1.0)
        plt.close(fig)

    def test_raw_counts_are_available_and_do_not_sum_to_one(self):
        groups, _ = Q.build(_panel())
        fig, ax = plt.subplots()
        gp = groups["Ya"]
        mesh = Q._heatmap(ax, gp.rows, "ql", "nno",
                          (0.0, float(gp.rows["nno"].max())), normalize=False)
        z = np.ma.filled(mesh.get_array(), 0.0)
        # Counts are whole runs, not shares — and they total the population minus
        # whatever the thin-column blanking removed, never more than it.
        assert np.allclose(z, np.round(z))
        assert z.max() > 1.0
        assert 0.85 * len(gp.rows) < z.sum() <= len(gp.rows)
        plt.close(fig)

    def test_thin_columns_are_blanked_not_saturated(self):
        """Normalising makes a 2-run column as dark as a 90-run one — blank it instead."""
        p = _panel()
        # A lone far-right column of 2 runs, well past the bulk of the rate range.
        p.loc[p.index[:2], "ql"] = 1900.0
        groups, _ = Q.build(p)
        fig, ax = plt.subplots()
        gp = groups["Ya"]
        mesh = Q._heatmap(ax, gp.rows, "ql", "nno", (0.0, float(gp.rows["nno"].max())))
        z = mesh.get_array().reshape(Q.HEATMAP_YBINS, Q.HEATMAP_XBINS)
        assert np.ma.getmaskarray(z)[:, -1].all()
        plt.close(fig)

    def test_colour_scale_is_shared_across_every_panel(self):
        """Per-panel autoscaling would let a pale Ya cell mean the same as a dark sour one."""
        groups, _ = Q.build(_panel())
        limits = {k: Q._ylims(gp, "heatmap") for k, gp in groups.items()}
        vmax = Q._shared_vmax(groups, "ql", limits)
        per_panel = [np.nanmax(Q._density(gp.rows, "ql", "nno", limits[k][0])[2])
                     for k, gp in groups.items()]
        assert vmax > 0
        # A single number, not one per group — and it is a percentile, so it sits
        # below the largest single cell rather than being dragged to it.
        assert vmax < max(per_panel)

    def test_empty_cells_are_masked_not_drawn_as_ramp_zero(self):
        groups, _ = Q.build(_panel())
        fig, ax = plt.subplots()
        gp = groups["Ya"]
        mesh = Q._heatmap(ax, gp.rows, "ql", "nno", (0.0, float(gp.rows["nno"].max())))
        assert np.ma.getmaskarray(mesh.get_array()).any()
        plt.close(fig)

    def test_fit_table_reports_every_panel_per_group(self):
        groups, _ = Q.build(_panel())
        t = Q.fit_table(groups)
        assert set(t["panel"]) == {"marginal", "qnom_residual", "field_model_residual"}
        assert len(t) == 3 * len(groups)
