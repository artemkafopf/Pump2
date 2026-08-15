"""Unit tests for the Свод TTF-by-design panels (Qnom, contractor) — no workbook."""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.workflows.production_risk import svod_ttf_by_design as D  # noqa: E402


@pytest.fixture(autouse=True)
def _cheap_field_models(monkeypatch):
    """Stand in for the real Ya fit (~130 s + a live warehouse); see the Ql test module."""
    def fake(ql, stratum):
        return (np.asarray(ql, float) / 250.0) ** (-0.35 if stratum == "nonsour" else -0.10)

    monkeypatch.setattr(D, "FIELD_MODELS",
                        {"Ya": ("fake Ya θ_Ql", fake), "Vt": ("fake Vt θ_Ql", fake)})
    monkeypatch.setattr("analysis.workflows.production_risk.svod_ttf_vs_ql.FIELD_MODELS",
                        {"Ya": ("fake Ya θ_Ql", fake), "Vt": ("fake Vt θ_Ql", fake)})
    # Ya's contractor factor comes from the same expensive fit; Vt's is closed form and
    # is left real so the stratum-aware tests below exercise the shipped numbers.
    fake_hr = {"brt": 1.0, "slb": 0.9, "oth": 0.6}
    monkeypatch.setitem(
        D.CONTRACTOR_MODELS, "Ya",
        lambda g: g["contractor"].map(fake_hr).astype(float))


def _panel(n: int = 900, seed: int = 0, qnom_slope: float = -90.0,
           ql_slope: float = 0.0, contractor_days: dict | None = None,
           all_pulls: bool = True) -> pd.DataFrame:
    """Synthetic Свод-shaped panel with independent Ql and Qnom, plus a contractor."""
    rng = np.random.default_rng(seed)
    field = rng.choice(["Ya", "Vt"], n)
    h2s = np.where(field == "Vt", rng.choice(["sour", "nonsour"], n), "nonsour")
    qnom = np.exp(rng.normal(5.4, 0.6, n))
    ql = np.exp(rng.normal(5.0, 0.6, n))          # independent of qnom by construction
    contractor = rng.choice(["brt", "slb", "oth"], n, p=[0.6, 0.3, 0.1])
    bump = pd.Series(contractor).map(contractor_days or {}).fillna(0.0).to_numpy()
    base = 400 + 7.0 * (abs(qnom_slope) + abs(ql_slope))
    nno = (base + qnom_slope * np.log(qnom) + ql_slope * np.log(ql) + bump
           + rng.normal(0, 20, n))
    assert (nno > 0).all()
    return pd.DataFrame({
        "code": [f"w{i}" for i in range(n)], "field": field, "h2s_class": h2s,
        "ql": ql, "log_ql": np.log(ql), "qnom": qnom, "log_qnom": np.log(qnom),
        "kpod": ql / qnom, "freq": 50.0, "nno": nno, "contractor": contractor,
        "event": rng.choice([0, 1], n) if all_pulls else np.ones(n, dtype=int),
        "install": pd.Timestamp("2024-06-01"), "end": pd.Timestamp("2025-06-01"),
    })


class TestQnomPanels:
    def test_marginal_slope_is_days_per_efold_of_qnom(self):
        groups, _ = D.build_qnom(_panel(qnom_slope=-90.0))
        f = groups["Ya"].marginal_fit
        assert f.beta1 == pytest.approx(-90.0, abs=8.0)
        assert f.p1 < 1e-6

    def test_legend_names_qnom_not_ql(self):
        """The shared fit object defaults to 'Ql'; a Qnom panel must not inherit that."""
        groups, _ = D.build_qnom(_panel())
        assert "per e-fold Qnom" in groups["Ya"].marginal_fit.label

    def test_log_ql_is_the_control_here_the_mirror_of_the_ql_figure(self):
        r = D._ql_residual(_panel())
        p = _panel()
        assert r.mean() == pytest.approx(0.0, abs=1e-8)
        assert np.corrcoef(r, np.log(p["ql"]))[0, 1] == pytest.approx(0.0, abs=1e-8)

    def test_pure_rate_effect_does_not_survive_the_control(self):
        p = _panel(qnom_slope=0.0, ql_slope=-100.0)
        groups, _ = D.build_qnom(p)
        assert groups["Ya"].residual_fit.p1 > 0.05

    def test_genuine_size_effect_survives_the_control(self):
        p = _panel(qnom_slope=-90.0, ql_slope=-100.0)
        groups, _ = D.build_qnom(p)
        f = groups["Ya"].residual_fit
        assert f.beta1 == pytest.approx(-90.0, abs=12.0)
        assert f.p1 < 1e-6

    def test_reference_is_the_fit_at_the_median_qnom(self):
        gp = D.build_qnom(_panel())[0]["Ya"]
        assert gp.qnom_median == pytest.approx(float(gp.rows["qnom"].median()))
        assert gp.ref_ttf == pytest.approx(gp.marginal_fit.predict(gp.qnom_median))

    def test_out_of_range_qnom_is_counted_not_silently_dropped(self):
        p = _panel()
        p.loc[p.index[:6], "qnom"] = 9000.0        # the sheet really carries these
        cov = D.build_qnom(p)[1].set_index("group")
        assert cov.loc[["Ya", "Vt (all)"], "n_dropped"].sum() == 6

    def test_fit_table_has_all_three_panels(self):
        groups, _ = D.build_qnom(_panel())
        t = D.qnom_fit_table(groups)
        assert set(t["panel"]) == {"marginal", "log_ql_residual",
                                   "field_model_rate_residual"}

    def test_every_overlay_renders(self, tmp_path):
        groups, _ = D.build_qnom(_panel())
        for overlay in D.OVERLAYS:
            out = D.plot_qnom(groups, tmp_path / f"{overlay}.png", suptitle="t",
                              overlay=overlay)
            assert out.stat().st_size > 0

    def test_unknown_overlay_is_rejected(self, tmp_path):
        groups, _ = D.build_qnom(_panel())
        with pytest.raises(ValueError, match="overlay must be"):
            D.plot_qnom(groups, tmp_path / "x.png", suptitle="t", overlay="hexbin")


class TestQnomVsCleanRate:
    """The symmetric contest: size vs the first-month rate, on identical rows."""

    @staticmethod
    def _with_start(p: pd.DataFrame, noise: float = 0.1, seed: int = 3) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        return p.assign(ql_start=p["ql"] * np.exp(rng.normal(0, noise, len(p))))

    def test_size_wins_when_only_size_drives_life(self):
        p = self._with_start(_panel(qnom_slope=-90.0, ql_slope=0.0))
        t = D.qnom_vs_ql_start(p).set_index("group")
        assert t.loc["Ya", "qnom_given_ql_start_p"] < 1e-4
        assert t.loc["Ya", "ql_start_given_qnom_p"] > 0.05

    def test_rate_wins_when_only_rate_drives_life(self):
        p = self._with_start(_panel(qnom_slope=0.0, ql_slope=-90.0))
        t = D.qnom_vs_ql_start(p).set_index("group")
        assert t.loc["Ya", "ql_start_given_qnom_p"] < 1e-4
        assert t.loc["Ya", "qnom_given_ql_start_p"] > 0.05

    def test_both_directions_use_the_same_rows(self):
        t = D.qnom_vs_ql_start(self._with_start(_panel())).set_index("group")
        # One n per group covers both directions — an asymmetric row count would make
        # the "which one survives" comparison meaningless.
        assert t.loc["Ya", "n"] > 0
        assert set(t.columns) >= {"qnom_given_ql_start_p", "ql_start_given_qnom_p",
                                  "corr_log_qnom_log_ql_start"}

    def test_thin_group_is_reported_without_a_verdict(self):
        p = self._with_start(_panel(n=60))
        p = pd.concat([p[p["field"] == "Ya"].head(10), p[p["field"] == "Vt"]])
        t = D.qnom_vs_ql_start(p).set_index("group")
        assert np.isnan(t.loc["Ya"].get("qnom_given_ql_start_days_per_efold", np.nan))


class TestContractorPanels:
    def test_reference_contractor_is_exactly_one(self):
        groups, _ = D.build_contractor(_panel(), n_boot=200)
        t = groups["Ya"].table.set_index("contractor")
        for col in ("raw_mult", "adj_mult", "model_mult"):
            assert t.loc["brt", col] == pytest.approx(1.0)

    def test_a_planted_contractor_gap_is_recovered(self):
        """oth 200 days shorter than brt on a ~1050-day base ⇒ ~0.81×."""
        p = _panel(qnom_slope=0.0, contractor_days={"brt": 0.0, "oth": -200.0})
        t = D.build_contractor(p, n_boot=200)[0]["Ya"].table.set_index("contractor")
        base = p.loc[(p["field"] == "Ya") & (p["contractor"] == "brt"), "nno"].mean()
        assert t.loc["oth", "raw_mult"] == pytest.approx((base - 200) / base, abs=0.02)
        assert t.loc["oth", "raw_hi"] < 1.0          # interval excludes parity

    def test_confounded_gap_disappears_once_the_operating_point_is_held(self):
        """oth given only big pumps must not read as a contractor effect after control."""
        p = _panel(qnom_slope=-90.0)
        ya_oth = (p["field"] == "Ya") & (p["contractor"] == "oth")
        p.loc[ya_oth, "qnom"] = p.loc[ya_oth, "qnom"] * 3.0
        p.loc[ya_oth, "nno"] = p.loc[ya_oth, "nno"] - 90.0 * np.log(3.0)
        t = D.build_contractor(p, n_boot=400)[0]["Ya"].table.set_index("contractor")
        assert t.loc["oth", "raw_mult"] < 0.95                       # looks bad raw
        assert t.loc["oth", "adj_lo"] < 1.0 < t.loc["oth", "adj_hi"]  # explained away

    def test_thin_cells_are_flagged_and_still_counted(self):
        p = _panel()
        keep = ~((p["field"] == "Ya") & (p["contractor"] == "oth"))
        p = pd.concat([p[keep], p[~keep].head(4)])
        t = D.build_contractor(p, n_boot=100)[0]["Ya"].table.set_index("contractor")
        assert t.loc["oth", "n"] == 4
        assert not t.loc["oth", "plotted"]

    def test_vt_model_multiplier_is_stratum_aware(self):
        """v3.2's contractor HR differs sour vs nonsour — a pooled number would hide it."""
        from analysis.workflows.production_risk import vt_composed_model as VC
        p = _panel()
        vt = p[p["field"] == "Vt"].copy()
        m = D.contractor_model_multiplier(vt)
        oth = vt["contractor"] == "oth"
        by_stratum = m[oth].groupby(vt.loc[oth, "h2s_class"]).mean()
        # Nonsour oth is the harsher penalty (HR 3.017 vs 1.507) ⇒ the shorter life.
        assert by_stratum["nonsour"] < by_stratum["sour"]
        assert VC.CONTRACTOR["nonsour"]["oth"] > VC.CONTRACTOR["sour"]["oth"]

    def test_model_multiplier_is_a_life_not_a_hazard_ratio(self):
        """A hazard ratio above 1 must come back as a life multiplier below 1."""
        from analysis.workflows.production_risk import vt_composed_model as VC
        p = _panel()
        vt = p[(p["field"] == "Vt") & (p["h2s_class"] == "nonsour")].copy()
        m = D.contractor_model_multiplier(vt)
        assert VC.CONTRACTOR["nonsour"]["oth"] > 1.0
        assert m[vt["contractor"] == "oth"].mean() < m[vt["contractor"] == "brt"].mean()

    def test_group_without_the_reference_contractor_is_skipped_with_a_note(self):
        p = _panel()
        p = p[~((p["field"] == "Ya") & (p["contractor"] == "brt"))]
        groups, cov = D.build_contractor(p, n_boot=100)
        assert "Ya" not in groups
        assert "no brt reference" in cov.set_index("group").loc["Ya", "note"]

    def test_figure_renders(self, tmp_path):
        groups, _ = D.build_contractor(_panel(), n_boot=100)
        out = D.plot_contractor(groups, tmp_path / "c.png", suptitle="t")
        assert out.stat().st_size > 0
