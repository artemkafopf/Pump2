"""Unit tests for the Свод TTF-vs-frequency panels — synthetic frames, no warehouse."""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.workflows.production_risk import svod_ttf_vs_freq as F  # noqa: E402

#: Shapes and betas the fake layers use, so tests can invert the offset by hand.
FAKE_BETA = 0.8
FAKE_EXP = {"nonsour": 0.35, "sour": 0.15}


@pytest.fixture(autouse=True)
def _cheap_layers(monkeypatch):
    """Substitute every shipped layer: each fits a model off the warehouse (Ya ~130 s)."""
    def offset(col):
        def fn(g):
            x = g[col].to_numpy(float)
            theta = np.array([(xi / 250.0) ** FAKE_EXP.get(s, 0.35)
                              for xi, s in zip(x, g["h2s_class"])])
            return theta, np.full(len(g), FAKE_BETA)
        return fn

    monkeypatch.setattr(F, "LAYERS", {
        "ql": {"Ya": ("fake Ya θ_Ql", offset("ql"), "ql"),
               "Vt": ("fake Vt θ_Ql", offset("ql"), "ql")},
        "qnom": {"Ya": ("fake Ya θ_Qnom", offset("qnom"), "qnom"),
                 "Vt": ("fake Vt θ_Qnom", offset("qnom"), "qnom")},
    })


def _panel(n: int = 900, seed: int = 0, rate_slope: float = -90.0,
           freq_curv: float = 0.0, freq_peak: float = 52.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    field = rng.choice(["Ya", "Vt"], n)
    h2s = np.where(field == "Vt", rng.choice(["sour", "nonsour"], n), "nonsour")
    ql = np.exp(rng.normal(5.0, 0.6, n))
    qnom = np.exp(rng.normal(5.4, 0.6, n))
    freq = rng.uniform(38.0, 62.0, n)
    nno = (1200.0 + rate_slope * np.log(ql) + freq_curv * (freq - freq_peak) ** 2
           + rng.normal(0, 20, n))
    assert (nno > 0).all()
    return pd.DataFrame({
        "code": [f"w{i}" for i in range(n)], "field": field, "h2s_class": h2s,
        "ql": ql, "qnom": qnom, "kpod": ql / qnom, "freq": freq, "nno": nno,
        "event": rng.choice([0, 1], n),
        "install": pd.Timestamp("2024-06-01"), "end": pd.Timestamp("2025-06-01"),
    })


class TestExtraction:
    def test_offset_is_multiplicative_on_time_not_subtractive(self):
        """t_adj = t·θ^(1/β): doubling every life must double every adjusted life."""
        p = _panel()
        a = F.extract_layer(p, "ql")
        b = F.extract_layer(p.assign(nno=p["nno"] * 2.0), "ql")
        assert np.allclose(b, 2.0 * a)

    def test_adjusted_life_is_positive_and_in_days(self):
        adj = F.extract_layer(_panel(), "ql")
        assert (adj > 0).all()
        assert adj.median() > 100.0        # still a life, not a standardised residual

    def test_high_rate_runs_are_scaled_up(self):
        """θ_Ql > 1 above the reference ⇒ the run's life is credited back upward."""
        p = _panel()
        adj = F.extract_layer(p, "ql")
        hi = p["ql"] > 250.0
        assert (adj[hi] > p.loc[hi, "nno"]).all()
        assert (adj[~hi] < p.loc[~hi, "nno"]).all()

    def test_extraction_removes_a_planted_rate_slope(self):
        """The whole point: after the offset, life should not vary with the rate."""
        exact = -90.0 * FAKE_BETA          # a slope the fake layer exactly undoes
        p = _panel(rate_slope=exact / FAKE_BETA * FAKE_BETA)
        p["nno"] = 1200.0 * (p["ql"] / 250.0) ** (-FAKE_EXP["nonsour"] / FAKE_BETA)
        ya = p[p["field"] == "Ya"]
        adj = F.extract_layer(ya, "ql")
        assert adj.std() / adj.mean() < 1e-6

    def test_mode_selects_which_covariate_is_removed(self):
        p = _panel()
        ql_adj = F.extract_layer(p, "ql")
        qn_adj = F.extract_layer(p, "qnom")
        assert not np.allclose(ql_adj, qn_adj)

    def test_qnom_mode_names_the_v4_and_v21_layers(self, monkeypatch):
        monkeypatch.undo()
        assert F.LAYERS["qnom"]["Vt"][0] == "Vt v4 θ_Qnom"
        assert F.LAYERS["qnom"]["Ya"][0] == "Ya v2.1 θ_Qnom"
        assert F.LAYERS["qnom"]["Vt"][2] == "qnom"
        assert F.LAYERS["ql"]["Vt"][0] == "Vt v3.2 θ_Ql"

    def test_legacy_alias_still_works(self):
        p = _panel()
        assert np.allclose(F.extract_ql_layer(p), F.extract_layer(p, "ql"))


class TestBuild:
    def test_a_planted_valley_survives_extraction_when_it_is_not_the_rate(self):
        """Frequency curvature independent of rate must still be there afterwards."""
        p = _panel(freq_curv=-1.5, freq_peak=52.0)
        gp = F.build(p)[0]["Ya"]
        assert gp.before_fit.beta2 < -0.5 and gp.before_fit.p_beta2 < 0.01
        assert gp.after_fit.beta2 < -0.5 and gp.after_fit.p_beta2 < 0.01
        assert gp.after_fit.vertex == pytest.approx(52.0, abs=3.0)

    def test_a_curve_that_is_only_the_rate_does_not_survive(self):
        """Make freq a pure proxy for Ql; the extraction must flatten it."""
        rng = np.random.default_rng(5)
        p = _panel()
        p["ql"] = 250.0 * np.exp(-((p["freq"] - 52.0) ** 2) / 60.0)
        p["nno"] = 1200.0 * (p["ql"] / 250.0) ** (-FAKE_EXP["nonsour"] / FAKE_BETA) \
            * np.exp(rng.normal(0, 0.01, len(p)))
        gp = F.build(p)[0]["Ya"]
        assert gp.before_fit.p_beta2 < 0.01                 # a strong raw shape...
        assert abs(gp.after_fit.beta2) < abs(gp.before_fit.beta2) / 5    # ...mostly gone

    def test_panels_share_one_row_set(self):
        gp = F.build(_panel())[0]["Ya"]
        assert len(gp.before) == len(gp.after) == len(gp.residual)
        assert gp.before_fit.n == gp.after_fit.n == gp.residual_fit.n

    def test_out_of_range_frequency_is_counted_not_silently_dropped(self):
        p = _panel()
        p.loc[p.index[:7], "freq"] = 236.0        # the sheet really carries these
        cov = F.build(p)[1].set_index("group")
        assert cov.loc[["Ya", "Vt (all)"], "n_dropped"].sum() == 7

    def test_fit_table_reports_all_three_panels_and_the_layer(self):
        t = F.fit_table(F.build(_panel())[0])
        assert set(t["panel"]) == {"before", "after_ql_extracted", "log_rate_residual"}
        assert set(t["rate_col"]) == {"ql"}
        t_q = F.fit_table(F.build(_panel(), mode="qnom")[0])
        assert set(t_q["rate_col"]) == {"qnom"}

    def test_curvature_not_log_slope_is_the_headline_on_this_axis(self):
        """Frequency layers are tents in Hz — the fit must be quadratic, not per-e-fold."""
        gp = F.build(_panel(freq_curv=-1.5))[0]["Ya"]
        assert hasattr(gp.before_fit, "p_beta2")
        assert gp.before_fit.p_headline == gp.before_fit.p_beta2


class TestFigures:
    def test_every_overlay_renders(self, tmp_path):
        groups, _ = F.build(_panel(freq_curv=-1.0))
        for overlay in F.OVERLAYS:
            out = F.plot(groups, tmp_path / f"{overlay}.png", suptitle="t",
                         overlay=overlay)
            assert out.stat().st_size > 0

    def test_unknown_overlay_is_rejected(self, tmp_path):
        groups, _ = F.build(_panel())
        with pytest.raises(ValueError, match="overlay must be"):
            F.plot(groups, tmp_path / "x.png", suptitle="t", overlay="hexbin")

    def test_before_and_after_share_a_day_axis(self):
        """Both are lives in days, so the pair must be read on one scale."""
        gp = F.build(_panel())[0]["Ya"]
        blim, alim, rlim = F._ylims(gp, "scatter")
        assert blim == alim
        assert rlim == (-blim[1] / 2.0, blim[1] / 2.0)
